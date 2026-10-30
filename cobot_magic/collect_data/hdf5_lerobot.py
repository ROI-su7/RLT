"""
Convert cobot_magic/Aloha-style HDF5 episodes to LeRobot format.

This script matches the HDF5 layout written by collect_data.py:
  /observations/qpos
  /observations/qvel
  /observations/effort
  /observations/images/{camera}
  /action

Example:
  python $RLT_ROOT/cobot_magic/collect_data/hdf5_lerobot.py \
      --datasetDir ~/rlt_data/aloha_mobile_dummy \
      --datasetName local/cobot_magic \
      --targetDir ~/rlt_data/lerobot_data \
      --instruction "Fold the cloth"
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 8
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    images: dict[str, np.ndarray]
    state: torch.Tensor
    action: torch.Tensor
    velocity: torch.Tensor | None = None
    effort: torch.Tensor | None = None
    policy_action: torch.Tensor | None = None
    leader_action: torch.Tensor | None = None
    intervention: torch.Tensor | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_MOTORS = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.as_posix())]


def find_episode_files(dataset_dir: Path) -> list[Path]:
    episode_files = sorted(dataset_dir.glob("episode_*.hdf5"), key=natural_key)
    if episode_files:
        return episode_files

    nested_episode_files = sorted(dataset_dir.glob("**/episode_*.hdf5"), key=natural_key)
    if nested_episode_files:
        return nested_episode_files

    return sorted(dataset_dir.glob("**/data.hdf5"), key=natural_key)


def get_cameras(ep_path: Path) -> list[str]:
    with h5py.File(ep_path, "r") as ep:
        if "/observations/images" not in ep:
            raise KeyError(f"{ep_path} is missing /observations/images")
        return sorted(ep["/observations/images"].keys())


def parse_camera_renames(values: list[str]) -> dict[str, str]:
    renames = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Camera rename must be OLD=NEW, got: {value}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"Camera rename must be OLD=NEW, got: {value}")
        renames[old] = new
    return renames


def _as_float_tensor(dataset: h5py.Dataset, expected_width: int, name: str) -> torch.Tensor:
    array = dataset[:]
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(f"{name} must have shape (T, {expected_width}), got {array.shape}")
    return torch.from_numpy(array.astype(np.float32, copy=False))


def load_images(ep: h5py.File, camera: str) -> np.ndarray:
    image_dataset = ep[f"/observations/images/{camera}"]
    if image_dataset.ndim == 4:
        images = image_dataset[:]
    else:
        import cv2

        decoded_images = []
        for encoded_image in image_dataset:
            image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode compressed image for camera {camera}")
            decoded_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        images = np.asarray(decoded_images, dtype=np.uint8)

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"/observations/images/{camera} must have shape (T, H, W, 3), got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"/observations/images/{camera} must be uint8, got {images.dtype}")
    return images


def load_episode(
    ep_path: Path,
    cameras: list[str],
    *,
    include_velocity: bool = False,
    include_effort: bool = False,
    include_intervention: bool = False,
    camera_renames: dict[str, str] | None = None,
) -> EpisodeData:
    camera_renames = camera_renames or {}
    with h5py.File(ep_path, "r") as ep:
        required_paths = ["/observations/qpos", "/action", "/observations/images"]
        missing = [path for path in required_paths if path not in ep]
        if missing:
            raise KeyError(f"{ep_path} is missing required datasets/groups: {missing}")

        state = _as_float_tensor(ep["/observations/qpos"], len(DEFAULT_MOTORS), "/observations/qpos")
        action = _as_float_tensor(ep["/action"], len(DEFAULT_MOTORS), "/action")
        num_frames = state.shape[0]
        if action.shape[0] != num_frames:
            raise ValueError(f"{ep_path} has {num_frames} qpos frames but {action.shape[0]} action frames")

        velocity = None
        if include_velocity:
            if "/observations/qvel" not in ep:
                raise KeyError(f"{ep_path} is missing /observations/qvel")
            velocity = _as_float_tensor(ep["/observations/qvel"], len(DEFAULT_MOTORS), "/observations/qvel")

        effort = None
        if include_effort:
            if "/observations/effort" not in ep:
                raise KeyError(f"{ep_path} is missing /observations/effort")
            effort = _as_float_tensor(ep["/observations/effort"], len(DEFAULT_MOTORS), "/observations/effort")

        policy_action = None
        leader_action = None
        intervention = None
        if include_intervention:
            missing_intervention = [
                path
                for path in ("/policy_action", "/leader_action", "/intervention_mask")
                if path not in ep
            ]
            if missing_intervention:
                raise KeyError(
                    f"{ep_path} is missing {missing_intervention}; it was not recorded by "
                    "takeover_collect_data.py"
                )
            policy_action = _as_float_tensor(ep["/policy_action"], len(DEFAULT_MOTORS), "/policy_action")
            leader_action = _as_float_tensor(ep["/leader_action"], len(DEFAULT_MOTORS), "/leader_action")
            mask = ep["/intervention_mask"][:]
            if mask.shape != (num_frames,):
                raise ValueError(
                    f"/intervention_mask must have shape ({num_frames},), got {mask.shape}"
                )
            intervention = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(1)

        images = {}
        for camera in cameras:
            image_path = f"/observations/images/{camera}"
            if image_path not in ep:
                raise KeyError(f"{ep_path} is missing {image_path}")

            camera_images = load_images(ep, camera)
            if camera_images.shape[0] != num_frames:
                raise ValueError(
                    f"{ep_path} camera {camera} has {camera_images.shape[0]} frames, expected {num_frames}"
                )
            images[camera_renames.get(camera, camera)] = camera_images

    return EpisodeData(
        images=images,
        state=state,
        action=action,
        velocity=velocity,
        effort=effort,
        policy_action=policy_action,
        leader_action=leader_action,
        intervention=intervention,
    )


def create_empty_dataset(
    *,
    repo_id: str,
    root: Path,
    robot_type: str,
    fps: int,
    cameras: list[str],
    include_velocity: bool,
    include_effort: bool,
    include_intervention: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [list(DEFAULT_MOTORS)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [list(DEFAULT_MOTORS)],
        },
    }
    if include_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [list(DEFAULT_MOTORS)],
        }
    if include_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [list(DEFAULT_MOTORS)],
        }

    if include_intervention:
        # policy_action is what the model wanted at that frame and action is what
        # the mux actually executed. During TAKEOVER they differ, and that pair
        # plus the mask is what intervention retraining consumes.
        for name in ("policy_action", "leader_action"):
            features[name] = {
                "dtype": "float32",
                "shape": (len(DEFAULT_MOTORS),),
                "names": [list(DEFAULT_MOTORS)],
            }
        features["intervention"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_takeover"],
        }

    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if root.exists():
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    cameras: list[str],
    *,
    include_velocity: bool,
    include_effort: bool,
    include_intervention: bool,
    camera_renames: dict[str, str],
) -> LeRobotDataset:
    for ep_path in tqdm.tqdm(hdf5_files, desc="Converting episodes"):
        episode = load_episode(
            ep_path,
            cameras,
            include_velocity=include_velocity,
            include_effort=include_effort,
            include_intervention=include_intervention,
            camera_renames=camera_renames,
        )
        for frame_idx in range(episode.state.shape[0]):
            frame = {
                "observation.state": episode.state[frame_idx],
                "action": episode.action[frame_idx],
                "task": task,
            }
            for camera, images in episode.images.items():
                frame[f"observation.images.{camera}"] = images[frame_idx]
            if episode.velocity is not None:
                frame["observation.velocity"] = episode.velocity[frame_idx]
            if episode.effort is not None:
                frame["observation.effort"] = episode.effort[frame_idx]
            if episode.policy_action is not None:
                frame["policy_action"] = episode.policy_action[frame_idx]
                frame["leader_action"] = episode.leader_action[frame_idx]
                frame["intervention"] = episode.intervention[frame_idx]
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def convert(args: argparse.Namespace) -> LeRobotDataset:
    dataset_dir = Path(args.datasetDir).expanduser()
    target_dir = Path(args.targetDir).expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Input dataset directory does not exist: {dataset_dir}")
    if dataset_dir.resolve() == target_dir.resolve():
        raise ValueError("--targetDir must be different from --datasetDir because it is recreated from scratch")

    hdf5_files = find_episode_files(dataset_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 or data.hdf5 files found under {dataset_dir}")

    camera_renames = parse_camera_renames(args.renameCamera)
    input_cameras = args.cameraNames or get_cameras(hdf5_files[0])
    output_cameras = [camera_renames.get(camera, camera) for camera in input_cameras]

    print(f"Found {len(hdf5_files)} HDF5 episodes")
    print(f"Input cameras: {input_cameras}")
    if camera_renames:
        print(f"Camera renames: {camera_renames}")
    print(f"Output cameras: {output_cameras}")

    dataset = create_empty_dataset(
        repo_id=args.datasetName,
        root=target_dir,
        robot_type=args.robotType,
        fps=args.fps,
        cameras=output_cameras,
        include_velocity=args.includeVelocity,
        include_effort=args.includeEffort,
        include_intervention=args.includeIntervention,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        args.instruction,
        input_cameras,
        include_velocity=args.includeVelocity,
        include_effort=args.includeEffort,
        include_intervention=args.includeIntervention,
        camera_renames=camera_renames,
    )
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=str, required=True, help="Directory containing episode_*.hdf5 files.")
    parser.add_argument("--datasetName", type=str, required=True, help="LeRobot repo_id/dataset name.")
    parser.add_argument("--targetDir", type=str, required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--instruction", type=str, default="DEBUG", help="Task language instruction saved per episode.")
    parser.add_argument("--robotType", type=str, default="cobot_magic")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cameraNames", nargs="+", default=None, help="Input camera names in the HDF5 file.")
    parser.add_argument(
        "--renameCamera",
        nargs="*",
        default=[],
        help="Optional camera renames, e.g. cam_high=cam_front cam_left_wrist=cam_left.",
    )
    parser.add_argument("--includeVelocity", action="store_true", help="Include /observations/qvel.")
    parser.add_argument("--includeEffort", action="store_true", help="Include /observations/effort.")
    parser.add_argument(
        "--includeIntervention",
        action="store_true",
        help="Include /policy_action, /leader_action and /intervention_mask from "
        "takeover_collect_data.py episodes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = convert(args)
    print(f"Created LeRobot dataset at {dataset.root}")
    print(f"Episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
