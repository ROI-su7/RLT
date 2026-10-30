import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

# Per arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (LeRobot degrees, gripper 0-100).
ARM_DIM = 6
STATE_DIM = 2 * ARM_DIM  # [left arm (6), right arm (6)]


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 dual-arm policy."""
    return {
        "state": np.ones((STATE_DIM,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class So101Inputs(transforms.DataTransformFn):
    """Inputs for the SO101 dual-arm policy.

    Expected inputs:
    - images: dict[name, img], img is [channel, height, width]; name in EXPECTED_CAMERAS
    - state: [12]
    - actions: [action_horizon, 12]
    The model pads state/actions to its action_dim, so no padding is done here.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = {name: _to_hwc(img) for name, img in data["images"].items()}
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, source in {"left_wrist_0_rgb": "cam_left_wrist", "right_wrist_0_rgb": "cam_right_wrist"}.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {"image": images, "image_mask": image_masks, "state": np.asarray(data["state"])}
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class So101Outputs(transforms.DataTransformFn):
    """Outputs for the SO101 dual-arm policy: keep the first 12 action dims."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :STATE_DIM])}


def _to_hwc(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")
