"""Piper RECAP data/labels/transforms and policy contracts without ROS or CUDA."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from rlinf.recap_piper.collect import decode_image
from rlinf.recap_piper.data import CAMERAS, EpisodeWriter, export_lerobot
from rlinf.recap_piper.policy import configure_trainable_parameters
from rlinf.recap_piper.preflight import check_dataset
from rlinf.recap_piper.source import native_config


@pytest.fixture(autouse=True)
def isolated_dataset_cache(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf_cache")


def make_frame(index=0, human=False):
    state = np.zeros(14, np.float32)
    state[[6, 13]] = 0.03
    action = state.copy()
    action[0] = 0.01 * index
    return {
        "state": state,
        "action": action,
        "policy_action": action,
        "leader_action": action,
        "timestamp": 10.0 + index / 15,
        "intervention": human,
        "leader_valid": True,
        "policy_valid": True,
        "images": {
            camera: np.full((24, 32, 3), color, np.uint8)
            for camera, color in zip(CAMERAS, (17, 83, 169), strict=True)
        },
    }


@pytest.fixture
def dataset(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "lerobot"
    for success in (True, False):
        writer = EpisodeWriter(raw, "insert the test wire")
        for index in range(4):
            writer.add_frame(make_frame(index, human=index >= 2))
        writer.finish(success)
    aborted = EpisodeWriter(raw, "insert the test wire")
    aborted.add_frame(make_frame())
    aborted.abort("operator quit")
    report = export_lerobot(raw, output, "local/piper_recap_test")
    assert (report["episodes"], report["successes"], report["failures"]) == (2, 1, 1)
    return output


def test_episodes_append_and_exports_never_replace(dataset):
    report = check_dataset(dataset)
    assert report == {"frames": 8, "episodes": 2, "positive_terminal_frames": 1}
    with pytest.raises(FileExistsError):
        export_lerobot(dataset.parent / "raw", dataset, "local/piper_recap_test")
    manifest = json.loads((dataset / "meta/piper_recap_manifest.json").read_text())
    tables = [
        pq.read_table(path).to_pylist()
        for path in sorted((dataset / "data").rglob("*.parquet"))
    ]
    assert sum(sum(bool(row["intervention"]) for row in rows) for rows in tables) == 4
    assert len(manifest["source_episodes"]) == 2


def test_episode_rejects_empty_outcome_and_bad_frames(tmp_path):
    writer = EpisodeWriter(tmp_path, "test")
    with pytest.raises(RuntimeError):
        writer.finish(True)
    bad = make_frame()
    bad["action"][0] = np.nan
    with pytest.raises(ValueError):
        writer.add_frame(bad)
    writer.add_frame(make_frame())
    with pytest.raises(ValueError, match="increasing"):
        writer.add_frame(make_frame())
    writer.abort("test")
    assert not (writer.path / "episode.json").exists()


def test_false_list_is_a_failure_and_returns_feed_value_dataset(dataset):
    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        process_dataset,
    )
    from rlinf.data.datasets.recap.value_dataset import ValueDataset

    process_dataset(
        dataset, None, "rollout", 1.0, -300.0, num_workers=1, tag="piper_recap_v1"
    )
    returns = pq.read_table(dataset / "meta/returns_piper_recap_v1.parquet").to_pydict()
    by_episode = {}
    for ep, fr, value in zip(
        returns["episode_index"], returns["frame_index"], returns["return"], strict=True
    ):
        by_episode.setdefault(ep, {})[fr] = value
    assert sorted(values[3] for values in by_episode.values()) == [-300.0, 0.0]
    assert sorted(values[0] for values in by_episode.values()) == [-303.0, -3.0]
    samples = ValueDataset(
        str(dataset),
        "piper",
        "pi05",
        action_horizon=2,
        action_dim=32,
        tag="piper_recap_v1",
        return_min=-303,
        return_max=0,
    )
    sample = samples[0]
    assert sample["prompt"] == "insert the test wire"
    assert len(sample["images"]) == 3
    # Cameras are exported as lossy AV1 videos, like the demonstrations.
    colours = sorted(
        int(np.asarray(image).reshape(-1)[0]) for image in sample["images"].values()
    )
    assert np.allclose(colours, [17, 83, 169], atol=3), colours
    assert -1 <= sample["target_values"] <= 0
    assert check_dataset(dataset, stage="value")["frames"] == 8


def test_advantage_coverage_and_flag_preservation(dataset):
    from rlinf.data.datasets.recap.cfg_model import AdvantagePreservingDataset

    # Minimal return sidecar is enough for this label-coverage contract.
    keys = [(ep, fr) for ep in range(2) for fr in range(4)]
    fields = {
        "episode_index": [key[0] for key in keys],
        "frame_index": [key[1] for key in keys],
    }
    pq.write_table(
        pa.table({**fields, "return": [-1.0] * 8}),
        dataset / "meta/returns_piper_recap_v1.parquet",
    )
    labels = [index % 2 == 0 for index in range(8)]
    pq.write_table(
        pa.table({**fields, "advantage": labels}),
        dataset / "meta/advantages_piper_recap_adv_v1.parquet",
    )
    assert check_dataset(dataset, stage="cfg")["frames"] == 8
    base = SimpleNamespace(hf_dataset={**fields})

    # The actual wrapper expects a dataset with len/indexing, and derives its
    # index length from the HF dataset rather than a dict's column count.
    class FakeHF:
        column_names = tuple(fields)

        def __len__(self):
            return 8

        def __getitem__(self, name):
            return fields[name]

    base.hf_dataset = FakeHF()
    transformed = [{"state": np.zeros(32)} for _ in range(8)]
    wrapped = AdvantagePreservingDataset(
        base, transformed, dict(zip(keys, labels, strict=True))
    )
    assert [wrapped[index]["advantage"] for index in range(8)] == labels
    pq.write_table(
        pa.table({**fields, "advantage": labels}).slice(1),
        dataset / "meta/advantages_piper_recap_adv_v1.parquet",
    )
    with pytest.raises(ValueError, match="cover every"):
        check_dataset(dataset, stage="cfg")


def test_value_training_and_inference_camera_transforms_agree():
    from openpi import transforms

    from examples.offline_rl.advantage_labeling.recap.process.compute_advantages import (
        build_obs,
    )
    from rlinf.data.datasets.recap.value_dataset import ValueDataset
    from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
        build_input_transforms,
    )

    frame = make_frame()
    sample = {
        "observation.state": frame["state"],
        "action": np.zeros((2, 14), np.float32),
        **{
            f"observation.images.{key}": value for key, value in frame["images"].items()
        },
        "prompt": "task",
        "task": "task",
    }
    train = ValueDataset._build_transform("piper", "pi05", 32, "task")(sample)
    raw = build_obs(sample, "piper", {})
    infer = transforms.compose(
        build_input_transforms("piper", "pi05", 32, "task", None, True)
    )(raw)
    for camera in train["image"]:
        np.testing.assert_array_equal(train["image"][camera], infer["image"][camera])
    np.testing.assert_array_equal(train["state"], infer["state"])


def test_only_action_expert_lora_is_trainable():
    from rlinf.models.embodiment.openpi.native_lora import JaxLoRALinear

    model = torch.nn.Module()
    model.gemma_expert = JaxLoRALinear(torch.nn.Linear(4, 4), rank=2, layout="standard")
    model.vision = torch.nn.Linear(4, 4)
    model.action_out_proj = torch.nn.Linear(4, 4)
    names = configure_trainable_parameters(model, lora_only=True)
    assert set(names) == {"gemma_expert.lora_a", "gemma_expert.lora_b"}
    torch.nn.init.normal_(model.gemma_expert.lora_a, std=0.01)
    before = model.vision.weight.detach().clone()
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model.gemma_expert(torch.randn(2, 4)).square().mean().backward()
    assert model.gemma_expert.lora_b.grad.abs().sum() > 0
    optim.step()
    np.testing.assert_array_equal(before.numpy(), model.vision.weight.detach().numpy())


def test_source_task_is_read_without_executing_file(tmp_path):
    source = tmp_path / "config.py"
    source.write_text(
        'raise RuntimeError("must not run")\nTrainConfig(name="wire", model=Pi0Config(pi05=True), data=DataConfig(default_prompt="exact task"))'
    )
    assert native_config(source, "wire") == ("exact task", {"pi05": True})
    with pytest.raises(ValueError, match="not found"):
        native_config(source, "missing")


def test_raw_ros_bgr_padding_is_decoded_without_cv_bridge():
    message = SimpleNamespace(
        encoding="bgr8", height=1, width=1, step=4, data=bytes([10, 20, 30, 0])
    )
    np.testing.assert_array_equal(decode_image(message), [[[30, 20, 10]]])


def test_returns_do_not_treat_false_list_as_success(tmp_path):
    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        _process_single_parquet,
    )

    path = tmp_path / "list_flags.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
                "is_success": [[False], [False], [False], [True]],
                "task_index": [0, 0, 0, 0],
            }
        ),
        path,
    )
    result = _process_single_parquet(str(path), "rollout", 1.0, -300, {0: "test"})
    assert result.column("return").to_pylist() == [-301, -300, -1, 0]


def test_cfg_websocket_restores_physical_targets_and_state_guidance():
    from openpi import transforms
    from openpi.shared.normalize import NormStats

    from rlinf.models.embodiment.openpi.policies.piper_joint_policy import (
        PiperJointInputs,
        PiperJointOutputs,
    )
    from rlinf.recap_piper.policy import PiperCFGPolicy

    frame = make_frame(1)
    targets = np.tile(frame["action"], (50, 1))
    targets[:, 1] = 0.4
    raw = {
        "observation/state": frame["state"],
        "observation/image": frame["images"][CAMERAS[0]],
        "observation/extra_view_image-0": frame["images"][CAMERAS[1]],
        "observation/extra_view_image-1": frame["images"][CAMERAS[2]],
        "actions": targets,
        "prompt": "exact task",
    }
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    stats = {
        key: NormStats(
            mean=np.zeros(14), std=np.ones(14), q01=-np.ones(14), q99=np.ones(14)
        )
        for key in ("state", "actions")
    }
    prepare = transforms.compose(
        [
            PiperJointInputs(32),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats, use_quantiles=True),
            transforms.PadStatesAndActions(32),
        ]
    )
    prepared = prepare(raw)
    prompts = []

    class Tokenizer:
        def tokenize(self, prompt, state=None):
            assert state is not None and state.shape == (14,)
            prompts.append(prompt)
            return np.ones(8, np.int32), np.ones(8, bool)

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._input_transform = transforms.compose(
                [
                    PiperJointInputs(32),
                    transforms.Normalize(stats, use_quantiles=True),
                    transforms.TokenizePrompt(Tokenizer(), discrete_state_input=True),
                    transforms.PadStatesAndActions(32),
                ]
            )
            self._output_transform = transforms.compose(
                [
                    transforms.Unnormalize(stats, use_quantiles=True),
                    transforms.AbsoluteActions(mask),
                    PiperJointOutputs(),
                ]
            )

        def sample_actions(self, observation):
            assert len(observation.images) == 3
            assert observation.tokenized_positive_guidance_prompt.shape == (1, 8)
            return {"actions": torch.from_numpy(prepared["actions"])[None]}

    policy = PiperCFGPolicy(Dummy(), task="exact task", device="cpu")
    response = policy.infer({"state": frame["state"], "images": frame["images"]})
    assert response["actions"].shape == (50, 14)
    np.testing.assert_allclose(response["actions"], targets, atol=1e-5)
    assert prompts == [
        "exact task",
        "exact task\nAdvantage: positive",
        "exact task\nAdvantage: negative",
    ]


def test_checkpoint_requires_matching_base_task_and_stats(tmp_path):
    from omegaconf import OmegaConf

    from rlinf.recap_piper.data import atomic_json
    from rlinf.recap_piper.policy import policy_provenance, validate_policy_provenance

    base = tmp_path / "base"
    base.mkdir()
    atomic_json(base / "config.json", {"converted_model_sha256": "test-source"})
    norm = tmp_path / "norm_stats.json"
    atomic_json(norm, {"norm_stats": {}})
    cfg = OmegaConf.create(
        {
            "model_path": str(base),
            "recap_lora_only": True,
            "openpi_data": {"norm_stats_path": str(norm), "default_prompt": "task"},
            "openpi": {"positive_only_conditional": True},
        }
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    atomic_json(checkpoint / "piper_recap_policy.json", policy_provenance(cfg))
    validate_policy_provenance(checkpoint, cfg)
    cfg.openpi_data.default_prompt = "wrong task"
    with pytest.raises(ValueError, match="differ"):
        validate_policy_provenance(checkpoint, cfg)


def test_active_gpu_model_is_detected_without_cuda(monkeypatch):
    from rlinf.recap_piper.preflight import gpu_busy_errors

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="123, 9000\n"),
    )
    assert "still active" in gpu_busy_errors()[0]
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="")
    )
    assert gpu_busy_errors() == []


def episode_order(dataset):
    """Map exported episode_index -> success; raw UUID order is not fixed."""
    rows = [
        row
        for path in sorted((dataset / "data").rglob("*.parquet"))
        for row in pq.read_table(
            path, columns=["episode_index", "is_success"]
        ).to_pylist()
    ]
    return {
        int(row["episode_index"]): bool(np.asarray(row["is_success"]).reshape(-1)[0])
        for row in rows
        if bool(np.asarray(row["is_success"]).reshape(-1)[0])
    }


def test_takeover_penalty_charges_step_before_onset(dataset):
    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        compute_returns_for_episode,
        process_dataset,
    )

    human = np.array([False, False, True, True])
    returns, rewards = compute_returns_for_episode(
        4, True, 1.0, -300.0, intervention=human, takeover_penalty=50.0
    )
    assert rewards.tolist() == [-1, -51, -1, 0]
    assert returns.tolist() == [-53, -52, -1, 0]
    # Disabled penalty and episodes that start under human control are unchanged.
    assert compute_returns_for_episode(4, True, 1.0, -300.0, intervention=human)[
        1
    ].tolist() == [-1, -1, -1, 0]
    assert compute_returns_for_episode(
        2, True, 1.0, -300.0, intervention=np.array([True, False]), takeover_penalty=9
    )[1].tolist() == [-1, 0]
    with pytest.raises(ValueError):
        compute_returns_for_episode(4, True, 1.0, -300.0, takeover_penalty=-1)

    # Exported frames 2..3 are human: onset is charged on frame 1.
    process_dataset(
        dataset, None, "rollout", 1.0, -300.0, 1, "piper_recap_v1", takeover_penalty=50
    )
    sidecar = pq.read_table(dataset / "meta/returns_piper_recap_v1.parquet")
    successful = episode_order(dataset)
    by_episode = {}
    for row in sidecar.to_pylist():
        by_episode.setdefault(row["episode_index"], []).append(row["reward"])
    for episode, values in by_episode.items():
        terminal = 0.0 if episode in successful else -300.0
        assert values == [-1, -51, -1, terminal]


def test_interventions_forced_positive_only_from_successes(dataset):
    import pandas as pd

    from examples.offline_rl.advantage_labeling.recap.process.compute_advantages import (
        load_intervention_frames,
        save_advantages_to_dataset,
    )

    (success,) = episode_order(dataset)
    assert load_intervention_frames(dataset) == {(success, 2), (success, 3)}
    assert len(load_intervention_frames(dataset, success_only=False)) == 4

    keys = [(ep, fr) for ep in range(2) for fr in range(4)]
    frame = pd.DataFrame(
        {
            "episode_index": [key[0] for key in keys],
            "frame_index": [key[1] for key in keys],
            "advantage": [0.0, 0.0, 0.0, 1.0] * 2,
        }
    )
    save_advantages_to_dataset(
        dataset,
        frame,
        threshold=0.5,
        tag="forced",
        forced_positive=load_intervention_frames(dataset),
    )
    saved = pd.read_parquet(dataset / "meta/advantages_forced.parquet")
    positive = {
        (int(row.episode_index), int(row.frame_index))
        for row in saved.itertuples()
        if row.advantage
    }
    # Threshold positives (frame 3) plus the successful correction at frame 2.
    assert positive == {(0, 3), (1, 3), (success, 2)}
    assert saved["forced_positive"].sum() == 2


def test_intervention_segments_export_as_success_episodes(tmp_path, dataset):
    from rlinf.recap_piper.data import intervention_segments

    flags = [{"intervention": value} for value in (False, True, True, False, True)]
    assert intervention_segments(flags, 1) == [(1, 3), (4, 5)]
    assert intervention_segments(flags, 2) == [(1, 3)]

    raw = dataset.parent / "raw"
    output = tmp_path / "interventions"
    report = export_lerobot(
        raw, output, "local/piper_int", interventions_only=True, min_segment_frames=2
    )
    # Only the successful episode's two human frames survive, as one episode.
    assert (report["episodes"], report["frames"], report["failures"]) == (1, 2, 0)
    assert report["source_episodes"][0]["frames"] == [2, 4]
    rows = [
        row
        for path in sorted((output / "data").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    assert all(bool(np.asarray(row["intervention"]).reshape(-1)[0]) for row in rows)
    assert [bool(np.asarray(row["is_success"]).reshape(-1)[0]) for row in rows] == [
        False,
        True,
    ]
    assert check_dataset(output) == {
        "frames": 2,
        "episodes": 1,
        "positive_terminal_frames": 1,
    }
    assert check_dataset(output, sft=True)["positive_terminal_frames"] == 1
    with pytest.raises(ValueError, match="No successful human-intervention"):
        export_lerobot(
            raw,
            tmp_path / "none",
            "local/piper_int",
            interventions_only=True,
            min_segment_frames=3,
        )


def test_sft_int_baseline_config_is_plain_unconditional_sft(monkeypatch):
    from pathlib import Path

    import hydra

    from rlinf.recap_piper.cli import SERVE_CONFIGS

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    config_dir = str(root / "examples/offline_rl/config")
    modes = {}
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        for mode, name in SERVE_CONFIGS.items():
            cfg = hydra.compose(config_name=name)
            modes[mode] = cfg
            assert cfg.actor.model.piper_policy_mode == mode
    baseline, recap = modes["sft_int"], modes["recap_cfg"]
    assert baseline.data.plain_sft is True
    assert baseline.actor.model.openpi.guidance_type == "no_guide"
    assert [entry.type for entry in baseline.data.train_data_paths] == ["sft", "sft"]
    # Same base, LoRA selection, horizon and optimizer as the RECAP policy.
    assert baseline.actor.model.recap_lora_only is recap.actor.model.recap_lora_only
    assert baseline.actor.optim == recap.actor.optim
    assert baseline.actor.model.openpi.action_horizon == 50
    assert not recap.data.get("plain_sft", False)


def test_plain_sft_labels_route_every_frame_unconditionally():
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        compute_cfg_routing_masks,
    )

    # plain_sft supplies all-False labels: no frame may see the CFG prompt.
    routing = compute_cfg_routing_masks(
        torch.zeros(64, dtype=torch.bool),
        positive_only_conditional=True,
        unconditional_prob=1.0,
    )
    assert not routing["positive_conditional_mask"].any()
    assert not routing["conditional_mask"].any()
    assert routing["negative_unconditional_mask"].all()


def test_three_insertion_score_gates_success_and_carries_failures():
    from rlinf.recap_piper.scoring import InsertionScore, insertions_per_frame

    score = InsertionScore(require_marks=True)
    score.begin()
    assert not score.can_succeed()
    score.mark(4)
    score.mark(9)
    assert score.done == 2 and not score.can_succeed()
    with pytest.raises(ValueError, match="Success needs 3/3"):
        score.finish(True, 12)
    failed = score.finish(False, 12)
    assert failed == {
        "total_insertions": 3,
        "start_insertions": 0,
        "insertions_done": 2,
        "milestone_frames": [4, 9],
        "marks_complete": True,
        "failed_insertion": 3,
    }
    # The two finished insertions stay in the scene for the continued episode.
    assert score.done == 2
    score.undo()
    assert score.done == 1
    score.mark(0)
    score.begin()
    assert score.start == 2
    score.mark(5)
    assert score.can_succeed()
    with pytest.raises(ValueError, match="already recorded"):
        score.mark(6)
    done = score.finish(True, 8)
    assert (done["start_insertions"], done["failed_insertion"]) == (2, None)
    assert score.done == 0
    assert insertions_per_frame(done, 7) == [2, 2, 2, 2, 2, 3, 3]
    assert insertions_per_frame(None, 2) == [0, 0]
    score.begin()
    score.mark(1)
    score.abort()
    assert score.done == 0
    with pytest.raises(ValueError):
        score.undo()


def test_scored_episodes_export_progress_and_per_insertion_report(tmp_path):
    from rlinf.recap_piper.scoring import InsertionScore, score_report

    raw = tmp_path / "raw"
    score = InsertionScore()
    # Episode 1: insertions at frames 2 and 4, human takeover during the third
    # attempt, then failure. Episode 2 continues from 2/3 and succeeds.
    for frames, marks, humans, success in (
        (6, [2, 4], {5}, False),
        (3, [2], set(), True),
    ):
        writer = EpisodeWriter(raw, "insert the test wire")
        score.begin()
        for index in range(frames):
            if index in marks:
                score.mark(index)
            writer.add_frame(make_frame(index, human=index in humans))
        writer.finish(success, score.finish(success, writer.count))
    report = score_report(raw)
    assert report["scored_episodes"] == 2 and report["successes"] == 1
    assert (report["from_start_episodes"], report["autonomous_successes"]) == (1, 0)
    third = report["per_insertion"]["3"]
    assert (third["attempts"], third["completed"], third["with_takeover"]) == (2, 1, 1)
    assert report["per_insertion"]["1"]["mean_seconds"] == pytest.approx(2 / 15)

    output = tmp_path / "lerobot"
    export_lerobot(raw, output, "local/piper_scored")
    rows = [
        row
        for path in sorted((output / "data").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_index"], []).append(
            (int(row["insertions_done"]), bool(row["insertion_event"]))
        )
    progress = sorted([count for count, _ in values] for values in by_episode.values())
    assert progress == [[0, 0, 1, 1, 2, 2], [2, 2, 3]]
    events = sorted(sum(event for _, event in values) for values in by_episode.values())
    assert events == [1, 2]


def test_failure_reward_grows_with_unfinished_insertions(tmp_path):
    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        _process_single_parquet,
    )

    path = tmp_path / "stages.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1, 2, 2],
                "frame_index": [0, 1, 0, 1, 0, 1],
                "is_success": [[False], [False], [False], [False], [False], [True]],
                "insertions_done": [[0], [0], [1], [2], [2], [3]],
                "task_index": [0] * 6,
            }
        ),
        path,
    )
    result = _process_single_parquet(
        str(path), "rollout", 1.0, -300, {0: "t"}, 0.0, 3, 100.0
    )
    # Failures with 3 and 1 insertions left; the success is unaffected.
    assert result.column("reward").to_pylist() == [-1, -600, -1, -400, -1, 0]
    plain = _process_single_parquet(str(path), "rollout", 1.0, -300, {0: "t"})
    assert plain.column("reward").to_pylist() == [-1, -300, -1, -300, -1, 0]


def test_piper_scoring_config_matches_recap(monkeypatch):
    from pathlib import Path

    import hydra

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    with hydra.initialize_config_dir(
        config_dir=str(root / "examples/offline_rl/config"), version_base="1.1"
    ):
        returns = hydra.compose(config_name="piper_recap_returns").data
        advantage = hydra.compose(config_name="piper_recap_advantages").advantage
    # pi*0.6 eq. 5 with gamma = 1; C_fail exceeds the longest demo (3336 steps).
    assert returns.gamma == 1.0 and returns.failure_reward <= -3337
    # The non-RECAP per-insertion failure cost is off by default.
    assert returns.stage_failure_cost == 0
    # "force I_t = True for all corrections", independent of episode outcome.
    assert advantage.force_intervention_positive
    assert not advantage.intervention_success_only


def test_inference_key_starts_resumes_and_requires_home_after_success():
    from rlinf.recap_piper.scoring import EpisodeGate

    gate = EpisodeGate()
    # I with no episode arms it; it opens only once the mux confirms.
    assert gate.inference_key(recording=False) == "start"
    assert not gate.should_start(recording=False, allowed=False)
    assert gate.should_start(recording=False, allowed=True)
    gate.started()
    # After T (or Space), I resumes the same episode.
    assert gate.inference_key(recording=True) == "resume"
    assert not gate.should_start(recording=False, allowed=True)
    # Failure: I may continue at once into a new episode.
    gate.finished(success=False)
    assert gate.inference_key(recording=False) == "start"
    gate.cancel()
    assert not gate.should_start(recording=False, allowed=True)
    # Success: the scene must be reset via HOME first.
    gate.finished(success=True)
    assert gate.inference_key(recording=False) == "blocked"
    assert not gate.should_start(recording=False, allowed=True)
    gate.homed()
    assert gate.inference_key(recording=False) == "start"


def test_home_after_failure_restarts_insertion_progress():
    from rlinf.recap_piper.scoring import InsertionScore

    score = InsertionScore()
    score.begin()
    score.mark(3)
    score.finish(False, 6)
    assert score.done == 1  # X then I would continue from the current scene
    score.reset_progress()  # X then H: the task restarts from scratch
    score.begin()
    assert score.start == 0


def test_c_and_x_alone_score_episodes_without_insertion_marks(tmp_path):
    from rlinf.recap_piper.scoring import InsertionScore, score_report

    score = InsertionScore()
    score.begin()
    assert score.can_succeed()
    success = score.finish(True, 5)
    assert (success["insertions_done"], success["marks_complete"]) == (3, False)
    score.begin()
    failure = score.finish(False, 5)
    assert failure["failed_insertion"] is None and not failure["marks_complete"]

    raw = tmp_path / "raw"
    for outcome in (True, False):
        writer = EpisodeWriter(raw, "insert the test wire")
        score.begin()
        for index in range(3):
            writer.add_frame(make_frame(index))
        writer.finish(outcome, score.finish(outcome, writer.count))
    report = score_report(raw)
    # Overall RECAP outcomes are counted; per-insertion figures need N marks.
    assert (report["scored_episodes"], report["successes"]) == (2, 1)
    assert report["autonomous_successes"] == 1
    assert report["per_insertion_episodes"] == 0 and report["per_insertion"] == {}


def test_recorder_prompts_reach_terminal_after_ros_logging_reset(tmp_path):
    import io
    import logging

    from rlinf.recap_piper.collect import attach_console

    # Emulate rospy.init_node: the root logger only writes to ~/.ros/log.
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    root.handlers = [logging.FileHandler(tmp_path / "ros.log")]
    target = logging.getLogger("piper_recap_console_test")
    stream = io.StringIO()
    handler = attach_console(target, stream)
    try:
        target.info("I: inference requested")
    finally:
        target.removeHandler(handler)
        root.handlers[0].close()
        root.handlers, root.level = saved
    assert "I: inference requested" in stream.getvalue()


def test_takeover_frames_record_despite_mux_stale_hold_flag():
    import threading

    from rlinf.recap_piper.collect import RosSamples

    samples = RosSamples.__new__(RosSamples)
    samples.lock = threading.RLock()

    def status(mode, phase, active, hold, home=False):
        samples.status = {
            "mode": mode,
            "phase": phase,
            "active": active,
            "hold": hold,
            "home": home,
        }
        return samples.recording_allowed(), samples.control_state()

    # Sequence published by piper_four_state_mux.py around a T press.
    assert status("INFERENCE", "IDLE", True, False) == (True, "INFERENCE/ACTIVE")
    # I resumes via HOLD whenever the mux is in any TAKEOVER phase.
    assert status("HOLD", "POLICY_STOP_REQUEST", False, True) == (False, "HOLD")
    allowed, control = status("TAKEOVER", "SWITCHING_MASTER", False, True)
    assert allowed is False and control.startswith("TAKEOVER")
    # hold_active stays latched True through TAKEOVER: human frames still record.
    assert status("TAKEOVER", "ACTIVE", False, True) == (True, "TAKEOVER/ACTIVE")
    # A real Space/HOLD or HOME stops recording.
    assert status("HOLD", "ACTIVE", False, True)[0] is False
    assert status("TAKEOVER", "ACTIVE", False, True, home=True)[0] is False
    assert status("INFERENCE", "IDLE", True, True)[0] is False
