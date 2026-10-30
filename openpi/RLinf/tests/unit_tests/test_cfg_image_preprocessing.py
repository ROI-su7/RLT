"""CFG image augmentation must accept mixed-precision FSDP inputs."""

from types import SimpleNamespace

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation,
    OpenPi0ForCFGActionPrediction,
)


def test_cfg_preprocess_bfloat16_images_with_rotation(monkeypatch):
    # Force a non-zero rotation, so the real OpenPI grid_sample path runs.
    def fixed_rand(*args, **kwargs):
        return torch.full(args, 0.75, device=kwargs.get("device"))

    monkeypatch.setattr(torch, "rand", fixed_rand)
    images = {
        k: torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16)
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    obs = Observation(
        images=images,
        image_masks={k: torch.ones(1, dtype=torch.bool) for k in images},
        state=torch.zeros((1, 32)),
    )
    result = OpenPi0ForCFGActionPrediction._preprocess_observation(
        None, obs, train=True
    )
    assert all(image.dtype == torch.float32 for image in result[0])
    assert all(image.shape == (1, 3, 224, 224) for image in result[0])
    assert all(image.dtype == torch.bfloat16 for image in obs.images.values())


def test_bfloat16_suffix_projections_accept_float32_flow():
    class FakeModel:
        pi05 = True
        config = SimpleNamespace(action_horizon=3)

        def __init__(self):
            self.action_in_proj = torch.nn.Linear(4, 8).to(torch.bfloat16)
            self.time_mlp_in = torch.nn.Linear(8, 8).to(torch.bfloat16)
            self.time_mlp_out = torch.nn.Linear(8, 8).to(torch.bfloat16)

        @staticmethod
        def _apply_checkpoint(fn, *args):
            return fn(*args)

        embed_suffix = PI0Pytorch.embed_suffix

    model = FakeModel()
    outputs = OpenPi0ForCFGActionPrediction._embed_suffix_mixed_precision(
        model, torch.zeros((2, 4)), torch.zeros((2, 3, 4)), torch.ones(2)
    )
    assert outputs[0].dtype == torch.bfloat16
    assert outputs[3].dtype == torch.bfloat16
