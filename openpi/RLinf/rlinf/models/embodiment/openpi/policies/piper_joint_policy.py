"""OpenPI transforms for the 14-D absolute-joint Piper interface."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.policies import aloha_policy


def _image_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _image_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = einops.rearrange(image, "h w c -> c h w")
    return image


def _extra_views(data: dict) -> tuple[np.ndarray, np.ndarray]:
    stacked = data.get("observation/extra_view_image")
    if stacked is not None:
        stacked = np.asarray(stacked)
        if stacked.shape[0] != 2:
            raise ValueError(f"Expected two Piper extra views, got {stacked.shape}.")
        return _image_hwc(stacked[0]), _image_hwc(stacked[1])
    return _image_hwc(data["observation/extra_view_image-0"]), _image_hwc(
        data["observation/extra_view_image-1"]
    )


@dataclasses.dataclass(frozen=True)
class PiperJointInputs(transforms.DataTransformFn):
    """Map RLinf Piper observations to the three-view OpenPI input schema."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-D Piper state, got {state.shape}.")
        # ``observation/image`` is the fixed/front camera.  The two extra
        # views are left wrist then right wrist, matching the LeRobot fields
        # in ``PiperJointDataConfig``.  Keeping this ordering is essential:
        # swapping front and left wrist makes a converted pi05_wire checkpoint
        # see a different camera layout from its LoRA training data.
        left_wrist, right_wrist = _extra_views(data)
        aloha_data = {
            "state": state,
            "images": {
                "cam_high": _image_chw(data["observation/image"]),
                "cam_left_wrist": _image_chw(left_wrist),
                "cam_right_wrist": _image_chw(right_wrist),
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 14:
                raise ValueError(
                    f"Expected Piper actions [horizon, 14], got {actions.shape}."
                )
            aloha_data["actions"] = actions
        if "prompt" in data:
            aloha_data["prompt"] = (
                data["prompt"].decode()
                if isinstance(data["prompt"], bytes)
                else data["prompt"]
            )
        # The source pi05_wire checkpoint was trained through
        # LeRobotAlohaDataConfig(adapt_to_pi=True).  Reusing that exact joint
        # sign and gripper transform is essential before applying its stats.
        return aloha_policy.AlohaInputs(adapt_to_pi=True)(aloha_data)


@dataclasses.dataclass(frozen=True)
class PiperJointOutputs(transforms.DataTransformFn):
    """Remove OpenPI action padding before sending commands to Piper."""

    def __call__(self, data: dict) -> dict:
        return aloha_policy.AlohaOutputs(adapt_to_pi=True)(data)
