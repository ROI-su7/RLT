"""Data config shared by Piper π0.5 LoRA/RLT SFT and online RLT rollout."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies.piper_joint_policy import (
    PiperJointInputs,
    PiperJointOutputs,
)


@dataclasses.dataclass(frozen=True)
class PiperJointDataConfig(DataConfigFactory):
    """Three-camera, 14-D absolute-joint Piper LeRobot data configuration."""

    # Match the prompt used by the validated examples/piper_real deployment.
    default_prompt: str | None = "insert the wire"
    output_action_dim: int = 14
    # Defaults match the existing FluxVLA Piper LeRobot export. They can be
    # overridden through actor.model.openpi_data without copying the dataset.
    image_key: str = "observation.images.cam_high"
    left_wrist_image_key: str = "observation.images.cam_left_wrist"
    right_wrist_image_key: str = "observation.images.cam_right_wrist"
    state_key: str = "observation.state"
    action_key: str = "action"
    task_key: str | None = "task"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_structure = {
            "observation/image": self.image_key,
            "observation/extra_view_image-0": self.left_wrist_image_key,
            "observation/extra_view_image-1": self.right_wrist_image_key,
            "observation/state": self.state_key,
            "actions": self.action_key,
        }
        if self.task_key is not None:
            repack_structure["task"] = self.task_key
        repack = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )
        transforms = _transforms.Group(
            inputs=[
                PiperJointInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[PiperJointOutputs()],
        )
        # pi05_wire used LeRobotAlohaDataConfig's default delta-joint action
        # representation: six delta arm joints, one absolute gripper, repeated
        # for the right arm.  AbsoluteActions converts model output back to
        # physical joint targets before PiperJointOutputs reverses the ALOHA
        # sign/gripper adaptation.
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
        transforms = transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=transforms,
            model_transforms=model_transforms,
            # This is the source LeRobot column used to build temporal action
            # chunks. RepackTransform below renames it to the model-facing
            # ``actions`` field only after the dataset item is loaded.
            action_sequence_keys=(self.action_key,),
        )
