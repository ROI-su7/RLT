import numpy as np
from openpi import transforms

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.aloha_policy import AlohaInputs
from rlinf.models.embodiment.openpi.policies.piper_joint_policy import (
    PiperJointInputs,
    PiperJointOutputs,
)

PIPER_TASK = "insert the wire"


def test_piper_transform_round_trip_matches_pi05_wire_action_space() -> None:
    state = np.array(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.04] * 2,
        dtype=np.float32,
    )
    actions = np.stack([state + 0.01, state + 0.02])
    actions[:, [6, 13]] = np.array([[0.03, 0.05], [0.02, 0.06]])
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    inputs = PiperJointInputs(action_dim=32)(
        {
            "observation/state": state,
            "observation/image": image,
            "observation/extra_view_image": np.stack([image, image]),
            "actions": actions,
            "prompt": PIPER_TASK,
        }
    )

    mask = transforms.make_bool_mask(6, -1, 6, -1)
    delta = transforms.DeltaActions(mask)(inputs.copy())
    arm_indices = np.r_[0:6, 7:13]
    arm_signs = np.array([1, -1, -1, 1, 1, 1] * 2)
    np.testing.assert_allclose(
        delta["actions"][:, arm_indices],
        np.stack([0.01 * arm_signs, 0.02 * arm_signs]),
        atol=1e-6,
    )

    absolute = transforms.AbsoluteActions(mask)(delta)
    outputs = PiperJointOutputs()(absolute)
    np.testing.assert_allclose(outputs["actions"], actions, atol=1e-6)


def test_piper_runtime_camera_and_state_contract_matches_aloha_input() -> None:
    """The real-world wrapper must preserve front, left, right and 14-D order."""
    raw_state = np.arange(14, dtype=np.float32)[None]
    front = np.full((1, 8, 9, 3), 11, dtype=np.uint8)
    left = np.full((1, 8, 9, 3), 22, dtype=np.uint8)
    right = np.full((1, 8, 9, 3), 33, dtype=np.uint8)
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.main_image_key = "cam_high"
    env.task_descriptions = [PIPER_TASK]

    wrapped = env._wrap_obs(
        {
            "state": {"joint_position": raw_state},
            "frames": {
                # Deliberately insert keys out of order. RealWorldEnv sorts
                # non-main views before stacking them.
                "cam_right_wrist": right,
                "cam_high": front,
                "cam_left_wrist": left,
            },
        }
    )

    np.testing.assert_array_equal(wrapped["states"].numpy(), raw_state)
    np.testing.assert_array_equal(wrapped["main_images"].numpy(), front)
    np.testing.assert_array_equal(wrapped["extra_view_images"][:, 0].numpy(), left)
    np.testing.assert_array_equal(wrapped["extra_view_images"][:, 1].numpy(), right)

    rlinf_input = PiperJointInputs(action_dim=32)(
        {
            "observation/state": wrapped["states"][0].numpy(),
            "observation/image": wrapped["main_images"][0].numpy(),
            "observation/extra_view_image": wrapped["extra_view_images"][0].numpy(),
            "prompt": PIPER_TASK,
        }
    )
    reference_input = AlohaInputs(adapt_to_pi=True)(
        {
            "state": raw_state[0].copy(),
            "images": {
                "cam_high": front[0].transpose(2, 0, 1),
                "cam_left_wrist": left[0].transpose(2, 0, 1),
                "cam_right_wrist": right[0].transpose(2, 0, 1),
            },
            "prompt": PIPER_TASK,
        }
    )

    np.testing.assert_array_equal(rlinf_input["state"], reference_input["state"])
    assert rlinf_input["prompt"] == reference_input["prompt"]
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        np.testing.assert_array_equal(rlinf_input["image"][key], reference_input["image"][key])


def test_piper_model_contract_matches_source_pi05_wire_checkpoint() -> None:
    config = get_openpi_config("pi05_piper_joint")

    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is True
    assert config.data.default_prompt == PIPER_TASK
