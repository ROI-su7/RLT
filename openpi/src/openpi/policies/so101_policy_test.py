import numpy as np

from openpi.policies import so101_policy
from openpi.training import config as _config


def test_inputs_outputs_roundtrip_shapes():
    out = so101_policy.So101Inputs()(so101_policy.make_so101_example())
    assert out["state"].shape == (12,)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert so101_policy.So101Outputs()({"actions": np.zeros((50, 32))})["actions"].shape == (50, 12)


def test_missing_wrist_camera_is_masked():
    ex = so101_policy.make_so101_example()
    del ex["images"]["cam_right_wrist"]
    out = so101_policy.So101Inputs()(ex)
    assert not out["image_mask"]["right_wrist_0_rgb"]


def test_train_configs_registered():
    cfg = _config.get_config("pi05_so101_dual_lora")
    assert cfg.data.repo_id == "./lerobot_data/so101_dual"
    assert cfg.model.action_horizon == 50
