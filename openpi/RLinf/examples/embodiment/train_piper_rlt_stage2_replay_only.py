"""Train Piper RLT Stage 2 from persisted replay without ROS or robot rollout."""

from __future__ import annotations

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.rlt_replay_only_runner import RLTReplayOnlyRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_piper_rlt_stage2_ac_mlp",
)
def main(cfg) -> None:
    replay_path = cfg.algorithm.replay_buffer.get("load_path", None)
    if not replay_path:
        raise ValueError(
            "Replay-only Stage 2 requires RLINF_REPLAY_LOAD_PATH or "
            "algorithm.replay_buffer.load_path."
        )
    replay_path = os.path.abspath(str(replay_path))
    for filename in ("metadata.json", "trajectory_index.json"):
        if not os.path.isfile(os.path.join(replay_path, filename)):
            raise FileNotFoundError(
                f"Persisted replay is incomplete: {os.path.join(replay_path, filename)}"
            )

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    actor = RLTACFSDPPolicy.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement,
    )
    actor.init_worker().wait()
    RLTReplayOnlyRunner(cfg, actor).run()


if __name__ == "__main__":
    main()
