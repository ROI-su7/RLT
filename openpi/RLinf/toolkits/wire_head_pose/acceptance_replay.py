#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Head pose on real Stage-2 transitions: what is misaligned when, and who fixes it.

Joins every replay transition of a run to its logged right-wrist frame
(``rlt_image_log.match_frame``), runs the head-pose detector and reports,
per episode outcome, the tip offset over the approach and at the moment the
operator takes over, and whether the Actor's / operator's corrections follow
the measured offset.

    python toolkits/wire_head_pose/acceptance_replay.py --run logs/<run>
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from rlinf.data.rlt_image_log import load_frames, match_frame  # noqa: E402
from rlinf.envs.realworld.piper.wire_head_pose import (  # noqa: E402
    HeadPoseNet,
    head_features,
    predict,
)

ARM = list(range(6)) + list(range(7, 13))
NAMES = [f"L{j}" for j in range(1, 7)] + [f"R{j}" for j in range(1, 7)]


def load_run(run: str, weights: str, first_id: int):
    replay = os.path.join(run, "replay_buffer", "rank_0")
    ti = json.load(open(os.path.join(replay, "trajectory_index.json")))[
        "trajectory_index"
    ]
    files = {
        int(os.path.basename(f).split("_")[1]): f
        for f in glob.glob(os.path.join(replay, "trajectory_*.pt"))
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeadPoseNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    frames_by_ep: dict[int, list] = {}
    rows = []
    for tid in sorted(int(k) for k in ti if int(k) >= first_id):
        info = ti[str(tid)]
        ep = info["rlt_episode_id"]
        ep_dir = os.path.join(run, "rlt_images", f"episode_{ep}")
        if not os.path.isdir(ep_dir):
            continue
        if ep not in frames_by_ep:
            frames_by_ep[ep] = load_frames(ep_dir)
        t = torch.load(files[tid], map_location="cpu", weights_only=False)
        hit = match_frame(
            frames_by_ep[ep], t["curr_obs"]["proprio"].reshape(-1).numpy()
        )
        feat = pred = None
        if hit is not None and "cam_right_wrist" in hit["cameras"]:
            image = cv2.imread(
                os.path.join(ep_dir, f"{hit['frame']:06d}_cam_right_wrist.jpg")
            )
            pred = predict(model, image, device)
            feat = head_features(pred)
        a = t["actions"].reshape(10, 14).numpy()
        ref = t["curr_obs"]["ref_chunk"].reshape(-1, 14)[:10].numpy()
        rows.append(
            {
                "tid": tid,
                "ep": ep,
                "human": bool(info["rlt_human"]),
                "online": bool(info["rlt_online"]),
                "reward": float(t["rewards"].sum()),
                "feat": feat,
                "pred": pred,
                "corr": (a - ref).mean(0)[ARM],
                "frame": None if hit is None else hit["frame"],
                "z": t["curr_obs"]["z_rl"].reshape(-1).numpy(),
                "proprio": t["curr_obs"]["proprio"].reshape(-1).numpy(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--weights", default=os.path.expanduser("~/rlt_data/wire_head_pose/head_pose_net.pt")
    )
    parser.add_argument(
        "--first-id", type=int, default=0, help="first replay id of this run"
    )
    parser.add_argument(
        "--sheet", default="", help="write a takeover contact sheet here"
    )
    args = parser.parse_args()

    rows = load_run(args.run, args.weights, args.first_id)
    eps = collections.OrderedDict()
    for r in rows:
        eps.setdefault(r["ep"], []).append(r)
    print(
        f"transitions with a matched frame: {sum(r['frame'] is not None for r in rows)}/{len(rows)}, "
        f"head pose visible: {sum(r['feat'] is not None for r in rows)}, episodes {len(eps)}"
    )

    summary = collections.defaultdict(list)
    takeovers = []
    for ep, rs in eps.items():
        took = any(r["human"] for r in rs)
        success = rs[-1]["reward"] > 0
        outcome = "takeover" if took else ("auto" if success else "fail")
        first_h = next((k for k, r in enumerate(rs) if r["human"]), len(rs))
        machine = [r for r in rs[:first_h] if not r["human"] and r["feat"] is not None]
        if machine:
            last = machine[-1]["feat"]
            summary[outcome].append((abs(last[0]), last[1], abs(last[3])))
        if took and machine:
            hum = [r for r in rs[first_h:] if r["human"]][:4]
            takeovers.append(
                {
                    "ep": ep,
                    "last": machine[-1],
                    "machine": machine[-4:],
                    "human": np.mean([r["corr"] for r in hum], 0) if hum else None,
                }
            )

    print(
        "\nLast machine-executed frame before takeover / end (detector, px; angle rad):"
    )
    for outcome, vals in summary.items():
        v = np.array(vals)
        print(
            f"  {outcome:8s} n={len(v):2d} |lateral| median {np.median(v[:, 0]):5.1f}  "
            f"along (other - own, y) median {np.median(v[:, 1]):6.1f}  |axis angle| median {np.median(v[:, 2]):.3f}"
        )

    # Does the correction follow the measured lateral offset?
    def corr_with_dx(rs, label):
        rs = [r for r in rs if r["feat"] is not None and abs(r["feat"][1]) < 40]
        if len(rs) < 8:
            print(f"  {label}: too few near-contact rows ({len(rs)})")
            return
        dx = np.array([r["feat"][0] for r in rs])
        c = np.array([r["corr"] for r in rs])
        rr = [np.corrcoef(dx, c[:, j])[0, 1] for j in range(12)]
        top = np.argsort(-np.abs(rr))[:4]
        print(
            f"  {label} (n={len(rs)}): corr(lateral offset, correction) strongest "
            + ", ".join(f"{NAMES[j]} {rr[j]:+.2f}" for j in top)
            + f" | mean |corr| {np.mean(np.abs(rr)):.2f}"
        )

    print("\nNear contact (|along| < 40 px):")
    corr_with_dx([r for r in rows if r["online"] and not r["human"]], "Actor rows   ")
    corr_with_dx([r for r in rows if r["human"]], "Operator rows")
    corr_with_dx(
        [r for r in rows if not r["online"] and not r["human"]], "JAX rows     "
    )

    if args.sheet and takeovers:
        tiles = []
        for t in takeovers[:16]:
            r = t["last"]
            ep_dir = os.path.join(args.run, "rlt_images", f"episode_{r['ep']}")
            im = cv2.imread(
                os.path.join(ep_dir, f"{r['frame']:06d}_cam_right_wrist.jpg")
            )
            for k, c in (
                ("own_tip", (0, 255, 0)),
                ("own_back", (0, 255, 0)),
                ("other_tip", (0, 0, 255)),
                ("other_back", (0, 0, 255)),
            ):
                x, y = (int(v) for v in r["pred"][k]["xy"])
                cv2.circle(im, (x, y), 4, c, -1 if k.endswith("tip") else 1)
            cv2.line(
                im,
                tuple(int(v) for v in r["pred"]["own_tip"]["xy"]),
                tuple(int(v) for v in r["pred"]["own_back"]["xy"]),
                (0, 255, 0),
                1,
            )
            cv2.line(
                im,
                tuple(int(v) for v in r["pred"]["other_tip"]["xy"]),
                tuple(int(v) for v in r["pred"]["other_back"]["xy"]),
                (0, 0, 255),
                1,
            )
            crop = cv2.resize(im[120:420, 170:470], (300, 300))
            f = r["feat"]
            cv2.putText(
                crop,
                f"dx {f[0]:+.0f} dy {f[1]:+.0f} ang {f[3]:+.2f}",
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            tiles.append(crop)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(
            args.sheet,
            np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]),
        )
        print(f"\ntakeover sheet: {args.sheet}")


if __name__ == "__main__":
    main()
