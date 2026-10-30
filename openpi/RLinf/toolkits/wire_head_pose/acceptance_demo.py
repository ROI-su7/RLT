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

"""Does the head-pose feature tell where to correct? Offline demo test.

On demonstrations the detector never saw, take right-wrist frames of the
first connection from ``--far-gap-mm`` before contact down to contact. At
each frame, predict the operator's arm motion over the next ``--horizon-s``
(12 joints) with grouped (per-demo) cross-validated ridge regression from:

    constant      mean motion only (the "average correction" baseline)
    proprio       joint angles
    head pose     detector features (tip offset, axis angles, gap)
    both          joint angles + detector features

and score the cosine between predicted and actual motion, on the full motion
and on its part that deviates from the mean motion (where the constant
baseline scores zero by construction).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import av
import numpy as np
import pandas as pd
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from rlinf.envs.realworld.piper.wire_head_pose import (  # noqa: E402
    HeadPoseNet,
    head_features,
    predict,
)
from toolkits.wire_head_pose.extract_contact_frames import (  # noqa: E402
    approach_frame,
    insertion_events,
)
from toolkits.wire_head_pose.piper_fk import distance_to_mated  # noqa: E402

ARM = list(range(6)) + list(range(7, 13))


def ridge_cv(x, y, groups, lam, folds=5, seed=0):
    ep = np.random.default_rng(seed).permutation(np.unique(groups))
    fold = {e: k % folds for k, e in enumerate(ep)}
    pred = np.zeros_like(y)
    for k in range(folds):
        te = np.array([fold[g] == k for g in groups])
        if x.shape[1] == 0:
            pred[te] = y[~te].mean(0)
            continue
        mu, sd = x[~te].mean(0), x[~te].std(0) + 1e-9
        xtr, xte = (x[~te] - mu) / sd, (x[te] - mu) / sd
        ym = y[~te].mean(0)
        w = np.linalg.solve(
            xtr.T @ xtr + lam * np.eye(x.shape[1]), xtr.T @ (y[~te] - ym)
        )
        pred[te] = xte @ w + ym
    return pred


def cosines(pred, y):
    return (pred * y).sum(1) / (
        np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1) + 1e-12
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset", default=os.path.expanduser("~/rlt_data/lerobot_data/all_wire_data")
    )
    parser.add_argument("--root", default=os.path.expanduser("~/rlt_data/wire_head_pose"))
    parser.add_argument("--far-gap-mm", type=float, default=20.0)
    parser.add_argument("--step", type=int, default=3, help="frames between samples")
    parser.add_argument("--horizon-s", type=float, default=0.5)
    parser.add_argument(
        "--cache", default=os.path.expanduser("~/rlt_data/wire_head_pose/acceptance_demo.npz")
    )
    args = parser.parse_args()

    if os.path.exists(args.cache):
        data = np.load(args.cache)
        feats, prop, motion, groups = (
            data["feats"],
            data["prop"],
            data["motion"],
            data["groups"],
        )
        frames = data["frames"]
    else:
        info = json.load(open(os.path.join(args.dataset, "meta", "info.json")))
        fps = float(info["fps"])
        horizon = int(round(args.horizon_s * fps))
        with open(os.path.join(args.root, "manifest.jsonl")) as stream:
            seen = {json.loads(line)["episode"] for line in stream}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HeadPoseNet(pretrained=False).to(device)
        model.load_state_dict(
            torch.load(os.path.join(args.root, "head_pose_net.pt"), map_location=device)
        )
        model.eval()
        feats, prop, motion, groups, frames = [], [], [], [], []
        for episode in range(int(info["total_episodes"])):
            if episode in seen:
                continue
            chunk = episode // int(info["chunks_size"])
            states = np.stack(
                pd.read_parquet(
                    os.path.join(
                        args.dataset,
                        info["data_path"].format(
                            episode_chunk=chunk, episode_index=episode
                        ),
                    )
                )["observation.state"].to_numpy()
            )
            centers = insertion_events(distance_to_mated(states), fps, 60.0)
            if len(centers) != 3:
                continue
            center = centers[0]
            start = approach_frame(
                distance_to_mated(states), center, args.far_gap_mm, int(8 * fps)
            )
            if start is None:
                continue
            wanted = set(range(start, center + 1, args.step))
            video = os.path.join(
                args.dataset,
                info["video_path"].format(
                    episode_chunk=chunk,
                    video_key="observation.images.cam_right_wrist",
                    episode_index=episode,
                ),
            )
            with av.open(video) as container:
                for index, frame in enumerate(
                    container.decode(container.streams.video[0])
                ):
                    if index in wanted and index + horizon < len(states):
                        f = head_features(
                            predict(model, frame.to_ndarray(format="bgr24"), device)
                        )
                        if f is not None:
                            feats.append(f)
                            prop.append(states[index, ARM])
                            motion.append(
                                states[index + horizon, ARM] - states[index, ARM]
                            )
                            groups.append(episode)
                            frames.append(index)
                    if index >= center:
                        break
        feats, prop, motion, groups, frames = map(
            np.array, (feats, prop, motion, groups, frames)
        )
        np.savez(
            args.cache,
            feats=feats,
            prop=prop,
            motion=motion,
            groups=groups,
            frames=frames,
        )

    print(f"frames {len(motion)} from {len(np.unique(groups))} unseen demos")
    # Does the operator remove the offset the detector sees? Offset now vs its
    # change over the horizon, both measured by the detector.
    horizon_frames = int(round(args.horizon_s * 30))
    lookup = {(g, f): k for k, (g, f) in enumerate(zip(groups, frames))}
    now, change = [], []
    for k, (g, f) in enumerate(zip(groups, frames)):
        later = lookup.get((g, f + horizon_frames))
        if later is not None:
            now.append(feats[k, :2])
            change.append(feats[later, :2] - feats[k, :2])
    now, change = np.array(now), np.array(change)
    for axis, name in ((0, "x (lateral)"), (1, "y (along)")):
        r = np.corrcoef(now[:, axis], change[:, axis])[0, 1]
        slope = np.polyfit(now[:, axis], change[:, axis], 1)[0]
        print(
            f"tip offset {name}: corr(offset, change over {args.horizon_s}s) {r:+.2f}, slope {slope:+.2f} (n={len(now)})"
        )
    keep = np.linalg.norm(motion, axis=1) > 1e-3  # the arms actually moved
    feats, prop, motion, groups = feats[keep], prop[keep], motion[keep], groups[keep]
    print(
        f"moving frames {len(motion)}; median |dx| {np.median(np.abs(feats[:, 0])):.1f}px"
    )
    inputs = {
        "constant": np.zeros((len(motion), 0)),
        "proprio": prop,
        "head pose": feats,
        "both": np.c_[prop, feats],
    }
    print(
        f"{'input':10s} {'cos(full)':>10s} {'same-sign':>10s} {'cos(deviation)':>15s} {'R2(deviation)':>14s}"
    )
    for name, x in inputs.items():
        full, dev, r2 = [], [], []
        for seed in range(5):
            pred = ridge_cv(x, motion, groups, lam=10.0, seed=seed)
            mean = motion.mean(0)
            full.append(cosines(pred, motion))
            dev.append(cosines(pred - mean, motion - mean))
            r2.append(1 - ((pred - motion) ** 2).sum() / ((motion - mean) ** 2).sum())
        full, dev = np.concatenate(full), np.concatenate(dev)
        print(
            f"{name:10s} {full.mean():+10.2f} {100 * (full > 0).mean():9.0f}% "
            f"{dev.mean():+15.2f} {np.mean(r2):+14.3f}"
        )


if __name__ == "__main__":
    main()
