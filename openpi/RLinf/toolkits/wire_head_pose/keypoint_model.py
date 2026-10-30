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

"""Keypoint detector for the two connector heads in the right wrist camera.

ResNet-18 (ImageNet weights) up to stride 16, upsampled to stride 4, predicts
one heatmap per keypoint (``annotate.POINTS``) and one visibility logit each.
The input is the central 320x480 strip of the 640x480 frame, where the own
head sits and the other head arrives.

    # grouped cross-validation (pixel errors on held-out demos)
    python toolkits/wire_head_pose/keypoint_model.py cv
    # train on all labels and save the weights
    python toolkits/wire_head_pose/keypoint_model.py train --out model.pt
    # predict unlabeled images into <root>/predictions.jsonl
    python toolkits/wire_head_pose/keypoint_model.py predict --weights model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from rlinf.envs.realworld.piper.wire_head_pose import (
    CROP_H,
    CROP_W,
    CROP_X0,
    POINTS,
    STRIDE,
    HeadPoseNet,
    normalize_crop,
    predict,
)  # noqa: E402

SIGMA = 2.0  # heatmap cells


def load_rows(root: str) -> list[dict]:
    """Latest non-skipped label per image with its manifest row."""
    manifest = {}
    with open(os.path.join(root, "manifest.jsonl")) as stream:
        for line in stream:
            row = json.loads(line)
            manifest[row["image"]] = row
    labels = {}
    with open(os.path.join(root, "labels.jsonl")) as stream:
        for line in stream:
            row = json.loads(line)
            labels[row["image"]] = row
    rows = []
    for image, label in labels.items():
        if label.get("skipped") or not label.get("points"):
            continue
        rows.append({**manifest[image], "points": label["points"]})
    return rows


def _affine(scale: float, angle_deg: float, shift: tuple[float, float]) -> np.ndarray:
    """2x3 matrix about the crop centre."""
    center = (CROP_W / 2.0, CROP_H / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale)
    matrix[:, 2] += shift
    return matrix


def prepare(image: np.ndarray, points: dict | None, augment: bool):
    """BGR frame -> normalized CHW crop, keypoints (4,2) in crop px, visibility (4,)."""
    crop = image[:, CROP_X0 : CROP_X0 + CROP_W]
    xy = np.zeros((len(POINTS), 2), np.float32)
    vis = np.zeros(len(POINTS), np.float32)
    if points:
        for k, name in enumerate(POINTS):
            p = points.get(name)
            if p is not None:
                xy[k] = (p[0] - CROP_X0, p[1])
                vis[k] = 1.0
    if augment:
        matrix = _affine(
            random.uniform(0.9, 1.1),
            random.uniform(-6, 6),
            (random.uniform(-16, 16), random.uniform(-16, 16)),
        )
        crop = cv2.warpAffine(
            crop, matrix, (CROP_W, CROP_H), borderMode=cv2.BORDER_REFLECT
        )
        xy = (np.c_[xy, np.ones(len(xy))] @ matrix.T).astype(np.float32)
        if random.random() < 0.5:
            crop = crop[:, ::-1]
            xy[:, 0] = CROP_W - 1 - xy[:, 0]
        hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV).astype(
            np.float32
        )
        hsv[..., 1] *= random.uniform(0.7, 1.3)
        hsv[..., 2] = hsv[..., 2] * random.uniform(0.7, 1.3) + random.uniform(-20, 20)
        crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if random.random() < 0.3:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < CROP_W)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < CROP_H)
        )
        vis = vis * inside
    return normalize_crop(crop), torch.from_numpy(xy), torch.from_numpy(vis)


def heatmaps(xy: torch.Tensor, vis: torch.Tensor, h: int, w: int) -> torch.Tensor:
    ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)
    cx = (xy[:, 0] / STRIDE).view(-1, 1, 1)
    cy = (xy[:, 1] / STRIDE).view(-1, 1, 1)
    maps = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * SIGMA**2))
    return maps * vis.view(-1, 1, 1)


class Frames(torch.utils.data.Dataset):
    def __init__(self, root: str, rows: list[dict], augment: bool):
        self.root, self.rows, self.augment = root, rows, augment
        self.images = [cv2.imread(os.path.join(root, r["image"])) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, xy, vis = prepare(self.images[i], self.rows[i]["points"], self.augment)
        return x, xy, vis


def train(
    root: str, rows: list[dict], epochs: int, device: str, seed: int = 0
) -> HeadPoseNet:
    torch.manual_seed(seed)
    random.seed(seed)
    model = HeadPoseNet().to(device)
    loader = torch.utils.data.DataLoader(
        Frames(root, rows, augment=True),
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=len(rows) > 16,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-3, total_steps=steps)
    model.train()
    for _ in range(epochs):
        for x, xy, vis in loader:
            x = x.to(device)
            heat, vis_logit = model(x)
            h, w = heat.shape[-2:]
            target = torch.stack([heatmaps(a, b, h, w) for a, b in zip(xy, vis)]).to(
                device
            )
            weight = vis.to(device).view(*vis.shape, 1, 1)
            # Visible points: fit the Gaussian; hidden points: push to zero.
            loss_heat = (((heat - target) ** 2) * (1 + 9 * weight)).mean() * 100
            loss_vis = F.binary_cross_entropy_with_logits(vis_logit, vis.to(device))
            optim.zero_grad()
            (loss_heat + loss_vis).backward()
            optim.step()
            sched.step()
    return model.eval()


def cross_validate(root: str, epochs: int, device: str, folds: int = 5) -> None:
    rows = load_rows(root)
    episodes = sorted({r["episode"] for r in rows})
    random.Random(0).shuffle(episodes)
    fold_of = {e: i % folds for i, e in enumerate(episodes)}
    errors = {name: [] for name in POINTS}
    lateral, vis_correct, vis_total = [], 0, 0
    for fold in range(folds):
        train_rows = [r for r in rows if fold_of[r["episode"]] != fold]
        test_rows = [r for r in rows if fold_of[r["episode"]] == fold]
        model = train(root, train_rows, epochs, device, seed=fold)
        for r in test_rows:
            pred = predict(model, cv2.imread(os.path.join(root, r["image"])), device)
            for name in POINTS:
                truth = r["points"].get(name)
                visible = pred[name]["p_visible"] > 0.5
                vis_correct += int(visible == (truth is not None))
                vis_total += 1
                if truth is not None:
                    errors[name].append(math.dist(pred[name]["xy"], truth))
            t, o = r["points"].get("other_tip"), r["points"].get("own_tip")
            if t and o:
                lateral.append(
                    (pred["other_tip"]["xy"][0] - pred["own_tip"]["xy"][0])
                    - (t[0] - o[0])
                )
        print(f"fold {fold}: train {len(train_rows)} test {len(test_rows)}")
    print(f"labeled images {len(rows)} from {len(episodes)} demos")
    for name in POINTS:
        e = np.array(errors[name])
        print(
            f"{name:10s} n={len(e):3d} median {np.median(e):5.1f}px  p90 {np.percentile(e, 90):5.1f}px"
        )
    lateral = np.abs(np.array(lateral))
    print(
        f"tip lateral offset (other - own, x) error: median {np.median(lateral):.1f}px "
        f"p90 {np.percentile(lateral, 90):.1f}px (n={len(lateral)})"
    )
    print(f"visibility accuracy {vis_correct / vis_total:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["cv", "train", "predict"])
    parser.add_argument("--root", default=os.path.expanduser("~/rlt_data/wire_head_pose"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--out", default=os.path.expanduser("~/rlt_data/wire_head_pose/head_pose_net.pt")
    )
    parser.add_argument(
        "--weights", default=os.path.expanduser("~/rlt_data/wire_head_pose/head_pose_net.pt")
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "cv":
        cross_validate(args.root, args.epochs, device)
    elif args.mode == "train":
        model = train(args.root, load_rows(args.root), args.epochs, device)
        torch.save(model.state_dict(), args.out)
        print(f"saved {args.out}")
    else:
        model = HeadPoseNet(pretrained=False).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()
        labeled = {r["image"] for r in load_rows(args.root)}
        with open(os.path.join(args.root, "manifest.jsonl")) as stream:
            manifest = [json.loads(line) for line in stream]
        out_path = os.path.join(args.root, "predictions.jsonl")
        with open(out_path, "w") as stream:
            for row in manifest:
                pred = predict(
                    model, cv2.imread(os.path.join(args.root, row["image"])), device
                )
                stream.write(
                    json.dumps(
                        {
                            "image": row["image"],
                            "labeled": row["image"] in labeled,
                            **pred,
                        }
                    )
                    + "\n"
                )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
