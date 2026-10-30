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

"""Connector-head keypoints in the right wrist camera, as a Stage-2 state.

ResNet-18 up to stride 16, upsampled to stride 4, predicts one heatmap and
one visibility logit per keypoint: the tip and a point farther back on the
axis of the head held by this camera's gripper ("own") and of the head
arriving from the other gripper ("other"). The input is the central 320x480
strip of the 640x480 frame. Training and labeling live in
``toolkits/wire_head_pose``.

The Stage-2 state is ``[dx, dy, dangle, visible]``: other tip minus own tip
in pixels, the angle between the two head axes in rad, and 1 when both tips
are visible (all zeros otherwise).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

POINTS = ("own_tip", "own_back", "other_tip", "other_back")
CROP_X0, CROP_W, CROP_H = 160, 320, 480
STRIDE = 4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HEAD_POSE_DIM = 4
HEAD_POSE_FIELDS = ("dx_px", "dy_px", "dangle_rad", "visible")
VISIBLE_THRESHOLD = 0.5


class HeadPoseNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.heat = nn.Conv2d(64, len(POINTS), 1)
        self.vis = nn.Linear(256, len(POINTS))

    def forward(self, x):
        x1 = self.layer1(self.stem(x))  # stride 4, 64 ch
        x2 = self.layer2(x1)  # stride 8, 128 ch
        x3 = self.layer3(x2)  # stride 16, 256 ch
        y = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, x2], 1))
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, x1], 1))
        return self.heat(y), self.vis(x3.mean((2, 3)))


def normalize_crop(crop: np.ndarray) -> torch.Tensor:
    """BGR uint8 crop -> ImageNet-normalized RGB CHW float tensor."""
    rgb = np.ascontiguousarray(crop[..., ::-1]).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def decode(heat: torch.Tensor) -> torch.Tensor:
    """(B,K,h,w) -> (B,K,2) crop px: weighted mean in a 5x5 window at the peak."""
    b, k, h, w = heat.shape
    flat = heat.view(b, k, -1)
    idx = flat.argmax(-1)
    py, px = (idx // w).float(), (idx % w).float()
    out = torch.zeros(b, k, 2)
    for i in range(b):
        for j in range(k):
            y0, x0 = int(py[i, j]), int(px[i, j])
            ys, ye = max(0, y0 - 2), min(h, y0 + 3)
            xs, xe = max(0, x0 - 2), min(w, x0 + 3)
            patch = heat[i, j, ys:ye, xs:xe].clamp(min=0)
            gy, gx = torch.meshgrid(
                torch.arange(ys, ye, dtype=torch.float32),
                torch.arange(xs, xe, dtype=torch.float32),
                indexing="ij",
            )
            total = patch.sum() + 1e-6
            out[i, j, 0] = (patch * gx).sum() / total
            out[i, j, 1] = (patch * gy).sum() / total
    # Heatmap cell centres sit at STRIDE * index + (STRIDE - 1) / 2 input px.
    return out * STRIDE + (STRIDE - 1) / 2.0


@torch.no_grad()
def predict(model: HeadPoseNet, image: np.ndarray, device: str) -> dict:
    """Keypoints of one 640x480 BGR frame in frame pixels, with visibility."""
    x = normalize_crop(image[:, CROP_X0 : CROP_X0 + CROP_W])
    heat, vis_logit = model(x[None].to(device))
    xy = decode(heat.float().cpu())[0]
    prob = torch.sigmoid(vis_logit.float().cpu())[0]
    return {
        name: {
            "xy": [round(float(xy[k, 0]) + CROP_X0, 1), round(float(xy[k, 1]), 1)],
            "p_visible": round(float(prob[k]), 3),
        }
        for k, name in enumerate(POINTS)
    }


def head_features(pred: dict) -> np.ndarray | None:
    """Tip offset, axis directions and own tip of one frame; None if hidden.

    ``[dx, dy, |d|, a_other - a_own, a_own, a_other, own_x, own_y]``.
    """
    if (
        pred["own_tip"]["p_visible"] < VISIBLE_THRESHOLD
        or pred["other_tip"]["p_visible"] < VISIBLE_THRESHOLD
    ):
        return None
    o, ob = np.array(pred["own_tip"]["xy"]), np.array(pred["own_back"]["xy"])
    t, tb = np.array(pred["other_tip"]["xy"]), np.array(pred["other_back"]["xy"])
    own_axis = o - ob
    other_axis = t - tb
    a_own = math.atan2(own_axis[0], -own_axis[1])
    a_other = math.atan2(other_axis[0], other_axis[1])
    d = t - o
    return np.array(
        [d[0], d[1], math.hypot(*d), a_other - a_own, a_own, a_other, *o],
        dtype=np.float64,
    )


def head_pose_state(pred: dict) -> np.ndarray:
    """``[dx, dy, dangle, visible]`` of one prediction (zeros when hidden)."""
    features = head_features(pred)
    if features is None:
        return np.zeros(HEAD_POSE_DIM, np.float32)
    return np.array([features[0], features[1], features[3], 1.0], np.float32)


class WireHeadPoseDetector:
    """Runs :class:`HeadPoseNet` on raw right-wrist frames for the Piper env."""

    def __init__(self, weights: str, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = HeadPoseNet(pretrained=False).to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.eval()
        self.last_ms = 0.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (CROP_H, CROP_X0 * 2 + CROP_W):
            raise ValueError(
                f"Head pose expects a 640x480 frame, got {image.shape[1]}x{image.shape[0]}."
            )
        started = time.monotonic()
        state = head_pose_state(predict(self.model, image, self.device))
        self.last_ms = (time.monotonic() - started) * 1000.0
        return state
