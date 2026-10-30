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

"""Click four keypoints of the two connector heads in each wrist frame.

Points are camera-relative, so male/female does not matter:

    1 own_tip     front of the head held by THIS camera's gripper (bottom of
                  the image): centre of its front-most visible point
    2 own_back    a point farther back on that head's axis
    3 other_tip   front of the head held by the OTHER gripper (comes in from
                  the top), same rule; half inserted, this is where its
                  visible part enters the bottom head
    4 other_back  a point farther back on that head's axis

Keys: left click = place the current point, v = current point not visible,
u / right click = undo, z = 2x zoom at the mouse (again to leave),
space / enter = save and next (all four points placed or marked v),
s = skip image, b = back, q / esc = quit. Upper and lower case both work.
Labels are appended to ``<root>/labels.jsonl`` (last row per image wins), so
quitting and starting again resumes where you stopped.

    source .venv-rlt/bin/activate
    python toolkits/wire_head_pose/annotate.py --root ~/rlt_data/wire_head_pose
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

POINTS = ("own_tip", "own_back", "other_tip", "other_back")
COLORS = {"own": (0, 200, 0), "other": (0, 0, 255)}
HELP = "v=hidden  u=undo  z=zoom  space=save  s=skip  b=back  q=quit"
# Qt key codes (OpenCV's Qt backend reports Enter/Esc as these).
_QT_ESCAPE, _QT_RETURN, _QT_ENTER = 0x01000000, 0x01000004, 0x01000005


def key_name(code: int) -> str | None:
    """Name of a cv2.waitKeyEx code across OpenCV GUI backends.

    The Qt backend of opencv-python reports letters as upper-case codes
    whether or not shift is held, and Enter/Esc as Qt key codes, so plain
    ``cv2.waitKey(...) == ord("v")`` never matches there.
    """
    if code is None or code < 0:
        return None
    if code in (13, 10, _QT_RETURN, _QT_ENTER):
        return "enter"
    if code in (27, _QT_ESCAPE):
        return "esc"
    if code == 32:
        return "space"
    if 33 <= code < 127:
        return chr(code).lower()
    return None


def load_labels(path: str) -> dict[str, dict]:
    """Image -> latest label row."""
    labels = {}
    if os.path.exists(path):
        with open(path) as stream:
            for line in stream:
                row = json.loads(line)
                labels[row["image"]] = row
    return labels


def append_label(
    path: str, image: str, points: dict | None, skipped: bool, source: str = "human"
) -> dict:
    row = {
        "image": image,
        "points": points,
        "skipped": skipped,
        "time": time.time(),
        "source": source,
    }
    with open(path, "a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def labeling_order(rows: list[dict], seed: int = 0) -> list[dict]:
    """Shuffle, then spread insertions and gap levels so early labels are diverse."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    # Round-robin over gap levels: every few images cover approach to mated.
    by_gap: dict[float, list[dict]] = {}
    for row in rows:
        by_gap.setdefault(row.get("gap_mm", 0.0), []).append(row)
    ordered = []
    while any(by_gap.values()):
        for gap in sorted(by_gap):
            if by_gap[gap]:
                ordered.append(by_gap[gap].pop())
    return ordered


class View:
    """Maps between display and original pixel coordinates (scale + zoom)."""

    def __init__(self, shape: tuple[int, int], scale: float):
        self.height, self.width = shape
        self.scale = scale
        self.zoom_origin: tuple[float, float] | None = None  # top-left, original px

    @property
    def factor(self) -> float:
        return self.scale * (2.0 if self.zoom_origin is not None else 1.0)

    def toggle_zoom(self, display_xy: tuple[int, int]) -> None:
        if self.zoom_origin is not None:
            self.zoom_origin = None
            return
        cx, cy = self.to_original(display_xy)
        w, h = self.width / 2.0, self.height / 2.0
        self.zoom_origin = (
            float(np.clip(cx - w / 2, 0, self.width - w)),
            float(np.clip(cy - h / 2, 0, self.height - h)),
        )

    def to_original(self, display_xy: tuple[int, int]) -> tuple[float, float]:
        ox, oy = self.zoom_origin or (0.0, 0.0)
        return ox + display_xy[0] / self.factor, oy + display_xy[1] / self.factor

    def to_display(self, xy: tuple[float, float]) -> tuple[int, int]:
        ox, oy = self.zoom_origin or (0.0, 0.0)
        return int(round((xy[0] - ox) * self.factor)), int(
            round((xy[1] - oy) * self.factor)
        )

    def render(self, image: np.ndarray) -> np.ndarray:
        if self.zoom_origin is not None:
            ox, oy = (int(v) for v in self.zoom_origin)
            image = image[oy : oy + self.height // 2, ox : ox + self.width // 2]
        return cv2.resize(
            image, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_LINEAR
        )


def draw(
    image: np.ndarray,
    view: View,
    points: dict,
    current: int,
    info: str,
    message: str = "",
) -> np.ndarray:
    canvas = view.render(image)
    for side in ("own", "other"):
        tip, back = points.get(f"{side}_tip"), points.get(f"{side}_back")
        if tip and back:
            cv2.line(
                canvas, view.to_display(tip), view.to_display(back), COLORS[side], 1
            )
        for name, filled in ((f"{side}_tip", True), (f"{side}_back", False)):
            if points.get(name):
                cv2.circle(
                    canvas,
                    view.to_display(points[name]),
                    5,
                    COLORS[side],
                    -1 if filled else 2,
                )
    prompt = (
        f"{current + 1}/4 click {POINTS[current]}  (v if hidden)"
        if current < len(POINTS)
        else "done: space=save  u=undo"
    )
    height = canvas.shape[0]
    lines = [((10, 25), prompt), ((10, 47), info), ((10, height - 12), HELP)]
    if message:
        lines.append(((10, 69), message))
    for (x, y), text in lines:
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=os.path.expanduser("~/rlt_data/wire_head_pose"))
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--review",
        action="store_true",
        help="go through the labels marked low-confidence; space keeps a "
        "label, u / clicks correct it, s marks the image unusable",
    )
    args = parser.parse_args()

    with open(os.path.join(args.root, "manifest.jsonl")) as stream:
        rows = [json.loads(line) for line in stream]
    label_path = os.path.join(args.root, "labels.jsonl")
    labels = load_labels(label_path)
    if args.review:
        order = [
            r for r in rows if labels.get(r["image"], {}).get("confidence") == "low"
        ]
        index = 0
    else:
        order = labeling_order(rows, args.seed)
        index = next(
            (i for i, r in enumerate(order) if r["image"] not in labels), len(order)
        )

    window = "wire head keypoints"
    # GUI_NORMAL: no Qt toolbar, so the wheel cannot zoom the view behind the
    # tool's back and shift where clicks land.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    state = {"mouse": (0, 0), "click": None, "undo": False}

    def on_mouse(event, x, y, flags, param):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["undo"] = True

    cv2.setMouseCallback(window, on_mouse)
    labeled = {image for image, row in labels.items() if not row["skipped"]}
    message = ""
    while 0 <= index < len(order):
        row = order[index]
        image = cv2.imread(os.path.join(args.root, row["image"]))
        view = View(image.shape[:2], args.scale)
        points: dict = {}
        current = 0
        previous = labels.get(row["image"])
        if previous and previous.get("points"):
            points = dict(previous["points"])
            current = len(POINTS)
        dirty = True
        note = (previous or {}).get("note", "") if args.review else ""
        while True:
            if dirty:
                info = (
                    f"image {index + 1}/{len(order)}  labeled {len(labeled)}  "
                    f"gap {row.get('gap_mm')}mm"
                )
                shown = message or (f"note: {note}" if note else "")
                cv2.imshow(window, draw(image, view, points, current, info, shown))
                dirty = False
            key = key_name(cv2.waitKeyEx(30))
            if state["click"] is not None:
                if current < len(POINTS):
                    x, y = view.to_original(state["click"])
                    points[POINTS[current]] = [round(x, 1), round(y, 1)]
                    current += 1
                    message = ""
                state["click"] = None
                dirty = True
            if state["undo"] or key == "u":
                state["undo"] = False
                if current > 0:
                    current -= 1
                    points.pop(POINTS[current], None)
                message, dirty = "", True
            if key is None:
                continue
            dirty = True
            if key == "v" and current < len(POINTS):
                points[POINTS[current]] = None
                current += 1
                message = ""
            elif key == "z":
                view.toggle_zoom(state["mouse"])
            elif key in ("space", "enter"):
                if current < len(POINTS):
                    message = (
                        f"need {len(POINTS) - current} more point(s): "
                        "click, or v if hidden"
                    )
                    continue
                labels[row["image"]] = append_label(
                    label_path, row["image"], points, False
                )
                labeled.add(row["image"])
                message = f"saved {row['image'].split('/')[-1]}"
                index += 1
                break
            elif key == "s":
                labels[row["image"]] = append_label(
                    label_path, row["image"], None, True
                )
                labeled.discard(row["image"])
                message = "skipped"
                index += 1
                break
            elif key == "b":
                index = max(0, index - 1)
                message = "back"
                break
            elif key in ("q", "esc"):
                cv2.destroyAllWindows()
                print(f"saved to {label_path}; {len(labeled)} images labeled")
                return
    cv2.destroyAllWindows()
    print("all images labeled")


if __name__ == "__main__":
    main()
