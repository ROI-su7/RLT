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

"""Frame selection and labeling helpers for the wire-head keypoint dataset."""

import numpy as np

from toolkits.wire_head_pose.annotate import (
    View,
    append_label,
    draw,
    key_name,
    labeling_order,
    load_labels,
)
from toolkits.wire_head_pose.extract_contact_frames import (
    approach_frame,
    insertion_events,
)
from toolkits.wire_head_pose.piper_fk import (
    MATED_FLANGE_OFFSET_MM,
    distance_to_mated,
    flange_pose,
)


def test_zero_pose_matches_piper_sdk_init_pos():
    # piper_sdk C_PiperForwardKinematics(dh_is_offset=1).init_pos, in mm.
    np.testing.assert_allclose(
        flange_pose(np.zeros(6))[0, :3, 3], [56.128, 0.0, 213.266], atol=1e-3
    )
    assert distance_to_mated(np.zeros((2, 14))).shape == (2,)
    assert MATED_FLANGE_OFFSET_MM.shape == (3,)


def test_each_insertion_is_found_at_its_closest_point():
    fps = 30.0
    distance = np.full(600, 300.0)
    distance[100:160] = np.linspace(55, 10, 60)  # insertion 1, closest at 159
    distance[160:200] = 10.0
    distance[400:430] = 30.0  # insertion 2
    distance[415] = 20.0
    distance[500:505] = 40.0  # 5 frames: too short to be an insertion
    assert insertion_events(distance, fps, threshold_mm=60.0) == [159, 415]


def test_approach_frame_is_the_last_one_still_farther_than_the_gap():
    distance = np.array([50, 40, 30, 20, 15, 12, 10], dtype=float)
    assert approach_frame(distance, 6, 0.0, 10) == 6
    assert approach_frame(distance, 6, 5.0, 10) == 4  # 15 >= 10 + 5
    assert approach_frame(distance, 6, 20.0, 10) == 2  # 30 >= 30
    assert approach_frame(distance, 6, 100.0, 10) is None
    assert approach_frame(distance, 6, 20.0, 3) is None  # outside the lookback


def test_clicks_map_back_to_original_pixels_with_and_without_zoom():
    view = View((480, 640), scale=1.5)
    assert view.to_original((150, 300)) == (100.0, 200.0)
    assert view.to_display((100.0, 200.0)) == (150, 300)
    view.toggle_zoom((480, 360))  # mouse over original (320, 240)
    assert view.zoom_origin == (160.0, 120.0)
    x, y = view.to_original((300, 300))  # factor 3
    assert (x, y) == (260.0, 220.0)
    assert view.to_display((x, y)) == (300, 300)
    assert view.render(np.zeros((480, 640, 3), np.uint8)).shape == (720, 960, 3)
    view.toggle_zoom((0, 0))
    assert view.zoom_origin is None


def test_labels_resume_from_the_last_row_per_image(tmp_path):
    path = str(tmp_path / "labels.jsonl")
    append_label(path, "images/a.jpg", None, skipped=True)
    points = {"own_tip": [1.0, 2.0], "own_back": None, "other_tip": [3, 4]}
    append_label(path, "images/a.jpg", points, skipped=False)
    append_label(path, "images/b.jpg", None, skipped=True)
    labels = load_labels(path)
    assert labels["images/a.jpg"]["points"] == points
    assert labels["images/b.jpg"]["skipped"] is True


def test_labeling_order_cycles_through_gap_levels():
    rows = [
        {"image": f"{gap}_{i}", "gap_mm": gap} for gap in (0, 5, 20) for i in range(4)
    ]
    order = labeling_order(rows, seed=1)
    assert sorted(r["image"] for r in order) == sorted(r["image"] for r in rows)
    assert [r["gap_mm"] for r in order[:3]] == [0, 5, 20]


def test_keys_work_with_the_qt_backend_codes():
    # opencv-python's Qt backend: letters upper-case, Enter/Esc as Qt codes.
    assert key_name(ord("V")) == key_name(ord("v")) == "v"
    assert key_name(ord("U")) == "u" and key_name(ord("z")) == "z"
    assert key_name(0x01000004) == key_name(13) == "enter"
    assert key_name(0x01000005) == "enter"
    assert key_name(0x01000000) == key_name(27) == "esc"
    assert key_name(32) == "space"
    assert key_name(-1) is None
    assert key_name(0x01000012) is None  # an arrow key


def test_overlay_shows_prompt_help_and_message():
    image = np.zeros((480, 640, 3), np.uint8)
    canvas = draw(image, View((480, 640), 1.5), {"own_tip": [10, 10]}, 1, "info", "msg")
    assert canvas.shape == (720, 960, 3)
    assert canvas.any()
