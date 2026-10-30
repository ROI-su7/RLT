#!/usr/bin/env python3
"""Quantify Piper arm "crawl" from rosbags of policy targets and joint feedback.

Record the same topics for every pipeline that is compared, e.g.::

    rosbag record -O ~/crawl_openpi.bag \
        /teleop/policy/joint_left /teleop/policy/joint_right \
        /puppet/joint_left /puppet/joint_right

and report them side by side with the ROS system Python::

    source /opt/ros/noetic/setup.bash
    /usr/bin/python3 toolkits/realworld_check/piper_motion_bag_report.py \
        ~/crawl_openpi.bag:openpi ~/crawl_rlinf_b10.bag:rlinf_b10

Every continuous policy-publishing segment (split at HOLD gaps) is reported
over fixed time windows, so the alignment phase at the end of an episode can
be compared with the approach phase. All times use the bag receive clock so
command and feedback topics share one time base.
"""

from __future__ import annotations

import argparse

import numpy as np
import rosbag

ARM = np.r_[0:6, 7:13]
GRID_HZ = 50.0


def _load_pair(bag_path, left_topic, right_topic):
    times = {left_topic: [], right_topic: []}
    values = {left_topic: [], right_topic: []}
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, stamp in bag.read_messages(topics=[left_topic, right_topic]):
            if len(msg.position) < 7:
                continue
            times[topic].append(stamp.to_sec())
            values[topic].append(np.asarray(msg.position[:7], dtype=np.float64))
    return [
        (np.asarray(times[t]), np.asarray(values[t]).reshape(-1, 7))
        for t in (left_topic, right_topic)
    ]


def _resample(pair, grid):
    out = np.empty((grid.size, 14))
    for side, (t, q) in enumerate(pair):
        for joint in range(7):
            out[:, side * 7 + joint] = np.interp(grid, t, q[:, joint])
    return out


def _segments(policy_times, max_gap_s, min_len_s):
    if policy_times.size == 0:
        return []
    breaks = np.where(np.diff(policy_times) > max_gap_s)[0]
    starts = np.r_[policy_times[0], policy_times[breaks + 1]]
    ends = np.r_[policy_times[breaks], policy_times[-1]]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len_s]


def _motion_stats(q, stop_rad_s):
    """Speed, stop share, backtracking share and pulsation of one arm trajectory."""
    dt = 1.0 / GRID_HZ
    vel = np.gradient(q[:, ARM], dt, axis=0)
    kernel = np.ones(3) / 3.0
    vel = np.stack(
        [np.convolve(vel[:, j], kernel, mode="same") for j in range(vel.shape[1])], 1
    )
    speed = np.linalg.norm(vel, axis=1)
    # Backtracking: 0.2-s steps that move against the 2-s local direction.
    step = int(round(0.2 * GRID_HZ))
    span = int(round(1.0 * GRID_HZ))
    backward = total = 0.0
    for i in range(span, q.shape[0] - span - step, step):
        direction = q[i + span, ARM] - q[i - span, ARM]
        norm = np.linalg.norm(direction)
        if norm < 1e-4:
            continue
        delta = q[i + step, ARM] - q[i, ARM]
        along = float(delta @ direction) / norm
        total += abs(along)
        backward += max(0.0, -along)
    # Pulsation: strongest autocorrelation peak of the speed in 0.5-2.5 s.
    # A replan-locked surge/stall cycle shows up at the replan period.
    centered = speed - speed.mean()
    period = strength = float("nan")
    if centered.size >= int(4 * GRID_HZ) and centered.std() > 1e-9:
        ac = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
        ac = ac / ac[0]
        lags = np.arange(int(0.5 * GRID_HZ), int(2.5 * GRID_HZ))
        lags = lags[lags + 1 < ac.size]
        peaks = lags[(ac[lags] > ac[lags - 1]) & (ac[lags] >= ac[lags + 1])]
        if peaks.size:
            lag = int(peaks[np.argmax(ac[peaks])])
            period, strength = lag / GRID_HZ, float(ac[lag])
    return {
        "speed": float(speed.mean()),
        "stop": float(np.mean(speed < stop_rad_s)),
        "back": backward / total if total > 0 else float("nan"),
        "period": period,
        "pulse": strength,
    }


def _report(label, bag_path, args):
    policy = _load_pair(bag_path, args.policy_left, args.policy_right)
    feedback = _load_pair(bag_path, args.feedback_left, args.feedback_right)
    if min(p[0].size for p in policy) == 0 or min(p[0].size for p in feedback) == 0:
        print(f"== {label}: missing policy or feedback messages in {bag_path}")
        return
    policy_times = np.sort(np.r_[policy[0][0], policy[1][0]])
    segments = _segments(policy_times, args.max_gap_s, args.min_segment_s)
    print(f"== {label}  ({bag_path})  {len(segments)} policy segment(s)")
    header = (
        "    window      | cmd rad/s  fb rad/s | fb stop  fb back | "
        "fb pulse period/strength"
    )
    for index, (start, end) in enumerate(segments):
        lo = max(start, max(p[0][0] for p in feedback))
        hi = min(end, min(p[0][-1] for p in feedback))
        grid = np.arange(lo, hi, 1.0 / GRID_HZ)
        if grid.size < int(4 * GRID_HZ):
            continue
        q_cmd = _resample(policy, grid)
        q_fb = _resample(feedback, grid)
        print(f"  segment {index}: {hi - lo:.1f}s")
        print(header)
        edges = list(np.arange(0.0, hi - lo, args.window_s)) + [hi - lo]
        if edges[-1] - edges[-2] < 2.0 and len(edges) > 2:
            edges.pop(-2)
        rows = [
            (edges[i], edges[i + 1], f"{edges[i]:5.1f}-{edges[i + 1]:5.1f}s")
            for i in range(len(edges) - 1)
        ]
        if len(rows) > 1:
            rows.append((0.0, hi - lo, "all"))
        for a, b, name in rows:
            m = (grid - lo >= a) & (grid - lo < b + 1e-9)
            if m.sum() < int(2 * GRID_HZ):
                continue
            cmd = _motion_stats(q_cmd[m], args.stop_rad_s)
            fb = _motion_stats(q_fb[m], args.stop_rad_s)
            print(
                f"    {name:>12s} | {cmd['speed']:8.4f} {fb['speed']:9.4f} | "
                f"{fb['stop']:6.0%} {fb['back']:7.0%} | "
                f"{fb['period']:5.2f}s / {fb['pulse']:+.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bags", nargs="+", help="bag path, optionally suffixed with :label"
    )
    parser.add_argument("--policy-left", default="/teleop/policy/joint_left")
    parser.add_argument("--policy-right", default="/teleop/policy/joint_right")
    parser.add_argument("--feedback-left", default="/puppet/joint_left")
    parser.add_argument("--feedback-right", default="/puppet/joint_right")
    parser.add_argument("--window-s", type=float, default=6.0)
    parser.add_argument(
        "--stop-rad-s",
        type=float,
        default=0.01,
        help="arm speed below this counts as stopped",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.5,
        help="policy gap that splits segments (HOLD)",
    )
    parser.add_argument("--min-segment-s", type=float, default=4.0)
    args = parser.parse_args()
    for item in args.bags:
        path, _, label = item.partition(".bag:")
        path = path + ".bag" if label else item
        _report(label or path, path, args)


if __name__ == "__main__":
    main()
