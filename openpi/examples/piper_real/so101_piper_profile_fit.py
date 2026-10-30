"""Build one SO-101/Piper mapping profile from paired capture samples.

Each mapped axis selects the sample whose target-axis motion dominates the
incidental movement of other mapped Piper axes. This tolerates small coupling
while avoiding samples where two large joints changed together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


AXES = (
    ("shoulder_pan", 0, "j1"),
    ("shoulder_lift", 1, "j2"),
    ("elbow_flex", 2, "j3"),
    ("wrist_flex", 4, "j5"),
    ("wrist_roll", 5, "j6"),
    ("gripper", 6, "gripper"),
)


def _shortest_deg_delta(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=tuple(name for _, _, name in AXES),
        help="Only fit these axes. Defaults to all axes.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        help="Existing profile whose non-selected mappings are preserved.",
    )
    parser.add_argument(
        "--piecewise-axes",
        nargs="*",
        default=(),
        choices=tuple(name for _, _, name in AXES),
        help="Fit selected axes as clamped piecewise-linear knots.",
    )
    parser.add_argument("--min-arm-span-deg", type=float, default=3.0)
    parser.add_argument("--min-gripper-span-m", type=float, default=0.005)
    args = parser.parse_args()

    samples = json.loads(args.input.read_text()).get("samples", [])
    if len(samples) < 2:
        raise ValueError("At least two paired samples are required")

    baseline = samples[0][args.side]
    baseline_so = baseline["so101_deg"]
    baseline_piper = np.asarray(baseline["piper_rad_gripper_m"], dtype=float)
    selected_axes = set(args.axes or [name for _, _, name in AXES])
    piecewise_axes = set(args.piecewise_axes)
    if not piecewise_axes.issubset(selected_axes):
        raise ValueError("--piecewise-axes must be included in --axes")
    mappings: list[dict] = []
    for leader_key, piper_index, name in AXES:
        if name not in selected_axes:
            continue
        piper_values = np.asarray(
            [sample[args.side]["piper_rad_gripper_m"][piper_index] for sample in samples],
            dtype=float,
        )
        deltas = piper_values - baseline_piper[piper_index]
        leader_values = np.asarray(
            [sample[args.side]["so101_deg"][leader_key] for sample in samples],
            dtype=float,
        )
        leader_reference = float(baseline_so[leader_key])
        if name != "gripper":
            leader_values = leader_reference + (
                (leader_values - leader_reference + 180.0) % 360.0 - 180.0
            )
        if name in piecewise_axes:
            order = np.argsort(leader_values)
            knots_x = leader_values[order]
            knots_y = piper_values[order]
            unique_x, inverse = np.unique(knots_x, return_inverse=True)
            unique_y = np.asarray(
                [np.mean(knots_y[inverse == index]) for index in range(len(unique_x))]
            )
            if len(unique_x) < 2:
                raise ValueError(f"Insufficient distinct leader values for {name}")
            mappings.append(
                {
                    "name": name,
                    "leader_key": leader_key,
                    "piper_index": piper_index,
                    "method": "piecewise_linear_clamped",
                    "leader_knots": unique_x.tolist(),
                    "piper_knots": unique_y.tolist(),
                    "selected_samples": (order + 1).tolist(),
                }
            )
            continue
        scores = np.zeros(len(samples), dtype=float)
        for sample_index, sample in enumerate(samples):
            all_delta = (
                np.asarray(sample[args.side]["piper_rad_gripper_m"], dtype=float)
                - baseline_piper
            )
            if name == "gripper":
                target_motion = abs(all_delta[6])
                other_motion = np.linalg.norm(
                    np.rad2deg(all_delta[[0, 1, 2, 4, 5]])
                ) / 1000.0
            else:
                target_motion = abs(np.rad2deg(all_delta[piper_index]))
                other_indices = [
                    value for value in (0, 1, 2, 4, 5) if value != piper_index
                ]
                other_motion = np.linalg.norm(np.rad2deg(all_delta[other_indices]))
            scores[sample_index] = target_motion / (other_motion + 1e-6)
        selected = int(np.argmax(scores))
        leader_value = float(leader_values[selected])
        if name == "gripper":
            leader_delta = leader_value - leader_reference
            min_span = args.min_gripper_span_m
            gain = float(deltas[selected] / leader_delta)
            unit = "piper_m_per_so101_unit"
        else:
            leader_delta = leader_value - leader_reference
            min_span = np.deg2rad(args.min_arm_span_deg)
            gain = float(deltas[selected] / np.deg2rad(leader_delta))
            unit = "piper_rad_per_so101_rad"
        if abs(deltas[selected]) < min_span or abs(leader_delta) < 1e-6:
            raise ValueError(f"Insufficient independent span for {name}")
        mappings.append(
            {
                "name": name,
                "leader_key": leader_key,
                "piper_index": piper_index,
                "leader_reference": leader_reference,
                "piper_reference": float(baseline_piper[piper_index]),
                "gain": gain,
                "gain_unit": unit,
                "selected_sample": selected + 1,
                "leader_delta": leader_delta,
                "piper_delta": float(deltas[selected]),
                "dominance_score": float(scores[selected]),
                "leader_observed_range": [
                    float(np.min(leader_values)),
                    float(np.max(leader_values)),
                ],
            }
        )

    if args.base_profile:
        base = json.loads(args.base_profile.read_text())
        if base.get("side") != args.side:
            raise ValueError("--base-profile side does not match --side")
        preserved = [
            mapping
            for mapping in base.get("mappings", [])
            if mapping.get("name") not in selected_axes
        ]
        mappings = preserved + mappings
    mappings.sort(key=lambda mapping: [name for _, _, name in AXES].index(mapping["name"]))

    profile = {
        "format_version": 1,
        "side": args.side,
        "source": str(args.input),
        "locked_piper_indices": [3],
        "mappings": mappings,
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"Wrote {args.side} profile to {args.output}")
    for mapping in mappings:
        if mapping.get("method") == "piecewise_linear_clamped":
            print(f"{mapping['name']}: {len(mapping['leader_knots'])} piecewise knots")
        else:
            print(
                f"{mapping['name']}: sample {mapping['selected_sample']}, "
                f"gain={mapping['gain']:+.8f}, dominance={mapping['dominance_score']:.3f}"
            )


if __name__ == "__main__":
    main()
