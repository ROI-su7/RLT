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

"""Per-mode success rates of Piper RLT Stage-2 episodes from run logs.

An episode ends with C (success), X or Q (failure). Its mode is:

* ``JAX``: no B, so the native-JAX policy ran the whole episode (not replayed);
* ``JAX_WARMUP``: B pressed before the Actor gate opened;
* ``ACTOR_EVAL`` / ``ACTOR_EXPLORE``: tagged by ``[RLT exploration]``;
* ``ACTOR``: Actor chunks in a log written before exploration tagging.

A success counts for the policy only when no T takeover happened; C after T
is reported as a human rescue.

Usage::

    python toolkits/realworld_check/rlt_episode_stats.py \\
        logs/<run1>/run_embodiment.log logs/<run2>/run_embodiment.log \\
        [--episodes] [--timeout-s 30]
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TIME = re.compile(r"\[INFO (\d\d):(\d\d):(\d\d)")
_MODES = ("JAX", "JAX_WARMUP", "ACTOR_EVAL", "ACTOR_EXPLORE", "ACTOR")


@dataclass
class Episode:
    source: str
    start_s: float | None = None
    b_s: float | None = None
    end_s: float | None = None
    recorded: bool = False
    takeover: bool = False
    actor: bool = False
    tag: str | None = None
    outcome: str | None = None
    events: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if not self.recorded:
            return "JAX"
        if self.tag is not None:
            return f"ACTOR_{self.tag}"
        return "ACTOR" if self.actor else "JAX_WARMUP"

    @property
    def duration_s(self) -> float | None:
        start = self.b_s if self.b_s is not None else self.start_s
        if start is None or self.end_s is None:
            return None
        return (self.end_s - start) % 86400.0


def _seconds(line: str) -> float | None:
    match = _TIME.search(line)
    if match is None:
        return None
    hours, minutes, seconds = (int(value) for value in match.groups())
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_episodes(lines, source: str = "") -> list[Episode]:
    """Split one run log into terminated and open episodes."""
    episodes: list[Episode] = []
    current: Episode | None = None
    for raw in lines:
        line = _ANSI.sub("", raw)
        now = _seconds(line)
        if "mux_rlt_wrapper" in line:
            message = line.split("] ", 2)[-1].strip()
            key = message[:2]
            if key == "I:":
                if current is None:
                    current = Episode(source=source, start_s=now)
            elif current is None:
                continue
            elif key == "B:":
                current.recorded = True
                current.b_s = now
            elif key == "T:":
                current.takeover = True
            elif key in {"C:", "X:", "Q:"}:
                current.outcome = "success" if key == "C:" else "failure"
                current.end_s = now
                episodes.append(current)
                current = None
                continue
            if current is not None:
                current.events.append(key[0])
        elif current is not None and "[RLT exploration]" in line:
            if current.tag is None:
                current.tag = "EVAL" if ": EVAL" in line else "EXPLORE"
        elif current is not None and re.search(r"source=(ACTOR|MIXED\(ACTOR)", line):
            current.actor = True
    if current is not None:
        episodes.append(current)
    return episodes


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a binomial success rate."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return centre - half / denominator, centre + half / denominator


def summarize(episodes: list[Episode], timeout_s: float) -> list[str]:
    rows = [
        f"{'mode':14s} {'n':>3s} {'policy_C':>8s} {'rate':>5s} {'95% CI':>13s} "
        f"{'rescued':>7s} {'X':>3s} {'>timeout':>8s} {'median_s':>8s}"
    ]
    for mode in _MODES:
        done = [e for e in episodes if e.mode == mode and e.outcome is not None]
        if not done:
            continue
        wins = sum(e.outcome == "success" and not e.takeover for e in done)
        rescued = sum(e.outcome == "success" and e.takeover for e in done)
        failures = sum(e.outcome == "failure" for e in done)
        durations = sorted(e.duration_s for e in done if e.duration_s is not None)
        slow = sum(d > timeout_s for d in durations)
        median = durations[len(durations) // 2] if durations else float("nan")
        low, high = wilson(wins, len(done))
        rows.append(
            f"{mode:14s} {len(done):3d} {wins:8d} {wins / len(done):5.0%} "
            f"[{low:4.0%}, {high:4.0%}] {rescued:7d} {failures:3d} {slow:8d} "
            f"{median:8.0f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="run_embodiment.log files in order")
    parser.add_argument("--episodes", action="store_true", help="list every episode")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="report episodes longer than this (from B, else from I)",
    )
    args = parser.parse_args()
    episodes: list[Episode] = []
    for path in args.logs:
        with open(path, errors="replace") as handle:
            episodes.extend(parse_episodes(handle, source=path))
    if args.episodes:
        for index, episode in enumerate(episodes, 1):
            duration = episode.duration_s
            print(
                f"{index:3d} {episode.mode:14s} "
                f"{episode.outcome or 'open':8s} "
                f"{'T' if episode.takeover else '-'} "
                f"{'' if duration is None else f'{duration:5.0f}s'}"
            )
        print()
    for row in summarize(episodes, args.timeout_s):
        print(row)


if __name__ == "__main__":
    main()
