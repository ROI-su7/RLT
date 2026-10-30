"""Three-insertion progress scoring for Piper RECAP episodes.

RECAP-only operator scoring; the RLT keyboard reward in
``rlinf.envs.realworld.piper.tasks`` is unchanged. ``C`` means all insertions
are complete (RECAP success label 1) and ``X`` means failure (label 0); these
alone define the RECAP reward. ``N``/``U`` optionally record each finished
insertion for per-insertion statistics; they never change the reward. With
``require_marks`` the recorder refuses ``C`` until every insertion is marked.
Milestone frame indices are saved in ``episode.json`` and expanded into a
per-frame ``insertions_done`` column at export time. No ROS or model imports.
"""

from __future__ import annotations

import json
from pathlib import Path

TOTAL_INSERTIONS = 3


class InsertionScore:
    """Operator-marked insertion progress for one recorder session."""

    def __init__(self, total: int = TOTAL_INSERTIONS, require_marks: bool = False):
        if total < 1:
            raise ValueError("total insertions must be positive.")
        self.total = total
        self.require_marks = bool(require_marks)
        # Insertions already complete when the next episode starts (e.g. the
        # scene left by a failed episode that is continued with I).
        self.pending_start = 0
        self.active = False
        self.start = 0
        self.milestones: list[int] = []

    @property
    def done(self) -> int:
        """Insertions complete in the current (or next) episode."""
        return self.start + len(self.milestones) if self.active else self.pending_start

    def begin(self) -> None:
        """Start scoring a new episode from the pending start progress."""
        self.active, self.start, self.milestones = True, self.pending_start, []
        self.pending_start = 0

    def mark(self, frame_index: int) -> str:
        """Record one completed insertion; frames >= frame_index include it."""
        if self.done >= self.total:
            raise ValueError(f"All {self.total} insertions are already recorded.")
        if not self.active:
            self.pending_start += 1
            return f"Next episode starts with {self.done}/{self.total} insertions done."
        if self.milestones and frame_index < self.milestones[-1]:
            raise ValueError("Insertion milestones must be in frame order.")
        self.milestones.append(int(frame_index))
        return f"Insertion {self.done}/{self.total} recorded at frame {frame_index}."

    def undo(self) -> str:
        """Remove the most recent insertion record."""
        if self.active and self.milestones:
            frame = self.milestones.pop()
            return f"Removed insertion record at frame {frame}; now {self.done}/{self.total}."
        if not self.active and self.pending_start:
            self.pending_start -= 1
            return f"Next episode starts with {self.done}/{self.total} insertions done."
        raise ValueError("No insertion record to undo.")

    def can_succeed(self) -> bool:
        """C is accepted unless marks are required and some are missing."""
        return self.active and (not self.require_marks or self.done == self.total)

    def finish(self, success: bool, frames: int) -> dict:
        """Close the episode and return its score for ``episode.json``.

        After a failure the completed insertions remain in the scene, so they
        become the next episode's start progress; after success it resets.
        """
        if not self.active:
            raise RuntimeError("No scored episode is active.")
        if success and self.require_marks and self.done != self.total:
            raise ValueError(
                f"Success needs {self.total}/{self.total} insertions, got {self.done}."
            )
        if any(frame > frames for frame in self.milestones):
            raise ValueError("Insertion milestone lies beyond the recorded frames.")
        # Without N the progress is unknown: only C/X (the RECAP label) is kept.
        tracked = self.require_marks or bool(self.milestones) or self.start > 0
        complete = tracked and (not success or self.done == self.total)
        score = {
            "total_insertions": self.total,
            "start_insertions": self.start,
            # C certifies every insertion even if some N presses were skipped.
            "insertions_done": self.total if success else self.done,
            "milestone_frames": list(self.milestones),
            "marks_complete": complete,
            "failed_insertion": self.done + 1 if not success and tracked else None,
        }
        self.active, self.milestones = False, []
        self.pending_start = (
            0 if success else min(score["insertions_done"], self.total - 1)
        )
        return score

    def reset_progress(self) -> None:
        """HOME restarts the task from scratch: no insertions carry over."""
        if not self.active:
            self.pending_start = 0

    def abort(self) -> None:
        """Drop an unlabelled episode; keep its start progress for the next one."""
        if self.active:
            self.pending_start = self.start
        self.active, self.milestones = False, []


def insertions_per_frame(score: dict | None, frames: int) -> list[int]:
    """Expand milestone frames into the insertion count completed at each frame."""
    if not score:
        return [0] * frames
    done, marks, counts = score["start_insertions"], score["milestone_frames"], []
    pointer = 0
    for index in range(frames):
        while pointer < len(marks) and marks[pointer] <= index:
            done, pointer = done + 1, pointer + 1
        counts.append(done)
    return counts


def score_report(raw_root: Path) -> dict:
    """Summarize committed raw episodes per insertion, without any export."""
    from rlinf.recap_piper.data import completed_episodes

    episodes, per_insertion = [], {}
    for path in completed_episodes(raw_root):
        meta = json.loads((path / "episode.json").read_text())
        score = meta.get("score")
        if score is None:
            continue
        rows = [
            json.loads(line)
            for line in (path / "frames.jsonl").read_text().splitlines()
        ]
        human = [bool(row["intervention"]) for row in rows]
        bounds = [0, *score["milestone_frames"], len(rows)]
        start, fps = score["start_insertions"], meta["fps"]
        # Per-insertion figures need every N; C/X alone still counts overall.
        marked = score.get("marks_complete", True)
        # Segment k covers the attempt at insertion start+k+1.
        for offset in range(len(bounds) - 1 if marked else 0):
            number = start + offset + 1
            if number > score["total_insertions"]:
                break
            completed = offset < len(score["milestone_frames"])
            item = per_insertion.setdefault(
                number,
                {"attempts": 0, "completed": 0, "with_takeover": 0, "seconds": []},
            )
            segment = human[bounds[offset] : bounds[offset + 1]]
            item["attempts"] += 1
            item["completed"] += int(completed)
            item["with_takeover"] += int(any(segment))
            if completed:
                item["seconds"].append(len(segment) / fps)
        episodes.append(
            {
                "success": bool(meta["is_success"]),
                "from_start": start == 0,
                "takeover": any(human),
                "marked": marked,
            }
        )
    from_start = [item for item in episodes if item["from_start"]]
    report = {
        "scored_episodes": len(episodes),
        "successes": sum(item["success"] for item in episodes),
        "from_start_episodes": len(from_start),
        "per_insertion_episodes": sum(item["marked"] for item in episodes),
        "autonomous_successes": sum(
            item["success"] and not item["takeover"] for item in from_start
        ),
        "per_insertion": {},
    }
    for number, item in sorted(per_insertion.items()):
        seconds = item.pop("seconds")
        item["completion_rate"] = item["completed"] / item["attempts"]
        item["takeover_rate"] = item["with_takeover"] / item["attempts"]
        item["mean_seconds"] = sum(seconds) / len(seconds) if seconds else None
        report["per_insertion"][str(number)] = item
    return report


class EpisodeGate:
    """Decide when an I press starts a RECAP episode.

    I is the start of RECAP's per-step reward (r_t = -1): with no episode
    open, I requests inference and the episode starts as soon as the mux
    confirms it. With an episode open (after T or Space), I resumes the policy
    inside the same episode. After a success the scene must be reset, so a new
    episode is refused until HOME.
    """

    def __init__(self):
        self.armed = False
        self.needs_home = False

    def inference_key(self, recording: bool) -> str:
        """Return ``resume``, ``start`` or ``blocked`` for an I press."""
        if recording:
            return "resume"
        if self.needs_home:
            return "blocked"
        self.armed = True
        return "start"

    def should_start(self, recording: bool, allowed: bool) -> bool:
        """Open the armed episode once inference/takeover is confirmed active."""
        return self.armed and not recording and allowed

    def started(self) -> None:
        self.armed = False

    def cancel(self) -> None:
        """Space/H/Q before confirmation: do not start an episode."""
        self.armed = False

    def finished(self, success: bool) -> None:
        self.armed = False
        self.needs_home = success

    def homed(self) -> None:
        self.needs_home = False
