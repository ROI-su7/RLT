"""RECAP keyboard/recorder using the existing Piper ROS mux, never direct CAN.

I/T/Space/H publish the same requests as piper_mode_keyboard.py. I also opens a
labelled RECAP episode (from then on r_t = -1 per step); C/X close it, and
N/U score the three-insertion progress
(rlinf.recap_piper.scoring; the RLT keyboard reward is separate and unchanged).
Run instead of, not alongside, the ROS keyboard.
The existing OpenPI real-robot client remains the only policy publisher.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import logging
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

from rlinf.recap_piper.data import CAMERAS, EpisodeWriter
from rlinf.recap_piper.scoring import TOTAL_INSERTIONS, EpisodeGate, InsertionScore

logger = logging.getLogger(__name__)


def decode_image(message) -> np.ndarray:
    """Decode RGB/BGR ROS Image without the Python-3.8 cv_bridge extension."""
    if message.encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"Expected rgb8/bgr8 camera, received {message.encoding}")
    data = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.step
    )
    image = data[:, : message.width * 3].reshape(message.height, message.width, 3)
    return image[..., ::-1].copy() if message.encoding == "bgr8" else image.copy()


class RosSamples:
    """Read-only timestamped sensor, mux action and ownership subscriptions."""

    def __init__(self, rospy, max_age: float, max_skew: float):
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Bool, Empty, String

        self.lock = threading.RLock()
        self.samples = collections.defaultdict(lambda: collections.deque(maxlen=60))
        self.status = {
            "mode": "UNKNOWN",
            "phase": "UNKNOWN",
            "active": False,
            "hold": True,
            "home": False,
        }
        self.home_serial = 0
        self.max_age, self.max_skew = max_age, max_skew
        self.last_frame = -1.0
        self.subscriptions = []
        for name, topic in zip(
            CAMERAS,
            (
                "/camera_f/color/image_raw",
                "/camera_l/color/image_raw",
                "/camera_r/color/image_raw",
            ),
            strict=True,
        ):
            self.subscriptions.append(
                rospy.Subscriber(
                    topic,
                    Image,
                    self._sample,
                    callback_args=name,
                    queue_size=2,
                    buff_size=4 * 1024 * 1024,
                    tcp_nodelay=True,
                )
            )
        for prefix, base in (
            ("state", "/puppet/joint_"),
            ("action", "/master/joint_"),
            ("policy", "/teleop/policy/joint_"),
            ("leader", "/teleop/leader/joint_"),
        ):
            for side in ("left", "right"):
                self.subscriptions.append(
                    rospy.Subscriber(
                        base + side,
                        JointState,
                        self._sample,
                        callback_args=f"{prefix}_{side}",
                        queue_size=10,
                        tcp_nodelay=True,
                    )
                )
        for key, topic, kind in (
            ("mode", "/teleop/mode", String),
            ("phase", "/teleop/takeover_phase", String),
            ("active", "/teleop/inference_active", Bool),
            ("hold", "/teleop/hold_active", Bool),
            ("home", "/teleop/home_active", Bool),
        ):
            self.subscriptions.append(
                rospy.Subscriber(
                    topic, kind, self._status, callback_args=key, queue_size=1
                )
            )
        self.subscriptions.append(
            rospy.Subscriber("/teleop/home_request", Empty, self._home, queue_size=1)
        )

    def _sample(self, message, name: str) -> None:
        stamp = message.header.stamp.to_sec()
        if stamp <= 0:
            return
        with self.lock:
            self.samples[name].append((stamp, time.monotonic(), message))

    def _status(self, message, key: str) -> None:
        with self.lock:
            self.status[key] = (
                message.data.strip().upper()
                if key in {"mode", "phase"}
                else bool(message.data)
            )

    def _home(self, _message) -> None:
        with self.lock:
            self.home_serial += 1

    def recording_allowed(self) -> bool:
        with self.lock:
            s = self.status
            if s["home"]:
                return False
            if s["mode"] == "INFERENCE":
                return s["active"] and not s["hold"]
            # piper_four_state_mux latches /teleop/hold_active=True when T is
            # accepted and clears it only on the next INFERENCE, so it stays
            # stale through TAKEOVER. A real HOLD changes the mode itself.
            return s["mode"] == "TAKEOVER" and s["phase"] == "ACTIVE"

    def control_state(self) -> str:
        """Short operator-facing ownership summary for transition logs."""
        with self.lock:
            s = self.status
            if s["mode"] == "TAKEOVER":
                return f"TAKEOVER/{s['phase']}"
            return s["mode"] + ("/ACTIVE" if s["active"] else "")

    def frame(self) -> dict | None:
        """Pair observations and commands on one camera timestamp, rejecting stale data."""
        with self.lock:
            if not self.recording_allowed() or not self.samples[CAMERAS[0]]:
                return None
            stamp = self.samples[CAMERAS[0]][-1][0]
            if stamp <= self.last_frame:
                return None
            now = time.monotonic()

            def nearest(name):
                history = self.samples[name]
                if not history:
                    return None
                sample = min(history, key=lambda sample: abs(sample[0] - stamp))
                if (
                    abs(sample[0] - stamp) > self.max_skew
                    or now - sample[1] > self.max_age
                ):
                    return None
                return sample[2]

            required = list(CAMERAS) + [
                f"{p}_{s}" for p in ("state", "action") for s in ("left", "right")
            ]
            messages = {name: nearest(name) for name in required}
            if any(value is None for value in messages.values()):
                return None

            def pair(prefix, required=False):
                values = [nearest(f"{prefix}_{side}") for side in ("left", "right")]
                good = all(
                    value is not None and len(value.position) == 7 for value in values
                )
                if not good:
                    if required:
                        raise ValueError(f"Missing/stale {prefix} joint pair")
                    return np.zeros(14, np.float32), False
                return np.asarray(
                    [v for message in values for v in message.position], np.float32
                ), True

            state, _ = pair("state", required=True)
            action, _ = pair("action", required=True)
            policy, policy_valid = pair("policy")
            human = self.status["mode"] == "TAKEOVER"
            leader, leader_valid = pair("leader", required=human)
            frame = {
                "state": state,
                "action": action,
                "policy_action": policy,
                "leader_action": leader,
                "leader_valid": leader_valid,
                "policy_valid": policy_valid,
                "timestamp": stamp,
                "intervention": human,
                "images": {name: decode_image(messages[name]) for name in CAMERAS},
            }
            self.last_frame = stamp
            return frame


def attach_console(target: logging.Logger, stream=None) -> logging.Handler:
    """Keep operator prompts on the terminal.

    ``rospy.init_node`` reconfigures the root logger to write only to
    ``~/.ros/log``, which would hide every key prompt from the operator.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    )
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    return handler


def main() -> None:
    """Own the terminal keyboard and record only explicitly scored episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", required=True, help="Use the source model config's exact prompt"
    )
    parser.add_argument(
        "--profile", choices=("takeover", "followers"), default="takeover"
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--insertions", type=int, default=TOTAL_INSERTIONS)
    parser.add_argument(
        "--require-marks",
        action="store_true",
        help="Refuse C until N marked every insertion (per-insertion statistics).",
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=0.0,
        help="RECAP success means finishing within this recorded time; the "
        "episode is scored as failure (X) when it is exceeded. 0 disables it.",
    )
    parser.add_argument("--max-age", type=float, default=0.5)
    parser.add_argument("--max-skew", type=float, default=0.08)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("An interactive terminal is required.")
    if args.fps <= 0 or args.max_age <= 0 or args.max_skew <= 0:
        parser.error("fps/max-age/max-skew must be positive.")
    if args.time_limit_s < 0:
        parser.error("time-limit-s must be >= 0.")
    import rospy
    from std_msgs.msg import Empty, String

    logging.basicConfig(level=logging.INFO)
    rospy.init_node("piper_recap_recorder", anonymous=True)
    attach_console(logger)
    samples = RosSamples(rospy, args.max_age, args.max_skew)
    mode = rospy.Publisher("/teleop/mode_request", String, queue_size=10)
    hold = rospy.Publisher("/teleop/hold_request", Empty, queue_size=1)
    home = rospy.Publisher("/teleop/home_request", Empty, queue_size=1)
    writer = None
    score = InsertionScore(args.insertions, require_marks=args.require_marks)
    gate = EpisodeGate()
    recorded = 0
    futures = collections.deque()
    last_frame_time, next_tick = time.monotonic(), time.monotonic()
    home_serial = samples.home_serial
    old_settings = termios.tcgetattr(sys.stdin)
    logger.info(
        "I=inference+start/resume episode T=takeover C=success(all %d) X=failure "
        "Space=HOLD H=HOME Q=quit; N/U=%s insertion marks",
        args.insertions,
        "required" if args.require_marks else "optional",
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    last_control = None
    # The mux resumes INFERENCE only from HOLD: I during TAKEOVER first requests
    # HOLD, then INFERENCE once HOLD is confirmed (was: Space, then I).
    resume_deadline = None
    operator_hold_at = -float("inf")

    def drain():
        while futures:
            futures.popleft().result()

    def start_episode():
        nonlocal writer, recorded, home_serial, last_frame_time
        writer = EpisodeWriter(args.output, args.task, args.fps)
        score.begin()
        gate.started()
        recorded = 0
        home_serial = samples.home_serial
        last_frame_time = time.monotonic()
        logger.info(
            "Episode %s started from %d/%d insertions done. RECAP: r_t = -1 per "
            "recorded step until C/X.",
            writer.path,
            score.done,
            score.total,
        )

    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            while futures and futures[0].done():
                futures.popleft().result()
            key = (
                sys.stdin.read(1).lower()
                if select.select([sys.stdin], [], [], 0.005)[0]
                else ""
            )
            if key in {"q", "\x03"}:
                break
            if (
                writer is not None
                and args.time_limit_s > 0
                and key != "c"
                and recorded >= args.time_limit_s * args.fps
            ):
                logger.warning(
                    "Time limit %.0f s reached without success: scoring as X.",
                    args.time_limit_s,
                )
                key = "x"
            if key == "i":
                action = gate.inference_key(writer is not None)
                if action == "blocked":
                    logger.warning(
                        "Last episode succeeded: press H to return home and reset "
                        "all %d wires before I starts a new episode.",
                        score.total,
                    )
                    continue
                if samples.control_state().startswith("TAKEOVER"):
                    hold.publish(Empty())
                    operator_hold_at = time.monotonic()
                    resume_deadline = operator_hold_at + 3.0
                    logger.info(
                        "I: TAKEOVER -> HOLD -> INFERENCE (automatic); the policy "
                        "resumes inside the same episode."
                    )
                    continue
                mode.publish(String(data="INFERENCE"))
                logger.info(
                    "I: policy resumes inside the current episode."
                    if action == "resume"
                    else "I: inference requested; the episode starts when the mux "
                    "confirms INFERENCE."
                )
            elif key == "t" and args.profile == "takeover":
                operator_hold_at = time.monotonic()
                mode.publish(String(data="TAKEOVER"))
                logger.info(
                    "Wait for the original mux's TAKEOVER ACTIVE before moving leaders. "
                    "RECAP: human frames get intervention=1 (forced positive I_t)."
                )
            elif key == " ":
                hold.publish(Empty())
                operator_hold_at = time.monotonic()
                resume_deadline = None
                gate.cancel()
                if writer is not None:
                    logger.info("HOLD: episode stays open; I resumes the policy.")
            elif key == "h":
                if not samples.status["hold"]:
                    logger.warning(
                        "Press Space or finish with C/X, wait for HOLD, then press H."
                    )
                    continue
                # No outcome is inferred from HOME; unlabelled work is excluded.
                hold.publish(Empty())
                if writer is not None:
                    drain()
                    writer.abort("HOME before C/X")
                    score.abort()
                    writer = None
                    logger.warning("HOME before C/X: the open episode was discarded.")
                gate.cancel()
                gate.homed()
                score.reset_progress()
                home.publish(Empty())
                logger.info(
                    "HOME requested. Reset all %d wires, then I starts a new episode "
                    "from the beginning.",
                    score.total,
                )
            elif key == "b" and writer is None:
                # Optional manual start, e.g. an episode that begins in TAKEOVER.
                if not samples.recording_allowed():
                    logger.warning(
                        "B needs confirmed active INFERENCE or TAKEOVER; press I and wait."
                    )
                else:
                    start_episode()
            elif key in {"n", "u"}:
                try:
                    logger.info(score.mark(recorded) if key == "n" else score.undo())
                except ValueError as exc:
                    logger.warning("%s", exc)
            elif key == "c" and writer is not None and not score.can_succeed():
                logger.warning(
                    "C needs %d/%d insertions; press N for each finished insertion, "
                    "or X for failure.",
                    score.total,
                    score.total,
                )
            elif key in {"c", "x"} and writer is not None:
                hold.publish(Empty())
                operator_hold_at = time.monotonic()
                resume_deadline = None
                drain()
                if writer.count:
                    result = score.finish(key == "c", writer.count)
                    logger.info(
                        "Saved %s outcome=%s frames=%d insertions=%d/%d. RECAP: "
                        "is_success=%d, terminal reward %s, every other step -1.",
                        writer.finish(key == "c", result),
                        key,
                        writer.count,
                        result["insertions_done"],
                        result["total_insertions"],
                        key == "c",
                        "0" if key == "c" else "= -C_fail (data.failure_reward)",
                    )
                    gate.finished(key == "c")
                    if key == "c":
                        logger.info(
                            "Success. Press H, reset all wires, then I for the next episode."
                        )
                    else:
                        logger.info(
                            "Failure saved%s. I continues in a new episode.",
                            ""
                            if result["failed_insertion"] is None
                            else f" on insertion {result['failed_insertion']}",
                        )
                else:
                    writer.abort("C/X before any valid sensor frame")
                    score.abort()
                writer = None
            control = samples.control_state()
            if resume_deadline is not None:
                if control == "HOLD":
                    mode.publish(String(data="INFERENCE"))
                    resume_deadline = None
                elif time.monotonic() > resume_deadline:
                    resume_deadline = None
                    logger.warning(
                        "The mux did not confirm HOLD within 3 s; press Space, then I."
                    )
            if control != last_control:
                if (
                    control == "HOLD"
                    and last_control == "INFERENCE/ACTIVE"
                    and writer is not None
                    and time.monotonic() - operator_hold_at > 1.5
                ):
                    logger.warning(
                        "The mux entered HOLD by itself (policy commands stopped for "
                        "> policy_input_timeout_s, see the client/server logs). The "
                        "episode stays open: press I to resume."
                    )
                elif control == "TAKEOVER/ACTIVE" and writer is not None:
                    logger.info(
                        "TAKEOVER ACTIVE: recording human frames (intervention=1). "
                        "Press I to hand back to the policy."
                    )
                elif last_control == "TAKEOVER/ACTIVE" and writer is not None:
                    logger.info("Left TAKEOVER (%s).", control)
                last_control = control
            if writer is None:
                if samples.home_serial != home_serial:
                    home_serial = samples.home_serial
                    gate.homed()
                    score.reset_progress()
                if gate.should_start(False, samples.recording_allowed()):
                    start_episode()
                else:
                    continue
            if samples.home_serial != home_serial:
                drain()
                writer.abort("External HOME before outcome")
                score.abort()
                writer = None
                continue
            now = time.monotonic()
            if not samples.recording_allowed():
                last_frame_time = now
                continue
            if now < next_tick:
                continue
            frame = samples.frame()
            if frame is None:
                if now - last_frame_time > args.max_age:
                    raise RuntimeError(
                        "Sensor/command stream stale during recording; episode aborted."
                    )
                continue
            # ~2 s of buffering absorbs disk jitter; a sustained backlog aborts.
            if len(futures) >= 2 * args.fps:
                raise RuntimeError(
                    "Disk writer cannot keep up; episode aborted instead of silently dropping frames."
                )
            futures.append(executor.submit(writer.add_frame, frame))
            recorded += 1
            last_frame_time = now
            next_tick = now + 1.0 / args.fps
    finally:
        hold.publish(Empty())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        try:
            drain()
        finally:
            if writer is not None:
                writer.abort("Recorder exit or exception before C/X")
            executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
