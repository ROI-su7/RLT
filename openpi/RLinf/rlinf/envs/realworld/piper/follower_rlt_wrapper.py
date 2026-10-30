"""Stage-2 keyboard collection for the two-follower ROS mux."""

from rlinf.envs.realworld.piper.mux_rlt_wrapper import PiperMuxRLTWrapper


class PiperFollowerRLTWrapper(PiperMuxRLTWrapper):
    """Reuse reward/replay handling while disabling human takeover requests."""

    def handle_key(self, key: str) -> None:
        """Ignore T; delegate I/B/C/X/Space/H to the existing episode control."""
        if key.lower() == "t":
            self._log("T is disabled in follower-only Stage 2.")
            return
        if key == "i":
            with self._lock:
                if self._resume_requested and self.mux.snapshot().hold:
                    # A stale-feedback rejection leaves the mux in HOLD.
                    # Let the operator retry I once feedback is restored.
                    self._resume_requested = False
        super().handle_key(key)

    def _log(self, message: str) -> None:
        message = message.replace("all four arms", "both followers").replace(
            "four-arm alignment", "fresh follower feedback"
        )
        super()._log(message)

    def _info(self, *, human: bool, valid: bool) -> dict:
        info = super()._info(human=human, valid=valid)
        info["rlt_keyboard_terminal"] = False
        return info

    def _terminal_step(self, *, truncated: bool = False):
        first = not self._terminal_reported
        result = super()._terminal_step(truncated=truncated)
        result[-1]["rlt_keyboard_terminal"] = (
            first and self._recording and self._terminal in {"success", "failure"}
        )
        return result
