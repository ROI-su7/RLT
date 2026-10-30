"""Compare requested and executor-filtered targets with aligned JAX references."""

from __future__ import annotations

import numpy as np


def summarize_correction(
    executed: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    human: np.ndarray,
    *,
    actor: bool,
) -> str:
    """Report command deviations, excluding padding and human takeover steps.

    These are environment-requested actions, not measured motor positions or
    the executor-filtered goals. The detailed chain reports the latter too.
    """
    executed = np.asarray(executed, dtype=np.float32).reshape(-1, 14)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1, 14)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    human = np.asarray(human, dtype=bool).reshape(-1)
    if executed.shape != reference.shape or len(valid) != len(executed):
        raise ValueError("Correction diagnostics require time-aligned 14D targets.")
    if human.shape != valid.shape:
        raise ValueError("Human and valid masks must have one flag per control step.")
    policy = valid & ~human
    human_steps = int((valid & human).sum())
    source = "ACTOR" if actor else "JAX_REFERENCE"
    if human_steps:
        source = "HUMAN" if not policy.any() else f"MIXED({source}+HUMAN)"
    prefix = (
        f"[RLT correction] source={source} policy_steps={int(policy.sum())} "
        f"human_steps={human_steps} comparison=recorded_action-minus-aligned_JAX"
    )
    if not policy.any():
        return prefix + " correction=not_applicable"
    delta = executed[policy] - reference[policy]
    parts = []
    for side, start in (("left", 0), ("right", 7)):
        arm = delta[:, start : start + 6]
        gripper = delta[:, start + 6]
        signed = ",".join(f"{v:+.5f}" for v in arm.mean(axis=0))
        parts.append(
            f"{side}: arm_mean_abs={np.abs(arm).mean():.5f}rad "
            f"arm_max_abs={np.abs(arm).max():.5f}rad "
            f"joint_mean_signed=[{signed}]rad "
            f"gripper_mean_signed={gripper.mean():+.5f}m "
            f"gripper_max_abs={np.abs(gripper).max():.5f}m"
        )
    return prefix + " | " + " | ".join(parts)


def format_actor_correction_chain(
    stages: dict[str, np.ndarray],
    reference: np.ndarray,
    rtc_request: np.ndarray,
    accepted: np.ndarray | None,
    valid: np.ndarray,
    human: np.ndarray,
    *,
    chunk_id: int,
    rtc_offset: int,
    executor_params: np.ndarray | None = None,
) -> list[str]:
    """Compare every stage at the SAME step/joint, never independent maxima.

    Executor targets are pre-ROS-mux goals after controller filtering, not
    measured motor motion or a counterfactual JAX-only execution.
    """
    reference = np.asarray(reference).reshape(-1, 14)
    rtc_request = np.asarray(rtc_request).reshape(-1, 14)
    if rtc_request.shape != reference.shape:
        raise ValueError("Actor diagnostics require time-aligned RTC targets.")
    for name in ("raw", "clipped", "smoothed", "tapered"):
        if np.asarray(stages[name]).shape != reference.shape:
            raise ValueError(f"Actor diagnostic stage {name} is not time-aligned.")
    mask = np.asarray(valid, bool).reshape(-1) & ~np.asarray(human, bool).reshape(-1)
    mask &= np.asarray(stages["coverage"], bool).reshape(-1)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    params = np.asarray(stages["params"]).reshape(-1)
    header = (
        f"[RLT actor chain] chunk={chunk_id} update={int(params[4])} rtc_offset={rtc_offset} "
        f"arm_clip={params[0]:.5f}rad gripper_clip={params[1]:.5f}m "
        f"actor_alpha={params[2]:.3f} taper_steps={int(params[3])} EMA/taper=arm-only "
        "rows=same-step/joint at raw-peak; executor=filtered goal BEFORE ROS mux"
    )
    if len(params) >= 8:
        header += f" basis_size={int(params[7])}"
    if len(params) >= 9:
        header += (
            f" explore_arm_std={params[8]:.4f}rad"
            if params[8] > 0
            else " explore=off(mean actor)"
        )
    if executor_params is not None:
        p = np.asarray(executor_params).reshape(-1, 3)[indices[0]]
        header += (
            f" executor_alpha={p[0]:.3f} arm_vmax={p[1]:.3f}rad/s interp={p[2]:.4f}s"
        )
    rows = [header]
    accepted = None if accepted is None else np.asarray(accepted).reshape(-1, 14)
    if accepted is not None and accepted.shape != reference.shape:
        raise ValueError("Executor diagnostics require time-aligned goals.")
    for side, start in (("L", 0), ("R", 7)):
        for joint in range(7):
            column = start + joint
            step = int(indices[np.argmax(np.abs(stages["raw"][indices, column]))])
            values = {
                key: float(stages[key][step, column])
                for key in (
                    "raw",
                    "projected",
                    "clipped",
                    "smoothed",
                    "tapered",
                    "slewed",
                )
                if key in stages
            }
            projected = (
                f"proj={values['projected']:+.5f} " if "projected" in values else ""
            )
            rtc_delta = float(rtc_request[step, column] - reference[step, column])
            unit = "m" if joint == 6 else "rad"
            name = "G" if joint == 6 else f"J{joint + 1}"
            factor = 1.0
            if joint != 6 and len(params) >= 7:
                n = min(int(params[3]), int(params[6]))
                if n:
                    factor = max(
                        0.0,
                        min(1.0, (params[5] + params[6] - 1 - rtc_offset - step) / n),
                    )
            final = (
                None
                if accepted is None or not np.isfinite(accepted[step, column])
                else float(accepted[step, column] - reference[step, column])
            )
            target = "n/a" if final is None else f"{accepted[step, column]:+.5f}"
            executor = "n/a" if final is None else f"{final:+.5f}"
            ratio = (
                "n/a"
                if final is None or abs(values["raw"]) < 1e-7
                else f"{100 * final / values['raw']:.1f}%"
            )
            rows.append(
                f"[RLT actor chain] {side}.{name} step={step} plan_step={rtc_offset + step} "
                f"ref={reference[step, column]:+.5f} target={target} | "
                f"raw={values['raw']:+.5f} {projected}clip={values['clipped']:+.5f} "
                f"smooth={values['smoothed']:+.5f} taper={values['tapered']:+.5f} "
                + (f"slew={values['slewed']:+.5f} " if "slewed" in values else "")
                + f"taper_weight={factor:.3f} "
                f"rtc={rtc_delta:+.5f} executor={executor} {unit} retained={ratio}"
            )
    return rows
