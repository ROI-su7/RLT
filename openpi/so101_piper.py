from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimitAnalysis:
    """一组映射参数与 follower 限位叠加后的派生结果。"""

    usable_leader_range_deg: tuple[float, float]
    piper_at_usable_range_deg: tuple[float, float]
    lower_limit_owner: str
    upper_limit_owner: str


@dataclass(frozen=True, kw_only=True)
class JointMappingSpec:
    """单个 SO-101/PiPER 关节对的不可变绝对映射事实。"""

    leader_key: str
    piper_key: str
    leader_reference_deg: float
    piper_reference_deg: float
    leader_range_deg: tuple[float, float]
    gain: float

    def __post_init__(self) -> None:
        values = (
            self.leader_reference_deg,
            self.piper_reference_deg,
            *self.leader_range_deg,
            self.gain,
        )
        if not self.leader_key or not self.piper_key:
            raise ValueError("mapping joint keys must not be empty")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("mapping values must be finite")
        if self.leader_range_deg[0] == self.leader_range_deg[1]:
            raise ValueError("leader range endpoints must differ")
        if self.gain == 0:
            raise ValueError("mapping gain must be non-zero")

    @property
    def offset_deg(self) -> float:
        return self.piper_reference_deg - self.gain * self.leader_reference_deg

    def raw_target(self, leader_deg: float) -> float:
        return self.piper_reference_deg + self.gain * (leader_deg - self.leader_reference_deg)

    def analyze(self, piper_range_deg: tuple[float, float]) -> JointLimitAnalysis:
        piper_as_leader = tuple(
            self.leader_reference_deg + (limit - self.piper_reference_deg) / self.gain
            for limit in piper_range_deg
        )
        piper_lower, piper_upper = sorted(piper_as_leader)
        leader_lower, leader_upper = sorted(self.leader_range_deg)
        usable = max(leader_lower, piper_lower), min(leader_upper, piper_upper)
        if usable[0] > usable[1]:
            raise ValueError(f"{self.leader_key} and {self.piper_key} ranges do not overlap")
        return JointLimitAnalysis(
            usable_leader_range_deg=usable,
            piper_at_usable_range_deg=tuple(self.raw_target(value) for value in usable),
            lower_limit_owner=("so101" if math.isclose(usable[0], leader_lower) else "piper"),
            upper_limit_owner=("so101" if math.isclose(usable[1], leader_upper) else "piper"),
        )


@dataclass(frozen=True)
class SO101PiperMappingProfile:
    """将特定 SO-101 calibration 与 PiPER 配对的不可变 profile。"""

    name: str
    calibration_id: str
    joints: tuple[JointMappingSpec, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.calibration_id or not self.joints:
            raise ValueError("mapping profile metadata and joints must not be empty")
        leader_keys = [joint.leader_key for joint in self.joints]
        piper_keys = [joint.piper_key for joint in self.joints]
        if len(set(leader_keys)) != len(leader_keys):
            raise ValueError("mapping profile contains duplicate leader keys")
        if len(set(piper_keys)) != len(piper_keys):
            raise ValueError("mapping profile contains duplicate PiPER keys")

    def first_joints(self, count: int) -> SO101PiperMappingProfile:
        return SO101PiperMappingProfile(
            name=f"{self.name}_{count}dof",
            calibration_id=self.calibration_id,
            joints=self.joints[:count],
        )

    def analyze(
        self, piper_ranges_deg: dict[str, tuple[float, float]]
    ) -> dict[str, JointLimitAnalysis]:
        if set(piper_ranges_deg) != {joint.piper_key for joint in self.joints}:
            raise ValueError("PiPER range keys must match mapping profile")
        return {
            joint.leader_key: joint.analyze(piper_ranges_deg[joint.piper_key])
            for joint in self.joints
        }


DEFAULT_SO101_PIPER_PROFILE = SO101PiperMappingProfile(
    name="so101_leader_to_piper_default",
    calibration_id="so101_leader",
    joints=(
        JointMappingSpec(
            leader_key="shoulder_pan.pos",
            piper_key="joint_1.pos",
            leader_reference_deg=0.0,
            piper_reference_deg=0.0,
            leader_range_deg=(-70.0, 70.0),
            gain=100.0 / 70.0,
        ),
        JointMappingSpec(
            leader_key="shoulder_lift.pos",
            piper_key="joint_2.pos",
            leader_reference_deg=-73.758,
            piper_reference_deg=0.0,
            leader_range_deg=(-260.0, 105.0),
            # SO-101 从参考姿态到实测正向端点时，PiPER joint2 到 170°，
            # 与 nominal 180° 限位保留 10° 余量。
            gain=170.0 / (105.0 - (-73.758)),
        ),
        JointMappingSpec(
            leader_key="elbow_flex.pos",
            piper_key="joint_3.pos",
            leader_reference_deg=60.879,
            piper_reference_deg=0.0,
            leader_range_deg=(-120.0, 240.0),
            # SO-101 从参考姿态伸到实测负向端点时，PiPER joint3 到 -160°，
            # 与 nominal -170° 限位保留 10° 余量。
            gain=160.0 / (60.879 - (-120.0)),
        ),
        JointMappingSpec(
            leader_key="wrist_flex.pos",
            piper_key="joint_5.pos",
            leader_reference_deg=33.0,
            piper_reference_deg=16.0,
            leader_range_deg=(-90.0, 85.0),
            gain=60.0 / 110.0,
        ),
        JointMappingSpec(
            leader_key="wrist_roll.pos",
            piper_key="joint_6.pos",
            leader_reference_deg=83.560,
            piper_reference_deg=0.0,
            leader_range_deg=(-10.0, 170.0),
            gain=200.0 / 180.0,
        ),
    ),
)
