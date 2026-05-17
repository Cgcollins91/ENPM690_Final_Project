"""

Runtime constants and pure task predicates for the topdown trainer

File map:

SUPPORTED_TOPDOWN_TASK:              Define supported topdown task constant
TOPDOWN_CURRICULUM_TASKS:            Define topdown curriculum tasks constant
TOPDOWN_SOURCE_BLOCK_NAMES:          Define topdown source block names constant
env_flag:                            Read a typed boolean environment override
env_float:                           Read a typed float environment override
TopdownTaskRuntime:                  Task routing state without Isaac side effects
topdown_source_block_name:           Return the display name for a source-pose block index
topdown_episode_failure_mode:        Classify the main topdown failure reason for one episode
topdown_lift_physical_success:       Return whether lift metrics satisfy the physical success gates
topdown_lift_physical_success_mask:  Return per-env physical success gates for lift metrics
radians_from_env_degrees:            Read a degree-valued environment override and return radians
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from collections.abc import Mapping

import torch


SUPPORTED_TOPDOWN_TASK = "Isaac-Topdown-Curriculum-G129-Dex3-Joint"
TOPDOWN_CURRICULUM_TASKS = frozenset({SUPPORTED_TOPDOWN_TASK})
TOPDOWN_SOURCE_BLOCK_NAMES = ("red", "yellow", "blue")


def env_flag(name: str, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    """Read a typed boolean environment override"""
    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float, env: Mapping[str, str] | None = None) -> float:
    """Read a typed float environment override"""
    source = os.environ if env is None else env
    raw = source.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class TopdownTaskRuntime:
    """Task routing state without Isaac side effects"""

    task: str                      = SUPPORTED_TOPDOWN_TASK  # string task value used by topdown task runtime
    env : Mapping[str, str] | None = None  # environment/backend object used by this runtime helper

    def validate_supported(self) -> None:
        """Raise when the task is outside the standalone trainer contract"""
        if self.task != SUPPORTED_TOPDOWN_TASK:
            raise RuntimeError(
                "ENPM690 standalone trainer supports only "
                f"{SUPPORTED_TOPDOWN_TASK}; got {self.task!r}"
            )

    @property
    def is_topdown_curriculum(self) -> bool:
        """Return whether the active task is the supported curriculum task"""
        return self.task in TOPDOWN_CURRICULUM_TASKS

    @property
    def is_topdown_curriculum_lift(self) -> bool:
        """Return whether lift mode is enabled for the supported curriculum task"""
        return self.is_topdown_curriculum and env_flag("TOPDOWN_LIFT_TASK", False, self.env)

    @property
    def use_topdown_contact_chain(self) -> bool:
        """Return whether contact metrics route through topdown sensors"""
        return True

    @property
    def lift_success_contact_min(self) -> float:
        """Return the lift success contact threshold"""
        return env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30, self.env)

    @property
    def lift_success_height(self) -> float:
        """Return the lift success height threshold"""
        return env_float("TOPDOWN_LIFT_SUCCESS_HEIGHT", 0.05, self.env)

    @property
    def lift_success_xy_drift_max(self) -> float:
        """Return the lift success XY drift threshold"""
        return env_float("TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX", 0.05, self.env)

    @property
    def lift_success_block_tilt_max_deg(self) -> float:
        """Return the lift success tilt threshold in degrees"""
        return env_float("TOPDOWN_LIFT_SUCCESS_BLOCK_TILT_MAX_DEG", 25.0, self.env)

    @property
    def block_drift_threshold(self) -> float:
        """Return the block drift failure threshold"""
        return env_float("CURRICULUM_BLOCK_DRIFT_THRESHOLD", 0.12, self.env)


def topdown_source_block_name(source_idx: int) -> str:
    """Return the display name for a source-pose block index"""
    if 0 <= source_idx < len(TOPDOWN_SOURCE_BLOCK_NAMES):
        return TOPDOWN_SOURCE_BLOCK_NAMES[source_idx]
    return "unknown"


def topdown_episode_failure_mode(
    *,
    success                      : bool = False,  # Param: boolean input controlling success
    physical_success             : bool = False,  # Param: boolean input controlling physical success
    off_table                    : bool = False,  # Param: boolean input controlling off table
    timeout                      : bool,  # Param: boolean input controlling timeout
    block_drift                  : bool,  # Param: boolean input controlling block drift
    best_stage                   : int,  # Param: integer input for best stage
    max_unlock                   : float,  # Param: floating-point input for max unlock
    best_contact                 : float,  # Param: floating-point input for best contact
    best_strict_contact          : float,  # Param: floating-point input for best strict contact
    best_lift                    : float,  # Param: floating-point input for best lift
    best_lift_with_strict_contact: float,  # Param: floating-point input for best lift with strict contact
    contact_min                  : float,  # Param: floating-point input for contact min
    lift_height_min              : float,  # Param: floating-point input for lift height min
) -> str:
    """Classify the main topdown failure reason for one episode

    Steps:
    - Resolve inputs for `topdown_episode_failure_mode` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if success or physical_success:
        return "success"
    if off_table:
        return "off_table"
    if block_drift:
        return "block_drift"
    if best_stage < 1:
        return "stage0_only"
    if best_stage < 2:
        return "no_stage2"
    if max_unlock < 0.95:
        return "no_finger_unlock"
    if best_contact < 0.5:
        return "no_contact"
    if best_strict_contact < contact_min:
        return "no_opposed_contact"
    if best_lift < lift_height_min:
        return "insufficient_lift"
    if best_lift_with_strict_contact < lift_height_min:
        return "lost_contact_during_lift"
    if timeout:
        return "timeout"
    return "other"


def topdown_lift_physical_success(
    *,
    best_lift_with_strict_contact: float,  # Param: floating-point input for best lift with strict contact
    best_block_disp              : float,  # Param: floating-point input for best block disp
    max_block_tilt_deg           : float,  # Param: floating-point input for max block tilt deg
    lift_height_min              : float,  # Param: floating-point input for lift height min
    xy_drift_max                 : float,  # Param: floating-point input for xy drift max
    block_tilt_max_deg           : float,  # Param: floating-point input for block tilt max deg
) -> bool:
    """Return whether lift metrics satisfy the physical success gates"""
    tilt_gate = float(block_tilt_max_deg) <= 0.0 or float(max_block_tilt_deg) <= float(block_tilt_max_deg)
    return (
        float(best_lift_with_strict_contact) >= float(lift_height_min)
        and float(best_block_disp) <= float(xy_drift_max)
        and tilt_gate
    )


def topdown_lift_physical_success_mask(
    *,
    best_lift_with_strict_contact: torch.Tensor,  # Param: tensor input carrying best lift with strict contact values
    best_block_disp              : torch.Tensor,  # Param: tensor input carrying best block disp values
    max_block_tilt_deg           : torch.Tensor,  # Param: tensor input carrying max block tilt deg values
    lift_height_min              : float,  # Param: floating-point input for lift height min
    xy_drift_max                 : float,  # Param: floating-point input for xy drift max
    block_tilt_max_deg           : float,  # Param: floating-point input for block tilt max deg
) -> torch.Tensor:
    """Return per-env physical success gates for lift metrics"""
    lift_ok = best_lift_with_strict_contact.to(dtype=torch.float32) >= float(lift_height_min)
    drift_ok = best_block_disp.to(device=lift_ok.device, dtype=torch.float32) <= float(xy_drift_max)
    if float(block_tilt_max_deg) <= 0.0:
        tilt_ok = torch.ones_like(lift_ok, dtype=torch.bool)
    else:
        tilt_ok = max_block_tilt_deg.to(device=lift_ok.device, dtype=torch.float32) <= float(block_tilt_max_deg)
    return lift_ok & drift_ok & tilt_ok


def radians_from_env_degrees(name: str, default_degrees: float, env: Mapping[str, str] | None = None) -> float:
    """Read a degree-valued environment override and return radians"""
    return math.radians(env_float(name, default_degrees, env))
