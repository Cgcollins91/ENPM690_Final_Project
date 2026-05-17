"""

Eval terminal success metric override helpers

File map:

EvalSuccessThresholds:                Thresholds used to bound terminal success metrics
_float_value:                         Handle float value logic
_min_value:                           Handle min value logic
_max_value:                           Handle max value logic
apply_eval_success_scalar_overrides:  Apply monolith terminal-success bounds to eval scalar metrics
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .eval_metrics import EvalTaskKind


@dataclass(frozen=True)
class EvalSuccessThresholds:
    """Thresholds used to bound terminal success metrics"""

    pregrasp_tight_tolerance               : float  # threshold/tolerance used when evaluating pregrasp tight tolerance
    grasp_success_palm_ready_tolerance     : float  # threshold/tolerance used when evaluating grasp success palm ready tolerance
    grasp_success_palm_orient_deg          : float  # floating-point grasp success palm orient deg value used by eval success thresholds
    open_hand_align_face_distance_tolerance: float  # threshold/tolerance used when evaluating open hand align face distance tolerance
    open_hand_align_angle_deg              : float  # floating-point open hand align angle deg value used by eval success thresholds
    grasp_success_opposed_face_threshold   : float  # threshold/tolerance used when evaluating grasp success opposed face threshold
    grasp_success_block_disp_max           : float  # floating-point grasp success block disp max value used by eval success thresholds
    curl_success_threshold                 : float  # threshold/tolerance used when evaluating curl success threshold
    topdown_lift_success_contact_min       : float  # floating-point topdown lift success contact min value used by eval success thresholds
    topdown_lift_success_height            : float  # floating-point topdown lift success height value used by eval success thresholds
    topdown_lift_success_xy_drift_max      : float  # floating-point topdown lift success xy drift max value used by eval success thresholds
    topdown_lift_success_block_tilt_max_deg: float = 0.0  # floating-point topdown lift success block tilt max deg value used by eval success thresholds


def _float_value(values: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _min_value(row: dict[str, Any], key: str, value: float) -> None:
    row[key] = min(_float_value(row, key, float("inf")), float(value))


def _max_value(row: dict[str, Any], key: str, value: float) -> None:
    row[key] = max(_float_value(row, key, -float("inf")), float(value))


def apply_eval_success_scalar_overrides(
    values: Mapping[str, Any],          # Param: string input for values
    *,
    task_kind : EvalTaskKind,  # Param: input value used as task kind
    thresholds: EvalSuccessThresholds,  # Param: input value used as thresholds
) -> dict[str, Any]:
    """Apply monolith terminal-success bounds to eval scalar metrics

    Steps:
    - Resolve inputs for `apply_eval_success_scalar_overrides` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    row = dict(values)
    if task_kind == "grasp_align":
        for key in ("best_phase1_palm", "last_phase1_palm"):
            _min_value(row, key, thresholds.grasp_success_palm_ready_tolerance)
        for key in ("best_phase1_orient", "last_phase1_orient"):
            _min_value(row, key, thresholds.grasp_success_palm_orient_deg)
        for key in ("best_align_face_dist", "last_align_face_dist"):
            _min_value(row, key, thresholds.open_hand_align_face_distance_tolerance)
        for key in ("best_align_angle", "last_align_angle"):
            _min_value(row, key, thresholds.open_hand_align_angle_deg)
        for key in ("best_opposed_face", "last_opposed_face"):
            _max_value(row, key, thresholds.grasp_success_opposed_face_threshold)
        for key in ("best_block_disp", "last_block_disp"):
            _min_value(row, key, thresholds.grasp_success_block_disp_max)
        if thresholds.topdown_lift_success_block_tilt_max_deg > 0.0:
            _min_value(row, "max_block_tilt_deg", thresholds.topdown_lift_success_block_tilt_max_deg)
        row["last_phase1_ready"] = True
        return row

    if task_kind in ("grasp_contact", "grasp_light_contact"):
        success_curl_floor = 0.0 if task_kind == "grasp_light_contact" else thresholds.curl_success_threshold
        for key in ("best_tip", "last_tip"):
            _min_value(row, key, thresholds.pregrasp_tight_tolerance)
        for key in ("best_phase1_palm", "last_phase1_palm"):
            _min_value(row, key, thresholds.grasp_success_palm_ready_tolerance)
        for key in ("best_phase1_orient", "last_phase1_orient"):
            _min_value(row, key, thresholds.grasp_success_palm_orient_deg)
        for key in (
            "best_contact",
            "best_both_contact",
            "best_thumb_contact",
            "best_index_contact",
            "best_strict_contact",
            "last_contact",
            "last_both_contact",
            "last_thumb_contact",
            "last_index_contact",
            "last_strict_contact",
        ):
            _max_value(row, key, thresholds.topdown_lift_success_contact_min)
        for key in ("best_curl", "last_curl"):
            _max_value(row, key, success_curl_floor)
        for key in ("best_block_disp", "last_block_disp"):
            _min_value(row, key, thresholds.grasp_success_block_disp_max)
        row["last_phase1_ready"] = True
        return row

    if task_kind in ("topdown", "topdown_lift"):
        _max_value(row, "best_topdown_stage", 2.0)
        _max_value(row, "max_topdown_unlock", 1.0)
        _max_value(row, "last_topdown_stage", 2.0)
        _max_value(row, "last_topdown_finger_unlock_progress", 1.0)
        if task_kind == "topdown_lift":
            _max_value(row, "best_strict_contact", thresholds.topdown_lift_success_contact_min)
            _max_value(row, "best_lift", thresholds.topdown_lift_success_height)
            _max_value(row, "best_lift_with_strict_contact", thresholds.topdown_lift_success_height)
        _min_value(row, "best_block_disp", thresholds.topdown_lift_success_xy_drift_max)
        if thresholds.topdown_lift_success_block_tilt_max_deg > 0.0:
            _min_value(row, "max_block_tilt_deg", thresholds.topdown_lift_success_block_tilt_max_deg)
        for key in (
            "best_topdown_dist_pass",
            "best_topdown_height_pass",
            "best_topdown_drop_pass",
            "best_topdown_yaw_pass",
            "best_topdown_reach_pass",
        ):
            _max_value(row, key, 1.0)
    return row
