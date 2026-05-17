"""

Eval summary row builders from per-env metric lists

File map:

EVAL_BOOL_KEYS:                      Define eval bool keys constant
EVAL_MEDIAN_KEYS:                    Define eval median keys constant
_as_list:                            Handle as list logic
_float_values:                       Handle float values logic
_bool_values:                        Handle bool values logic
_rate:                               Handle rate logic
_median:                             Handle median logic
build_eval_summary_from_env_values:  Build an eval_summary row from per-env values
add_topdown_lift_success_summary:    Add topdown lift physical success and failure-mode fields
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import statistics

from ..core.runtime import topdown_episode_failure_mode, topdown_lift_physical_success


EVAL_BOOL_KEYS = (
    "success",
    "off_table",
    "phase15_shell_drift",
    "block_drift",
    "timeout",
    "done",
)

EVAL_MEDIAN_KEYS = (
    "best_tip",
    "final_tip",
    "best_phase1_palm_dist",
    "final_phase1_palm_dist",
    "best_phase1_orient_deg",
    "final_phase1_orient_deg",
    "best_contact",
    "final_contact",
    "best_both_contact",
    "final_both_contact",
    "best_any_contact",
    "final_any_contact",
    "best_hand_contact",
    "final_hand_contact",
    "best_thumb_contact",
    "final_thumb_contact",
    "best_index_contact",
    "final_index_contact",
    "final_thumb_contact_force_N",
    "final_index_contact_force_N",
    "best_strict_light_contact",
    "final_strict_light_contact",
    "best_lift",
    "best_lift_with_strict_contact",
    "final_lift",
    "best_curl",
    "final_curl",
    "best_align_face_dist",
    "final_align_face_dist",
    "best_align_angle",
    "final_align_angle",
    "best_opposite_face",
    "final_opposite_face",
    "best_block_disp",
    "final_block_disp",
    "max_block_tilt_deg",
    "final_block_tilt_deg",
    "best_teacher_ik_task_space_center_err",
    "final_teacher_ik_task_space_q",
    "final_teacher_ik_task_space_center_err",
    "final_teacher_ik_task_space_center_err_after",
    "final_teacher_ik_task_space_span_z",
    "final_teacher_ik_task_space_span_z_after",
    "final_teacher_ik_task_space_drop_err",
    "final_teacher_ik_task_space_drop_err_after",
    "final_topdown_palm_local_grip_offset_live_blend",
)


def _as_list(values: Sequence[object] | object) -> list[object]:
    if isinstance(values, (str, bytes)):
        return [values]
    if isinstance(values, Sequence):
        return list(values)
    return [values]


def _float_values(values: Sequence[object]) -> list[float]:
    return [float(value) for value in values]


def _bool_values(values: Sequence[object]) -> list[bool]:
    return [bool(value) for value in values]


def _rate(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean([1.0 if value else 0.0 for value in values]))


def _median(values: Sequence[object]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(_float_values(values)))


def build_eval_summary_from_env_values(
    *,
    global_step     : int,  # Param: current absolute training step
    eval_episode_idx: int,  # Param: evaluation episode index associated with the record
    env_values      : Mapping[str, Sequence[object] | object],  # Param: string input for env values
    extra           : Mapping[str, object] | None = None,  # Param: string input for extra
) -> dict[str, object]:
    """Build an eval_summary row from per-env values

    Steps:
    - Resolve inputs for `build_eval_summary_from_env_values` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    values = {str(key): _as_list(value) for key, value in env_values.items()}
    env_count = max((len(value) for value in values.values()), default=0)
    returns = _float_values(values.get("return", []))
    steps = [int(value) for value in values.get("steps", [])]
    done_values = _bool_values(values.get("done", []))
    summary: dict[str, object] = {
        "mode"                     : "eval_summary",
        "global_step"              : int(global_step),
        "eval_episode_idx"         : int(eval_episode_idx),
        "eval_env_count"           : int(env_count),
        "eval_return"              : float(statistics.mean(returns)) if returns else 0.0,
        "eval_steps"               : int(max(steps)) if steps else 0,
        "eval_final_geometry_frame": "mixed" if any(done_values) else "live_state",
    }
    for key in EVAL_BOOL_KEYS:
        key_values = _bool_values(values.get(key, []))
        if not key_values:
            continue
        summary[f"eval_{key}"] = bool(all(key_values)) if key == "done" else bool(any(key_values))
        summary[f"eval_{key}_count"] = int(sum(1 for value in key_values if value))
        summary[f"eval_{key}_rate"] = _rate(key_values)
    final_phase1_ready = _bool_values(values.get("final_phase1_ready", []))
    if final_phase1_ready:
        summary["eval_final_phase1_ready"] = bool(any(final_phase1_ready))
        summary["eval_final_phase1_ready_rate"] = _rate(final_phase1_ready)
    for key in EVAL_MEDIAN_KEYS:
        key_values = values.get(key)
        if key_values:
            summary[f"eval_{key}"] = _median(key_values)
    for key, key_values in values.items():
        summary[f"eval_env_{key}"] = key_values
    if extra:
        summary.update(dict(extra))
    return summary


def add_topdown_lift_success_summary(
    summary: dict[str, object],                       # Param: string input for summary
    *,
    env_success                  : Sequence[object],  # Param: ordered input collection of env success entries
    env_off_table                : Sequence[object],  # Param: ordered input collection of env off table entries
    env_block_drift              : Sequence[object],  # Param: ordered input collection of env block drift entries
    env_timeout                  : Sequence[object],  # Param: ordered input collection of env timeout entries
    best_topdown_stage           : Sequence[object],  # Param: ordered input collection of best topdown stage entries
    max_topdown_unlock           : Sequence[object],  # Param: ordered input collection of max topdown unlock entries
    best_contact                 : Sequence[object],  # Param: ordered input collection of best contact entries
    best_strict_contact          : Sequence[object],  # Param: ordered input collection of best strict contact entries
    best_lift                    : Sequence[object],  # Param: ordered input collection of best lift entries
    best_lift_with_strict_contact: Sequence[object],  # Param: ordered input collection of best lift with strict contact entries
    best_block_disp              : Sequence[object],  # Param: ordered input collection of best block disp entries
    max_block_tilt_deg           : Sequence[object],  # Param: ordered input collection of max block tilt deg entries
    lift_height_min              : float,  # Param: floating-point input for lift height min
    xy_drift_max                 : float,  # Param: floating-point input for xy drift max
    block_tilt_max_deg           : float,  # Param: floating-point input for block tilt max deg
    contact_min                  : float,  # Param: floating-point input for contact min
) -> dict[str, object]:
    """Add topdown lift physical success and failure-mode fields

    Steps:
    - Resolve inputs for `add_topdown_lift_success_summary` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    count = min(
        len(env_success),
        len(env_off_table),
        len(env_block_drift),
        len(env_timeout),
        len(best_topdown_stage),
        len(max_topdown_unlock),
        len(best_contact),
        len(best_strict_contact),
        len(best_lift),
        len(best_lift_with_strict_contact),
        len(best_block_disp),
        len(max_block_tilt_deg),
    )
    physical_values: list[bool] = []
    failure_modes  : list[str]  = []
    for env_id in range(count):
        physical = topdown_lift_physical_success(
            best_lift_with_strict_contact=float(best_lift_with_strict_contact[env_id]),
            best_block_disp=float(best_block_disp[env_id]),
            max_block_tilt_deg=float(max_block_tilt_deg[env_id]),
            lift_height_min=lift_height_min,
            xy_drift_max=xy_drift_max,
            block_tilt_max_deg=block_tilt_max_deg,
        )
        physical_values.append(bool(physical))
        failure_modes.append(
            topdown_episode_failure_mode(
                success=bool(env_success[env_id]),
                physical_success=bool(physical),
                off_table=bool(env_off_table[env_id]),
                block_drift=bool(env_block_drift[env_id]),
                timeout=bool(env_timeout[env_id]),
                best_stage=int(best_topdown_stage[env_id]),
                max_unlock=float(max_topdown_unlock[env_id]),
                best_contact=float(best_contact[env_id]),
                best_strict_contact=float(best_strict_contact[env_id]),
                best_lift=float(best_lift[env_id]),
                best_lift_with_strict_contact=float(best_lift_with_strict_contact[env_id]),
                contact_min=contact_min,
                lift_height_min=lift_height_min,
            )
        )
    summary.update(
        {
            "eval_physical_success"     : bool(any(physical_values)),
            "eval_physical_success_rate": _rate(physical_values),
            "eval_lift_failure_mode": (
                failure_modes[0]
                if failure_modes and len(set(failure_modes)) == 1
                else "mixed"
            ),
            "eval_env_physical_success" : physical_values,
            "eval_env_lift_failure_mode": failure_modes,
        }
    )
    return summary
