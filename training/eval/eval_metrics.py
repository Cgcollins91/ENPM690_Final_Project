"""

Evaluation aggregation and best-checkpoint selection helpers

File map:

EvalAggregationOptions:       Task knobs needed for pure eval aggregation
EvalComparison:               Lexicographic eval comparison result
eval_metric_values:           Return scalar metric values with eval_env list expansion
eval_metric_int_values:       Return int metric values with eval_env list expansion
eval_metric_text_values:      Return text metric values with eval_env list expansion
_median:                      Handle median logic
_mean:                        Handle mean logic
_rate_ge:                     Handle rate ge logic
_rate_true:                   Handle rate true logic
aggregate_eval_summaries:     Aggregate eval episode rows into the trainer summary schema
eval_state_from_summary:      Convert an eval summary into best-checkpoint comparison state
initial_best_eval_state:      Return the initial best-checkpoint comparison floor
best_checkpoint_tiebreakers:  Return task-specific best-checkpoint tiebreak keys
compare_eval_states:          Compare two eval states using strict lexicographic ordering
best_checkpoint_message:      Format the best-checkpoint status line
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import statistics
from typing import Literal

from ..geometry.topdown_metrics import topdown_eval_source_conditioned_metrics


MetricOrder = Literal["higher", "lower"]
EvalTaskKind = Literal[
    "topdown_lift",
    "topdown",
    "grasp_align",
    "grasp_light_contact",
    "grasp_contact",
    "default",
]


@dataclass(frozen=True)
class EvalAggregationOptions:
    """Task knobs needed for pure eval aggregation"""

    task_kind               : EvalTaskKind = "topdown_lift"  # stores task kind for eval aggregation options
    contact_success_strength: float        = 0.30  # floating-point contact success strength value used by eval aggregation options


@dataclass(frozen=True)
class EvalComparison:
    """Lexicographic eval comparison result"""

    is_better    : bool  # boolean value indicating the is better state for eval comparison
    decisive_key : str | None   = None  # string decisive key value used by eval comparison
    current_value: float | None = None  # floating-point current value value used by eval comparison
    best_value   : float | None = None  # floating-point best value value used by eval comparison


def eval_metric_values(
    summaries: Sequence[Mapping[str, object]],  # Param: string input for summaries
    key      : str,  # Param: mapping key being read or written
    default  : float,  # Param: fallback value used when the input omits or rejects a setting
) -> list[float]:
    """Return scalar metric values with eval_env list expansion"""
    env_key = key.replace("eval_", "eval_env_", 1)
    values: list[float] = []
    for row in summaries:
        env_values = row.get(env_key)
        if isinstance(env_values, list):
            values.extend(float(value) for value in env_values)
        else:
            values.append(float(row.get(key, default)))
    return values


def eval_metric_int_values(
    summaries: Sequence[Mapping[str, object]],  # Param: string input for summaries
    key      : str,  # Param: mapping key being read or written
    default  : int,  # Param: fallback value used when the input omits or rejects a setting
) -> list[int]:
    """Return int metric values with eval_env list expansion"""
    env_key = key.replace("eval_", "eval_env_", 1)
    values: list[int] = []
    for row in summaries:
        env_values = row.get(env_key)
        if isinstance(env_values, list):
            values.extend(int(value) for value in env_values)
        else:
            values.append(int(row.get(key, default)))
    return values


def eval_metric_text_values(
    summaries: Sequence[Mapping[str, object]],  # Param: string input for summaries
    key      : str,  # Param: mapping key being read or written
    default  : str,  # Param: fallback value used when the input omits or rejects a setting
) -> list[str]:
    """Return text metric values with eval_env list expansion"""
    env_key = key.replace("eval_", "eval_env_", 1)
    values: list[str] = []
    for row in summaries:
        env_values = row.get(env_key)
        if isinstance(env_values, list):
            values.extend(str(value) for value in env_values)
        else:
            values.append(str(row.get(key, default)))
    return values


def _median(values: Sequence[float], default: float) -> float:
    if not values:
        return float(default)
    return float(statistics.median(float(value) for value in values))


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(statistics.mean(float(value) for value in values))


def _rate_ge(values: Sequence[float], threshold: float) -> tuple[int, float]:
    count = sum(1 for value in values if float(value) >= float(threshold))
    return int(count), float(count / max(len(values), 1))


def _rate_true(values: Sequence[float]) -> tuple[int, float]:
    return _rate_ge(values, 0.5)


def aggregate_eval_summaries(
    summaries: Sequence[Mapping[str, object]],      # Param: string input for summaries
    *,
    global_step: int,  # Param: current absolute training step
    options    : EvalAggregationOptions | None = None,  # Param: input value used as options
) -> dict[str, object]:
    """Aggregate eval episode rows into the trainer summary schema

    Steps:
    - Resolve inputs for `aggregate_eval_summaries` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    opts = EvalAggregationOptions() if options is None else options
    success_values = eval_metric_values(summaries, "eval_success", 0.0)
    done_values = eval_metric_values(summaries, "eval_done", 0.0)
    off_table_values = eval_metric_values(summaries, "eval_off_table", 0.0)
    phase15_values = eval_metric_values(summaries, "eval_phase15_shell_drift", 0.0)
    block_drift_values = eval_metric_values(summaries, "eval_block_drift", 0.0)
    timeout_values = eval_metric_values(summaries, "eval_timeout", 0.0)
    total_eval_envs = max(len(success_values), 1)

    success_count, success_rate = _rate_true(success_values)
    done_count, done_rate = _rate_true(done_values)
    off_table_count, off_table_rate = _rate_true(off_table_values)
    phase15_count, phase15_rate = _rate_true(phase15_values)
    block_drift_count, block_drift_rate = _rate_true(block_drift_values)
    timeout_count, timeout_rate = _rate_true(timeout_values)

    contact_key = (
        "eval_best_strict_light_contact"
        if opts.task_kind == "grasp_light_contact"
        else "eval_best_contact"
    )
    contact_values = eval_metric_values(summaries, contact_key, 0.0)
    contact_episode_count, contact_episode_rate = _rate_ge(
        contact_values,
        opts.contact_success_strength,
    )

    best_tips = eval_metric_values(summaries, "eval_best_tip", float("inf"))
    best_phase1_palms = eval_metric_values(summaries, "eval_best_phase1_palm_dist", float("inf"))
    best_phase1_orients = eval_metric_values(summaries, "eval_best_phase1_orient_deg", float("inf"))
    best_contacts = eval_metric_values(summaries, "eval_best_contact", 0.0)
    best_strict_contacts = eval_metric_values(summaries, "eval_best_strict_light_contact", 0.0)
    best_lifts = eval_metric_values(summaries, "eval_best_lift", 0.0)
    best_lifts_strict = eval_metric_values(summaries, "eval_best_lift_with_strict_contact", 0.0)
    max_block_tilts = eval_metric_values(summaries, "eval_max_block_tilt_deg", float("inf"))
    best_curls = eval_metric_values(summaries, "eval_best_curl", 0.0)
    best_align_face_dists = eval_metric_values(summaries, "eval_best_align_face_dist", float("inf"))
    best_align_angles = eval_metric_values(summaries, "eval_best_align_angle", float("inf"))
    best_opposed_faces = eval_metric_values(summaries, "eval_best_opposite_face", 0.0)
    best_block_disps = eval_metric_values(summaries, "eval_best_block_disp", float("inf"))
    final_tips = eval_metric_values(summaries, "eval_final_tip", float("inf"))
    final_phase1_palms = eval_metric_values(summaries, "eval_final_phase1_palm_dist", float("inf"))
    final_phase1_orients = eval_metric_values(summaries, "eval_final_phase1_orient_deg", float("inf"))
    final_contacts = eval_metric_values(summaries, "eval_final_contact", 0.0)
    final_strict_contacts = eval_metric_values(summaries, "eval_final_strict_light_contact", 0.0)
    final_lifts = eval_metric_values(summaries, "eval_final_lift", 0.0)
    final_block_tilts = eval_metric_values(summaries, "eval_final_block_tilt_deg", float("inf"))
    final_curls = eval_metric_values(summaries, "eval_final_curl", 0.0)
    final_align_face_dists = eval_metric_values(summaries, "eval_final_align_face_dist", float("inf"))
    final_align_angles = eval_metric_values(summaries, "eval_final_align_angle", float("inf"))
    final_opposed_faces = eval_metric_values(summaries, "eval_final_opposite_face", 0.0)
    final_block_disps = eval_metric_values(summaries, "eval_final_block_disp", float("inf"))

    aggregate: dict[str, object] = {
        "mode"                                     : "eval_aggregate",
        "global_step"                              : int(global_step),
        "eval_episodes"                            : len(summaries),
        "eval_env_episodes"                        : len(success_values),
        "eval_success_rate"                        : success_rate,
        "eval_success_count"                       : success_count,
        "eval_done_rate"                           : done_rate,
        "eval_done_count"                          : done_count,
        "eval_off_table_rate"                      : off_table_rate,
        "eval_off_table_count"                     : off_table_count,
        "eval_phase15_shell_drift_rate"            : phase15_rate,
        "eval_phase15_shell_drift_count"           : phase15_count,
        "eval_block_drift_rate"                    : block_drift_rate,
        "eval_block_drift_count"                   : block_drift_count,
        "eval_timeout_rate"                        : timeout_rate,
        "eval_timeout_count"                       : timeout_count,
        "eval_contact_episode_rate"                : contact_episode_rate,
        "eval_contact_episode_count"               : contact_episode_count,
        "eval_median_best_tip"                     : _median(best_tips, float("inf")),
        "eval_median_best_phase1_palm_dist"        : _median(best_phase1_palms, float("inf")),
        "eval_median_best_phase1_orient_deg"       : _median(best_phase1_orients, float("inf")),
        "eval_median_best_contact"                 : _median(best_contacts, 0.0),
        "eval_median_best_strict_light_contact"    : _median(best_strict_contacts, 0.0),
        "eval_median_best_lift"                    : _median(best_lifts, 0.0),
        "eval_median_best_lift_with_strict_contact": _median(best_lifts_strict, 0.0),
        "eval_median_max_block_tilt_deg"           : _median(max_block_tilts, float("inf")),
        "eval_median_best_curl"                    : _median(best_curls, 0.0),
        "eval_median_best_align_face_dist"         : _median(best_align_face_dists, float("inf")),
        "eval_median_best_align_angle"             : _median(best_align_angles, float("inf")),
        "eval_median_best_opposite_face"           : _median(best_opposed_faces, 0.0),
        "eval_median_best_block_disp"              : _median(best_block_disps, float("inf")),
        "eval_median_final_tip"                    : _median(final_tips, float("inf")),
        "eval_median_final_phase1_palm_dist"       : _median(final_phase1_palms, float("inf")),
        "eval_median_final_phase1_orient_deg"      : _median(final_phase1_orients, float("inf")),
        "eval_median_final_contact"                : _median(final_contacts, 0.0),
        "eval_median_final_strict_light_contact"   : _median(final_strict_contacts, 0.0),
        "eval_median_final_lift"                   : _median(final_lifts, 0.0),
        "eval_median_final_block_tilt_deg"         : _median(final_block_tilts, float("inf")),
        "eval_median_final_curl"                   : _median(final_curls, 0.0),
        "eval_median_final_align_face_dist"        : _median(final_align_face_dists, float("inf")),
        "eval_median_final_align_angle"            : _median(final_align_angles, float("inf")),
        "eval_median_final_opposite_face"          : _median(final_opposed_faces, 0.0),
        "eval_median_final_block_disp"             : _median(final_block_disps, float("inf")),
        "eval_best_tip"                            : float(min(best_tips)) if best_tips else float("inf"),
        "eval_best_phase1_palm_dist"               : float(min(best_phase1_palms)) if best_phase1_palms else float("inf"),
        "eval_best_phase1_orient_deg"              : float(min(best_phase1_orients)) if best_phase1_orients else float("inf"),
        "eval_best_contact"                        : float(max(best_contacts)) if best_contacts else 0.0,
        "eval_best_strict_light_contact"           : float(max(best_strict_contacts)) if best_strict_contacts else 0.0,
        "eval_best_lift"                           : float(max(best_lifts)) if best_lifts else 0.0,
        "eval_best_lift_with_strict_contact"       : float(max(best_lifts_strict)) if best_lifts_strict else 0.0,
        "eval_max_block_tilt_deg"                  : float(max(max_block_tilts)) if max_block_tilts else float("inf"),
        "eval_best_curl"                           : float(max(best_curls)) if best_curls else 0.0,
        "eval_best_align_face_dist"                : float(min(best_align_face_dists)) if best_align_face_dists else float("inf"),
        "eval_best_align_angle"                    : float(min(best_align_angles)) if best_align_angles else float("inf"),
        "eval_best_opposite_face"                  : float(max(best_opposed_faces)) if best_opposed_faces else 0.0,
        "eval_best_block_disp"                     : float(min(best_block_disps)) if best_block_disps else float("inf"),
    }

    if opts.task_kind == "grasp_align":
        aggregate["eval_open_hand_align_ready_rate"] = success_rate
    elif opts.task_kind == "grasp_light_contact":
        aggregate["eval_light_contact_ready_rate"] = success_rate
    elif opts.task_kind == "grasp_contact":
        aggregate["eval_grasp_ready_rate"] = success_rate

    if opts.task_kind in ("topdown", "topdown_lift"):
        best_topdown_stages = eval_metric_int_values(summaries, "eval_best_topdown_stage", -1)
        final_topdown_stages = eval_metric_int_values(summaries, "eval_final_topdown_stage", -1)
        max_topdown_unlocks = eval_metric_values(
            summaries,
            "eval_max_topdown_finger_unlock_progress",
            0.0,
        )
        stage1_count = sum(1 for stage in best_topdown_stages if stage >= 1)
        stage2_count = sum(1 for stage in best_topdown_stages if stage >= 2)
        aggregate.update(
            {
                "eval_topdown_stage1_episode_rate"              : stage1_count / max(total_eval_envs, 1),
                "eval_topdown_stage2_episode_rate"              : stage2_count / max(total_eval_envs, 1),
                "eval_topdown_stage1_episode_count"             : int(stage1_count),
                "eval_topdown_stage2_episode_count"             : int(stage2_count),
                "eval_median_best_topdown_stage"                : _median(best_topdown_stages, -1.0),
                "eval_median_final_topdown_stage"               : _median(final_topdown_stages, -1.0),
                "eval_median_max_topdown_finger_unlock_progress": _median(max_topdown_unlocks, 0.0),
                "eval_topdown_reach_ready_rate": _mean(
                    eval_metric_values(summaries, "eval_topdown_reach_pass", 0.0),
                    0.0,
                ),
            }
        )
        aggregate.update(
            topdown_eval_source_conditioned_metrics(
                summaries,
                contact_threshold=opts.contact_success_strength,
            )
        )

    if opts.task_kind == "topdown_lift":
        physical_successes = eval_metric_values(summaries, "eval_physical_success", 0.0)
        clean_lifts = eval_metric_values(summaries, "eval_clean_lift_episode", 0.0)
        upright_clean_lifts = eval_metric_values(summaries, "eval_upright_clean_lift_episode", 0.0)
        xy_gates = eval_metric_values(summaries, "eval_lift_xy_drift_success_gate", 0.0)
        tilt_gates = eval_metric_values(summaries, "eval_lift_block_tilt_success_gate", 0.0)
        aggregate.update(
            {
                "eval_physical_success_rate"            : _mean(physical_successes, 0.0),
                "eval_clean_lift_episode_rate"          : _mean(clean_lifts, 0.0),
                "eval_upright_clean_lift_episode_rate"  : _mean(upright_clean_lifts, 0.0),
                "eval_lift_xy_drift_success_gate_rate"  : _mean(xy_gates, 0.0),
                "eval_lift_block_tilt_success_gate_rate": _mean(tilt_gates, 0.0),
            }
        )

    return aggregate


def eval_state_from_summary(eval_summary: Mapping[str, object], *, global_step: int) -> dict[str, float | int]:
    """Convert an eval summary into best-checkpoint comparison state"""
    return {
        "success_rate"        : float(eval_summary.get("eval_success_rate", 0.0)),
        "contact_episode_rate": float(eval_summary.get("eval_contact_episode_rate", 0.0)),
        "median_best_lift"    : float(eval_summary.get("eval_median_best_lift", 0.0)),
        "median_best_lift_with_strict_contact": float(
            eval_summary.get("eval_median_best_lift_with_strict_contact", 0.0)
        ),
        "median_best_contact": float(eval_summary.get("eval_median_best_contact", 0.0)),
        "median_best_strict_light_contact": float(
            eval_summary.get("eval_median_best_strict_light_contact", 0.0)
        ),
        "median_best_curl": float(eval_summary.get("eval_median_best_curl", 0.0)),
        "median_best_tip" : float(eval_summary.get("eval_median_best_tip", float("inf"))),
        "median_best_align_face_dist": float(
            eval_summary.get("eval_median_best_align_face_dist", float("inf"))
        ),
        "median_best_align_angle": float(eval_summary.get("eval_median_best_align_angle", float("inf"))),
        "median_best_phase1_palm_dist": float(
            eval_summary.get("eval_median_best_phase1_palm_dist", float("inf"))
        ),
        "median_best_phase1_orient_deg": float(
            eval_summary.get("eval_median_best_phase1_orient_deg", float("inf"))
        ),
        "median_final_block_disp"  : float(eval_summary.get("eval_median_final_block_disp", float("inf"))),
        "median_best_block_disp"   : float(eval_summary.get("eval_median_best_block_disp", float("inf"))),
        "median_max_block_tilt_deg": float(eval_summary.get("eval_median_max_block_tilt_deg", float("inf"))),
        "physical_success_rate"    : float(eval_summary.get("eval_physical_success_rate", 0.0)),
        "upright_clean_lift_episode_rate": float(
            eval_summary.get("eval_upright_clean_lift_episode_rate", 0.0)
        ),
        "clean_lift_episode_rate": float(eval_summary.get("eval_clean_lift_episode_rate", 0.0)),
        "lift_xy_drift_success_gate_rate": float(
            eval_summary.get("eval_lift_xy_drift_success_gate_rate", 0.0)
        ),
        "lift_block_tilt_success_gate_rate": float(
            eval_summary.get("eval_lift_block_tilt_success_gate_rate", 0.0)
        ),
        "topdown_stage1_episode_rate": float(eval_summary.get("eval_topdown_stage1_episode_rate", 0.0)),
        "topdown_stage2_episode_rate": float(eval_summary.get("eval_topdown_stage2_episode_rate", 0.0)),
        "median_best_topdown_stage"  : float(eval_summary.get("eval_median_best_topdown_stage", -1.0)),
        "median_max_topdown_finger_unlock_progress": float(
            eval_summary.get("eval_median_max_topdown_finger_unlock_progress", 0.0)
        ),
        "global_step": int(global_step),
    }


def initial_best_eval_state() -> dict[str, float]:
    """Return the initial best-checkpoint comparison floor"""
    return {
        "success_rate"                             : 0.0,
        "contact_episode_rate"                     : 0.0,
        "median_best_lift"                         : 0.0,
        "median_best_lift_with_strict_contact"     : 0.0,
        "median_best_contact"                      : 0.0,
        "median_best_strict_light_contact"         : 0.0,
        "median_best_curl"                         : 0.0,
        "median_best_tip"                          : float("inf"),
        "median_best_align_face_dist"              : float("inf"),
        "median_best_align_angle"                  : float("inf"),
        "median_best_phase1_palm_dist"             : float("inf"),
        "median_best_phase1_orient_deg"            : float("inf"),
        "median_final_block_disp"                  : float("inf"),
        "median_best_block_disp"                   : float("inf"),
        "median_max_block_tilt_deg"                : float("inf"),
        "physical_success_rate"                    : 0.0,
        "upright_clean_lift_episode_rate"          : 0.0,
        "clean_lift_episode_rate"                  : 0.0,
        "lift_xy_drift_success_gate_rate"          : 0.0,
        "lift_block_tilt_success_gate_rate"        : 0.0,
        "topdown_stage1_episode_rate"              : 0.0,
        "topdown_stage2_episode_rate"              : 0.0,
        "median_best_topdown_stage"                : -1.0,
        "median_max_topdown_finger_unlock_progress": 0.0,
        "global_step"                              : -1,
    }


def best_checkpoint_tiebreakers(task_kind: EvalTaskKind) -> tuple[tuple[str, MetricOrder], ...]:
    """Return task-specific best-checkpoint tiebreak keys

    Steps:
    - Resolve inputs for `best_checkpoint_tiebreakers` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if task_kind == "topdown_lift":
        return (
            ("success_rate", "higher"),
            ("physical_success_rate", "higher"),
            ("upright_clean_lift_episode_rate", "higher"),
            ("clean_lift_episode_rate", "higher"),
            ("topdown_stage2_episode_rate", "higher"),
            ("topdown_stage1_episode_rate", "higher"),
            ("median_best_topdown_stage", "higher"),
            ("median_max_topdown_finger_unlock_progress", "higher"),
            ("median_best_strict_light_contact", "higher"),
            ("median_best_contact", "higher"),
            ("median_best_lift_with_strict_contact", "higher"),
            ("median_best_lift", "higher"),
            ("lift_block_tilt_success_gate_rate", "higher"),
            ("lift_xy_drift_success_gate_rate", "higher"),
            ("median_max_block_tilt_deg", "lower"),
            ("median_best_block_disp", "lower"),
            ("median_best_align_face_dist", "lower"),
            ("median_best_align_angle", "lower"),
        )
    if task_kind == "topdown":
        return (
            ("success_rate", "higher"),
            ("topdown_stage2_episode_rate", "higher"),
            ("topdown_stage1_episode_rate", "higher"),
            ("median_best_topdown_stage", "higher"),
            ("median_best_contact", "higher"),
            ("median_best_align_face_dist", "lower"),
            ("median_best_align_angle", "lower"),
            ("median_best_phase1_palm_dist", "lower"),
            ("median_best_phase1_orient_deg", "lower"),
        )
    if task_kind == "grasp_align":
        return (
            ("success_rate", "higher"),
            ("median_best_align_face_dist", "lower"),
            ("median_best_align_angle", "lower"),
            ("median_best_phase1_palm_dist", "lower"),
            ("median_best_phase1_orient_deg", "lower"),
            ("median_final_block_disp", "lower"),
        )
    if task_kind == "grasp_light_contact":
        return (
            ("success_rate", "higher"),
            ("median_best_strict_light_contact", "higher"),
            ("median_best_contact", "higher"),
            ("median_best_phase1_palm_dist", "lower"),
            ("median_best_phase1_orient_deg", "lower"),
            ("median_best_tip", "lower"),
            ("median_final_block_disp", "lower"),
            ("contact_episode_rate", "higher"),
        )
    if task_kind == "grasp_contact":
        return (
            ("success_rate", "higher"),
            ("median_best_contact", "higher"),
            ("median_best_curl", "higher"),
            ("median_best_phase1_palm_dist", "lower"),
            ("median_best_phase1_orient_deg", "lower"),
            ("median_best_tip", "lower"),
            ("median_final_block_disp", "lower"),
            ("contact_episode_rate", "higher"),
        )
    return (
        ("success_rate", "higher"),
        ("contact_episode_rate", "higher"),
        ("median_best_contact", "higher"),
        ("median_best_curl", "higher"),
        ("median_best_tip", "lower"),
        ("median_best_lift", "higher"),
    )


def compare_eval_states(
    current    : Mapping[str, float | int],  # Param: integer input for current
    best       : Mapping[str, float | int],  # Param: integer input for best
    tiebreakers: Sequence[tuple[str, MetricOrder]],  # Param: string input for tiebreakers
) -> EvalComparison:
    """Compare two eval states using strict lexicographic ordering"""
    for key, order in tiebreakers:
        cur_value = float(current.get(key, math.nan))
        best_value = float(best.get(key, math.nan))
        if order == "higher":
            if cur_value > best_value:
                return EvalComparison(True, key, cur_value, best_value)
            if cur_value < best_value:
                return EvalComparison(False, key, cur_value, best_value)
        else:
            if cur_value < best_value:
                return EvalComparison(True, key, cur_value, best_value)
            if cur_value > best_value:
                return EvalComparison(False, key, cur_value, best_value)
    return EvalComparison(False)


def best_checkpoint_message(
    state: Mapping[str, float | int],  # Param: mutable or immutable runtime state read by this helper
    *,
    step     : int,  # Param: integer input for step
    task_kind: EvalTaskKind,  # Param: input value used as task kind
    best_path: str,  # Param: filesystem path for best
) -> str:
    """Format the best-checkpoint status line"""
    if task_kind in ("topdown", "topdown_lift"):
        lift_extra = ""
        if task_kind == "topdown_lift":
            lift_extra = (
                f" clean_lift={float(state.get('clean_lift_episode_rate', 0.0)):.2f} "
                f"upright_clean={float(state.get('upright_clean_lift_episode_rate', 0.0)):.2f} "
                f"tilt_gate={float(state.get('lift_block_tilt_success_gate_rate', 0.0)):.2f} "
                f"drift_gate={float(state.get('lift_xy_drift_success_gate_rate', 0.0)):.2f} "
                f"median_tilt={float(state.get('median_max_block_tilt_deg', float('inf'))):.1f} "
                f"median_disp={float(state.get('median_best_block_disp', float('inf'))):.3f}"
            )
        return (
            f"best_checkpoint step={int(step):05d} "
            f"success_rate={float(state.get('success_rate', 0.0)):.2f} "
            f"stage1_rate={float(state.get('topdown_stage1_episode_rate', 0.0)):.2f} "
            f"stage2_rate={float(state.get('topdown_stage2_episode_rate', 0.0)):.2f} "
            f"median_best_stage={float(state.get('median_best_topdown_stage', -1.0)):.1f} "
            f"median_best_contact={float(state.get('median_best_contact', 0.0)):.4f} "
            f"median_best_align={float(state.get('median_best_align_face_dist', float('inf'))):.4f} "
            f"median_best_angle={float(state.get('median_best_align_angle', float('inf'))):.2f} "
            f"median_best_palm={float(state.get('median_best_phase1_palm_dist', float('inf'))):.4f} "
            f"median_best_orient={float(state.get('median_best_phase1_orient_deg', float('inf'))):.2f} "
            f"{lift_extra} -> {best_path}"
        )
    if task_kind == "grasp_align":
        return (
            f"best_checkpoint step={int(step):05d} "
            f"success_rate={float(state.get('success_rate', 0.0)):.2f} "
            f"median_best_align={float(state.get('median_best_align_face_dist', float('inf'))):.4f} "
            f"median_best_angle={float(state.get('median_best_align_angle', float('inf'))):.2f} "
            f"median_best_palm={float(state.get('median_best_phase1_palm_dist', float('inf'))):.4f} "
            f"median_best_orient={float(state.get('median_best_phase1_orient_deg', float('inf'))):.2f} "
            f"median_final_blk_disp={float(state.get('median_final_block_disp', float('inf'))):.4f} "
            f"-> {best_path}"
        )
    if task_kind in ("grasp_contact", "grasp_light_contact"):
        return (
            f"best_checkpoint step={int(step):05d} "
            f"success_rate={float(state.get('success_rate', 0.0)):.2f} "
            f"contact_rate={float(state.get('contact_episode_rate', 0.0)):.2f} "
            f"median_best_contact={float(state.get('median_best_contact', 0.0)):.4f} "
            f"median_best_strict={float(state.get('median_best_strict_light_contact', 0.0)):.4f} "
            f"median_best_curl={float(state.get('median_best_curl', 0.0)):.4f} "
            f"median_best_palm={float(state.get('median_best_phase1_palm_dist', float('inf'))):.4f} "
            f"median_best_orient={float(state.get('median_best_phase1_orient_deg', float('inf'))):.2f} "
            f"median_best_tip={float(state.get('median_best_tip', float('inf'))):.4f} "
            f"median_final_blk_disp={float(state.get('median_final_block_disp', float('inf'))):.4f} "
            f"-> {best_path}"
        )
    return (
        f"best_checkpoint step={int(step):05d} "
        f"success_rate={float(state.get('success_rate', 0.0)):.2f} "
        f"contact_rate={float(state.get('contact_episode_rate', 0.0)):.2f} "
        f"median_best_lift={float(state.get('median_best_lift', 0.0)):.4f} "
        f"median_best_contact={float(state.get('median_best_contact', 0.0)):.4f} "
        f"median_best_curl={float(state.get('median_best_curl', 0.0)):.4f} "
        f"median_best_tip={float(state.get('median_best_tip', float('inf'))):.4f} "
        f"-> {best_path}"
    )
