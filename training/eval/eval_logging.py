"""

Evaluation status line formatting helpers

File map:

EvalStepLineSummary:     Scalar fields printed during eval rollout
_eval_step_prefix:       Handle eval step prefix logic
format_eval_start_line:  Format eval rollout start line
should_log_eval_step:    Return whether an eval step should print a trace line
format_eval_step_line:   Format one eval rollout progress line
_float:                  Handle float logic
_int:                    Handle int logic
_eval_end_prefix:        Handle eval end prefix logic
format_eval_end_line:    Format final eval rollout status line
format_eval_pass_line:   Format aggregate eval pass status line
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .eval_metrics import EvalTaskKind


@dataclass(frozen=True)
class EvalStepLineSummary:
    """Scalar fields printed during eval rollout"""

    global_step       : int  # training step associated with this record or action
    eval_episode_idx  : int  # evaluation episode index for this record
    eval_step         : int  # step count used for eval step scheduling or reporting
    reward            : float  # reward tensor or scalar produced by the environment step
    done              : bool  # done flag tensor or scalar returned by the environment step
    success           : bool  # success flag or rate for the rollout/evaluation record
    tip               : float = 0.0  # floating-point tip value used by eval step line summary
    thumb_err         : float = 0.0  # floating-point thumb err value used by eval step line summary
    idx_err           : float = 0.0  # floating-point idx err value used by eval step line summary
    palm              : float = 0.0  # floating-point palm value used by eval step line summary
    orient_deg        : float = 0.0  # floating-point orient deg value used by eval step line summary
    phase1_ready      : bool  = False  # boolean/tensor readiness state for phase1
    align_face        : float = 0.0  # block face selected for alignment scoring
    align_angle       : float = 0.0  # alignment angle value used by topdown/contact metrics
    opposed_face      : float = 0.0  # block face opposite the active contact/alignment face
    contact           : float = 0.0  # floating-point contact value used by eval step line summary
    strict_contact    : float = 0.0  # floating-point strict contact value used by eval step line summary
    both_contact      : float = 0.0  # floating-point both contact value used by eval step line summary
    any_contact       : float = 0.0  # floating-point any contact value used by eval step line summary
    hand_contact      : float = 0.0  # floating-point hand contact value used by eval step line summary
    thumb_contact     : float = 0.0  # contact strength observed at the thumb side
    index_contact     : float = 0.0  # contact strength observed at the index-finger side
    curl              : float = 0.0  # floating-point curl value used by eval step line summary
    lift              : float = 0.0  # floating-point lift value used by eval step line summary
    hand_force        : float = 0.0  # aggregate hand/contact force used for diagnostics or gates
    block_disp        : float = 0.0  # block displacement value used by metrics or summaries
    thumb_to_block    : float = 0.0  # floating-point thumb to block value used by eval step line summary
    index_to_block    : float = 0.0  # floating-point index to block value used by eval step line summary
    shell_drift       : bool  = False  # boolean value indicating the shell drift state for eval step line summary
    off_table         : bool  = False  # flag indicating that the block left the table/work surface
    block_drift       : bool  = False  # measured block drift used by diagnostics or success checks
    topdown_stage     : int   = -1  # current topdown curriculum stage per environment
    best_topdown_stage: int   = -1  # highest topdown curriculum stage reached so far
    reach_hold        : int   = 0  # integer reach hold value tracked by eval step line summary
    align_hold        : int   = 0  # integer align hold value tracked by eval step line summary
    unlock_progress   : float = 0.0  # floating-point unlock progress value used by eval step line summary
    drop_axis_deg     : float = 0.0  # floating-point drop axis deg value used by eval step line summary
    yaw_axis_deg      : float = 0.0  # floating-point yaw axis deg value used by eval step line summary
    spread_axis_deg   : float = 0.0  # floating-point spread axis deg value used by eval step line summary


def _eval_step_prefix(summary: EvalStepLineSummary) -> str:
    return (
        f"eval step={int(summary.global_step):05d}:"
        f"{int(summary.eval_episode_idx):02d}:"
        f"{int(summary.eval_step):03d} "
        f"reward={float(summary.reward):+.3f} "
    )


def format_eval_start_line(
    *,
    global_step     : int,  # Param: current absolute training step
    eval_episode_idx: int,  # Param: evaluation episode index associated with the record
    max_steps       : int,  # Param: step count used for max steps
    preroll_steps   : int,  # Param: step count used for preroll steps
) -> str:
    """Format eval rollout start line"""
    return (
        f"eval_start step={int(global_step):05d} "
        f"ep={int(eval_episode_idx):02d} "
        f"max_steps={int(max_steps)} "
        f"preroll={int(preroll_steps)}"
    )


def should_log_eval_step(
    *,
    eval_step   : int,  # Param: step count used for eval step
    log_every   : int,  # Param: global-step interval used to decide when progress rows are emitted
    eval_success: bool,  # Param: boolean input controlling eval success
    eval_done   : bool,  # Param: boolean input controlling eval done
) -> bool:
    """Return whether an eval step should print a trace line"""
    return (
        (int(log_every) > 0 and int(eval_step) % int(log_every) == 0)
        or bool(eval_success)
        or bool(eval_done)
    )


def format_eval_step_line(summary: EvalStepLineSummary, *, task_kind: EvalTaskKind) -> str:
    """Format one eval rollout progress line

    Steps:
    - Resolve inputs for `format_eval_step_line` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    prefix = _eval_step_prefix(summary)
    done_value = int(bool(summary.done))
    success_value = int(bool(summary.success))
    if task_kind in ("topdown", "topdown_lift"):
        base = (
            f"{prefix}"
            f"stage={int(summary.topdown_stage)} "
            f"best_stage={int(summary.best_topdown_stage)} "
            f"reach_hold={int(summary.reach_hold)} "
            f"align_hold={int(summary.align_hold)} "
            f"unlock={float(summary.unlock_progress):.2f} "
            f"palm={float(summary.palm):.3f} "
            f"orient={float(summary.orient_deg):.1f} "
            f"drop={float(summary.drop_axis_deg):.1f} "
            f"yaw={float(summary.yaw_axis_deg):.1f} "
            f"spread={float(summary.spread_axis_deg):.1f} "
            f"align={float(summary.align_face):.3f} "
            f"align_angle={float(summary.align_angle):.1f} "
            f"opp={float(summary.opposed_face):.3f} "
            f"contact={float(summary.contact):.3f} "
            f"strict={float(summary.strict_contact):.3f} "
        )
        if summary.done:
            return f"{base}success={success_value} done=1"
        return (
            f"{base}"
            f"thumb_c={float(summary.thumb_contact):.3f} "
            f"idx_c={float(summary.index_contact):.3f} "
            f"hand_N={float(summary.hand_force):.2f} "
            f"blk_disp={float(summary.block_disp):.3f} "
            f"success={success_value} "
            f"block_drift={int(bool(summary.block_drift))} "
            f"done=0"
        )
    if task_kind == "grasp_align":
        if summary.done:
            return (
                f"{prefix}"
                f"last_live_tip={float(summary.tip):.3f} "
                f"palm={float(summary.palm):.3f} "
                f"orient={float(summary.orient_deg):.1f} "
                f"phase1={int(bool(summary.phase1_ready))} "
                f"align={float(summary.align_face):.3f} "
                f"align_angle={float(summary.align_angle):.1f} "
                f"opp={float(summary.opposed_face):.3f} "
                f"curl={float(summary.curl):.3f} "
                f"last_live_blk_disp={float(summary.block_disp):.3f} "
                f"success={success_value} done=1"
            )
        return (
            f"{prefix}"
            f"tip={float(summary.tip):.3f} "
            f"palm={float(summary.palm):.3f} "
            f"orient={float(summary.orient_deg):.1f} "
            f"phase1={int(bool(summary.phase1_ready))} "
            f"align={float(summary.align_face):.3f} "
            f"align_angle={float(summary.align_angle):.1f} "
            f"opp={float(summary.opposed_face):.3f} "
            f"contact={float(summary.contact):.3f} "
            f"curl={float(summary.curl):.3f} "
            f"blk_disp={float(summary.block_disp):.3f} "
            f"success={success_value} done=0"
        )
    if task_kind in ("grasp_contact", "grasp_light_contact"):
        if summary.done:
            return (
                f"{prefix}"
                f"last_live_tip={float(summary.tip):.3f} "
                f"thumb_err={float(summary.thumb_err):.3f} "
                f"idx_err={float(summary.idx_err):.3f} "
                f"palm={float(summary.palm):.3f} "
                f"orient={float(summary.orient_deg):.1f} "
                f"phase1={int(bool(summary.phase1_ready))} "
                f"last_live_contact={float(summary.contact):.3f} "
                f"strict={float(summary.strict_contact):.3f} "
                f"thumb_c={float(summary.thumb_contact):.3f} "
                f"idx_c={float(summary.index_contact):.3f} "
                f"last_live_curl={float(summary.curl):.3f} "
                f"last_live_blk_disp={float(summary.block_disp):.3f} "
                f"success={success_value} "
                f"shell_drift={int(bool(summary.shell_drift))} "
                f"off_table={int(bool(summary.off_table))} "
                f"block_drift={int(bool(summary.block_drift))} "
                f"done=1"
            )
        return (
            f"{prefix}"
            f"tip={float(summary.tip):.3f} "
            f"thumb_err={float(summary.thumb_err):.3f} "
            f"idx_err={float(summary.idx_err):.3f} "
            f"palm={float(summary.palm):.3f} "
            f"orient={float(summary.orient_deg):.1f} "
            f"phase1={int(bool(summary.phase1_ready))} "
            f"contact={float(summary.contact):.3f} "
            f"strict={float(summary.strict_contact):.3f} "
            f"thumb_c={float(summary.thumb_contact):.3f} "
            f"idx_c={float(summary.index_contact):.3f} "
            f"curl={float(summary.curl):.3f} "
            f"hand_N={float(summary.hand_force):.2f} "
            f"blk_disp={float(summary.block_disp):.3f} "
            f"thumb_blk={float(summary.thumb_to_block):.3f} "
            f"idx_blk={float(summary.index_to_block):.3f} "
            f"success={success_value} "
            f"shell_drift={int(bool(summary.shell_drift))} "
            f"off_table={int(bool(summary.off_table))} "
            f"block_drift={int(bool(summary.block_drift))} "
            f"done=0"
        )
    if summary.done:
        return (
            f"{prefix}"
            f"last_live_tip={float(summary.tip):.3f} "
            f"thumb_err={float(summary.thumb_err):.3f} "
            f"idx_err={float(summary.idx_err):.3f} "
            f"last_live_contact={float(summary.contact):.3f} "
            f"both={float(summary.both_contact):.3f} "
            f"any={float(summary.any_contact):.3f} "
            f"hand={float(summary.hand_contact):.3f} "
            f"opp={float(summary.opposed_face):.3f} "
            f"last_live_lift={float(summary.lift):.3f} "
            f"last_live_curl={float(summary.curl):.3f} "
            f"success={success_value} "
            f"off_table={int(bool(summary.off_table))} "
            f"block_drift={int(bool(summary.block_drift))} "
            f"done=1"
        )
    return (
        f"{prefix}"
        f"tip={float(summary.tip):.3f} "
        f"thumb_err={float(summary.thumb_err):.3f} "
        f"idx_err={float(summary.idx_err):.3f} "
        f"contact={float(summary.contact):.3f} "
        f"both={float(summary.both_contact):.3f} "
        f"any={float(summary.any_contact):.3f} "
        f"hand={float(summary.hand_contact):.3f} "
        f"opp={float(summary.opposed_face):.3f} "
        f"lift={float(summary.lift):.3f} "
        f"curl={float(summary.curl):.3f} "
        f"hand_N={float(summary.hand_force):.2f} "
        f"blk_disp={float(summary.block_disp):.3f} "
        f"thumb_blk={float(summary.thumb_to_block):.3f} "
        f"idx_blk={float(summary.index_to_block):.3f} "
        f"success={success_value} "
        f"off_table={int(bool(summary.off_table))} "
        f"block_drift={int(bool(summary.block_drift))} "
        f"done=0"
    )


def _float(summary: Mapping[str, object], key: str, default: float = 0.0) -> float:
    return float(summary.get(key, default))


def _int(summary: Mapping[str, object], key: str, default: int = 0) -> int:
    return int(summary.get(key, default))


def _eval_end_prefix(
    summary: Mapping[str, object],  # Param: string input for summary
    *,
    global_step     : int,  # Param: current absolute training step
    eval_episode_idx: int,  # Param: evaluation episode index associated with the record
) -> str:
    return (
        f"eval_end step={int(global_step):05d} "
        f"ep={int(eval_episode_idx):02d} "
        f"envs={_int(summary, 'eval_env_count', 1)} "
        f"steps={_int(summary, 'eval_steps', 0)} "
        f"return={_float(summary, 'eval_return', 0.0):+.3f} "
        f"success_rate={_float(summary, 'eval_success_rate', 0.0):.2f} "
    )


def format_eval_end_line(
    summary: Mapping[str, object],  # Param: string input for summary
    *,
    global_step     : int,  # Param: current absolute training step
    eval_episode_idx: int,  # Param: evaluation episode index associated with the record
    task_kind       : EvalTaskKind,  # Param: input value used as task kind
) -> str:
    """Format final eval rollout status line

    Steps:
    - Resolve inputs for `format_eval_end_line` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    prefix = _eval_end_prefix(
        summary,
        global_step=global_step,
        eval_episode_idx=eval_episode_idx,
    )
    if task_kind in ("topdown", "topdown_lift"):
        return (
            f"{prefix}"
            f"best_stage={_float(summary, 'eval_best_topdown_stage', -1.0):.1f} "
            f"final_stage={_float(summary, 'eval_final_topdown_stage', -1.0):.1f} "
            f"max_unlock={_float(summary, 'eval_max_topdown_finger_unlock_progress', 0.0):.2f} "
            f"best_palm={_float(summary, 'eval_best_phase1_palm_dist', 0.0):.3f} "
            f"best_orient={_float(summary, 'eval_best_phase1_orient_deg', 0.0):.1f} "
            f"best_drop={_float(summary, 'eval_best_topdown_drop_axis_deg', 0.0):.1f} "
            f"best_yaw={_float(summary, 'eval_best_topdown_yaw_axis_deg', 0.0):.1f} "
            f"best_spread={_float(summary, 'eval_best_topdown_spread_axis_deg', 0.0):.1f} "
            f"yaw_pass={_float(summary, 'eval_topdown_yaw_pass', 0.0):.2f} "
            f"reach_pass={_float(summary, 'eval_topdown_reach_pass', 0.0):.2f} "
            f"best_align={_float(summary, 'eval_best_align_face_dist', 0.0):.3f} "
            f"best_angle={_float(summary, 'eval_best_align_angle', 0.0):.1f} "
            f"best_opp={_float(summary, 'eval_best_opposite_face', 0.0):.3f} "
            f"best_contact={_float(summary, 'eval_best_contact', 0.0):.3f} "
            f"best_strict={_float(summary, 'eval_best_strict_light_contact', 0.0):.3f} "
            f"best_lift={_float(summary, 'eval_best_lift', 0.0):.3f} "
            f"best_xy_drift={_float(summary, 'eval_best_block_disp', 0.0):.3f} "
            f"max_tilt={_float(summary, 'eval_max_block_tilt_deg', 0.0):.1f}"
        )
    if task_kind == "grasp_align":
        return (
            f"{prefix}"
            f"best_palm={_float(summary, 'eval_best_phase1_palm_dist', 0.0):.3f} "
            f"best_orient={_float(summary, 'eval_best_phase1_orient_deg', 0.0):.1f} "
            f"best_align={_float(summary, 'eval_best_align_face_dist', 0.0):.3f} "
            f"best_angle={_float(summary, 'eval_best_align_angle', 0.0):.1f} "
            f"best_opp={_float(summary, 'eval_best_opposite_face', 0.0):.3f} "
            f"best_curl={_float(summary, 'eval_best_curl', 0.0):.3f} "
            f"best_blk_disp={_float(summary, 'eval_best_block_disp', 0.0):.3f}"
        )
    if task_kind in ("grasp_contact", "grasp_light_contact"):
        return (
            f"{prefix}"
            f"best_tip={_float(summary, 'eval_best_tip', 0.0):.3f} "
            f"best_palm={_float(summary, 'eval_best_phase1_palm_dist', 0.0):.3f} "
            f"best_orient={_float(summary, 'eval_best_phase1_orient_deg', 0.0):.1f} "
            f"best_contact={_float(summary, 'eval_best_contact', 0.0):.3f} "
            f"best_strict={_float(summary, 'eval_best_strict_light_contact', 0.0):.3f} "
            f"best_thumb_c={_float(summary, 'eval_best_thumb_contact', 0.0):.3f} "
            f"best_idx_c={_float(summary, 'eval_best_index_contact', 0.0):.3f} "
            f"best_curl={_float(summary, 'eval_best_curl', 0.0):.3f} "
            f"best_blk_disp={_float(summary, 'eval_best_block_disp', 0.0):.3f}"
        )
    return (
        f"{prefix}"
        f"best_tip={_float(summary, 'eval_best_tip', 0.0):.3f} "
        f"best_contact={_float(summary, 'eval_best_contact', 0.0):.3f} "
        f"best_lift={_float(summary, 'eval_best_lift', 0.0):.3f} "
        f"best_curl={_float(summary, 'eval_best_curl', 0.0):.3f}"
    )


def format_eval_pass_line(summary: Mapping[str, object], *, global_step: int) -> str:
    """Format an aggregate eval pass summary for stdout."""
    return (
        "eval_pass "
        f"step={int(global_step)} "
        f"episodes={_int(summary, 'eval_episodes', 0)} "
        f"env_episodes={_int(summary, 'eval_env_episodes', 0)} "
        f"success_rate={_float(summary, 'eval_success_rate', 0.0):.3f} "
        f"physical_success_rate={_float(summary, 'eval_physical_success_rate', 0.0):.3f} "
        f"contact_episode_rate={_float(summary, 'eval_contact_episode_rate', 0.0):.3f} "
        f"stage1_rate={_float(summary, 'eval_topdown_stage1_episode_rate', 0.0):.3f} "
        f"stage2_rate={_float(summary, 'eval_topdown_stage2_episode_rate', 0.0):.3f} "
        f"median_best_stage={_float(summary, 'eval_median_best_topdown_stage', -1.0):.1f} "
        f"median_best_contact={_float(summary, 'eval_median_best_contact', 0.0):.3f} "
        f"best_contact={_float(summary, 'eval_best_contact', 0.0):.3f} "
        f"median_best_strict={_float(summary, 'eval_median_best_strict_light_contact', 0.0):.3f} "
        f"best_strict={_float(summary, 'eval_best_strict_light_contact', 0.0):.3f} "
        f"median_best_lift={_float(summary, 'eval_median_best_lift', 0.0):.4f} "
        f"best_lift={_float(summary, 'eval_best_lift', 0.0):.4f} "
        f"median_strict_lift={_float(summary, 'eval_median_best_lift_with_strict_contact', 0.0):.4f} "
        f"best_strict_lift={_float(summary, 'eval_best_lift_with_strict_contact', 0.0):.4f} "
        f"median_best_blk_disp={_float(summary, 'eval_median_best_block_disp', 0.0):.4f} "
        f"best_blk_disp={_float(summary, 'eval_best_block_disp', 0.0):.4f} "
        f"median_best_align={_float(summary, 'eval_median_best_align_face_dist', 0.0):.4f} "
        f"best_align={_float(summary, 'eval_best_align_face_dist', 0.0):.4f} "
        f"median_best_align_angle={_float(summary, 'eval_median_best_align_angle', 0.0):.2f} "
        f"best_align_angle={_float(summary, 'eval_best_align_angle', 0.0):.2f} "
        f"clean_lift_rate={_float(summary, 'eval_clean_lift_episode_rate', 0.0):.3f} "
        f"upright_clean_lift_rate={_float(summary, 'eval_upright_clean_lift_episode_rate', 0.0):.3f} "
        f"xy_gate_rate={_float(summary, 'eval_lift_xy_drift_success_gate_rate', 0.0):.3f} "
        f"tilt_gate_rate={_float(summary, 'eval_lift_block_tilt_success_gate_rate', 0.0):.3f} "
        f"median_max_tilt={_float(summary, 'eval_median_max_block_tilt_deg', 0.0):.1f} "
        f"max_tilt={_float(summary, 'eval_max_block_tilt_deg', 0.0):.1f}"
    )
