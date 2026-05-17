"""

Episode end line formatting helpers

File map:

EpisodeEndSummary:               Scalar fields printed at episode end
_episode_prefix:                 Handle episode prefix logic
format_topdown_episode_end:      Format topdown curriculum episode end line
format_grasp_align_episode_end:  Format grasp align episode end line
format_default_episode_end:      Format default grasp episode end line
format_episode_end_line:         Select task-family episode end formatter
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeEndSummary:
    """Scalar fields printed at episode end"""

    episode_idx                  : int  # training episode index associated with this record
    episode_step                 : int  # per-env step count inside the current episode
    episode_return               : float  # floating-point episode return value used by episode end summary
    best_tip                     : float = 0.0  # floating-point best tip value used by episode end summary
    best_phase1_palm             : float = 0.0  # floating-point best phase1 palm value used by episode end summary
    best_phase1_orient           : float = 0.0  # floating-point best phase1 orient value used by episode end summary
    thumb_err                    : float = 0.0  # floating-point thumb err value used by episode end summary
    idx_err                      : float = 0.0  # floating-point idx err value used by episode end summary
    align_face                   : float = 0.0  # block face selected for alignment scoring
    align_angle                  : float = 0.0  # alignment angle value used by topdown/contact metrics
    best_contact                 : float = 0.0  # floating-point best contact value used by episode end summary
    strict_contact               : float = 0.0  # floating-point strict contact value used by episode end summary
    best_strict_contact          : float = 0.0  # floating-point best strict contact value used by episode end summary
    best_lift                    : float = 0.0  # floating-point best lift value used by episode end summary
    best_lift_with_strict_contact: float = 0.0  # floating-point best lift with strict contact value used by episode end summary
    best_curl                    : float = 0.0  # floating-point best curl value used by episode end summary
    success                      : bool  = False  # success flag or rate for the rollout/evaluation record
    physical_success             : bool  = False  # boolean value indicating the physical success state for episode end summary
    off_table                    : bool  = False  # flag indicating that the block left the table/work surface
    block_drift                  : bool  = False  # measured block drift used by diagnostics or success checks
    best_topdown_stage           : int   = -1  # highest topdown curriculum stage reached so far
    max_topdown_unlock           : float = 0.0  # floating-point max topdown unlock value used by episode end summary
    failure_mode                 : str   = "unknown"  # string failure mode value used by episode end summary


def _episode_prefix(summary: EpisodeEndSummary) -> str:
    return (
        f"episode_end env=000 ep={int(summary.episode_idx):03d} "
        f"steps={int(summary.episode_step)} "
        f"return={float(summary.episode_return):+.3f}"
    )


def format_topdown_episode_end(summary: EpisodeEndSummary) -> str:
    """Format topdown curriculum episode end line"""
    return (
        f"{_episode_prefix(summary)} "
        f"best_stage={int(summary.best_topdown_stage)} "
        f"max_unlock={float(summary.max_topdown_unlock):.2f} "
        f"best_palm={float(summary.best_phase1_palm):.3f} "
        f"best_orient={float(summary.best_phase1_orient):.1f} "
        f"align={float(summary.align_face):.3f} "
        f"align_angle={float(summary.align_angle):.1f} "
        f"best_contact={float(summary.best_contact):.3f} "
        f"strict={float(summary.strict_contact):.3f} "
        f"best_strict={float(summary.best_strict_contact):.3f} "
        f"best_lift={float(summary.best_lift):.3f} "
        f"strict_lift={float(summary.best_lift_with_strict_contact):.3f} "
        f"success={int(bool(summary.success))} "
        f"physical_success={int(bool(summary.physical_success))} "
        f"block_drift={int(bool(summary.block_drift))} "
        f"fail={summary.failure_mode}"
    )


def format_grasp_align_episode_end(summary: EpisodeEndSummary) -> str:
    """Format grasp align episode end line"""
    return (
        f"{_episode_prefix(summary)} "
        f"best_tip={float(summary.best_tip):.3f} "
        f"best_palm={float(summary.best_phase1_palm):.3f} "
        f"best_orient={float(summary.best_phase1_orient):.1f} "
        f"thumb_err={float(summary.thumb_err):.3f} "
        f"idx_err={float(summary.idx_err):.3f} "
        f"align={float(summary.align_face):.3f} "
        f"align_angle={float(summary.align_angle):.1f} "
        f"best_contact={float(summary.best_contact):.3f} "
        f"best_curl={float(summary.best_curl):.3f} "
        f"success={int(bool(summary.success))}"
    )


def format_default_episode_end(summary: EpisodeEndSummary) -> str:
    """Format default grasp episode end line"""
    return (
        f"{_episode_prefix(summary)} "
        f"best_tip={float(summary.best_tip):.3f} "
        f"best_palm={float(summary.best_phase1_palm):.3f} "
        f"best_orient={float(summary.best_phase1_orient):.1f} "
        f"thumb_err={float(summary.thumb_err):.3f} "
        f"idx_err={float(summary.idx_err):.3f} "
        f"best_contact={float(summary.best_contact):.3f} "
        f"best_lift={float(summary.best_lift):.3f} "
        f"best_curl={float(summary.best_curl):.3f} "
        f"success={int(bool(summary.success))} "
        f"off_table={int(bool(summary.off_table))} "
        f"block_drift={int(bool(summary.block_drift))}"
    )


def format_episode_end_line(
    summary: EpisodeEndSummary,     # Param: input value used as summary
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    grasp_align_task       : bool,  # Param: boolean input controlling grasp align task
) -> str:
    """Select task-family episode end formatter"""
    if topdown_curriculum_task:
        return format_topdown_episode_end(summary)
    if grasp_align_task:
        return format_grasp_align_episode_end(summary)
    return format_default_episode_end(summary)
