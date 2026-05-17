"""

Shared curriculum phase data structures.

File map:

PhaseSpec:  One autonomous run-level curriculum phase
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSpec:
    """One autonomous run-level curriculum phase."""

    name                      : str
    description               : str
    steps                     : int
    start_steps               : int
    bc_only_steps             : int
    rl_phase_start_steps      : int
    assist_mix                : float
    assist_floor              : float
    assist_decay_steps        : int
    success_height            : float
    success_hold_steps        : int
    min_success_rate          : float
    min_median_lift           : float
    max_median_disp           : float
    score_drop_limit          : float
    regression_patience_steps : int
    env                       : dict[str, str]
    args                      : dict[str, str | int | float]
    force_dagger_after_resume : bool = False
    reset_optimizers_on_resume: bool = False
    remove_args               : tuple[tuple[str, bool], ...] = ()
