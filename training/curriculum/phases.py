"""

Default v35 phased curriculum ladder.

File map:

build_default_phases:            Build the default phase ladder from CLI inputs
build_pure_rl_v35_phases:        Build the single-phase pure-RL v35 baseline from phase-1 CLI inputs
build_reanchor_recovery_phases:  Build the single-stage recovery run from a replay checkpoint
"""

from __future__ import annotations

import argparse

from training.curriculum import p01_transfer_lift02_noblock
from training.curriculum import p02_reanchor_full_weights
from training.curriculum import p03_autonomy_lift02_noblock
from training.curriculum import p04_soft_drift_penalty
from training.curriculum import p05_vertical_lift04
from training.curriculum import p06_strict_lift07
from training.curriculum import pure_rl_v35
from training.curriculum.spec import PhaseSpec


def build_default_phases(args: argparse.Namespace) -> list[PhaseSpec]:
    """Build the default phase ladder from CLI inputs.

    The individual phase modules own their defaults directly.  This function
    only wires CLI overrides into those explicit phase builders.
    """

    return [
        p01_transfer_lift02_noblock.build_phase(
            steps=args.phase1_steps,
            start_steps=args.phase1_start_steps,
            bc_only_steps=args.phase1_bc_steps,
            rl_phase_start_steps=args.phase1_rl_phase_start_steps,
            assist_mix=args.phase1_assist_mix,
            assist_floor=args.phase1_assist_floor,
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase1_min_success,
            min_median_lift=args.phase1_min_lift,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
        ),
        p02_reanchor_full_weights.build_phase(
            steps=args.phase2_steps,
            start_steps=args.phase2_start_steps,
            bc_only_steps=args.phase2_bc_steps,
            rl_phase_start_steps=args.phase2_rl_phase_start_steps,
            assist_mix=args.phase2_assist_mix,
            assist_floor=args.phase2_assist_floor,
            assist_decay_steps=(
                None if int(args.phase2_assist_decay_steps) < 0 else int(args.phase2_assist_decay_steps)
            ),
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase2_min_success,
            min_median_lift=args.phase2_min_lift,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
        ),
        p03_autonomy_lift02_noblock.build_phase(
            steps=args.phase3_steps,
            assist_mix=args.phase3_assist_mix,
            assist_floor=args.phase3_assist_floor,
            rl_n_step=args.phase3_rl_n_step,
            rl_updates_per_step=args.phase3_rl_updates_per_step,
            rl_gamma=args.phase3_rl_gamma,
            rl_policy_noise=args.phase3_rl_policy_noise,
            rl_policy_noise_finger=args.phase3_rl_policy_noise_finger,
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase3_min_success,
            min_median_lift=args.phase3_min_lift,
            max_median_disp=args.phase3_max_disp,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
            force_dagger_after_resume=args.phase3_force_dagger_after_resume,
        ),
        p04_soft_drift_penalty.build_phase(
            steps=args.phase4_steps,
            assist_floor=args.phase4_assist_floor,
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase4_min_success,
            min_median_lift=args.phase4_min_lift,
            max_median_disp=args.phase4_max_disp,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
        ),
        p05_vertical_lift04.build_phase(
            steps=args.phase5_steps,
            assist_floor=args.phase5_assist_floor,
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase5_min_success,
            min_median_lift=args.phase5_min_lift,
            max_median_disp=args.phase5_max_disp,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
        ),
        p06_strict_lift07.build_phase(
            steps=args.phase6_steps,
            assist_floor=args.phase6_assist_floor,
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase6_min_success,
            min_median_lift=args.phase6_min_lift,
            max_median_disp=args.phase6_max_disp,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
        ),
    ]


def build_pure_rl_v35_phases(args: argparse.Namespace) -> list[PhaseSpec]:
    """Build the single-phase pure-RL v35 baseline from phase-1 CLI inputs."""

    return pure_rl_v35.build_phases(
        steps=args.phase1_steps,
        start_steps=0,
        success_hold_steps=args.lift_hold_steps,
        min_success_rate=args.phase1_min_success,
        min_median_lift=args.phase1_min_lift,
        score_drop_limit=args.score_drop_limit,
        regression_patience_steps=args.regression_patience_steps,
    )


def build_reanchor_recovery_phases(args: argparse.Namespace) -> list[PhaseSpec]:
    """Build the single-stage recovery run from a replay checkpoint."""

    return [
        p02_reanchor_full_weights.build_phase(
            steps=args.phase2_steps,
            start_steps=args.phase2_start_steps,
            bc_only_steps=args.phase2_bc_steps,
            rl_phase_start_steps=args.phase2_rl_phase_start_steps,
            assist_mix=args.phase2_assist_mix,
            assist_floor=args.phase2_assist_floor,
            success_hold_steps=args.lift_hold_steps,
            min_success_rate=args.phase2_min_success,
            min_median_lift=args.phase2_min_lift,
            score_drop_limit=args.score_drop_limit,
            regression_patience_steps=args.regression_patience_steps,
        )
    ]
