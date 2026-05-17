"""

Phase 1: transfer v35 teacher lift with block penalties disabled


build_phase:  Build the phase specification for the teacher/BC/DAgger transfer stage with block penalties disabled, 
                where success is defined as sustaining a 2cm lift with opposed/strict contact.
"""

from __future__ import annotations

import math

from training.curriculum.envs import default_phase_env, no_block_penalty_env
from training.curriculum.spec import PhaseSpec


def build_phase(
    *,
    steps                             : int        = 3_000_000,
    start_steps                       : int        = 100_000,
    bc_only_steps                     : int        = 300_000,
    rl_phase_start_steps              : int        = -1,
    assist_mix                        : float      = 1.0,
    assist_floor                      : float      = 0.92,
    assist_decay_steps                : int | None = None,
    success_mode                      : str        = "gated",
    success_height                    : float      = 0.020,
    success_hold_steps                : int        = 30,
    success_contact_mode              : str        = "opposed",
    success_contact_min               : float      = 0.30,
    success_xy_drift_max              : float      = 999.0,
    min_success_rate                  : float      = 0.80,
    min_median_lift                   : float      = 0.020,
    max_median_disp                   : float      = math.inf,
    score_drop_limit                  : float      = 6.0,
    regression_patience_steps         : int        = 200_000,
    block_drift_threshold             : float      = 999.0,
    contact_block_disp_max            : float      = 999.0,
    lift_terminate_drop_from_max      : float      = 0.0,
    lift_terminate_drop_min_peak      : float      = 999.0,
    lift_terminate_drop_hold_steps    : int        = 999999,
    contact_centered_contact           : float     = 2.0,
    lift_xy_drift_penalty             : float      = 0.0,
    block_tilt_lift_penalty           : float      = 0.0,
    uncentered_lift_penalty           : float      = 0.0,
    block_xy_velocity_penalty         : float      = 0.0,
    block_angular_velocity_penalty    : float      = 0.0,
    block_drop_penalty                : float      = 0.0,
    contact_one_sided                 : float      = -2.0,
    contact_bilateral_imbalance       : float      = -2.0,
    alignment_degradation             : float      = 0.0,
    lift_height_progress_requires_grip: int        = 1,
    lift_height_progress              : float      = 4.0,
    block_off_table_bonus             : float      = 20.0,
) -> PhaseSpec:
    env = {
        **default_phase_env(),
        "TOPDOWN_LIFT_SUCCESS_MODE"            : success_mode,
        "TOPDOWN_LIFT_SUCCESS_REQUIRES_CONTACT": "1",
        "TOPDOWN_LIFT_SUCCESS_CONTACT_MODE"    : success_contact_mode,
        "TOPDOWN_LIFT_SUCCESS_CONTACT_MIN"     : str(success_contact_min),
        "TOPDOWN_LIFT_SUCCESS_HEIGHT"          : str(success_height),
        "TOPDOWN_LIFT_SUCCESS_HOLD_STEPS"      : str(success_hold_steps),
        "TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX"    : str(success_xy_drift_max),
        **no_block_penalty_env(
            block_drift_threshold              = block_drift_threshold,
            contact_block_disp_max             = contact_block_disp_max,
            lift_terminate_drop_from_max       = lift_terminate_drop_from_max,
            lift_terminate_drop_min_peak       = lift_terminate_drop_min_peak,
            lift_terminate_drop_hold_steps     = lift_terminate_drop_hold_steps,
            lift_xy_drift_penalty              = lift_xy_drift_penalty,
            block_tilt_lift_penalty            = block_tilt_lift_penalty,
            uncentered_lift_penalty            = uncentered_lift_penalty,
            block_xy_velocity_penalty          = block_xy_velocity_penalty,
            block_angular_velocity_penalty     = block_angular_velocity_penalty,
            block_drop_penalty                 = block_drop_penalty,
            contact_one_sided                  = contact_one_sided,
            contact_bilateral_imbalance        = contact_bilateral_imbalance,
            alignment_degradation              = alignment_degradation,
            lift_height_progress_requires_grip = lift_height_progress_requires_grip,
            lift_height_progress               = lift_height_progress,
            block_off_table_bonus              = block_off_table_bonus,
            contact_centered_contact           = contact_centered_contact,
        ),
    }
    return PhaseSpec(
        name="p01_transfer_lift02_noblock",
        description=(
            "Teacher/BC/DAgger transfer, success is sustained 2cm lift with "
            "opposed/strict contact; block penalties/terminations off."
        ),
        steps=steps,
        start_steps=start_steps,
        bc_only_steps=bc_only_steps,
        rl_phase_start_steps=rl_phase_start_steps,
        assist_mix=assist_mix,
        assist_floor=assist_floor,
        assist_decay_steps=assist_decay_steps if assist_decay_steps is not None else max(1, steps - start_steps),
        success_height=success_height,
        success_hold_steps=success_hold_steps,
        min_success_rate=min_success_rate,
        min_median_lift=min_median_lift,
        max_median_disp=max_median_disp,
        score_drop_limit=score_drop_limit,
        regression_patience_steps=regression_patience_steps,
        env=env,
        args={},
    )
