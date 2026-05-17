"""

Recovery phase: teacher re-anchor with full reward and BC weights


build_phase:  Build the phase specification for the teacher/BC/DAgger transfer stage with block penalties disabled, 
                where success is defined as sustaining a 2cm lift with opposed/strict contact.
"""

from __future__ import annotations

import math

from training.curriculum.envs import default_phase_env
from training.curriculum.spec import PhaseSpec


def build_phase(
    *,
    steps                         : int        = 1_000_000,
    start_steps                   : int        = 100_000,
    bc_only_steps                 : int        = 100_000,
    rl_phase_start_steps          : int        = 100_000,
    assist_mix                    : float      = 0.90,
    assist_floor                  : float      = 0.90,
    assist_decay_steps            : int | None = None,
    success_height                : float      = 0.020,
    success_hold_steps            : int        = 30,
    success_contact_mode          : str        = "opposed",
    success_contact_min           : float      = 0.30,
    success_xy_drift_max          : float      = 0.12,
    min_success_rate              : float      = 0.10,
    min_median_lift               : float      = 0.020,
    max_median_disp               : float      = math.inf,
    score_drop_limit              : float      = 6.0,
    regression_patience_steps     : int        = 1_000_000,
    block_drift_threshold         : float      = 0.50,
    contact_block_disp_max        : float      = 0.06,
    contact_opposed_contact       : float      = 24.0,
    contact_bilateral_contact     : float      = 8.0,
    contact_bilateral_imbalance   : float      = -2.0,
    contact_centered_contact      : float      = 12.0,
    contact_smooth_success_pose   : float      = 10.0,
    contact_smooth_success_contact: float      = 16.0,
    contact_success_now_continuous: float      = 12.0,
    light_contact_success_bonus   : float      = 80.0,
    lift_terminate_drop_from_max  : float      = 0.025,
    lift_terminate_drop_min_peak  : float      = 0.035,
    lift_terminate_drop_hold_steps: int        = 2,
    force_dagger_after_resume     : bool       = False,
    reset_optimizers_on_resume    : bool       = True,
) -> PhaseSpec:
    resolved_assist_decay_steps = (
        max(1, int(steps) - int(start_steps))
        if assist_decay_steps is None
        else int(assist_decay_steps)
    )
    env = {
        **default_phase_env(),
        "TOPDOWN_LIFT_SUCCESS_MODE"                 : "gated",
        "TOPDOWN_LIFT_SUCCESS_REQUIRES_CONTACT"     : "1",
        "TOPDOWN_LIFT_SUCCESS_CONTACT_MODE"         : success_contact_mode,
        "TOPDOWN_LIFT_SUCCESS_CONTACT_MIN"          : str(success_contact_min),
        "TOPDOWN_LIFT_SUCCESS_HEIGHT"               : str(success_height),
        "TOPDOWN_LIFT_SUCCESS_HOLD_STEPS"           : str(success_hold_steps),
        "TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX"         : str(success_xy_drift_max),
        "CURRICULUM_BLOCK_DRIFT_THRESHOLD"          : str(block_drift_threshold),
        "CURRICULUM_CONTACT_BLOCK_DISP_MAX"         : str(contact_block_disp_max),
        "TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX"      : str(lift_terminate_drop_from_max),
        "TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK"      : str(lift_terminate_drop_min_peak),
        "TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS"    : str(lift_terminate_drop_hold_steps),
        "CURRICULUM_LIFT_HEIGHT_PROGRESS_REQUIRES_GRIP": "1",
        "CURRICULUM_W_ALIGNMENT_DEGRADATION"        : "-25.0",
        "CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY": "-1",
        "CURRICULUM_W_BLOCK_DROP_PENALTY"           : "-5",
        "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS"        : "15",
        "CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY"      : "-15",
        "CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY"    : "-2",
        "CURRICULUM_W_CENTERED_UPRIGHT_LIFT_BONUS"  : "20",
        "CURRICULUM_W_CONTACT_OPPOSED_CONTACT"      : str(contact_opposed_contact),
        "CURRICULUM_W_CONTACT_BILATERAL_CONTACT"    : str(contact_bilateral_contact),
        "CURRICULUM_W_CONTACT_BILATERAL_IMBALANCE"  : str(contact_bilateral_imbalance),
        "CURRICULUM_W_CONTACT_CENTERED_CONTACT"     : str(contact_centered_contact),
        "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_POSE"  : str(contact_smooth_success_pose),
        "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_WITH_CONTACT": str(contact_smooth_success_contact),
        "CURRICULUM_W_CONTACT_SUCCESS_NOW_CONTINUOUS": str(contact_success_now_continuous),
        "CURRICULUM_W_LIGHT_CONTACT_SUCCESS_BONUS"  : str(light_contact_success_bonus),
        "CURRICULUM_W_CONTACT_ONE_SIDED"            : "-4",
        "CURRICULUM_W_LIFT_HEIGHT_PROGRESS"         : "2",
        "CURRICULUM_W_LIFT_WITH_GRIP"               : "8",
        "CURRICULUM_W_LIFT_XY_DRIFT_PENALTY"        : "-12",
        "CURRICULUM_W_UNCENTERED_LIFT_PENALTY"      : "-8",
        "TEACHER_BC_WEIGHT"                         : "0.0",
        "TEACHER_BC_ARM_WEIGHT"                     : "10.0",
        "TEACHER_BC_FINGER_WEIGHT"                  : "4.0",
        "TEACHER_BC_DECAY_STEPS"                    : "0",
        "RL_TEACHER_BC_WEIGHT"                      : "0.0",
        "RL_TEACHER_BC_ARM_WEIGHT"                  : "10.0",
        "RL_TEACHER_BC_FINGER_WEIGHT"               : "4.0",
        "RL_TEACHER_BC_DECAY_STEPS"                 : "0",
        "RL_ACTOR_FREEZE_STEPS"                     : "0",
        "POLICY_BC_RELABEL"                         : "1",
        "RL_POLICY_BC_RELABEL"                      : "1",
    }
    args: dict[str, int | float] = {
        "--teacher-bc-weight"          : 0.0,
        "--teacher-bc-arm-weight"      : 10.0,
        "--teacher-bc-finger-weight"   : 4.0,
        "--teacher-bc-decay-steps"     : 0,
        "--rl-teacher-bc-weight"       : 0.0,
        "--rl-teacher-bc-arm-weight"   : 10.0,
        "--rl-teacher-bc-finger-weight": 4.0,
        "--rl-teacher-bc-decay-steps"  : 0,
        "--rl-actor-freeze-steps"      : 0,
        "--policy-bc-relabel"          : 1,
        "--rl-policy-bc-relabel"       : 1,
    }
    return PhaseSpec(
        name="p02_reanchor_full_weights",
        description=(
            "Recovery stage from replay: 100k teacher-only warmup, then linear "
            "teacher-assist bleed from the configured peak to floor with full "
            "teacher BC weights and restored v35 block drift/drop/tilt reward penalties."
        ),
        steps=steps,
        start_steps=start_steps,
        bc_only_steps=bc_only_steps,
        rl_phase_start_steps=rl_phase_start_steps,
        assist_mix=assist_mix,
        assist_floor=assist_floor,
        assist_decay_steps=resolved_assist_decay_steps,
        success_height=success_height,
        success_hold_steps=success_hold_steps,
        min_success_rate=min_success_rate,
        min_median_lift=min_median_lift,
        max_median_disp=max_median_disp,
        score_drop_limit=score_drop_limit,
        regression_patience_steps=regression_patience_steps,
        env=env,
        args=args,
        force_dagger_after_resume=force_dagger_after_resume,
        reset_optimizers_on_resume=reset_optimizers_on_resume,
    )


__all__ = ["build_phase"]
