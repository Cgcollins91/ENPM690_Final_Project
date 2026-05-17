"""

Phase 6: target 7cm lift with drift pressure restored.

File map:

build_phase:  Handle build phase logic
"""

from __future__ import annotations

from training.curriculum.envs import default_phase_env
from training.curriculum.spec import PhaseSpec


def build_phase(
    *,
    steps                         : int        = 500_000,
    start_steps                   : int        = 0,
    bc_only_steps                 : int        = 0,
    rl_phase_start_steps          : int        = 0,
    assist_mix                    : float      = 1.0,
    assist_floor                  : float      = 0.75,
    assist_decay_steps            : int | None = None,
    success_height                : float      = 0.070,
    success_hold_steps            : int        = 30,
    min_success_rate              : float      = 0.010,
    min_median_lift               : float      = 0.060,
    max_median_disp               : float      = 0.08,
    score_drop_limit              : float      = 6.0,
    regression_patience_steps     : int        = 200_000,
    block_drift_threshold         : float      = 999.0,
    lift_terminate_drop_from_max  : float      = 0.025,
    lift_terminate_drop_min_peak  : float      = 0.05,
    lift_terminate_drop_hold_steps: int        = 6,
    lift_xy_drift_penalty         : float      = -5.0,
    block_tilt_lift_penalty       : float      = -3.0,
    uncentered_lift_penalty       : float      = -3.0,
    block_xy_velocity_penalty     : float      = -1.0,
    block_angular_velocity_penalty: float      = -0.75,
    block_drop_penalty            : float      = -20.0,
    lift_height_progress          : float      = 8.0,
    block_off_table_bonus         : float      = 28.0,
    force_dagger_after_resume     : bool       = True,
    reset_optimizers_on_resume    : bool       = False,
) -> PhaseSpec:
    env = {
        **default_phase_env(),
        "TOPDOWN_LIFT_SUCCESS_MODE"                  : "height_only",
        "TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX"          : str(max_median_disp),
        "CURRICULUM_BLOCK_DRIFT_THRESHOLD"           : str(block_drift_threshold),
        "TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX"       : str(lift_terminate_drop_from_max),
        "TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK"       : str(lift_terminate_drop_min_peak),
        "TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS"     : str(lift_terminate_drop_hold_steps),
        "CURRICULUM_W_LIFT_XY_DRIFT_PENALTY"         : str(lift_xy_drift_penalty),
        "CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY"       : str(block_tilt_lift_penalty),
        "CURRICULUM_W_UNCENTERED_LIFT_PENALTY"       : str(uncentered_lift_penalty),
        "CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY"     : str(block_xy_velocity_penalty),
        "CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY": str(block_angular_velocity_penalty),
        "CURRICULUM_W_BLOCK_DROP_PENALTY"            : str(block_drop_penalty),
        "CURRICULUM_W_LIFT_HEIGHT_PROGRESS"          : str(lift_height_progress),
        "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS"         : str(block_off_table_bonus),
    }
    return PhaseSpec(
        name="p06_strict_lift07",
        description="7cm lift target with drift as success gate but still not a hard termination.",
        steps=steps,
        start_steps=start_steps,
        bc_only_steps=bc_only_steps,
        rl_phase_start_steps=rl_phase_start_steps,
        assist_mix=assist_mix,
        assist_floor=assist_floor,
        assist_decay_steps=assist_decay_steps if assist_decay_steps is not None else steps,
        success_height=success_height,
        success_hold_steps=success_hold_steps,
        min_success_rate=min_success_rate,
        min_median_lift=min_median_lift,
        max_median_disp=max_median_disp,
        score_drop_limit=score_drop_limit,
        regression_patience_steps=regression_patience_steps,
        env=env,
        args={},
        force_dagger_after_resume=force_dagger_after_resume,
        reset_optimizers_on_resume=reset_optimizers_on_resume,
    )
