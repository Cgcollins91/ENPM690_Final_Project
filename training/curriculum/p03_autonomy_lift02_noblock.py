"""

Phase 3: warm-start RL autonomy for 2cm lift with block penalties disabled.

File map:

build_phase:  Handle build phase logic
"""

from __future__ import annotations

import math

from training.curriculum.envs import default_phase_env, lift_first_reward_env, no_block_penalty_env
from training.curriculum.spec import PhaseSpec


def build_phase(
    *,
    steps                             : int        = 500_000,
    start_steps                       : int        = 0,
    bc_only_steps                     : int        = 0,
    rl_phase_start_steps              : int        = 0,
    assist_mix                        : float      = 1.0,
    assist_floor                      : float      = 0.90,
    assist_decay_steps                : int | None = None,
    rl_n_step                         : int | None = None,
    rl_updates_per_step               : int | None = None,
    rl_gamma                          : float | None = None,
    rl_policy_noise                   : float | None = None,
    rl_policy_noise_finger            : float | None = None,
    success_height                    : float      = 0.020,
    success_hold_steps                : int        = 30,
    min_success_rate                  : float      = 0.30,
    min_median_lift                   : float      = 0.020,
    max_median_disp                   : float      = math.inf,
    score_drop_limit                  : float      = 6.0,
    regression_patience_steps         : int        = 200_000,
    force_dagger_after_resume         : bool       = True,
    reset_optimizers_on_resume        : bool       = False,
    block_drift_threshold             : float      = 999.0,
    contact_block_disp_max            : float      = 999.0,
    lift_terminate_drop_from_max      : float      = 0.0,
    lift_terminate_drop_min_peak      : float      = 999.0,
    lift_terminate_drop_hold_steps    : int        = 999999,
    lift_xy_drift_penalty             : float      = 0.0,
    block_tilt_lift_penalty           : float      = 0.0,
    uncentered_lift_penalty           : float      = 0.0,
    block_xy_velocity_penalty         : float      = 0.0,
    block_angular_velocity_penalty    : float      = 0.0,
    block_drop_penalty                : float      = 0.0,
    contact_one_sided                 : float      = 0.0,
    contact_bilateral_imbalance       : float      = 0.0,
    alignment_degradation             : float      = 0.0,
    lift_height_progress_requires_grip: int        = 0,
    lift_height_progress              : float      = 40.0,
    block_off_table_bonus             : float      = 200.0,
    contact_target_distance           : float      = -1.0,
    contact_vertical_gap              : float      = -1.0,
    contact_thumb_contact             : float      = 0.10,
    contact_index_contact             : float      = 0.10,
    contact_opposed_contact           : float      = 2.0,
    contact_lift_progress             : float      = 150.0,
    lift_with_grip                    : float      = 80.0,
    centered_lift_progress            : float      = 0.0,
    centered_upright_lift_bonus       : float      = 0.0,
    stage2_floor                      : float      = 0.0,
    contact_centered_contact          : float      = 1.0,
    light_contact_success_bonus       : float      = 0.0,
    contact_smooth_success_pose       : float      = 2.0,
    contact_smooth_success_with_contact: float     = 4.0,
) -> PhaseSpec:
    """Process for `build_phase`

    Steps:
    - Resolve inputs for `build_phase` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    env = {
        **default_phase_env(),
        **no_block_penalty_env(
            block_drift_threshold=block_drift_threshold,
            contact_block_disp_max=contact_block_disp_max,
            lift_terminate_drop_from_max=lift_terminate_drop_from_max,
            lift_terminate_drop_min_peak=lift_terminate_drop_min_peak,
            lift_terminate_drop_hold_steps=lift_terminate_drop_hold_steps,
            lift_xy_drift_penalty=lift_xy_drift_penalty,
            block_tilt_lift_penalty=block_tilt_lift_penalty,
            uncentered_lift_penalty=uncentered_lift_penalty,
            block_xy_velocity_penalty=block_xy_velocity_penalty,
            block_angular_velocity_penalty=block_angular_velocity_penalty,
            block_drop_penalty=block_drop_penalty,
            contact_one_sided=contact_one_sided,
            contact_bilateral_imbalance=contact_bilateral_imbalance,
            alignment_degradation=alignment_degradation,
            lift_height_progress_requires_grip=lift_height_progress_requires_grip,
            lift_height_progress=lift_height_progress,
            block_off_table_bonus=block_off_table_bonus,
        ),
        **lift_first_reward_env(
            contact_target_distance=contact_target_distance,
            contact_vertical_gap=contact_vertical_gap,
            contact_thumb_contact=contact_thumb_contact,
            contact_index_contact=contact_index_contact,
            contact_opposed_contact=contact_opposed_contact,
            contact_lift_progress=contact_lift_progress,
            lift_with_grip=lift_with_grip,
            centered_lift_progress=centered_lift_progress,
            centered_upright_lift_bonus=centered_upright_lift_bonus,
            block_off_table_bonus=block_off_table_bonus,
            stage2_floor=stage2_floor,
            contact_centered_contact=contact_centered_contact,
            light_contact_success_bonus=light_contact_success_bonus,
            contact_smooth_success_pose=contact_smooth_success_pose,
            contact_smooth_success_with_contact=contact_smooth_success_with_contact,
        ),
    }
    args: dict[str, int | float] = {}
    if rl_n_step is not None:
        args["--rl-n-step"] = rl_n_step
    if rl_updates_per_step is not None:
        args["--rl-updates-per-step"] = rl_updates_per_step
    if rl_gamma is not None:
        args["--rl-gamma"] = rl_gamma
    if rl_policy_noise is not None:
        args["--rl-policy-noise"] = rl_policy_noise
    if rl_policy_noise_finger is not None:
        args["--rl-policy-noise-finger"] = rl_policy_noise_finger

    return PhaseSpec(
        name="p03_autonomy_lift02_noblock",
        description="Warm-start RL autonomy for the 2cm lift behavior, still no block penalties.",
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
        args=args,
        force_dagger_after_resume=force_dagger_after_resume,
        reset_optimizers_on_resume=reset_optimizers_on_resume,
    )
