"""

Shared environment override groups for curriculum phases


default_phase_env:           Defaults copied from the v35 curriculum runner, not imported from profiles
no_block_penalty_env:        Disable block drift/drop penalties for pure lift transfer phases
lift_first_reward_env:       Phase-2 reward shape that makes any physical lift dominate contact farming
strict_contact_no_lift_env:  Reward strict centered contact while leaving lift neutral
"""

from __future__ import annotations


def default_phase_env(
    *,
    jsonl_per_env_rows          : int   = 0,
    jsonl_eval_step_rows        : int   = 0,
    jsonl_preroll_event_rows    : int   = 0,
    verbose_jsonl               : int   = 0,
    lift_terminate_tilt_deg     : float = 0.0,
    align_failfast_after_seconds: float = 0.0,
) -> dict[str, str]:
    """Defaults copied from the v35 curriculum runner, not imported from profiles."""

    return {
        "TOPDOWN_JSONL_PER_ENV_ROWS"             : str(jsonl_per_env_rows),
        "TOPDOWN_JSONL_EVAL_STEP_ROWS"           : str(jsonl_eval_step_rows),
        "TOPDOWN_JSONL_PREROLL_EVENT_ROWS"       : str(jsonl_preroll_event_rows),
        "TOPDOWN_VERBOSE_JSONL"                  : str(verbose_jsonl),
        "TOPDOWN_LIFT_TERMINATE_TILT_DEG"        : str(lift_terminate_tilt_deg),
        "CURRICULUM_ALIGN_FAILFAST_AFTER_SECONDS": str(align_failfast_after_seconds),
    }


def no_block_penalty_env(
    *,
    block_drift_threshold             : float = 999.0,
    contact_block_disp_max            : float = 999.0,
    lift_terminate_drop_from_max      : float = 0.0,
    lift_terminate_drop_min_peak      : float = 999.0,
    lift_terminate_drop_hold_steps    : int   = 999999,
    lift_xy_drift_penalty             : float = 0.0,
    block_tilt_lift_penalty           : float = 0.0,
    uncentered_lift_penalty           : float = 0.0,
    block_xy_velocity_penalty         : float = 0.0,
    block_angular_velocity_penalty    : float = 0.0,
    block_drop_penalty                : float = 0.0,
    contact_one_sided                 : float = 0.0,
    contact_bilateral_imbalance       : float = 0.0,
    contact_centered_contact          : float = 0.0,
    alignment_degradation             : float = 0.0,
    lift_height_progress_requires_grip: int   = 0,
    lift_height_progress              : float = 4.0,
    block_off_table_bonus             : float = 20.0,
) -> dict[str, str]:
    """Disable block drift/drop penalties for pure lift transfer phases."""

    return {
        "CURRICULUM_BLOCK_DRIFT_THRESHOLD"             : str(block_drift_threshold),
        "CURRICULUM_CONTACT_BLOCK_DISP_MAX"            : str(contact_block_disp_max),
        "TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX"         : str(lift_terminate_drop_from_max),
        "TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK"         : str(lift_terminate_drop_min_peak),
        "TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS"       : str(lift_terminate_drop_hold_steps),
        "CURRICULUM_W_LIFT_XY_DRIFT_PENALTY"           : str(lift_xy_drift_penalty),
        "CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY"         : str(block_tilt_lift_penalty),
        "CURRICULUM_W_UNCENTERED_LIFT_PENALTY"         : str(uncentered_lift_penalty),
        "CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY"       : str(block_xy_velocity_penalty),
        "CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY"  : str(block_angular_velocity_penalty),
        "CURRICULUM_W_BLOCK_DROP_PENALTY"              : str(block_drop_penalty),
        "CURRICULUM_W_CONTACT_ONE_SIDED"               : str(contact_one_sided),
        "CURRICULUM_W_CONTACT_BILATERAL_IMBALANCE"     : str(contact_bilateral_imbalance),
        "CURRICULUM_W_ALIGNMENT_DEGRADATION"           : str(alignment_degradation),
        "CURRICULUM_LIFT_HEIGHT_PROGRESS_REQUIRES_GRIP": str(lift_height_progress_requires_grip),
        "CURRICULUM_W_LIFT_HEIGHT_PROGRESS"            : str(lift_height_progress),
        "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS"           : str(block_off_table_bonus),
        "CURRICULUM_W_CONTACT_CENTERED_CONTACT"        : str(contact_centered_contact),
    }


def lift_first_reward_env(
    *,
    contact_target_distance              : float = -1.0,
    contact_vertical_gap                 : float = -1.0,
    contact_thumb_contact                : float = 0.10,
    contact_index_contact                : float = 0.10,
    contact_opposed_contact              : float = 2.0,
    contact_lift_progress                : float = 150.0,
    lift_with_grip                       : float = 80.0,
    centered_lift_progress               : float = 0.0,
    centered_upright_lift_bonus          : float = 0.0,
    block_off_table_bonus                : float = 200.0,
    stage2_floor                         : float = 0.0,
    contact_centered_contact             : float = 1.0,
    light_contact_success_bonus          : float = 0.0,
    contact_smooth_success_pose          : float = 2.0,
    contact_smooth_success_with_contact  : float = 4.0,
) -> dict[str, str]:
    """Phase-2 reward shape that makes any physical lift dominate contact farming."""

    return {
        "CURRICULUM_W_CONTACT_TARGET_DISTANCE"            : str(contact_target_distance),
        "CURRICULUM_W_CONTACT_VERTICAL_GAP"               : str(contact_vertical_gap),
        "CURRICULUM_W_CONTACT_THUMB_CONTACT"              : str(contact_thumb_contact),
        "CURRICULUM_W_CONTACT_INDEX_CONTACT"              : str(contact_index_contact),
        "CURRICULUM_W_CONTACT_OPPOSED_CONTACT"            : str(contact_opposed_contact),
        "CURRICULUM_W_CONTACT_LIFT_PROGRESS"              : str(contact_lift_progress),
        "CURRICULUM_W_LIFT_WITH_GRIP"                     : str(lift_with_grip),
        "CURRICULUM_W_CENTERED_LIFT_PROGRESS"             : str(centered_lift_progress),
        "CURRICULUM_W_CENTERED_UPRIGHT_LIFT_BONUS"        : str(centered_upright_lift_bonus),
        "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS"              : str(block_off_table_bonus),
        "CURRICULUM_W_STAGE2_FLOOR"                       : str(stage2_floor),
        "CURRICULUM_W_CONTACT_CENTERED_CONTACT"           : str(contact_centered_contact),
        "CURRICULUM_W_LIGHT_CONTACT_SUCCESS_BONUS"        : str(light_contact_success_bonus),
        "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_POSE"        : str(contact_smooth_success_pose),
        "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_WITH_CONTACT": str(contact_smooth_success_with_contact),
    }


def strict_contact_no_lift_env(
    *,
    contact_target_distance            : float = -6.0,
    contact_vertical_gap               : float = -6.0,
    contact_thumb_contact              : float = 1.5,
    contact_index_contact              : float = 1.5,
    contact_opposed_contact            : float = 24.0,
    contact_bilateral_contact          : float = 8.0,
    contact_bilateral_imbalance        : float = -2.0,
    contact_one_sided                  : float = -2.0,
    contact_one_sided_flip             : float = -1.0,
    contact_deep_shell                 : float = 8.0,
    contact_centered_contact           : float = 12.0,
    contact_smooth_success_pose        : float = 10.0,
    contact_smooth_success_with_contact: float = 16.0,
    light_contact_success_bonus        : float = 80.0,
    stage2_floor                       : float = 0.75,
    lift_height_progress               : float = 0.0,
    contact_lift_progress              : float = 0.0,
    lift_with_grip                     : float = 0.0,
    centered_lift_progress             : float = 0.0,
    centered_upright_lift_bonus        : float = 0.0,
    block_off_table_bonus              : float = 0.0,
    block_xy_velocity_penalty          : float = 0.0,
    block_angular_velocity_penalty     : float = 0.0,
    lift_xy_drift_penalty              : float = 0.0,
    block_tilt_lift_penalty            : float = 0.0,
    uncentered_lift_penalty            : float = 0.0,
    block_drop_penalty                 : float = 0.0,
) -> dict[str, str]:
    """Reward strict centered contact while leaving lift neutral."""

    return {
        "CURRICULUM_W_CONTACT_TARGET_DISTANCE"            : str(contact_target_distance),
        "CURRICULUM_W_CONTACT_VERTICAL_GAP"               : str(contact_vertical_gap),
        "CURRICULUM_W_CONTACT_THUMB_CONTACT"              : str(contact_thumb_contact),
        "CURRICULUM_W_CONTACT_INDEX_CONTACT"              : str(contact_index_contact),
        "CURRICULUM_W_CONTACT_OPPOSED_CONTACT"            : str(contact_opposed_contact),
        "CURRICULUM_W_CONTACT_BILATERAL_CONTACT"          : str(contact_bilateral_contact),
        "CURRICULUM_W_CONTACT_BILATERAL_IMBALANCE"        : str(contact_bilateral_imbalance),
        "CURRICULUM_W_CONTACT_ONE_SIDED"                  : str(contact_one_sided),
        "CURRICULUM_W_CONTACT_ONE_SIDED_FLIP"             : str(contact_one_sided_flip),
        "CURRICULUM_W_CONTACT_DEEP_SHELL"                 : str(contact_deep_shell),
        "CURRICULUM_W_CONTACT_CENTERED_CONTACT"           : str(contact_centered_contact),
        "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_POSE"        : str(contact_smooth_success_pose),
        "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_WITH_CONTACT": str(contact_smooth_success_with_contact),
        "CURRICULUM_W_LIGHT_CONTACT_SUCCESS_BONUS"        : str(light_contact_success_bonus),
        "CURRICULUM_W_STAGE2_FLOOR"                       : str(stage2_floor),
        "CURRICULUM_W_LIFT_HEIGHT_PROGRESS"               : str(lift_height_progress),
        "CURRICULUM_W_CONTACT_LIFT_PROGRESS"              : str(contact_lift_progress),
        "CURRICULUM_W_LIFT_WITH_GRIP"                     : str(lift_with_grip),
        "CURRICULUM_W_CENTERED_LIFT_PROGRESS"             : str(centered_lift_progress),
        "CURRICULUM_W_CENTERED_UPRIGHT_LIFT_BONUS"        : str(centered_upright_lift_bonus),
        "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS"              : str(block_off_table_bonus),
        "CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY"          : str(block_xy_velocity_penalty),
        "CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY"     : str(block_angular_velocity_penalty),
        "CURRICULUM_W_LIFT_XY_DRIFT_PENALTY"              : str(lift_xy_drift_penalty),
        "CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY"            : str(block_tilt_lift_penalty),
        "CURRICULUM_W_UNCENTERED_LIFT_PENALTY"            : str(uncentered_lift_penalty),
        "CURRICULUM_W_BLOCK_DROP_PENALTY"                 : str(block_drop_penalty),
    }
