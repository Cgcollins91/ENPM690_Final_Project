"""

Reset cleanup helpers for trainer-owned environment state

This module provides helper functions and data structures for clearing trainer-owned environment state on reset,
used by the training loop and environment wrapper

File map:

ZERO_RESET_ATTR_NAMES:       Define zero reset attr names constant
MINUS_ONE_RESET_ATTR_NAMES:  Define minus one reset attr names constant
fill_env_tensor_attr:        Fill one tensor-like environment attribute when present
clear_reset_tensor_attrs:    Clear tensor attributes and return names that were touched
clear_touch_counters:        Clear cached contact lift teacher and action history state
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..actions.action_history import clear_policy_level_action_history
from ..teacher.teacher_cache import clear_cached_teacher_action


ZERO_RESET_ATTR_NAMES = (
    "_grasp_off_table_counter",
    "_grasp_lift_high_contact_counter",
    "_grasp_drop_high_contact_counter",
    "_topdown_lift_drop_high_contact_counter",
    "_grasp_lift_success_counter",
    "_grasp_success_hold_counter",
    "_touch_teacher_contact_latched",
    "_touch_teacher_hold_fraction",
    "_open_hand_alignment_success_counter",
    "_topdown_light_contact_success_counter",
    "_topdown_light_contact_success_cached_step",
    "_topdown_light_contact_success_cached_value",
    "_topdown_light_contact_prev_success",
    "_topdown_light_contact_shell_drift_counter",
    "_topdown_lift_success_counter",
    "_topdown_lift_prev_success",
    "_contact_preroll_touch_phase_latched",
    "_contact_handoff_action_valid",
    "_arm_lift_latched",
    "_arm_lift_latch_signal",
    "_arm_lift_contact_counter",
    "_arm_lift_target_xy",
    "_arm_lift_target_base_z",
    "_arm_lift_target_nominal_z",
    "_arm_lift_block_xy_latch",
    "_teacher_ik_topdown_lift_progress",
    "_teacher_ik_topdown_nominal_z_blend_progress",
    "_teacher_ik_topdown_lift_freeze_active",
    "_teacher_ik_topdown_block_xy_stabilizer_m",
    "_teacher_ik_topdown_block_xy_stabilizer_active",
    "_topdown_contact_teacher_thumb_fraction",
    "_topdown_contact_teacher_index_fraction",
    "_topdown_contact_teacher_middle_fraction",
    "_topdown_contact_teacher_thumb_latched",
    "_topdown_contact_teacher_index_latched",
    "_topdown_contact_teacher_middle_latched",
    "_topdown_contact_teacher_ready",
    "_topdown_contact_teacher_finger_ready",
    "_topdown_contact_teacher_center_gate",
    "_topdown_contact_teacher_finger_close_gate",
    "_topdown_contact_teacher_wrist_yaw_release_gate",
    "_topdown_contact_teacher_descent_ready",
    "_topdown_contact_teacher_descent_ready_age",
    "_topdown_contact_teacher_descent_closure_gate",
    "_topdown_contact_teacher_descent_z",
    "_topdown_contact_teacher_descent_z_need",
    "_topdown_contact_teacher_prelift_contact_loss",
    "_topdown_contact_teacher_inward_m",
    "_topdown_contact_teacher_xy_offset",
    "_topdown_contact_teacher_tip_servo",
    "_topdown_contact_teacher_tip_servo_m",
    "_topdown_contact_teacher_post_latch_servo",
    "_topdown_contact_teacher_post_latch_servo_m",
    "_topdown_contact_teacher_post_latch_servo_active",
    "_topdown_contact_teacher_precenter_servo_m",
    "_topdown_contact_teacher_precenter_active",
    "_topdown_contact_teacher_center_servo_m",
    "_topdown_contact_teacher_center_servo_active",
    "_topdown_contact_teacher_center_err_xy",
    "_topdown_contact_teacher_live_thumb_missing",
    "_topdown_contact_teacher_live_index_missing",
    "_topdown_contact_teacher_live_middle_missing",
    "_topdown_finger_close_gate",
    "_topdown_finger_xyz_close_gate",
    "_topdown_finger_xyz_error",
    "_topdown_finger_thumb_xyz_error",
    "_topdown_finger_index_xyz_error",
    "_topdown_finger_front_gate",
    "_topdown_finger_thumb_front_margin",
    "_topdown_finger_index_front_margin",
    "_topdown_contact_teacher_servo_thumb_missing",
    "_topdown_contact_teacher_servo_index_missing",
    "_topdown_contact_teacher_servo_middle_missing",
    "_topdown_contact_teacher_arm_hold_unlock_fallback",
    "_topdown_contact_teacher_thumb_missing",
    "_topdown_contact_teacher_index_missing",
    "_topdown_contact_teacher_middle_missing",
    "_topdown_contact_teacher_thumb_z_gap",
    "_topdown_contact_teacher_index_z_gap",
    "_topdown_contact_teacher_middle_z_gap",
    "_topdown_contact_teacher_thumb_geom_done",
    "_topdown_contact_teacher_index_geom_done",
    "_topdown_contact_teacher_middle_geom_done",
)

MINUS_ONE_RESET_ATTR_NAMES = (
    "_arm_lift_latch_step",
    "_topdown_contact_teacher_thumb_hold_fraction",
    "_topdown_contact_teacher_index_hold_fraction",
    "_topdown_contact_teacher_middle_hold_fraction",
    "_topdown_contact_teacher_thumb_lift_freeze_fraction",
    "_topdown_contact_teacher_index_lift_freeze_fraction",
    "_topdown_contact_teacher_middle_lift_freeze_fraction",
)


def fill_env_tensor_attr(
    env,                                  # Param: environment or backend object used for runtime calls
    attr_name: str,  # Param: string input for attr name
    value    : float | bool,  # Param: input value normalized or converted by this helper
    env_ids  : torch.Tensor | None = None,  # Param: tensor input carrying env ids values
) -> bool:
    """Fill one tensor-like environment attribute when present

    Steps:
    - Resolve inputs for `fill_env_tensor_attr` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    tensor = getattr(env, attr_name, None)
    if tensor is None:
        return False
    if env_ids is None:
        if hasattr(tensor, "fill_"):
            tensor.fill_(value)
            return True
        return False
    if env_ids.numel() == 0:
        return False
    tensor[env_ids.to(device=tensor.device)] = value
    return True


def clear_reset_tensor_attrs(
    env,                                                                 # Param: environment or backend object used for runtime calls
    env_ids: torch.Tensor | None = None,                                 # Param: tensor input carrying env ids values
    *,
    zero_attr_names     : tuple[str, ...] = ZERO_RESET_ATTR_NAMES,  # Param: ordered candidate names used to resolve zero attr
    minus_one_attr_names: tuple[str, ...] = MINUS_ONE_RESET_ATTR_NAMES,  # Param: ordered candidate names used to resolve minus one attr
) -> tuple[str, ...]:
    """Clear tensor attributes and return names that were touched"""
    touched: list[str] = []
    for attr_name in zero_attr_names:
        if fill_env_tensor_attr(env, attr_name, 0, env_ids):
            touched.append(attr_name)
    for attr_name in minus_one_attr_names:
        if fill_env_tensor_attr(env, attr_name, -1.0, env_ids):
            touched.append(attr_name)
    return tuple(touched)


def clear_touch_counters(
    env,                                                                                  # Param: environment or backend object used for runtime calls
    env_ids: torch.Tensor | None = None,                                                  # Param: tensor input carrying env ids values
    *,
    clear_teacher_action: Callable[[object], None] | None                      = clear_cached_teacher_action,  # Param: callback used to compute or fetch clear teacher action
    clear_policy_history: Callable[[object, torch.Tensor | None], None] | None = (  # Param: callback used to compute or fetch clear policy history
        clear_policy_level_action_history
    ),
) -> tuple[str, ...]:
    """Clear cached contact lift teacher and action history state"""
    touched = clear_reset_tensor_attrs(env, env_ids)
    if clear_teacher_action is not None:
        clear_teacher_action(env)
    if clear_policy_history is not None:
        clear_policy_history(env, env_ids)
    return touched
