"""

Teacher IK mask helpers

File map:

tensor_bool_attr_mask:               Return a bool tensor attr mask or zeros
topdown_arm_hold_frozen_mask:        Return rows where in-pocket arm hold captured a frozen pose
topdown_lift_latched_mask:           Return rows where lift latch should disable prehold servos
topdown_lift_servo_correction_mask:  Return rows where lift latch disables one servo correction
topdown_prehold_position_only_mask:  Return rows that should use position-only prehold IK
"""

from __future__ import annotations

import torch


def tensor_bool_attr_mask(
    env,                # Param: environment or backend object used for runtime calls
    attr_name: str,  # Param: string input for attr name
    shape    : torch.Size,  # Param: input value used as shape
) -> torch.Tensor:
    """Return a bool tensor attr mask or zeros"""
    value = getattr(env, attr_name, None)
    if torch.is_tensor(value) and value.shape == shape:
        return value.to(device=env.device, dtype=torch.bool)
    return torch.zeros(shape, dtype=torch.bool, device=env.device)


def topdown_arm_hold_frozen_mask(env, shape: torch.Size) -> torch.Tensor:
    """Return rows where in-pocket arm hold captured a frozen pose"""
    return tensor_bool_attr_mask(env, "_inpocket_arm_hold_frozen", shape)


def topdown_lift_latched_mask(
    env,                                       # Param: environment or backend object used for runtime calls
    shape: torch.Size,                         # Param: input value used as shape
    *,
    topdown_curriculum_lift_task      : bool,  # Param: boolean input controlling topdown curriculum lift task
    disable_prehold_servos_after_latch: bool,  # Param: boolean input controlling disable prehold servos after latch
) -> torch.Tensor:
    """Return rows where lift latch should disable prehold servos"""
    if not bool(topdown_curriculum_lift_task) or not bool(disable_prehold_servos_after_latch):
        return torch.zeros(shape, dtype=torch.bool, device=env.device)
    return tensor_bool_attr_mask(env, "_arm_lift_latched", shape)


def topdown_lift_servo_correction_mask(
    env,                                 # Param: environment or backend object used for runtime calls
    shape: torch.Size,                   # Param: input value used as shape
    *,
    topdown_curriculum_lift_task: bool,  # Param: boolean input controlling topdown curriculum lift task
    disable_after_latch         : bool,  # Param: boolean input controlling disable after latch
) -> torch.Tensor:
    """Return rows where lift latch disables one servo correction"""
    if not bool(topdown_curriculum_lift_task) or not bool(disable_after_latch):
        return torch.zeros(shape, dtype=torch.bool, device=env.device)
    return tensor_bool_attr_mask(env, "_arm_lift_latched", shape)


def topdown_prehold_position_only_mask(
    env,                                       # Param: environment or backend object used for runtime calls
    *,
    enabled                           : bool,  # Param: boolean input controlling enabled
    topdown_curriculum_task           : bool,  # Param: boolean input controlling topdown curriculum task
    stage_min                         : int,  # Param: integer input for stage min
    topdown_curriculum_lift_task      : bool,  # Param: boolean input controlling topdown curriculum lift task
    disable_prehold_servos_after_latch: bool,  # Param: boolean input controlling disable prehold servos after latch
) -> torch.Tensor:
    """Return rows that should use position-only prehold IK

    Steps:
    - Resolve inputs for `topdown_prehold_position_only_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not bool(enabled) or not bool(topdown_curriculum_task):
        return mask
    stage = getattr(env, "_topdown_stage", None)
    if torch.is_tensor(stage) and stage.shape[0] == env.num_envs:
        mask = stage.to(device=env.device) >= int(stage_min)
    mask = mask & (~topdown_arm_hold_frozen_mask(env, mask.shape))
    mask = mask & (
        ~topdown_lift_latched_mask(
            env,
            mask.shape,
            topdown_curriculum_lift_task=topdown_curriculum_lift_task,
            disable_prehold_servos_after_latch=disable_prehold_servos_after_latch,
        )
    )
    return mask
