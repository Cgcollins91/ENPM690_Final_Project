"""

Topdown pre-roll state readers with lazy task-state routing

File map:

_topdown_state_machine:                    Handle topdown state machine logic
_env_tensor_attr:                          Handle env tensor attr logic
topdown_curriculum_preroll_shell_tensors:  Return topdown pre-roll release diagnostics from state-machine tensors
topdown_curriculum_preroll_release_mask:   Return rows ready to leave topdown pre-roll
"""

from __future__ import annotations

from types import ModuleType

import torch


def _topdown_state_machine() -> ModuleType:
    from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp import (
        state_machine,
    )

    return state_machine


def _env_tensor_attr(
    env,                          # Param: environment or backend object used for runtime calls
    attr_name: str,               # Param: string input for attr name
    *,
    dtype  : torch.dtype,  # Param: torch dtype used when converting or allocating tensors
    default: float | int | bool,  # Param: fallback value used when the input omits or rejects a setting
) -> torch.Tensor:
    value = getattr(env, attr_name, None)
    if torch.is_tensor(value) and value.shape[0] == env.num_envs:
        return value.to(device=env.device, dtype=dtype)
    return torch.full((env.num_envs,), default, dtype=dtype, device=env.device)


def topdown_curriculum_preroll_shell_tensors(
    env,                                             # Param: environment or backend object used for runtime calls
    *,
    unlock_progress     : float,  # Param: floating-point input for unlock progress
    state_machine_module: ModuleType | None = None,  # Param: input value used as state machine module
) -> dict[str, torch.Tensor]:
    """Return topdown pre-roll release diagnostics from state-machine tensors

    Steps:
    - Resolve inputs for `topdown_curriculum_preroll_shell_tensors` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    sm = _topdown_state_machine() if state_machine_module is None else state_machine_module
    sm.ensure_curriculum_stage_updated(env)

    stage = _env_tensor_attr(env, "_topdown_stage", dtype=torch.long, default=-1)
    unlock = _env_tensor_attr(
        env,
        "_topdown_finger_unlock_progress",
        dtype=torch.float32,
        default=0.0,
    )
    contact_pose_ready = _env_tensor_attr(
        env,
        "_topdown_contact_pose_ready",
        dtype=torch.bool,
        default=False,
    )
    contact_pose_hold = _env_tensor_attr(
        env,
        "_topdown_contact_pose_hold",
        dtype=torch.float32,
        default=0.0,
    )
    contact_pose_age = _env_tensor_attr(
        env,
        "_topdown_contact_pose_age",
        dtype=torch.float32,
        default=0.0,
    )

    palm_d = sm.palm_distance_contact(env)
    palm_h = sm.palm_height_error_contact(env)
    drop_deg = torch.rad2deg(sm.palm_drop_axis_error_rad(env))
    yaw_deg = torch.rad2deg(sm.palm_yaw_axis_error_rad(env))
    align_e = sm.open_hand_alignment_error(env)
    opposed = sm.opposite_face_gate(env)
    blk_disp = sm.block_displacement(env)
    lift = sm.block_lift_height(env)
    shell_now = sm.contact_pose_shell_now(env)
    ready = shell_now & contact_pose_ready & (unlock >= float(unlock_progress))
    return {
        "ready"                 : ready,
        "contact_pose_shell_now": shell_now,
        "contact_pose_ready"    : contact_pose_ready,
        "contact_pose_hold"     : contact_pose_hold,
        "contact_pose_age"      : contact_pose_age,
        "stage"                 : stage.to(dtype=torch.float32),
        "unlock"                : unlock,
        "palm_contact_dist"     : palm_d,
        "palm_contact_height"   : palm_h,
        "drop_deg"              : drop_deg,
        "yaw_deg"               : yaw_deg,
        "align"                 : align_e,
        "opposed"               : opposed,
        "block_displacement"    : blk_disp,
        "lift"                  : lift,
    }


def topdown_curriculum_preroll_release_mask(
    env,                                             # Param: environment or backend object used for runtime calls
    *,
    enabled             : bool,  # Param: boolean input controlling enabled
    unlock_progress     : float,  # Param: floating-point input for unlock progress
    state_machine_module: ModuleType | None = None,  # Param: input value used as state machine module
) -> torch.Tensor:
    """Return rows ready to leave topdown pre-roll"""
    if not bool(enabled):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return topdown_curriculum_preroll_shell_tensors(
        env,
        unlock_progress=unlock_progress,
        state_machine_module=state_machine_module,
    )["ready"]
