"""

Import-safe teacher IK state planning helpers

File map:

TeacherIkBodyIndices:      Resolved body and Jacobian indices for teacher IK
TeacherIkStatePlan:        Import-safe tensor plan for teacher IK state construction
jacobian_body_index:       Map a body index to PhysX Jacobian body index
teacher_ik_body_indices:   Resolve body indices needed by topdown teacher IK
palm_local_offsets:        Return thumb and index positions in palm-local coordinates
teacher_ik_joint_weights:  Resolve teacher IK servo joint weights from a comma list
teacher_ik_state_plan:     Build the pure teacher IK state plan without Isaac controllers
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..geometry.geometry import joint_selection_weights, parse_joint_name_list, quat_wxyz_to_matrix


@dataclass(frozen=True)
class TeacherIkBodyIndices:
    """Resolved body and Jacobian indices for teacher IK"""

    palm_body_idx                   : int  # Field: index identifying the palm body entry
    palm_jacobian_body_idx          : int  # Field: index identifying the palm jacobian body entry
    contact_thumb_jacobian_body_idx : int  # Field: index identifying the contact thumb jacobian body entry
    contact_index_jacobian_body_idx : int  # Field: index identifying the contact index jacobian body entry
    contact_middle_jacobian_body_idx: int  # Field: index identifying the contact middle jacobian body entry
    thumb_body_idx                  : int  # Field: index identifying the thumb body entry
    index_body_idx                  : int  # Field: index identifying the index body entry


@dataclass(frozen=True)
class TeacherIkStatePlan:
    """Import-safe tensor plan for teacher IK state construction"""

    arm_joint_ids             : tuple[int, ...]  # Field: integer arm joint ids value tracked by teacher ik state plan
    body_indices              : TeacherIkBodyIndices  # Field: stores body indices for teacher ik state plan
    target_quat               : torch.Tensor  # Field: tensor containing target quat values for batched env rows
    thumb_local_offset        : torch.Tensor  # Field: tensor containing thumb local offset values for batched env rows
    index_local_offset        : torch.Tensor  # Field: tensor containing index local offset values for batched env rows
    align_servo_joint_weights : torch.Tensor  # Field: weight applied to align servo joint terms
    planar_align_joint_weights: torch.Tensor  # Field: weight applied to planar align joint terms


def jacobian_body_index(body_idx: int, *, fixed_base: bool) -> int:
    """Map a body index to PhysX Jacobian body index"""
    return int(body_idx) - 1 if bool(fixed_base) else int(body_idx)


def teacher_ik_body_indices(
    body_names: Sequence[str],  # Param: ordered candidate names used to resolve body
    *,
    palm_link          : str,  # Param: string input for palm link
    thumb_link         : str,  # Param: string input for thumb link
    index_link         : str,  # Param: string input for index link
    contact_thumb_link : str,  # Param: string input for contact thumb link
    contact_index_link : str,  # Param: string input for contact index link
    contact_middle_link: str,  # Param: string input for contact middle link
    fixed_base         : bool,  # Param: boolean input controlling fixed base
) -> TeacherIkBodyIndices:
    """Resolve body indices needed by topdown teacher IK

    Steps:
    - Resolve inputs for `teacher_ik_body_indices` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    palm_body_idx = body_names.index(palm_link)
    contact_thumb_body_idx = body_names.index(contact_thumb_link)
    contact_index_body_idx = body_names.index(contact_index_link)
    contact_middle_body_idx = body_names.index(contact_middle_link)
    return TeacherIkBodyIndices(
        palm_body_idx=palm_body_idx,
        palm_jacobian_body_idx=jacobian_body_index(palm_body_idx, fixed_base=fixed_base),
        contact_thumb_jacobian_body_idx=jacobian_body_index(contact_thumb_body_idx, fixed_base=fixed_base),
        contact_index_jacobian_body_idx=jacobian_body_index(contact_index_body_idx, fixed_base=fixed_base),
        contact_middle_jacobian_body_idx=jacobian_body_index(contact_middle_body_idx, fixed_base=fixed_base),
        thumb_body_idx=body_names.index(thumb_link),
        index_body_idx=body_names.index(index_link),
    )


def palm_local_offsets(
    *,
    palm_pos : torch.Tensor,  # Param: tensor input carrying palm pos values
    palm_quat: torch.Tensor,  # Param: tensor input carrying palm quat values
    thumb_pos: torch.Tensor,  # Param: tensor input carrying thumb pos values
    index_pos: torch.Tensor,  # Param: tensor input carrying index pos values
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return thumb and index positions in palm-local coordinates

    Steps:
    - Resolve inputs for `palm_local_offsets` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    rot = quat_wxyz_to_matrix(palm_quat.unsqueeze(0))[0]
    rot_t = rot.transpose(0, 1)
    thumb_offset = rot_t @ (thumb_pos - palm_pos)
    index_offset = rot_t @ (index_pos - palm_pos)
    return thumb_offset, index_offset


def teacher_ik_joint_weights(
    arm_joint_names: Sequence[str],      # Param: ordered candidate names used to resolve arm joint
    *,
    raw_spec    : str | None,  # Param: string input for raw spec
    default_spec: str,  # Param: string input for default spec
    device      : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype       : torch.dtype = torch.float32,  # Param: torch dtype used when converting or allocating tensors
) -> torch.Tensor:
    """Resolve teacher IK servo joint weights from a comma list"""
    selected = parse_joint_name_list(raw_spec, default_spec)
    return joint_selection_weights(arm_joint_names, selected, device=device, dtype=dtype)


def teacher_ik_state_plan(
    *,
    mapped_indices      : torch.Tensor,  # Param: tensor input carrying mapped indices values
    num_arm             : int,  # Param: number of arm action dimensions in the active layout
    body_names          : Sequence[str],  # Param: ordered candidate names used to resolve body
    arm_joint_names     : Sequence[str],  # Param: ordered candidate names used to resolve arm joint
    body_link_pose_w    : torch.Tensor,  # Param: tensor input carrying body link pose w values
    fixed_base          : bool,  # Param: boolean input controlling fixed base
    palm_link           : str,  # Param: string input for palm link
    thumb_link          : str,  # Param: string input for thumb link
    index_link          : str,  # Param: string input for index link
    contact_thumb_link  : str,  # Param: string input for contact thumb link
    contact_index_link  : str,  # Param: string input for contact index link
    contact_middle_link : str,  # Param: string input for contact middle link
    align_joint_spec    : str | None,  # Param: string input for align joint spec
    align_joint_default : str,  # Param: string input for align joint default
    planar_joint_spec   : str | None,  # Param: string input for planar joint spec
    planar_joint_default: str,  # Param: string input for planar joint default
    device              : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype               : torch.dtype = torch.float32,  # Param: torch dtype used when converting or allocating tensors
) -> TeacherIkStatePlan:
    """Build the pure teacher IK state plan without Isaac controllers

    Steps:
    - Resolve inputs for `teacher_ik_state_plan` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    body_indices = teacher_ik_body_indices(
        body_names,
        palm_link=palm_link,
        thumb_link=thumb_link,
        index_link=index_link,
        contact_thumb_link=contact_thumb_link,
        contact_index_link=contact_index_link,
        contact_middle_link=contact_middle_link,
        fixed_base=fixed_base,
    )
    arm_joint_ids = tuple(int(value) for value in mapped_indices[: int(num_arm)].tolist())
    default_ee_pos = body_link_pose_w[0, body_indices.palm_body_idx, :3].detach().clone()
    default_ee_quat = body_link_pose_w[0, body_indices.palm_body_idx, 3:7].detach().clone()
    thumb_default_pos = body_link_pose_w[0, body_indices.thumb_body_idx, :3].detach().clone()
    index_default_pos = body_link_pose_w[0, body_indices.index_body_idx, :3].detach().clone()
    thumb_local_offset, index_local_offset = palm_local_offsets(
        palm_pos=default_ee_pos,
        palm_quat=default_ee_quat,
        thumb_pos=thumb_default_pos,
        index_pos=index_default_pos,
    )
    align_weights = teacher_ik_joint_weights(
        arm_joint_names,
        raw_spec=align_joint_spec,
        default_spec=align_joint_default,
        device=device,
        dtype=dtype,
    )
    planar_weights = teacher_ik_joint_weights(
        arm_joint_names,
        raw_spec=planar_joint_spec,
        default_spec=planar_joint_default,
        device=device,
        dtype=dtype,
    )
    return TeacherIkStatePlan(
        arm_joint_ids=arm_joint_ids,
        body_indices=body_indices,
        target_quat=default_ee_quat,
        thumb_local_offset=thumb_local_offset.to(device=device, dtype=dtype),
        index_local_offset=index_local_offset.to(device=device, dtype=dtype),
        align_servo_joint_weights=align_weights,
        planar_align_joint_weights=planar_weights,
    )
