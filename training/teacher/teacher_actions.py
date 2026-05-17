"""

Teacher reduced-action assembly helpers

File map:

TopdownContactTeacherParts:                Closed-loop topdown contact teacher action parts
topdown_contact_teacher_parts_from_tuple:  Convert legacy contact-teacher tuple to named parts
teacher_finger_action:                     Return teacher finger action for contact or scheduled teacher
apply_stage2_teacher_gate:                 Gate closure and descent on curriculum stage 2
assemble_teacher_reduced_action:           Concatenate and clamp teacher arm and finger actions
zero_finger_action_like:                   Return zero finger action matching arm batch
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopdownContactTeacherParts:
    """Closed-loop topdown contact teacher action parts"""

    finger_action   : torch.Tensor  # Field: tensor containing finger action values for batched env rows
    closure_fraction: torch.Tensor  # Field: normalized finger-closure progress fraction
    descent         : torch.Tensor  # Field: tensor containing descent values for batched env rows
    xy_offset       : torch.Tensor  # Field: tensor containing xy offset values for batched env rows
    inward          : torch.Tensor  # Field: tensor containing inward values for batched env rows
    tip_servo       : torch.Tensor  # Field: tensor containing tip servo values for batched env rows


def topdown_contact_teacher_parts_from_tuple(
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  # Param: tensor input carrying values values
) -> TopdownContactTeacherParts:
    """Convert legacy contact-teacher tuple to named parts"""
    finger_action, closure_fraction, descent, xy_offset, inward, tip_servo = values
    return TopdownContactTeacherParts(
        finger_action=finger_action,
        closure_fraction=closure_fraction,
        descent=descent,
        xy_offset=xy_offset,
        inward=inward,
        tip_servo=tip_servo,
    )


def teacher_finger_action(
    *,
    topdown_contact_teacher_enabled: bool,  # Param: boolean input enabling topdown contact teacher
    contact_parts                  : TopdownContactTeacherParts | None,  # Param: input value used as contact parts
    closure_fraction               : torch.Tensor | None,  # Param: tensor input carrying closure fraction values
    compute_in_current_mode        : Callable[[torch.Tensor], torch.Tensor],  # Param: callback used to compute or fetch compute in current mode
) -> torch.Tensor:
    """Return teacher finger action for contact or scheduled teacher"""
    if bool(topdown_contact_teacher_enabled):
        if contact_parts is None:
            raise ValueError("contact_parts are required when contact teacher is enabled")
        return contact_parts.finger_action
    if closure_fraction is None:
        raise ValueError("closure_fraction is required when contact teacher is disabled")
    return compute_in_current_mode(closure_fraction)


def apply_stage2_teacher_gate(
    *,
    closure_fraction       : torch.Tensor,  # Param: tensor input carrying closure fraction values
    descent                : float | torch.Tensor | None,  # Param: tensor input carrying descent values
    stage                  : torch.Tensor | None,  # Param: tensor input carrying stage values
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gate closure and descent on curriculum stage 2

    Steps:
    - Resolve inputs for `apply_stage2_teacher_gate` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not bool(topdown_curriculum_task) or stage is None:
        return closure_fraction, descent if torch.is_tensor(descent) else (
            None if descent is None else torch.full_like(closure_fraction, float(descent))
        )
    mask = (stage.to(device=closure_fraction.device) >= 2).to(dtype=closure_fraction.dtype)
    gated_closure = closure_fraction * mask
    if descent is None:
        return gated_closure, None
    if torch.is_tensor(descent):
        descent_tensor = descent.to(device=closure_fraction.device, dtype=closure_fraction.dtype).reshape(-1)
    else:
        descent_tensor = torch.full_like(closure_fraction, float(descent))
    return gated_closure, descent_tensor * mask


def assemble_teacher_reduced_action(
    *,
    arm_action   : torch.Tensor,  # Param: tensor input carrying arm action values
    finger_action: torch.Tensor,  # Param: tensor input carrying finger action values
) -> torch.Tensor:
    """Concatenate and clamp teacher arm and finger actions"""
    return torch.cat([arm_action, finger_action], dim=-1).clamp(-1.0, 1.0)


def zero_finger_action_like(
    arm_action: torch.Tensor,  # Param: tensor input carrying arm action values
    *,
    num_fingers: int,          # Param: number of finger action dimensions in the active layout
) -> torch.Tensor:
    """Return zero finger action matching arm batch"""
    return torch.zeros(
        (arm_action.shape[0], int(num_fingers)),
        device=arm_action.device,
        dtype=arm_action.dtype,
    )
