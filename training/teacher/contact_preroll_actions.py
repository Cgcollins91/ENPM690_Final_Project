"""

Contact pre-roll action assembly helpers

File map:

open_hand_action_from_arm:      Append zero finger commands to an arm action
zero_preroll_finger_columns:    Zero finger columns in a contact pre-roll action
select_contact_preroll_action:  Select contact action rows only during touch phase
"""

from __future__ import annotations

import torch


def open_hand_action_from_arm(
    arm_action: torch.Tensor,  # Param: tensor input carrying arm action values
    *,
    num_fingers: int,          # Param: number of finger action dimensions in the active layout
) -> torch.Tensor:
    """Append zero finger commands to an arm action"""
    finger_action = torch.zeros(
        (arm_action.shape[0], int(num_fingers)),
        device=arm_action.device,
        dtype=arm_action.dtype,
    )
    return torch.cat([arm_action, finger_action], dim=-1).clamp(-1.0, 1.0)


def zero_preroll_finger_columns(
    action: torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    *,
    num_arm    : int,  # Param: number of arm action dimensions in the active layout
    num_fingers: int,  # Param: number of finger action dimensions in the active layout
) -> torch.Tensor:
    """Zero finger columns in a contact pre-roll action"""
    if action.shape[-1] < int(num_arm) + int(num_fingers):
        return action
    out = action.clone()
    out[..., int(num_arm) : int(num_arm) + int(num_fingers)] = 0.0
    return out.clamp(-1.0, 1.0)


def select_contact_preroll_action(
    *,
    open_action   : torch.Tensor,  # Param: tensor input carrying open action values
    contact_action: torch.Tensor,  # Param: tensor input carrying contact action values
    touch_phase   : torch.Tensor,  # Param: tensor input carrying touch phase values
) ->  torch.Tensor:
    """
    Select contact action rows only during touch phase
    """

    mask = touch_phase.to(device=open_action.device, dtype=torch.bool).unsqueeze(-1)
    return torch.where(mask, contact_action.to(device=open_action.device, dtype=open_action.dtype), open_action).clamp(
        -1.0,
        1.0,
    )
