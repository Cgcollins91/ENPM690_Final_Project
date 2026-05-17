"""

Loss helpers that do not depend on Isaac runtime state


resolved_bc_component_weights:  Resolve arm and finger BC weights
weighted_bc_loss:               Return weighted BC loss plus unweighted arm and finger diagnostics
"""

from __future__ import annotations

import torch


def resolved_bc_component_weights(base_weight: float, arm_weight: float, finger_weight: float) -> tuple[float, float]:
    """Resolve arm and finger BC weights"""
    base = max(0.0, float(base_weight))
    arm = base if float(arm_weight) < 0.0 else max(0.0, float(arm_weight))
    finger = base if float(finger_weight) < 0.0 else max(0.0, float(finger_weight))
    return arm, finger


def weighted_bc_loss(
    actor_bc     : torch.Tensor,  # Param: tensor input carrying actor bc values
    target_bc    : torch.Tensor,  # Param: tensor input carrying target bc values
    base_weight  : float,  # Param: weight applied to base
    arm_weight   : float,  # Param: weight applied to arm
    finger_weight: float,  # Param: weight applied to finger
    *,
    num_arm    : int,  # Param: number of arm action dimensions in the active layout
    num_fingers: int,  # Param: number of finger action dimensions in the active layout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, float, float]:
    """Return weighted BC loss plus unweighted arm and finger diagnostics

    Steps:
    - Resolve inputs for `weighted_bc_loss` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    diff_sq = (actor_bc - target_bc).pow(2)
    base = max(0.0, float(base_weight))
    arm_w, finger_w = resolved_bc_component_weights(base, arm_weight, finger_weight)
    weights = torch.full_like(diff_sq, base)
    if diff_sq.shape[-1] >= num_arm:
        weights[:, :num_arm] = arm_w
    if diff_sq.shape[-1] >= num_arm + num_fingers:
        weights[:, num_arm : num_arm + num_fingers] = finger_w
        arm_loss = diff_sq[:, :num_arm].mean()
        finger_loss = diff_sq[:, num_arm : num_arm + num_fingers].mean()
    elif diff_sq.shape[-1] == num_fingers:
        weights[:] = finger_w
        arm_loss = None
        finger_loss = diff_sq.mean()
    else:
        arm_loss = diff_sq[:, :num_arm].mean() if diff_sq.shape[-1] >= num_arm else None
        finger_loss = None
    return (diff_sq * weights).mean(), diff_sq.mean(), arm_loss, finger_loss, arm_w, finger_w
