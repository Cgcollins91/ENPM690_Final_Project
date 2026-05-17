"""

Frozen policy-arm teacher handoff helpers


policy_arm_pre_latch_mask:         Return rows where frozen policy should still own arm
policy_arm_ik_tight_contact_gate:  Return near-contact gate for policy-arm IK residuals
one_contact_only_mix_mask:         Apply one-contact-only gate to arm IK mix rows
mix_policy_arm_into_teacher:       Mix frozen policy arm and IK teacher arm columns
"""

from __future__ import annotations

import torch


def policy_arm_pre_latch_mask(
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    latched : torch.Tensor | None = None,  # Param: tensor input carrying latched values
) -> torch.Tensor:
    """Return rows where frozen policy should still own arm"""
    if latched is None or latched.shape[0] != int(num_envs):
        return torch.ones(int(num_envs), dtype=torch.bool, device=device)
    return ~latched.to(device=device, dtype=torch.bool)


def policy_arm_ik_tight_contact_gate(
    *,
    num_envs   : int,  # Param: number of parallel environment rows represented
    device     : torch.device | str,  # Param: torch device where tensors are read or allocated
    enabled    : bool,  # Param: boolean input controlling enabled
    palm_dist  : torch.Tensor | None = None,  # Param: tensor input carrying palm dist values
    height_err : torch.Tensor | None = None,  # Param: tensor input carrying height err values
    align_err  : torch.Tensor | None = None,  # Param: tensor input carrying align err values
    opposed    : torch.Tensor | None = None,  # Param: tensor input carrying opposed values
    palm_max   : float               = -1.0,  # Param: floating-point input for palm max
    height_max : float               = -1.0,  # Param: floating-point input for height max
    align_max  : float               = -1.0,  # Param: floating-point input for align max
    opposed_min: float               = -1.0,  # Param: floating-point input for opposed min
) -> torch.Tensor:
    """Return near-contact gate for policy-arm IK residuals

    Steps:
    - Resolve inputs for `policy_arm_ik_tight_contact_gate` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    gate = torch.ones(int(num_envs), dtype=torch.bool, device=device)
    if not bool(enabled):
        return gate
    if float(palm_max) >= 0.0 and palm_dist is not None:
        gate &= palm_dist.to(device=device).detach() <= float(palm_max)
    if float(height_max) >= 0.0 and height_err is not None:
        gate &= height_err.to(device=device).detach() <= float(height_max)
    if float(align_max) >= 0.0 and align_err is not None:
        gate &= align_err.to(device=device).detach() <= float(align_max)
    if float(opposed_min) >= 0.0 and opposed is not None:
        gate &= opposed.to(device=device).detach() >= float(opposed_min)
    return gate


def one_contact_only_mix_mask(
    *,
    base_mask     : torch.Tensor,  # Param: boolean mask selecting base rows
    thumb_strength: torch.Tensor,  # Param: tensor input carrying thumb strength values
    back_strength : torch.Tensor,  # Param: tensor input carrying back strength values
    threshold     : float,  # Param: cutoff used by the comparison or gate
    enabled       : bool,  # Param: boolean input controlling enabled
) -> torch.Tensor:
    """Apply one-contact-only gate to arm IK mix rows

    Steps:
    - Resolve inputs for `one_contact_only_mix_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    mask = base_mask.to(dtype=torch.bool)
    if not bool(enabled):
        return mask
    thumb_hit = thumb_strength.to(device=mask.device).detach() >= float(threshold)
    back_hit = back_strength.to(device=mask.device).detach() >= float(threshold)
    return mask & (thumb_hit ^ back_hit)


def mix_policy_arm_into_teacher(
    *,
    teacher_action   : torch.Tensor,  # Param: teacher action used for override or behavior-cloning targets
    policy_arm_action: torch.Tensor,  # Param: tensor input carrying policy arm action values
    pre_latch_mask   : torch.Tensor,  # Param: boolean mask selecting pre latch rows
    num_arm          : int,  # Param: number of arm action dimensions in the active layout
    arm_ik_mix       : float               = 0.0,  # Param: floating-point input for arm ik mix
    mix_mask         : torch.Tensor | None = None,  # Param: boolean mask selecting mix rows
) -> torch.Tensor:
    """Mix frozen policy arm and IK teacher arm columns

    Steps:
    - Resolve inputs for `mix_policy_arm_into_teacher` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    out = teacher_action.clone()
    arm_end = int(num_arm)
    pre_latch = pre_latch_mask.to(device=out.device, dtype=torch.bool)
    if float(arm_ik_mix) > 0.0:
        mix = pre_latch if mix_mask is None else pre_latch & mix_mask.to(device=out.device, dtype=torch.bool)
        alpha = max(0.0, min(1.0, float(arm_ik_mix)))
        out[mix, :arm_end] = ((1.0 - alpha) * policy_arm_action[mix] + alpha * out[mix, :arm_end]).clamp(-1.0, 1.0)
        policy_only = pre_latch & (~mix)
        out[policy_only, :arm_end] = policy_arm_action[policy_only]
    else:
        out[pre_latch, :arm_end] = policy_arm_action[pre_latch]
    return out.clamp(-1.0, 1.0)
