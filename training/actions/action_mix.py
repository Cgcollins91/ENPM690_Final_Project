"""

Policy teacher action mixing helpers


PrerollActionOverride:              Action tensors after applying preroll rows
assist_action_requested:            Return whether any assist mix requests teacher blending
bc_relabel_requested:               Return whether teacher labels should replace policy BC labels
mix_policy_teacher_actions:         Blend policy and teacher actions with component overrides
apply_policy_arm_teacher_override:  Hard-copy teacher arm columns when soft assist is disabled
add_warmup_teacher_noise:           Add warmup teacher noise with optional arm preservation
apply_preroll_action_override:      Overwrite preroll rows across rollout replay BC and teacher actions
actor_teacher_mse:                  Return arm and finger MSE diagnostics between actor and teacher
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PrerollActionOverride:
    """Action tensors after applying preroll rows"""

    policy_level_action: torch.Tensor  # tensor containing policy level action values for batched env rows
    replay_action      : torch.Tensor  # tensor containing replay action values for batched env rows
    bc_action          : torch.Tensor  # behavior-cloning target action tensor
    teacher_action     : torch.Tensor | None  # teacher action tensor used for override or behavior cloning
    all_preroll        : bool  # boolean value indicating the all preroll state for preroll action override


def assist_action_requested(*mixes: float) -> bool:
    """Return whether any assist mix requests teacher blending"""
    return max((float(value) for value in mixes), default=0.0) > 0.0


def bc_relabel_requested(
    *,
    policy_bc_relabel       : bool,  # Param: boolean input controlling policy bc relabel
    bc_only_steps           : int,  # Param: step count used for bc only steps
    teacher_bc_weight       : float,  # Param: weight applied to teacher bc
    teacher_bc_arm_weight   : float,  # Param: weight applied to teacher bc arm
    teacher_bc_finger_weight: float,  # Param: weight applied to teacher bc finger
) -> bool:
    """Return whether teacher labels should replace policy BC labels"""
    return bool(policy_bc_relabel) and (
        int(bc_only_steps) > 0
        or float(teacher_bc_weight) > 0.0
        or float(teacher_bc_arm_weight) >= 0.0
        or float(teacher_bc_finger_weight) >= 0.0
    )


def mix_policy_teacher_actions(
    policy_action : torch.Tensor,  # Param: raw actor action before teacher, gate, or smoothing overrides
    teacher_action: torch.Tensor,  # Param: teacher action used for override or behavior-cloning targets
    *,
    assist_mix       : float,  # Param: floating-point input for assist mix
    assist_arm_mix   : float,  # Param: floating-point input for assist arm mix
    assist_finger_mix: float,  # Param: floating-point input for assist finger mix
    num_arm          : int,  # Param: number of arm action dimensions in the active layout
    num_fingers      : int,  # Param: number of finger action dimensions in the active layout
) -> torch.Tensor:
    """Blend policy and teacher actions with component overrides

    Steps:
    - Resolve inputs for `mix_policy_teacher_actions` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    base_mix = float(assist_mix)
    mixed = (1.0 - base_mix) * policy_action + base_mix * teacher_action
    if policy_action.shape[-1] < int(num_arm) + int(num_fingers):
        return mixed
    mixed = mixed.clone()
    arm_end = int(num_arm)
    finger_end = int(num_arm) + int(num_fingers)
    arm_mix = float(assist_arm_mix)
    finger_mix = float(assist_finger_mix)
    mixed[:, :arm_end] = (1.0 - arm_mix) * policy_action[:, :arm_end] + arm_mix * teacher_action[:, :arm_end]
    mixed[:, arm_end:finger_end] = (
        (1.0 - finger_mix) * policy_action[:, arm_end:finger_end]
        + finger_mix * teacher_action[:, arm_end:finger_end]
    )
    return mixed


def apply_policy_arm_teacher_override(
    action        : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    teacher_action: torch.Tensor,  # Param: teacher action used for override or behavior-cloning targets
    *,
    num_arm               : int,  # Param: number of arm action dimensions in the active layout
    use_policy_arm_teacher: bool,  # Param: boolean input selecting whether policy arm teacher is used
    soft_policy_arm_assist: bool,  # Param: boolean input controlling soft policy arm assist
) -> torch.Tensor:
    """Hard-copy teacher arm columns when soft assist is disabled"""
    if not use_policy_arm_teacher or soft_policy_arm_assist or action.shape[-1] < int(num_arm):
        return action
    out = action.clone()
    out[:, : int(num_arm)] = teacher_action[:, : int(num_arm)]
    return out


def add_warmup_teacher_noise(
    teacher_action: torch.Tensor,  # Param: teacher action used for override or behavior-cloning targets
    *,
    sigma       : float,  # Param: floating-point input for sigma
    num_arm     : int,  # Param: number of arm action dimensions in the active layout
    preserve_arm: bool,  # Param: boolean input controlling preserve arm
) -> torch.Tensor:
    """Add warmup teacher noise with optional arm preservation"""
    if float(sigma) <= 0.0:
        return teacher_action
    noise = torch.randn_like(teacher_action) * float(sigma)
    if preserve_arm and teacher_action.shape[-1] >= int(num_arm):
        noise[:, : int(num_arm)] = 0.0
    return teacher_action + noise


def apply_preroll_action_override(
    *,
    policy_level_action: torch.Tensor,  # Param: tensor input carrying policy level action values
    replay_action      : torch.Tensor,  # Param: tensor input carrying replay action values
    bc_action          : torch.Tensor,  # Param: behavior-cloning target action
    teacher_action     : torch.Tensor | None,  # Param: teacher action used for override or behavior-cloning targets
    preroll_action     : torch.Tensor,  # Param: tensor input carrying preroll action values
    preroll_mask       : torch.Tensor,  # Param: boolean mask selecting preroll rows
) -> PrerollActionOverride:
    """Overwrite preroll rows across rollout replay BC and teacher actions

    Steps:
    - Resolve inputs for `apply_preroll_action_override` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    mask = preroll_mask.to(device=policy_level_action.device, dtype=torch.bool)
    policy_out = policy_level_action.clone()
    replay_out = replay_action.clone()
    bc_out = bc_action.clone()
    policy_out[mask] = preroll_action[mask]
    replay_out[mask] = preroll_action[mask]
    bc_out[mask] = preroll_action[mask]
    teacher_out = None
    if teacher_action is not None:
        teacher_out = teacher_action.clone()
        teacher_out[mask] = preroll_action[mask]
    return PrerollActionOverride(
        policy_level_action=policy_out,
        replay_action=replay_out,
        bc_action=bc_out,
        teacher_action=teacher_out,
        all_preroll=bool(mask.all().item()),
    )


def actor_teacher_mse(
    actor_action  : torch.Tensor,  # Param: tensor input carrying actor action values
    teacher_action: torch.Tensor,  # Param: teacher action used for override or behavior-cloning targets
    *,
    num_arm    : int,  # Param: number of arm action dimensions in the active layout
    num_fingers: int,  # Param: number of finger action dimensions in the active layout
    mask       : torch.Tensor | None = None,  # Param: boolean mask selecting mask rows
) -> tuple[float, float]:
    """Return arm and finger MSE diagnostics between actor and teacher

    Steps:
    - Resolve inputs for `actor_teacher_mse` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if mask is not None:
        selected = mask.to(device=actor_action.device, dtype=torch.bool)
        if not bool(selected.any().item()):
            return math.nan, math.nan
        actor_action = actor_action[selected]
        teacher_action = teacher_action[selected]
    if actor_action.numel() == 0 or teacher_action.numel() == 0:
        return math.nan, math.nan
    if actor_action.shape[-1] >= int(num_arm) + int(num_fingers):
        arm_end = int(num_arm)
        finger_end = int(num_arm) + int(num_fingers)
        arm_mse = torch.mean((actor_action[:, :arm_end] - teacher_action[:, :arm_end]) ** 2)
        finger_mse = torch.mean(
            (actor_action[:, arm_end:finger_end] - teacher_action[:, arm_end:finger_end]) ** 2
        )
        return float(arm_mse.item()), float(finger_mse.item())
    finger_mse = torch.mean((actor_action - teacher_action) ** 2)
    return math.nan, float(finger_mse.item())
