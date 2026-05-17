"""

Evaluation action selection and teacher-assist helpers

This module provides helper functions and data structures for selecting eval actions,
applying teacher assist, and computing diagnostics comparing policy and teacher actions,
used by the evaluation loop in the legacy monolith trainer.

File map:

EvalActionDiagnostics:          Teacher comparison diagnostics for one eval action
EvalActionResult:               Policy and mixed action tensors for eval
clamp_eval_teacher_assist_mix:  Clamp eval teacher assist mix to [0, 1]
apply_action_processors:        Apply action processors in order
eval_action_diagnostics:        Compute eval actor teacher action diagnostics
mix_eval_teacher_action:        Blend deterministic eval policy action with teacher action
select_eval_action:             Select and optionally teacher-assist one eval action
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch


ActionProcessor = Callable[[torch.Tensor], torch.Tensor]
TeacherActionFn = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class EvalActionDiagnostics:
    """Teacher comparison diagnostics for one eval action"""

    actor_teacher_arm_mse   : float = math.nan  # floating-point actor teacher arm mse value used by eval action diagnostics
    actor_teacher_finger_mse: float = math.nan  # floating-point actor teacher finger mse value used by eval action diagnostics
    teacher_action_l2       : float = math.nan  # floating-point teacher action l2 value used by eval action diagnostics
    teacher_available       : bool  = False     # whether teacher action data is available for this row


@dataclass(frozen=True)
class EvalActionResult:
    """Policy and mixed action tensors for eval"""

    policy_action : torch.Tensor  # raw policy action before teacher or gate overrides
    action        : torch.Tensor  # environment action tensor selected for the step
    teacher_action: torch.Tensor | None  # teacher action tensor used for override or behavior cloning
    diagnostics   : EvalActionDiagnostics  # structured diagnostic values captured with the result


def clamp_eval_teacher_assist_mix(value: float) -> float:
    """Clamp eval teacher assist mix to [0, 1]"""
    return max(0.0, min(1.0, float(value)))


def apply_action_processors(
    action    : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    processors: Sequence[ActionProcessor],  # Param: ordered input collection of processors entries
) -> torch.Tensor:
    """Apply action processors in order"""
    out = action
    for processor in processors:
        out = processor(out)
    return out


def eval_action_diagnostics(
    policy_action : torch.Tensor,  # Param: raw actor action before teacher, gate, or smoothing overrides
    teacher_action: torch.Tensor | None,  # Param: teacher action used for override or behavior-cloning targets
    *,
    num_arm    : int,  # Param: number of arm action dimensions in the active layout
    num_fingers: int,  # Param: number of finger action dimensions in the active layout
) -> EvalActionDiagnostics:
    """Compute eval actor teacher action diagnostics

    Steps:
    - Resolve inputs for `eval_action_diagnostics` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if teacher_action is None:
        return EvalActionDiagnostics()
    teacher = teacher_action.to(device=policy_action.device, dtype=policy_action.dtype)
    teacher_l2 = float(torch.linalg.norm(teacher[0]).item()) if teacher.numel() > 0 else math.nan
    action_width = int(policy_action.shape[-1])
    split_width = int(num_arm) + int(num_fingers)
    if action_width >= split_width:
        arm_mse = float(torch.mean((policy_action[:, : int(num_arm)] - teacher[:, : int(num_arm)]) ** 2).item())
        finger_mse = float(
            torch.mean(
                (
                    policy_action[:, int(num_arm) : split_width]
                    - teacher[:, int(num_arm) : split_width]
                )
                ** 2
            ).item()
        )
    else:
        arm_mse = math.nan
        finger_mse = float(torch.mean((policy_action - teacher) ** 2).item())
    return EvalActionDiagnostics(
        actor_teacher_arm_mse=arm_mse,
        actor_teacher_finger_mse=finger_mse,
        teacher_action_l2=teacher_l2,
        teacher_available=True,
    )


def mix_eval_teacher_action(
    policy_action : torch.Tensor,         # Param: raw actor action before teacher, gate, or smoothing overrides
    teacher_action: torch.Tensor | None,  # Param: teacher action used for override or behavior-cloning targets
    *,
    teacher_assist_mix: float,            # Param: floating-point input for teacher assist mix
) -> torch.Tensor:
    """Blend deterministic eval policy action with teacher action"""
    mix = clamp_eval_teacher_assist_mix(teacher_assist_mix)
    if mix <= 0.0 or teacher_action is None:
        return policy_action
    teacher = teacher_action.to(device=policy_action.device, dtype=policy_action.dtype)
    return (policy_action * (1.0 - mix) + teacher * mix).clamp(-1.0, 1.0)


def select_eval_action(
    *,
    obs_tensor             : torch.Tensor,                            # Param: policy observation tensor used by actor or replay logic
    select_policy_action_fn: Callable[[torch.Tensor], torch.Tensor],  # Param: callback used to compute or fetch select policy action
    policy_processors      : Sequence[ActionProcessor] = (),          # Param: ordered input collection of policy processors entries
    teacher_action_fn      : TeacherActionFn | None    = None,        # Param: callback used to compute or fetch teacher action
    teacher_processors     : Sequence[ActionProcessor] = (),          # Param: ordered input collection of teacher processors entries
    teacher_assist_mix     : float                     = 0.0,         # Param: floating-point input for teacher assist mix
    num_arm                : int                       = 0,           # Param: number of arm action dimensions in the active layout
    num_fingers            : int                       = 0,       # Param: number of finger action dimensions in the active layout
) -> EvalActionResult:

    """Select and optionally teacher-assist one eval action

    1. Compute raw policy action from the observation tensor using the provided callback, and clamp to [-1, 1]
    2. Apply policy action processors in order
    3. If a teacher action callback is provided, compute the teacher action, clamp to [-1, 1], and apply teacher action processors in order,
       with error handling to skip teacher
    4. Compute diagnostics comparing the policy and teacher actions
    5. Blend the policy and teacher actions using the provided teacher assist mix value, with clamping to [0, 1] and defaulting
       to the policy action if the teacher action is not available
    6. Return the final mixed action, raw policy action, teacher action, and diagnostics in a structured result

    Steps:
    - Resolve inputs for `select_eval_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    policy_action = select_policy_action_fn(obs_tensor).clamp(-1.0, 1.0)
    policy_action = apply_action_processors(policy_action, policy_processors)
    teacher_action = None
    teacher_required = clamp_eval_teacher_assist_mix(teacher_assist_mix) >= 1.0
    if teacher_action_fn is not None:
        try:
            teacher_action = teacher_action_fn().clamp(-1.0, 1.0)
            teacher_action = apply_action_processors(teacher_action, teacher_processors)
        except Exception as exc:
            if teacher_required:
                raise RuntimeError("eval teacher action failed while teacher assist mix is 1.0") from exc
            teacher_action = None
    elif teacher_required:
        raise RuntimeError("eval teacher assist mix is 1.0, but no teacher_action_fn is installed")
    diagnostics = eval_action_diagnostics(
        policy_action,
        teacher_action,
        num_arm=num_arm,
        num_fingers=num_fingers,
    )
    mixed_action = mix_eval_teacher_action(
        policy_action,
        teacher_action,
        teacher_assist_mix=teacher_assist_mix,
    )
    return EvalActionResult(
        policy_action=policy_action,
        action=mixed_action,
        teacher_action=teacher_action,
        diagnostics=diagnostics,
    )
