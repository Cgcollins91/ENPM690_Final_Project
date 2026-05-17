"""

Native rollout action selection and teacher mixing

File map:

NativeActionDiagnostics:        Action diagnostics for one native rollout step
NativeActionSelection:          Policy teacher replay and BC actions for one rollout step
NativeActionMixConfig:          Teacher assist and BC relabel settings
_component_mix:                 Handle component mix logic
_apply_assist_noise:            Handle apply assist noise logic
phase1_open_preroll_action:     Build open-hand Phase 1 pre-roll action from frozen policy teacher
phase1_contact_preroll_action:  Select open Phase 1 or contact action by touch-phase rows
_teacher_required:              Handle teacher required logic
_teacher_or_none:               Handle teacher or none logic
select_native_rollout_action:   Select native rollout action and training labels
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch

from ..actions.action_mix import (
    PrerollActionOverride,
    actor_teacher_mse,
    assist_action_requested,
    apply_policy_arm_teacher_override,
    apply_preroll_action_override,
    bc_relabel_requested,
    mix_policy_teacher_actions,
)
from ..teacher.contact_preroll_actions import open_hand_action_from_arm, select_contact_preroll_action
from ..eval.eval_actions import ActionProcessor, apply_action_processors


PolicyActionFn = Callable[[torch.Tensor], torch.Tensor]
TeacherActionFn = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class NativeActionDiagnostics:
    """Action diagnostics for one native rollout step"""

    actor_teacher_arm_mse   : float = math.nan  # Field: floating-point actor teacher arm mse value used by native action diagnostics
    actor_teacher_finger_mse: float = math.nan  # Field: floating-point actor teacher finger mse value used by native action diagnostics
    teacher_available       : bool  = False  # Field: whether teacher action data is available for this row
    bc_relabel_applied      : bool  = False  # Field: boolean value indicating the bc relabel applied state for native action diagnostics
    preroll_all             : bool  = False  # Field: boolean value indicating the preroll all state for native action diagnostics
    teacher_warmup          : bool  = False  # Field: whether this step used the teacher-only warmup path
    assist_mix              : float = 0.0  # Field: effective whole-action assist mix used for this step
    assist_arm_mix          : float = 0.0  # Field: effective arm assist mix used for this step
    assist_finger_mix       : float = 0.0  # Field: effective finger assist mix used for this step


@dataclass(frozen=True)
class NativeActionSelection:
    """Policy teacher replay and BC actions for one rollout step"""

    policy_action : torch.Tensor  # Field: raw policy action before teacher or gate overrides
    mixed_action  : torch.Tensor  # Field: tensor containing mixed action values for batched env rows
    replay_action : torch.Tensor  # Field: tensor containing replay action values for batched env rows
    bc_action     : torch.Tensor  # Field: behavior-cloning target action tensor
    teacher_action: torch.Tensor | None  # Field: teacher action tensor used for override or behavior cloning
    diagnostics   : NativeActionDiagnostics  # Field: structured diagnostic values captured with the result


@dataclass(frozen=True)
class NativeActionMixConfig:
    """Teacher assist and BC relabel settings"""

    assist_mix              : float = 0.0  # Field: floating-point assist mix value used by native action mix config
    assist_arm_mix          : float = 0.0  # Field: floating-point assist arm mix value used by native action mix config
    assist_finger_mix       : float = 0.0  # Field: floating-point assist finger mix value used by native action mix config
    global_step             : int   = 0  # Field: global transition step used for rollout schedules
    start_steps             : int   = 0  # Field: teacher-only warmup step count
    policy_bc_relabel       : bool  = False  # Field: boolean value indicating the policy bc relabel state for native action mix config
    bc_only_steps           : int   = 0  # Field: step count used for bc only steps scheduling or reporting
    teacher_bc_weight       : float = 0.0  # Field: weight applied to teacher bc terms
    teacher_bc_arm_weight   : float = -1.0  # Field: weight applied to teacher bc arm terms
    teacher_bc_finger_weight: float = -1.0  # Field: weight applied to teacher bc finger terms
    assist_noise_arm        : float = 0.0  # Field: noise sigma applied to assisted arm actions
    assist_noise_finger     : float = 0.0  # Field: noise sigma applied to assisted finger actions
    assist_noise_clean_bc_target: bool = True  # Field: keep BC target equal to the clean teacher action when assist noise is applied
    use_policy_arm_teacher  : bool  = False  # Field: boolean value indicating the use policy arm teacher state for native action mix config
    soft_policy_arm_assist  : bool  = False  # Field: boolean value indicating the soft policy arm assist state for native action mix config


def _component_mix(value: float, fallback: float) -> float:
    return float(fallback) if float(value) < 0.0 else float(value)


def _apply_assist_noise(
    action: torch.Tensor,
    *,
    num_arm: int,
    num_fingers: int,
    arm_sigma: float,
    finger_sigma: float,
) -> torch.Tensor:
    """Process for `_apply_assist_noise`

    Steps:
    - Resolve inputs for `_apply_assist_noise` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if float(arm_sigma) <= 0.0 and float(finger_sigma) <= 0.0:
        return action
    noisy = action.clone()
    width = noisy.shape[-1]
    arm_end = min(int(num_arm), width)
    finger_end = min(int(num_arm) + int(num_fingers), width)
    if float(arm_sigma) > 0.0 and arm_end > 0:
        noisy[:, :arm_end] += torch.randn_like(noisy[:, :arm_end]) * float(arm_sigma)
    if float(finger_sigma) > 0.0 and finger_end > arm_end:
        noisy[:, arm_end:finger_end] += (
            torch.randn_like(noisy[:, arm_end:finger_end]) * float(finger_sigma)
        )
    return noisy.clamp(-1.0, 1.0)


def phase1_open_preroll_action(
    phase1_policy_teacher,     # Param: input value used as phase1 policy teacher
    obs_tensor: torch.Tensor,  # Param: policy observation tensor used by actor or replay logic
    *,
    num_fingers: int,          # Param: number of finger action dimensions in the active layout
) -> torch.Tensor:
    """Build open-hand Phase 1 pre-roll action from frozen policy teacher"""
    return open_hand_action_from_arm(
        phase1_policy_teacher.arm_action(obs_tensor),
        num_fingers=num_fingers,
    )


def phase1_contact_preroll_action(
    phase1_policy_teacher,         # Param: input value used as phase1 policy teacher
    obs_tensor: torch.Tensor,      # Param: policy observation tensor used by actor or replay logic
    *,
    contact_action: torch.Tensor,  # Param: tensor input carrying contact action values
    touch_phase   : torch.Tensor,  # Param: tensor input carrying touch phase values
    num_fingers   : int,  # Param: number of finger action dimensions in the active layout
) -> torch.Tensor:
    """Select open Phase 1 or contact action by touch-phase rows"""
    return select_contact_preroll_action(
        open_action=phase1_open_preroll_action(
            phase1_policy_teacher,
            obs_tensor,
            num_fingers=num_fingers,
        ),
        contact_action=contact_action,
        touch_phase=touch_phase,
    )


def _teacher_required(mix_config: NativeActionMixConfig) -> bool:
    """Process for `_teacher_required`

    Steps:
    - Resolve inputs for `_teacher_required` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    assist_mix = float(mix_config.assist_mix)
    assist_arm_mix = _component_mix(mix_config.assist_arm_mix, assist_mix)
    assist_finger_mix = _component_mix(mix_config.assist_finger_mix, assist_mix)
    teacher_warmup = int(mix_config.global_step) < int(mix_config.start_steps)
    relabel = bc_relabel_requested(
        policy_bc_relabel=mix_config.policy_bc_relabel,
        bc_only_steps=mix_config.bc_only_steps,
        teacher_bc_weight=mix_config.teacher_bc_weight,
        teacher_bc_arm_weight=mix_config.teacher_bc_arm_weight,
        teacher_bc_finger_weight=mix_config.teacher_bc_finger_weight,
    )
    return bool(
        teacher_warmup
        or relabel
        or assist_action_requested(assist_mix, assist_arm_mix, assist_finger_mix)
    )


def _teacher_or_none(
    teacher_action_fn: TeacherActionFn | None,
    *,
    required: bool,
) -> torch.Tensor | None:
    if teacher_action_fn is None:
        if bool(required):
            raise RuntimeError(
                "teacher assist/BC was requested, but no native teacher_action_fn is installed"
            )
        return None
    try:
        return teacher_action_fn().clamp(-1.0, 1.0)
    except Exception as exc:
        if bool(required):
            raise RuntimeError("native teacher action failed while teacher assist was required") from exc
        return None


def select_native_rollout_action(
    *,
    obs_tensor        : torch.Tensor,  # Param: policy observation tensor used by actor or replay logic
    policy_action_fn  : PolicyActionFn,  # Param: callback used to compute or fetch policy action
    policy_processors : Sequence[ActionProcessor] = (),  # Param: ordered input collection of policy processors entries
    teacher_action_fn : TeacherActionFn | None    = None,  # Param: callback used to compute or fetch teacher action
    teacher_processors: Sequence[ActionProcessor] = (),  # Param: ordered input collection of teacher processors entries
    mix_config        : NativeActionMixConfig     = NativeActionMixConfig(),  # Param: input value used as mix config
    num_arm           : int,  # Param: number of arm action dimensions in the active layout
    num_fingers       : int,  # Param: number of finger action dimensions in the active layout
    preroll_action    : torch.Tensor | None       = None,  # Param: tensor input carrying preroll action values
    preroll_mask      : torch.Tensor | None       = None,  # Param: boolean mask selecting preroll rows
) -> NativeActionSelection:
    """Select native rollout action and training labels

    Steps:
    - Resolve inputs for `select_native_rollout_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    policy_action = policy_action_fn(obs_tensor).clamp(-1.0, 1.0)
    policy_action = apply_action_processors(policy_action, policy_processors)
    teacher_action = _teacher_or_none(teacher_action_fn, required=_teacher_required(mix_config))
    if teacher_action is not None:
        teacher_action = apply_action_processors(teacher_action, teacher_processors)

    assist_mix = float(mix_config.assist_mix)
    assist_arm_mix = _component_mix(mix_config.assist_arm_mix, assist_mix)
    assist_finger_mix = _component_mix(mix_config.assist_finger_mix, assist_mix)
    teacher_warmup = teacher_action is not None and int(mix_config.global_step) < int(mix_config.start_steps)
    if teacher_action is not None:
        if teacher_warmup:
            mixed = teacher_action.clone()
        else:
            mixed = mix_policy_teacher_actions(
                policy_action,
                teacher_action,
                assist_mix=assist_mix,
                assist_arm_mix=assist_arm_mix,
                assist_finger_mix=assist_finger_mix,
                num_arm=num_arm,
                num_fingers=num_fingers,
            ).clamp(-1.0, 1.0)
            mixed = apply_policy_arm_teacher_override(
                mixed,
                teacher_action,
                num_arm=num_arm,
                use_policy_arm_teacher=mix_config.use_policy_arm_teacher,
                soft_policy_arm_assist=mix_config.soft_policy_arm_assist,
            )
    else:
        mixed = policy_action

    relabel = teacher_action is not None and bc_relabel_requested(
        policy_bc_relabel=mix_config.policy_bc_relabel,
        bc_only_steps=mix_config.bc_only_steps,
        teacher_bc_weight=mix_config.teacher_bc_weight,
        teacher_bc_arm_weight=mix_config.teacher_bc_arm_weight,
        teacher_bc_finger_weight=mix_config.teacher_bc_finger_weight,
    )
    assisted = teacher_action is not None and assist_action_requested(
        assist_mix,
        assist_arm_mix,
        assist_finger_mix,
    )
    teacher_bc_target = teacher_action is not None and (teacher_warmup or relabel or assisted)
    bc_action = teacher_action.clone() if teacher_bc_target else policy_action.clone()
    replay_action = mixed.clone()
    if teacher_action is not None and (teacher_warmup or assisted):
        noised = _apply_assist_noise(
            mixed,
            num_arm=num_arm,
            num_fingers=num_fingers,
            arm_sigma=mix_config.assist_noise_arm,
            finger_sigma=mix_config.assist_noise_finger,
        )
        if noised is not mixed:
            mixed = noised
            replay_action = noised.clone()
            if not mix_config.assist_noise_clean_bc_target:
                bc_action = noised.clone()
    preroll_all = False
    if preroll_action is not None and preroll_mask is not None:
        override = apply_preroll_action_override(
            policy_level_action=mixed,
            replay_action=replay_action,
            bc_action=bc_action,
            teacher_action=teacher_action,
            preroll_action=preroll_action,
            preroll_mask=preroll_mask,
        )
        mixed = override.policy_level_action
        replay_action = override.replay_action
        bc_action = override.bc_action
        teacher_action = override.teacher_action
        preroll_all = override.all_preroll

    arm_mse, finger_mse = (
        actor_teacher_mse(policy_action, teacher_action, num_arm=num_arm, num_fingers=num_fingers)
        if teacher_action is not None
        else (math.nan, math.nan)
    )
    return NativeActionSelection(
        policy_action=policy_action,
        mixed_action=mixed.clamp(-1.0, 1.0),
        replay_action=replay_action.clamp(-1.0, 1.0),
        bc_action=bc_action.clamp(-1.0, 1.0),
        teacher_action=teacher_action,
        diagnostics=NativeActionDiagnostics(
            actor_teacher_arm_mse=arm_mse,
            actor_teacher_finger_mse=finger_mse,
            teacher_available=teacher_action is not None,
            bc_relabel_applied=bool(relabel),
            preroll_all=bool(preroll_all),
            teacher_warmup=bool(teacher_warmup),
            assist_mix=assist_mix,
            assist_arm_mix=assist_arm_mix,
            assist_finger_mix=assist_finger_mix,
        ),
    )
