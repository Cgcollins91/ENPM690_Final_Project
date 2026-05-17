"""

Native Phase 1 warm-start and frozen-teacher startup

File map:

NativePhase1StartupResult:    Summary of native Phase 1 startup handling
policy_arm_joint_names:       Return policy action arm-joint prefix
policy_arm_action_scales:     Return policy action arm-scale prefix
_phase1_checkpoint:           Handle phase1 checkpoint logic
_actor_skip_reason:           Handle actor skip reason logic
apply_native_phase1_startup:  Apply Phase 1 warm-start and construct frozen policy teacher
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_components import NativeTrainingComponents
from ..io.warmstart import (
    ActorWarmStartConfig,
    make_frozen_policy_teacher_from_checkpoint,
    warm_start_actor_from_checkpoint,
    warm_start_obs_stats_from_checkpoint,
)


ActorWarmStartFn = Callable[..., Mapping[str, object]]
ObsStatsWarmStartFn = Callable[..., Mapping[str, object]]
PolicyTeacherFn = Callable[..., Any]


@dataclass(frozen=True)
class NativePhase1StartupResult:
    """Summary of native Phase 1 startup handling"""

    requested             : bool                                 # Field: boolean value indicating the requested state for native phase1 startup result
    skipped_ik            : bool                        = False  # Field: boolean value indicating the skipped ik state for native phase1 startup result
    actor_copy_skipped    : bool                        = False  # Field: boolean value indicating the actor copy skipped state for native phase1 startup result
    actor_copy_skip_reason: str | None                  = None   # Field: string actor copy skip reason value used by native phase1 startup result
    actor_copy_applied    : Mapping[str, object] | None = None   # Field: string actor copy applied value used by native phase1 startup result
    obs_stats_applied     : Mapping[str, object] | None = None   # Field: string obs stats applied value used by native phase1 startup result
    policy_teacher        : Any                         = None   # Field: stores policy teacher for native phase1 startup result


def policy_arm_joint_names(context: TrainerRuntimeContext, *, finger_count: int = 7) -> tuple[str, ...]:
    """Return policy action arm-joint prefix"""
    joints = tuple(context.action.policy_action_spec.joint_names)
    if len(joints) <= int(finger_count):
        return ()
    return joints[: -int(finger_count)]


def policy_arm_action_scales(context: TrainerRuntimeContext, *, finger_count: int = 7) -> tuple[float, ...]:
    """Return policy action arm-scale prefix"""
    scales = tuple(float(value) for value in context.action.policy_action_spec.scales)
    if len(scales) <= int(finger_count):
        return ()
    return scales[: -int(finger_count)]


def _phase1_checkpoint(components: NativeTrainingComponents) -> Mapping[str, object]:
    checkpoint = components.checkpoints.phase1
    if checkpoint is None:
        raise RuntimeError("phase1 checkpoint was requested but not loaded")
    return checkpoint


def _actor_skip_reason(components: NativeTrainingComponents) -> str | None:
    """Process for `_actor_skip_reason`

    Steps:
    - Resolve inputs for `_actor_skip_reason` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    plan = components.checkpoint_plan
    if plan.phase1_teacher_only:
        return "phase1_teacher_only"
    if plan.resume_requested:
        return "resume_checkpoint"
    if plan.actor_init_requested:
        return "actor_init_checkpoint"
    return None


def apply_native_phase1_startup(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    components: NativeTrainingComponents,  # Param: input value used as components
    *,
    actor_warm_start_fn    : ActorWarmStartFn    = warm_start_actor_from_checkpoint,  # Param: callback used to compute or fetch actor warm start
    obs_stats_warm_start_fn: ObsStatsWarmStartFn = warm_start_obs_stats_from_checkpoint,  # Param: callback used to compute or fetch obs stats warm start
    policy_teacher_fn      : PolicyTeacherFn     = make_frozen_policy_teacher_from_checkpoint,  # Param: callback used to compute or fetch policy teacher
) -> NativePhase1StartupResult:
    """Apply Phase 1 warm-start and construct frozen policy teacher

    Steps:
    - Resolve inputs for `apply_native_phase1_startup` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    plan = components.checkpoint_plan
    if not plan.phase1_requested:
        return NativePhase1StartupResult(requested=False)
    if plan.phase1_ik_skip:
        return NativePhase1StartupResult(requested=True, skipped_ik=True)

    checkpoint = _phase1_checkpoint(components)
    arm_joints = policy_arm_joint_names(context)
    arm_scales = policy_arm_action_scales(context)
    actor_copy: Mapping[str, object] | None = None
    obs_stats : Mapping[str, object] | None = None
    skip_reason = _actor_skip_reason(components)

    if plan.phase1_actor_copy_allowed:
        actor_copy = actor_warm_start_fn(
            components.agent.actor,
            config=ActorWarmStartConfig(
                arm_joint_names=arm_joints,
                expected_arm_scales=arm_scales,
            ),
            checkpoint=checkpoint,
        )
        components.agent.actor_target.load_state_dict(components.agent.actor.state_dict())

    if bool(context.args.get("observation_normalization", False)) and not (
        plan.resume_requested or plan.actor_init_requested
    ):
        obs_stats = obs_stats_warm_start_fn(
            components.agent.obs_stats,
            checkpoint=checkpoint,
        )

    policy_teacher = policy_teacher_fn(
        checkpoint,
        current_obs_keys=context.obs_keys,
        arm_joint_names=arm_joints,
        device=context.device,
    )
    return NativePhase1StartupResult(
        requested=True,
        skipped_ik=False,
        actor_copy_skipped=skip_reason is not None,
        actor_copy_skip_reason=skip_reason,
        actor_copy_applied=actor_copy,
        obs_stats_applied=obs_stats,
        policy_teacher=policy_teacher,
    )
