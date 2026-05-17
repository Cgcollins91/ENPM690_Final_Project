"""

Native checkpoint startup application orchestration

File map:

NativeCheckpointStartupResult:    Summary of checkpoint state applied during native startup
_required_checkpoint:             Handle required checkpoint logic
_checkpoint_global_step:          Handle checkpoint global step logic
apply_native_checkpoint_startup:  Apply native startup checkpoints to agent replay and play state
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..io.checkpoint_apply import (
    ActorInitApplyResult,
    PlayCheckpointApplyResult,
    ResumeApplyResult,
    apply_actor_init_checkpoint,
    apply_play_checkpoint,
    apply_play_skip_checkpoint,
    apply_resume_checkpoint,
    reset_obs_stats_after_checkpoint_load,
)
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_components import NativeTrainingComponents
from ..io.replay_startup import (
    HandoffReuseResult,
    ReplayResumeResult,
    apply_handoff_checkpoint_reuse,
    apply_resume_replay,
)


@dataclass(frozen=True)
class NativeCheckpointStartupResult:
    """Summary of checkpoint state applied during native startup"""

    resume                           : ResumeApplyResult | None         = None  # stores resume for native checkpoint startup result
    actor_init                       : ActorInitApplyResult | None      = None  # stores actor init for native checkpoint startup result
    replay_resume                    : ReplayResumeResult | None        = None  # stores replay resume for native checkpoint startup result
    handoff_reuse                    : HandoffReuseResult | None        = None  # stores handoff reuse for native checkpoint startup result
    play                             : PlayCheckpointApplyResult | None = None  # integer play value tracked by native checkpoint startup result
    play_skip_checkpoint             : bool                             = False  # boolean value indicating the play skip checkpoint state for native checkpoint startup result
    obs_stats_reset                  : bool                             = False  # boolean value indicating the obs stats reset state for native checkpoint startup result
    transitions_collected            : int                              = 0  # number of replay transitions collected so far
    replay_size                      : int                              = 0  # configured or observed replay-buffer size
    auto_handoff_loaded              : bool                             = False  # boolean value indicating the auto handoff loaded state for native checkpoint startup result
    skip_training_after_handoff_reuse: bool                             = False  # boolean value indicating the skip training after handoff reuse state for native checkpoint startup result


def _required_checkpoint(checkpoint: Mapping[str, object] | None, label: str) -> Mapping[str, object]:
    if checkpoint is None:
        raise RuntimeError(f"{label} checkpoint was requested but not loaded")
    return checkpoint


def _checkpoint_global_step(checkpoint: Mapping[str, object] | None) -> int:
    if checkpoint is None:
        return 0
    return max(0, int(checkpoint.get("global_step", 0)))


def apply_native_checkpoint_startup(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    components: NativeTrainingComponents,  # Param: input value used as components
    *,
    current_handoff_compatibility: Mapping[str, object] | None = None,  # Param: string input for current handoff compatibility
    handoff_checkpoint           : Mapping[str, object] | None = None,  # Param: string input for handoff checkpoint
    play_checkpoint              : Mapping[str, object] | None = None,  # Param: string input for play checkpoint
) -> NativeCheckpointStartupResult:
    """Apply native startup checkpoints to agent replay and play state

    Steps:
    - Resolve inputs for `apply_native_checkpoint_startup` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    plan = components.checkpoint_plan
    agent = components.agent
    replay = components.replay
    handoff = {} if current_handoff_compatibility is None else current_handoff_compatibility

    resume_result    : ResumeApplyResult | None         = None
    actor_init_result: ActorInitApplyResult | None      = None
    replay_result    : ReplayResumeResult | None        = None
    handoff_result   : HandoffReuseResult | None        = None
    play_result      : PlayCheckpointApplyResult | None = None
    play_skip = False
    obs_reset = False
    transitions_collected = 0
    replay_size = 0
    auto_handoff_loaded = False
    skip_after_handoff = False

    resume_checkpoint = components.checkpoints.resume
    if plan.resume_requested:
        resume_checkpoint = _required_checkpoint(resume_checkpoint, "resume")
        resume_result = apply_resume_checkpoint(
            agent,
            resume_checkpoint,
            resume_replay=configs.checkpoint.resume_replay,
            resume_global_step=configs.checkpoint.resume_global_step,
            reset_optimizers_on_resume=configs.checkpoint.reset_optimizers_on_resume,
        )
        if configs.checkpoint.resume_global_step:
            transitions_collected = _checkpoint_global_step(resume_checkpoint)

    if plan.replay_resume_requested:
        resume_checkpoint = _required_checkpoint(resume_checkpoint, "resume")
        replay_result = apply_resume_replay(
            replay,
            resume_checkpoint,
            handoff,
        )
        replay_size = replay_result.replay_size
        if configs.checkpoint.resume_global_step:
            transitions_collected = max(transitions_collected, replay_result.source_global_step)

    if handoff_checkpoint is not None:
        handoff_result = apply_handoff_checkpoint_reuse(
            agent=agent,
            replay=replay,
            checkpoint=handoff_checkpoint,
            current_handoff_compatibility=handoff,
            ignore_source_hashes=configs.checkpoint.allow_handoff_source_hash_mismatch,
            stop_after_handoff_checkpoint=configs.checkpoint.stop_after_handoff_checkpoint,
        )
        if handoff_result.reused:
            auto_handoff_loaded = True
            transitions_collected = max(transitions_collected, handoff_result.transitions_collected)
            replay_size = handoff_result.replay_size
            skip_after_handoff = handoff_result.skip_training_after_reuse

    if plan.actor_init_requested:
        actor_init_checkpoint = _required_checkpoint(components.checkpoints.actor_init, "actor-init")
        actor_init_result = apply_actor_init_checkpoint(
            agent,
            actor_init_checkpoint,
            observation_normalization=bool(context.args.get("observation_normalization", False)),
        )

    if plan.reset_obs_stats_after_load:
        obs_reset = reset_obs_stats_after_checkpoint_load(
            agent,
            enabled=True,
            reason="native_checkpoint_startup",
        )

    if plan.play_requested:
        if configs.eval.play_skip_checkpoint:
            apply_play_skip_checkpoint(agent, eval_teacher_assist_mix=configs.eval.eval_teacher_assist_mix)
            play_skip = True
        else:
            play_result = apply_play_checkpoint(
                agent,
                _required_checkpoint(play_checkpoint, "play"),
                current_arm_controller=configs.teacher.arm_controller,
                observation_normalization=bool(context.args.get("observation_normalization", False)),
                reward_normalization=bool(context.args.get("reward_normalization", False)),
            )

    return NativeCheckpointStartupResult(
        resume=resume_result,
        actor_init=actor_init_result,
        replay_resume=replay_result,
        handoff_reuse=handoff_result,
        play=play_result,
        play_skip_checkpoint=play_skip,
        obs_stats_reset=obs_reset,
        transitions_collected=transitions_collected,
        replay_size=replay_size,
        auto_handoff_loaded=auto_handoff_loaded,
        skip_training_after_handoff_reuse=skip_after_handoff,
    )
