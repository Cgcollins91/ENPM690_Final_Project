"""

Checkpoint application helpers for startup branches

This module provides helper functions and data structures for applying checkpoint state to the agent at startup



CheckpointAgent:                        Agent surface needed by startup checkpoint application
ResumeApplyResult:                      Outcome of loading a resume checkpoint
ActorInitApplyResult:                   Outcome of loading actor-init checkpoint state
PlayCheckpointApplyResult:              Outcome of loading play-mode actor checkpoint state
optimizer_state_attrs:                  Return optimizer attributes available on an agent
clear_optimizer_states:                 Clear optimizer state dicts and return cleared attribute names
_agent_state:                           Handle agent state logic
apply_resume_checkpoint:                Apply full agent state from a resume checkpoint
apply_actor_init_checkpoint:            Apply actor and optional obs stats from an actor-init checkpoint
reset_obs_stats_after_checkpoint_load:  Reset observation stats after warm checkpoint load when requested
apply_play_skip_checkpoint:             Prepare play mode without loading a checkpoint
apply_play_checkpoint:                  Apply actor and optional stats for deterministic play mode
_default_module_loader:                 Handle default module loader logic
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from ..model.networks import load_module_state_allow_env_count_buffers
from ..model.normalization import reset_obs_stats_for_actor_rollout


class CheckpointAgent(Protocol):
    """Agent surface needed by startup checkpoint application"""

    train_step  : int  # Field: step count used for train step scheduling or reporting
    actor       : torch.nn.Module  # Field: actor network or actor checkpoint payload
    actor_target: torch.nn.Module  # Field: stores actor target for checkpoint agent
    obs_stats   : object  # Field: stores obs stats for checkpoint agent
    reward_stats: object  # Field: stores reward stats for checkpoint agent

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load full agent state"""
        ...


@dataclass(frozen=True)
class ResumeApplyResult:
    """Outcome of loading a resume checkpoint"""

    resume_loaded           : bool  # Field: boolean value indicating the resume loaded state for resume apply result
    train_step_reset        : bool  # Field: boolean value indicating the train step reset state for resume apply result
    optimizer_states_cleared: tuple[str, ...]  # Field: string optimizer states cleared value used by resume apply result
    source_train_step       : object  # Field: step count used for source train step scheduling or reporting
    source_global_step      : object  # Field: step count used for source global step scheduling or reporting


@dataclass(frozen=True)
class ActorInitApplyResult:
    """Outcome of loading actor-init checkpoint state"""

    actor_init_loaded : bool  # Field: boolean value indicating the actor init loaded state for actor init apply result
    obs_stats_loaded  : bool  # Field: boolean value indicating the obs stats loaded state for actor init apply result
    source_global_step: object  # Field: step count used for source global step scheduling or reporting


@dataclass(frozen=True)
class PlayCheckpointApplyResult:
    """Outcome of loading play-mode actor checkpoint state"""

    checkpoint_loaded  : bool  # Field: boolean value indicating the checkpoint loaded state for play checkpoint apply result
    obs_stats_loaded   : bool  # Field: boolean value indicating the obs stats loaded state for play checkpoint apply result
    reward_stats_loaded: bool  # Field: boolean value indicating the reward stats loaded state for play checkpoint apply result
    source_train_step  : object  # Field: step count used for source train step scheduling or reporting


def optimizer_state_attrs(agent: object) -> tuple[str, ...]:
    """Return optimizer attributes available on an agent"""
    names = ("actor_opt", "critic1_opt", "critic2_opt", "critic_opt")
    return tuple(name for name in names if hasattr(agent, name))


def clear_optimizer_states(agent: object, attrs: Iterable[str] | None = None) -> tuple[str, ...]:
    """Clear optimizer state dicts and return cleared attribute names"""
    cleared: list[str] = []
    for name in optimizer_state_attrs(agent) if attrs is None else tuple(attrs):
        opt = getattr(agent, name, None)
        state = getattr(opt, "state", None)
        if hasattr(state, "clear"):
            state.clear()
            cleared.append(str(name))
    return tuple(cleared)


def _agent_state(checkpoint: Mapping[str, object], *, context: str) -> Mapping[str, object]:
    agent_state = checkpoint.get("agent", {})
    if not isinstance(agent_state, Mapping):
        raise RuntimeError(f"{context} checkpoint is missing agent state")
    return agent_state


def apply_resume_checkpoint(
    agent     : CheckpointAgent,  # Param: TD3 agent whose networks, optimizers, or stats are used
    checkpoint: Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    *,
    resume_replay             : bool,  # Param: boolean input controlling resume replay
    resume_global_step        : bool,  # Param: step count used for resume global step
    reset_optimizers_on_resume: bool,  # Param: boolean input controlling reset optimizers on resume
) -> ResumeApplyResult:
    """Apply full agent state from a resume checkpoint

    Steps:
    - Resolve inputs for `apply_resume_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    agent_state = _agent_state(checkpoint, context="resume")
    agent.load_state_dict(agent_state)
    train_step_reset = False
    if not (bool(resume_replay) or bool(resume_global_step)):
        agent.train_step = 0
        train_step_reset = True
    cleared = clear_optimizer_states(agent) if reset_optimizers_on_resume else ()
    return ResumeApplyResult(
        resume_loaded=True,
        train_step_reset=train_step_reset,
        optimizer_states_cleared=cleared,
        source_train_step=agent_state.get("train_step", "?"),
        source_global_step=checkpoint.get("global_step", "?"),
    )


def apply_actor_init_checkpoint(
    agent     : CheckpointAgent,  # Param: TD3 agent whose networks, optimizers, or stats are used
    checkpoint: Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    *,
    observation_normalization: bool,  # Param: boolean input controlling observation normalization
    module_loader            : Callable[[torch.nn.Module, dict, str], None] | None = None,  # Param: callback used to compute or fetch module loader
) -> ActorInitApplyResult:
    """Apply actor and optional obs stats from an actor-init checkpoint

    Steps:
    - Resolve inputs for `apply_actor_init_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    agent_state = _agent_state(checkpoint, context="actor_init")
    actor_state = agent_state.get("actor")
    if not isinstance(actor_state, Mapping):
        raise RuntimeError("actor_init_checkpoint is missing agent.actor state")
    loader = _default_module_loader if module_loader is None else module_loader
    loader(agent.actor, dict(actor_state), "actor_init_checkpoint actor")
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    obs_stats_loaded = False
    if observation_normalization:
        ckpt_obs_stats = agent_state.get("obs_stats")
        if ckpt_obs_stats is None:
            raise RuntimeError("actor_init_checkpoint has no obs_stats but observation normalization is enabled")
        agent.obs_stats.load_state_dict(ckpt_obs_stats)
        obs_stats_loaded = True
    return ActorInitApplyResult(
        actor_init_loaded=True,
        obs_stats_loaded=obs_stats_loaded,
        source_global_step=checkpoint.get("global_step", "?"),
    )


def reset_obs_stats_after_checkpoint_load(
    agent: CheckpointAgent,  # Param: TD3 agent whose networks, optimizers, or stats are used
    *,
    enabled: bool,  # Param: boolean input controlling enabled
    reason : str,  # Param: string input for reason
) -> bool:
    """Reset observation stats after warm checkpoint load when requested"""
    if not enabled:
        return False
    reset_obs_stats_for_actor_rollout(agent, reason=reason)
    return True


def apply_play_skip_checkpoint(agent: CheckpointAgent, *, eval_teacher_assist_mix: float) -> None:
    """Prepare play mode without loading a checkpoint"""
    if float(eval_teacher_assist_mix) < 1.0:
        raise RuntimeError("--play-skip-checkpoint requires --eval-teacher-assist-mix=1.0")
    agent.actor.eval()


def apply_play_checkpoint(
    agent     : CheckpointAgent,  # Param: TD3 agent whose networks, optimizers, or stats are used
    checkpoint: Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    *,
    current_arm_controller   : str,  # Param: string input for current arm controller
    observation_normalization: bool,  # Param: boolean input controlling observation normalization
    reward_normalization     : bool,  # Param: boolean input controlling reward normalization
    module_loader            : Callable[[torch.nn.Module, dict, str], None] | None = None,  # Param: callback used to compute or fetch module loader
) -> PlayCheckpointApplyResult:
    """Apply actor and optional stats for deterministic play mode

    Steps:
    - Resolve inputs for `apply_play_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ckpt_arm_controller = str(checkpoint.get("arm_controller", current_arm_controller))
    if ckpt_arm_controller != str(current_arm_controller):
        raise RuntimeError(
            "checkpoint arm_controller mismatch: "
            f"checkpoint={ckpt_arm_controller} current={current_arm_controller}"
        )
    agent_state = _agent_state(checkpoint, context="play")
    actor_state = agent_state.get("actor")
    if not isinstance(actor_state, Mapping):
        raise RuntimeError("play checkpoint is missing agent.actor state")
    loader = _default_module_loader if module_loader is None else module_loader
    loader(agent.actor, dict(actor_state), "play checkpoint actor")
    agent.actor.eval()

    obs_stats_loaded = False
    ckpt_obs_stats = agent_state.get("obs_stats")
    if ckpt_obs_stats and observation_normalization:
        agent.obs_stats.load_state_dict(ckpt_obs_stats)
        obs_stats_loaded = True

    reward_stats_loaded = False
    ckpt_reward_stats = agent_state.get("reward_stats")
    if ckpt_reward_stats and reward_normalization:
        agent.reward_stats.load_state_dict(ckpt_reward_stats)
        reward_stats_loaded = True

    return PlayCheckpointApplyResult(
        checkpoint_loaded=True,
        obs_stats_loaded=obs_stats_loaded,
        reward_stats_loaded=reward_stats_loaded,
        source_train_step=agent_state.get("train_step", "?"),
    )


def _default_module_loader(module: torch.nn.Module, state: dict, context: str) -> None:
    load_module_state_allow_env_count_buffers(module, state, context=context)
