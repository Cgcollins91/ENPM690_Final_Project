"""

Native env-step collection for the callback trainer

File map:

SizedReplayTarget:                 Replay target with a public size counter
NativeEnvActionAssemblyConfig:     Action assembly inputs needed before env step
NativeEnvStepCallbacks:            Injected callbacks for one native env step
NativeEnvStepRequest:              Inputs for one vectorized native env step
NativeEnvStepPayload:              Normalized result from env step
NativeEnvStepResult:               Collected transition details for one env step
_policy_obs:                       Handle policy obs logic
_tensor_1d_bool:                   Handle tensor 1d bool logic
_reward_tensor:                    Handle reward tensor logic
_replay_size:                      Handle replay size logic
normalize_native_env_step_result:  Normalize common gym step result shapes
assemble_native_env_action:        Assemble the env action for a native step
collect_native_env_step:           Step env collect transitions and return the native loop batch
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..actions.action_assembly import assemble_env_reduced_action
from ..actions.action_gates import ActionGateConfig
from .native_actions import NativeActionSelection
from .native_loop import NativeLoopStepBatch, native_loop_step_batch
from ..env.observations import flatten_policy_obs, flatten_privileged_obs
from ..state.replay import flush_ready_n_step_transitions
from ..state.replay_batches import ReplayAddTarget, add_ready_transitions_to_replay
from ..env.terminal_observations import (
    TerminalNextObservation,
    substitute_terminal_next_observations,
    terminal_observation_from_info,
)
from ..state.transition_collection import TransitionCollectionResult, append_step_transitions


EnvStepFn = Callable[[torch.Tensor], object]
ActionAssemblyFn = Callable[[torch.Tensor], torch.Tensor]


class SizedReplayTarget(ReplayAddTarget, Protocol):
    """Replay target with a public size counter"""

    size : int  # integer size value tracked by sized replay target


@dataclass(frozen=True)
class NativeEnvActionAssemblyConfig:
    """Action assembly inputs needed before env step"""

    gate_config                    : ActionGateConfig  # stores gate config for native env action assembly config
    arm_controller                 : str  # string arm controller value used by native env action assembly config
    finger_action_mode             : str                     = "absolute"  # configured interpretation of finger action columns
    arm_reduced_action             : torch.Tensor | None     = None  # tensor containing arm reduced action values for batched env rows
    env                            : Any                     = None  # environment/backend object used by this runtime helper
    mapped_indices                 : torch.Tensor | None     = None  # column indices used to map between action layouts
    mapped_scales                  : torch.Tensor | None     = None  # scales applied while mapping action columns
    finger_delta_scale             : float                   = 0.05  # scale applied to finger delta action columns
    contact_finger_open_until_ready: ActionAssemblyFn | None = None  # boolean/tensor readiness state for contact finger open until
    align_open_hand_action         : ActionAssemblyFn | None = None  # stores align open hand action for native env action assembly config


@dataclass(frozen=True)
class NativeEnvStepCallbacks:
    """Injected callbacks for one native env step"""

    env_step_fn           : EnvStepFn  # callback used for the env step fn operation
    assemble_env_action_fn: ActionAssemblyFn | None = None  # callback used for the assemble env action fn operation


@dataclass(frozen=True)
class NativeEnvStepRequest:
    """Inputs for one vectorized native env step"""

    obs                      : Mapping[str, object]  # policy observation tensor or observation payload for this transition
    obs_tensor               : torch.Tensor  # policy observation tensor passed to the actor or replay path
    priv_obs_tensor          : torch.Tensor | None  # privileged observation tensor passed to critic-side logic
    action_selection         : NativeActionSelection  # stores action selection for native env step request
    preroll_mask_before      : torch.Tensor  # tensor containing preroll mask before values for batched env rows
    action_source            : str  # string action source value used by native env step request
    replay                   : ReplayAddTarget  # stores replay for native env step request
    n_step_queues            : Sequence[deque]  # ordered collection of n step queues entries for native env step request
    obs_keys                 : Sequence[str]  # ordered keys used to resolve obs values
    gamma                    : float  # discount factor used by TD3 updates
    n_step                   : int  # step count used for n step scheduling or reporting
    privileged_critic        : bool                                 = False  # boolean value indicating the privileged critic state for native env step request
    active_env_mask          : torch.Tensor | None                  = None  # mask selecting env rows that are still active
    existing_checkpoint_names: tuple[str, ...]                      = ()  # ordered names used to resolve existing checkpoint attributes
    action_assembly          : NativeEnvActionAssemblyConfig | None = None  # stores action assembly for native env step request


@dataclass(frozen=True)
class NativeEnvStepPayload:
    """Normalized result from env step"""

    next_obs  : Mapping[str, object]  # next policy observation tensor after the transition step
    reward    : torch.Tensor  # reward tensor or scalar produced by the environment step
    terminated: torch.Tensor  # tensor containing terminated values for batched env rows
    timeout   : torch.Tensor  # tensor containing timeout values for batched env rows
    info      : object  # auxiliary info mapping returned by the environment or backend


@dataclass(frozen=True)
class NativeEnvStepResult:
    """Collected transition details for one env step"""

    env_action          : torch.Tensor  # tensor containing env action values for batched env rows
    next_obs            : Mapping[str, object]  # next policy observation tensor after the transition step
    next_obs_tensor     : torch.Tensor  # tensor containing next obs tensor values for batched env rows
    next_priv_obs_tensor: torch.Tensor | None  # tensor containing next priv obs tensor values for batched env rows
    reward_tensor       : torch.Tensor  # tensor containing reward tensor values for batched env rows
    terminated_flags    : torch.Tensor  # flag values describing terminated state for native env step result
    timeout_flags       : torch.Tensor  # per-env timeout flags returned by the environment step
    terminal_next       : TerminalNextObservation  # stores terminal next for native env step result
    collection          : TransitionCollectionResult  # stores collection for native env step result
    ready_count         : int  # count of ready values
    inserted_count      : int  # count of inserted values
    batch               : NativeLoopStepBatch  # stores batch for native env step result


def _policy_obs(obs: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    policy = obs.get("policy")
    if not isinstance(policy, Mapping):
        raise KeyError("env observation is missing policy observations")
    return policy  # type: ignore[return-value]


def _tensor_1d_bool(value: object, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.bool).reshape(-1)
    return torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)


def _reward_tensor(value: object, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        reward = value.to(device=device, dtype=torch.float32)
    else:
        reward = torch.as_tensor(value, device=device, dtype=torch.float32)
    return reward.reshape(-1, 1)


def _replay_size(replay: ReplayAddTarget) -> int:
    return max(0, int(getattr(replay, "size", 0)))


def normalize_native_env_step_result(raw: object, *, device: torch.device) -> NativeEnvStepPayload:
    """Normalize common gym step result shapes"""
    if isinstance(raw, NativeEnvStepPayload):
        return raw
    if isinstance(raw, tuple):
        if len(raw) == 5:
            next_obs, reward, terminated, timeout, info = raw
        elif len(raw) == 4:
            next_obs, reward, done, info = raw
            terminated = done
            timeout = torch.zeros_like(_tensor_1d_bool(done, device=device))
        else:
            raise TypeError(f"env step tuple must have length 4 or 5, got {len(raw)}")
    elif isinstance(raw, Mapping):
        next_obs = raw["next_obs"]
        reward = raw["reward"]
        terminated = raw.get("terminated", raw.get("done"))
        if terminated is None:
            raise KeyError("env step mapping is missing terminated or done")
        timeout = raw.get("timeout", raw.get("truncated"))
        if timeout is None:
            timeout = torch.zeros_like(_tensor_1d_bool(terminated, device=device))
        info = raw.get("info", {})
    else:
        raise TypeError(f"unsupported env step result type {type(raw)!r}")

    if not isinstance(next_obs, Mapping):
        raise TypeError("env step next_obs must be a mapping")
    return NativeEnvStepPayload(
        next_obs=next_obs,
        reward=_reward_tensor(reward, device=device),
        terminated=_tensor_1d_bool(terminated, device=device),
        timeout=_tensor_1d_bool(timeout, device=device),
        info=info,
    )


def assemble_native_env_action(
    policy_level_action: torch.Tensor,               # Param: tensor input carrying policy level action values
    *,
    callbacks: NativeEnvStepCallbacks,  # Param: input value used as callbacks
    assembly : NativeEnvActionAssemblyConfig | None,  # Param: input value used as assembly
) -> torch.Tensor:
    """Assemble the env action for a native step"""
    if callbacks.assemble_env_action_fn is not None:
        return callbacks.assemble_env_action_fn(policy_level_action).clamp(-1.0, 1.0)
    if assembly is None:
        return policy_level_action.clamp(-1.0, 1.0)
    return assemble_env_reduced_action(
        policy_level_action,
        assembly.gate_config,
        arm_controller=assembly.arm_controller,
        finger_action_mode=assembly.finger_action_mode,
        arm_reduced_action=assembly.arm_reduced_action,
        env=assembly.env,
        mapped_indices=assembly.mapped_indices,
        mapped_scales=assembly.mapped_scales,
        finger_delta_scale=assembly.finger_delta_scale,
        contact_finger_open_until_ready=assembly.contact_finger_open_until_ready,
        align_open_hand_action=assembly.align_open_hand_action,
    )


def collect_native_env_step(
    request  : NativeEnvStepRequest,  # Param: normalized request object passed into this helper
    callbacks: NativeEnvStepCallbacks,  # Param: input value used as callbacks
) -> NativeEnvStepResult:
    """Step env collect transitions and return the native loop batch

    Steps:
    - Resolve inputs for `collect_native_env_step` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    device = request.obs_tensor.device
    env_action = assemble_native_env_action(
        request.action_selection.mixed_action,
        callbacks=callbacks,
        assembly=request.action_assembly,
    )
    payload = normalize_native_env_step_result(callbacks.env_step_fn(env_action), device=device)
    next_obs_tensor = flatten_policy_obs(_policy_obs(payload.next_obs), request.obs_keys)
    next_priv_obs_tensor = (
        flatten_privileged_obs(payload.next_obs) if request.privileged_critic else None
    )
    terminal_next = substitute_terminal_next_observations(
        live_next_obs=next_obs_tensor,
        live_next_privileged_obs=next_priv_obs_tensor,
        terminal_obs=terminal_observation_from_info(payload.info),
        timeout_flags=payload.timeout,
        obs_keys=request.obs_keys,
        privileged_critic=request.privileged_critic,
    )
    collection = append_step_transitions(
        n_step_queues=request.n_step_queues,
        preroll_mask_before=request.preroll_mask_before,
        obs_tensor=request.obs_tensor,
        replay_action=request.action_selection.replay_action,
        bc_action=request.action_selection.bc_action,
        reward_tensor=payload.reward,
        replay_next_obs_tensor=terminal_next.next_obs,
        terminated_flags=payload.terminated,
        timeout_flags=payload.timeout,
        priv_obs_tensor=request.priv_obs_tensor,
        replay_next_priv_obs_tensor=terminal_next.next_privileged_obs,
        teacher_action_present=request.action_selection.teacher_action is not None,
        action_source=request.action_source,
    )
    ready = flush_ready_n_step_transitions(
        request.n_step_queues,
        gamma=request.gamma,
        n_step=request.n_step,
    )
    inserted = add_ready_transitions_to_replay(request.replay, ready)
    done_flags = payload.terminated | payload.timeout
    batch = native_loop_step_batch(
        num_added=inserted,
        replay_size=_replay_size(request.replay),
        done_flags=done_flags,
        active_env_mask=request.active_env_mask,
        existing_checkpoint_names=request.existing_checkpoint_names,
    )
    return NativeEnvStepResult(
        env_action=env_action,
        next_obs=payload.next_obs,
        next_obs_tensor=next_obs_tensor,
        next_priv_obs_tensor=next_priv_obs_tensor,
        reward_tensor=payload.reward,
        terminated_flags=payload.terminated,
        timeout_flags=payload.timeout,
        terminal_next=terminal_next,
        collection=collection,
        ready_count=len(ready),
        inserted_count=inserted,
        batch=batch,
    )
