"""

Checkpoint payload and RNG helpers for import-safe training code

File map:

StateDictSource:             Object that can expose serializable training state
TrainingCheckpointMetadata:  Metadata fields persisted beside learner state
capture_rng_state:           Capture Python and CPU torch RNG state
restore_rng_state:           Restore Python and CPU torch RNG state when present
_metadata_int:               Handle metadata int logic
_state_dict_payload:         Handle state dict payload logic
_replay_size:                Handle replay size logic
build_checkpoint_payload:    Build the topdown trainer checkpoint payload
save_training_checkpoint:    Build and write a topdown trainer checkpoint
load_training_checkpoint:    Load a checkpoint with Python objects allowed
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
import random
from typing import Any, Protocol

import torch

from ..actions.action_space import ReducedActionSpec


class StateDictSource(Protocol):
    """Object that can expose serializable training state"""

    def state_dict(self) -> Mapping[str, object]:
        """Return a checkpoint-compatible state mapping"""
        ...


@dataclass(frozen=True)
class TrainingCheckpointMetadata:
    """Metadata fields persisted beside learner state"""

    task                 : str  # Field: string task value used by training checkpoint metadata
    global_step          : int  # Field: training step associated with this record or action
    episode_idx          : int | torch.Tensor  # Field: training episode index associated with this record
    arm_controller       : str  # Field: string arm controller value used by training checkpoint metadata
    td3_backend          : str  # Field: string td3 backend value used by training checkpoint metadata
    obs_schema_version   : int  # Field: integer obs schema version value tracked by training checkpoint metadata
    obs_keys             : tuple[str, ...]  # Field: ordered keys used to resolve obs values
    obs_dim              : int  # Field: width of the policy observation vector
    priv_obs_dim         : int  # Field: width of the privileged observation vector
    policy_action_spec   : ReducedActionSpec  # Field: action layout spec expected by the policy output
    env_action_spec      : ReducedActionSpec  # Field: action layout spec expected by the environment
    log_jsonl            : str | None                  = None  # Field: JSONL log path or enablement flag for structured logging
    args                 : Mapping[str, object]        = field(default_factory=dict)  # Field: parsed CLI/config arguments passed into this helper
    ik_arm_joints        : tuple[str, ...]             = ()  # Field: string ik arm joints value used by training checkpoint metadata
    handoff_compatibility: Mapping[str, object] | None = None  # Field: string handoff compatibility value used by training checkpoint metadata


def capture_rng_state() -> dict[str, object]:
    """Capture Python and CPU torch RNG state"""
    return {
        "python": random.getstate(),
        "torch" : torch.get_rng_state(),
    }


def restore_rng_state(state: Mapping[str, object] | None) -> None:
    """Restore Python and CPU torch RNG state when present

    Steps:
    - Resolve inputs for `restore_rng_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not isinstance(state, Mapping):
        return
    python_state = state.get("python")
    torch_state = state.get("torch")
    if python_state is not None:
        random.setstate(python_state)
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.cpu())


def _metadata_int(value: int | torch.Tensor) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _state_dict_payload(source: Mapping[str, object] | StateDictSource) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    if hasattr(source, "state_dict"):
        return source.state_dict()
    raise TypeError(f"checkpoint source {type(source).__name__!r} does not provide state_dict")


def _replay_size(replay_state: Mapping[str, object] | None, replay: object | None) -> int | None:
    if replay_state is not None and "size" in replay_state:
        return int(replay_state["size"])
    if replay is not None and hasattr(replay, "size"):
        return int(getattr(replay, "size"))
    return None


def build_checkpoint_payload(
    *,
    metadata    : TrainingCheckpointMetadata,  # Param: integer input for metadata
    agent_state : Mapping[str, object],  # Param: string input for agent state
    replay_state: Mapping[str, object] | None = None,  # Param: string input for replay state
    replay_size : int | None                  = None,  # Param: number of transitions currently available in replay
    rng_state   : Mapping[str, object] | None = None,  # Param: string input for rng state
    extra_fields: Mapping[str, Any] | None    = None,  # Param: string input for extra fields
) -> dict[str, object]:
    """Build the topdown trainer checkpoint payload

    Steps:
    - Resolve inputs for `build_checkpoint_payload` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    policy_action_joints = tuple(metadata.policy_action_spec.joint_names)
    policy_action_scales = tuple(float(x) for x in metadata.policy_action_spec.scales)
    env_action_joints = tuple(metadata.env_action_spec.joint_names)
    env_action_scales = tuple(float(x) for x in metadata.env_action_spec.scales)
    payload: dict[str, object] = {
        "task"                     : metadata.task,
        "global_step"              : int(metadata.global_step),
        "episode_idx"              : _metadata_int(metadata.episode_idx),
        "arm_controller"           : metadata.arm_controller,
        "td3_backend"              : metadata.td3_backend,
        "obs_schema_version"       : int(metadata.obs_schema_version),
        "policy_obs_keys"          : tuple(metadata.obs_keys),
        "obs_keys"                 : tuple(metadata.obs_keys),
        "obs_dim"                  : int(metadata.obs_dim),
        "priv_obs_dim"             : int(metadata.priv_obs_dim),
        "policy_action_joints"     : policy_action_joints,
        "policy_action_scales"     : policy_action_scales,
        "policy_action_dim"        : len(policy_action_joints),
        "reduced_action_joints"    : policy_action_joints,
        "reduced_action_scales"    : policy_action_scales,
        "env_reduced_action_joints": env_action_joints,
        "env_reduced_action_scales": env_action_scales,
        "env_reduced_action_dim"   : len(env_action_joints),
        "ik_arm_joints"            : tuple(metadata.ik_arm_joints),
        "log_jsonl"                : metadata.log_jsonl,
        "agent"                    : dict(agent_state),
        "args"                     : dict(metadata.args),
    }
    if metadata.handoff_compatibility is not None:
        handoff = dict(metadata.handoff_compatibility)
        payload["handoff_compatibility"] = handoff
        payload["handoff_compatibility_digest"] = handoff.get("digest")
    if replay_state is not None:
        payload["replay"] = dict(replay_state)
        payload["replay_size"] = int(replay_size) if replay_size is not None else _replay_size(replay_state, None)
    if rng_state is not None:
        payload["rng_state"] = dict(rng_state)
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


def save_training_checkpoint(
    path: str | os.PathLike[str],                                  # Param: filesystem path read or written by this helper
    *,
    metadata      : TrainingCheckpointMetadata,  # Param: integer input for metadata
    agent         : Mapping[str, object] | StateDictSource,  # Param: TD3 agent whose networks, optimizers, or stats are used
    replay        : Mapping[str, object] | StateDictSource | None = None,  # Param: replay buffer or replay target used for transition storage
    include_replay: bool                                          = False,  # Param: boolean input controlling include replay
    rng_state     : Mapping[str, object] | None                   = None,  # Param: string input for rng state
    extra_fields  : Mapping[str, Any] | None                      = None,  # Param: string input for extra fields
) -> dict[str, object]:
    """Build and write a topdown trainer checkpoint

    Steps:
    - Resolve inputs for `save_training_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    agent_state = _state_dict_payload(agent)
    if include_replay and replay is None:
        raise RuntimeError(f"include_replay requested for checkpoint {os.fspath(path)!r}, but replay is None")
    replay_state = _state_dict_payload(replay) if include_replay and replay is not None else None
    payload = build_checkpoint_payload(
        metadata     = metadata,
        agent_state  = agent_state,
        replay_state = replay_state,
        replay_size  = _replay_size(replay_state, replay),
        rng_state    = rng_state,
        extra_fields = extra_fields,
    )
    target = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    torch.save(payload, target)
    return payload


def load_training_checkpoint(
    path: str | os.PathLike[str],              # Param: filesystem path read or written by this helper
    *,
    map_location: str | torch.device = "cpu",  # Param: string input for map location
) -> dict[str, object]:
    """Load a checkpoint with Python objects allowed"""
    target = os.fspath(path)
    try:
        loaded = torch.load(target, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(target, map_location=map_location)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"checkpoint payload must be a dict, got {type(loaded).__name__}")
    return loaded
