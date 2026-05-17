"""

Startup observation and action contracts for the refactored trainer

File map:

StartupObservationContract:             Resolved startup observation contract
StartupActionContract:                  Resolved startup action dimensions
resolve_startup_observation_contract:   Resolve and validate initial observation tensors from env reset
validate_topdown_observation_contract:  Validate fixed topdown observation dimensions and column offsets
resolve_startup_action_contract:        Resolve action dimensions after env action space is available
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..actions.action_space import (
    TOPDOWN_FINGER_UNLOCK_PROGRESS_COL,
    TOPDOWN_POLICY_OBS_DIM,
    TOPDOWN_PRIVILEGED_OBS_DIM,
    TOPDOWN_STAGE_ONE_HOT_OBS_COL,
    ActionLayout,
)
from .observations import (
    ObservationColumnLayout,
    flatten_policy_obs,
    flatten_privileged_obs,
    resolve_observation_columns,
    resolve_policy_obs_keys,
)


@dataclass(frozen=True)
class StartupObservationContract:
    """Resolved startup observation contract"""

    obs_keys             : tuple[str, ...]  # ordered keys used to resolve obs values
    obs_tensor           : torch.Tensor  # policy observation tensor passed to the actor or replay path
    obs_dim              : int  # width of the policy observation vector
    privileged_obs_tensor: torch.Tensor | None  # tensor containing privileged obs tensor values for batched env rows
    privileged_obs_dim   : int  # integer privileged obs dim value tracked by startup observation contract
    columns              : ObservationColumnLayout  # stores columns for startup observation contract


@dataclass(frozen=True)
class StartupActionContract:
    """Resolved startup action dimensions"""

    action_dim          : int  # width of the policy action vector
    full_action_dim     : int  # integer full action dim value tracked by startup action contract
    policy_action_joints: tuple[str, ...]  # string policy action joints value used by startup action contract
    env_action_joints   : tuple[str, ...]  # string env action joints value used by startup action contract


def resolve_startup_observation_contract(
    obs: Mapping[str, object],        # Param: observation payload returned by the environment or replay path
    *,
    privileged_critic : bool,  # Param: boolean input controlling privileged critic
    topdown_curriculum: bool = True,  # Param: boolean input controlling topdown curriculum
) -> StartupObservationContract:
    """Resolve and validate initial observation tensors from env reset

    Steps:
    - Resolve inputs for `resolve_startup_observation_contract` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    policy_obs = obs.get("policy")
    if not isinstance(policy_obs, Mapping):
        raise TypeError("obs['policy'] must be a mapping")
    obs_keys = resolve_policy_obs_keys(policy_obs, topdown_curriculum=topdown_curriculum)
    columns = resolve_observation_columns(
        policy_obs,
        obs_keys,
        topdown_curriculum=topdown_curriculum,
    )
    obs_tensor = flatten_policy_obs(policy_obs, obs_keys)
    obs_dim = int(obs_tensor.shape[-1])
    privileged_obs_tensor = flatten_privileged_obs(obs) if privileged_critic else None
    privileged_obs_dim = int(privileged_obs_tensor.shape[-1]) if privileged_obs_tensor is not None else 0
    validate_topdown_observation_contract(
        obs_dim=obs_dim,
        privileged_obs_dim=privileged_obs_dim,
        privileged_critic=privileged_critic,
        columns=columns,
        topdown_curriculum=topdown_curriculum,
    )
    return StartupObservationContract(
        obs_keys=obs_keys,
        obs_tensor=obs_tensor,
        obs_dim=obs_dim,
        privileged_obs_tensor=privileged_obs_tensor,
        privileged_obs_dim=privileged_obs_dim,
        columns=columns,
    )


def validate_topdown_observation_contract(
    *,
    obs_dim           : int,  # Param: integer input for obs dim
    privileged_obs_dim: int,  # Param: integer input for privileged obs dim
    privileged_critic : bool,  # Param: boolean input controlling privileged critic
    columns           : ObservationColumnLayout,  # Param: input value used as columns
    topdown_curriculum: bool = True,  # Param: boolean input controlling topdown curriculum
) -> None:
    """Validate fixed topdown observation dimensions and column offsets

    Steps:
    - Resolve inputs for `validate_topdown_observation_contract` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not topdown_curriculum:
        return
    if int(obs_dim) != TOPDOWN_POLICY_OBS_DIM:
        raise RuntimeError(
            f"topdown policy observation width mismatch: expected {TOPDOWN_POLICY_OBS_DIM}, got {obs_dim}"
        )
    if columns.finger_unlock_progress_col != TOPDOWN_FINGER_UNLOCK_PROGRESS_COL:
        raise RuntimeError(
            "topdown finger_unlock_progress column mismatch: expected "
            f"{TOPDOWN_FINGER_UNLOCK_PROGRESS_COL}, got {columns.finger_unlock_progress_col}"
        )
    if columns.stage_one_hot_col != TOPDOWN_STAGE_ONE_HOT_OBS_COL:
        raise RuntimeError(
            "topdown stage_one_hot column mismatch: expected "
            f"{TOPDOWN_STAGE_ONE_HOT_OBS_COL}, got {columns.stage_one_hot_col}"
        )
    if privileged_critic and int(privileged_obs_dim) != TOPDOWN_PRIVILEGED_OBS_DIM:
        raise RuntimeError(
            f"topdown privileged observation width mismatch: expected "
            f"{TOPDOWN_PRIVILEGED_OBS_DIM}, got {privileged_obs_dim}"
        )


def resolve_startup_action_contract(
    *,
    action_layout    : ActionLayout,  # Param: input value used as action layout
    full_action_shape: Sequence[int],  # Param: integer input for full action shape
) -> StartupActionContract:
    """Resolve action dimensions after env action space is available"""
    if len(full_action_shape) == 0:
        raise ValueError("full_action_shape must include an action dimension")
    return StartupActionContract(
        action_dim=len(action_layout.policy_action_spec.joint_names),
        full_action_dim=int(full_action_shape[-1]),
        policy_action_joints=tuple(action_layout.policy_action_spec.joint_names),
        env_action_joints=tuple(action_layout.env_action_spec.joint_names),
    )
