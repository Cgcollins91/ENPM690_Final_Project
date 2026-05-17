"""

Observation ordering and flattening helpers for trainer modules

File map:

PRIVILEGED_OBS_KEYS:          Define privileged obs keys constant
ObservationColumnLayout:      Resolved special observation column offsets
_term_width:                  Handle term width logic
resolve_policy_obs_keys:      Resolve the ordered observation terms used by the policy input
observation_term_offsets:     Return starting flat-column offset for each observation term
resolve_observation_columns:  Resolve special flat-observation columns used by action gates
flatten_policy_obs:           Flatten selected policy observation terms into one tensor
flatten_privileged_obs:       Flatten privileged observation group when present
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..actions.action_space import (
    DEFAULT_POLICY_OBS_KEYS,
    REMOVED_TEACHER_STATE_OBS_KEYS,
    TOPDOWN_POLICY_OBS_KEYS,
)


PRIVILEGED_OBS_KEYS = (
    "controller_state_scalars",
    "teacher_contact_state_scalars",
    "teacher_ik_state_scalars",
    "arm_hold_action_scalars",
)


@dataclass(frozen=True)
class ObservationColumnLayout:
    """Resolved special observation column offsets"""

    stage_one_hot_col         : int | None = None  # integer stage one hot col value tracked by observation column layout
    contact_finger_unlock_col : int | None = None  # integer contact finger unlock col value tracked by observation column layout
    finger_unlock_progress_col: int | None = None  # integer finger unlock progress col value tracked by observation column layout


def _term_width(term) -> int:
    shape = getattr(term, "shape", None)
    if shape is None:
        raise TypeError(f"observation term has no shape: {type(term)!r}")
    return int(shape[-1])


def resolve_policy_obs_keys(
    policy_obs: Mapping[str, torch.Tensor],  # Param: tensor input carrying policy obs values
    *,
    topdown_curriculum: bool = True,         # Param: boolean input controlling topdown curriculum
) -> tuple[str, ...]:
    """Resolve the ordered observation terms used by the policy input

    Steps:
    - Resolve inputs for `resolve_policy_obs_keys` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    available_keys = tuple(policy_obs.keys())
    if len(available_keys) == 0:
        raise ValueError("policy observation dict is empty")
    if topdown_curriculum:
        removed_present = [key for key in REMOVED_TEACHER_STATE_OBS_KEYS if key in policy_obs]
        if removed_present:
            raise RuntimeError(
                "topdown policy observation still exposes teacher/controller "
                f"terms: {removed_present}"
            )
        expected = set(TOPDOWN_POLICY_OBS_KEYS)
        available = set(available_keys)
        if available != expected:
            missing = sorted(expected - available)
            extra = sorted(available - expected)
            raise RuntimeError(
                "topdown policy observation keys changed: "
                f"missing={missing} extra={extra}"
            )
        return TOPDOWN_POLICY_OBS_KEYS
    preferred = [key for key in DEFAULT_POLICY_OBS_KEYS if key in policy_obs]
    extras = [key for key in available_keys if key not in preferred]
    return tuple(preferred + extras)


def observation_term_offsets(
    policy_obs: Mapping[str, torch.Tensor],  # Param: tensor input carrying policy obs values
    obs_keys  : Sequence[str],  # Param: ordered mapping keys used to resolve obs
) -> dict[str, int]:
    """Return starting flat-column offset for each observation term"""
    offset = 0
    offsets: dict[str, int] = {}
    for key in obs_keys:
        if key not in policy_obs:
            raise KeyError(f"policy observation missing key {key!r}")
        offsets[key] = offset
        offset += _term_width(policy_obs[key])
    return offsets


def resolve_observation_columns(
    policy_obs: Mapping[str, torch.Tensor],  # Param: tensor input carrying policy obs values
    obs_keys  : Sequence[str],  # Param: ordered mapping keys used to resolve obs
    *,
    topdown_curriculum: bool = True,  # Param: boolean input controlling topdown curriculum
    contact_task      : bool = False,  # Param: boolean input controlling contact task
) -> ObservationColumnLayout:
    """Resolve special flat-observation columns used by action gates

    Steps:
    - Resolve inputs for `resolve_observation_columns` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    offsets = observation_term_offsets(policy_obs, obs_keys)
    stage_one_hot_col         : int | None = None
    contact_finger_unlock_col : int | None = None
    finger_unlock_progress_col: int | None = None

    if topdown_curriculum:
        if "stage_one_hot" not in offsets:
            raise RuntimeError("topdown curriculum task observation is missing 'stage_one_hot'")
        if _term_width(policy_obs["stage_one_hot"]) != 3:
            raise RuntimeError(f"stage_one_hot must be width 3; got {_term_width(policy_obs['stage_one_hot'])}")
        stage_one_hot_col = offsets["stage_one_hot"]

        if "finger_unlock_progress" not in offsets:
            raise RuntimeError("topdown curriculum task observation is missing 'finger_unlock_progress'")
        if _term_width(policy_obs["finger_unlock_progress"]) != 1:
            raise RuntimeError(
                "finger_unlock_progress must be width 1; "
                f"got {_term_width(policy_obs['finger_unlock_progress'])}"
            )
        finger_unlock_progress_col = offsets["finger_unlock_progress"]

    if contact_task:
        if "contact_stage_scalars" not in offsets:
            raise RuntimeError("contact-stage observation is missing contact_stage_scalars")
        width = _term_width(policy_obs["contact_stage_scalars"])
        if width < 4:
            raise RuntimeError(f"contact_stage_scalars must include unlock gate at index 3; got width={width}")
        contact_finger_unlock_col = offsets["contact_stage_scalars"] + 3

    return ObservationColumnLayout(
        stage_one_hot_col=stage_one_hot_col,
        contact_finger_unlock_col=contact_finger_unlock_col,
        finger_unlock_progress_col=finger_unlock_progress_col,
    )


def flatten_policy_obs(
    policy_obs: Mapping[str, torch.Tensor],  # Param: tensor input carrying policy obs values
    obs_keys  : Sequence[str],  # Param: ordered mapping keys used to resolve obs
) -> torch.Tensor:
    """Flatten selected policy observation terms into one tensor"""
    missing = [key for key in obs_keys if key not in policy_obs]
    if missing:
        raise KeyError(
            f"policy observation missing keys {missing}; available={tuple(policy_obs.keys())}"
        )
    return torch.cat([policy_obs[key] for key in obs_keys], dim=-1)


def flatten_privileged_obs(
    obs: Mapping[str, object],                                 # Param: observation payload returned by the environment or replay path
    *,
    privileged_obs_keys: Sequence[str] = PRIVILEGED_OBS_KEYS,  # Param: ordered mapping keys used to resolve privileged obs
) -> torch.Tensor | None:
    """Flatten privileged observation group when present

    Steps:
    - Resolve inputs for `flatten_privileged_obs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    priv = obs.get("privileged") if isinstance(obs, Mapping) else None
    if priv is None:
        return None
    if not isinstance(priv, Mapping):
        raise TypeError("obs['privileged'] must be a mapping")
    missing = [key for key in privileged_obs_keys if key not in priv]
    if missing:
        raise KeyError(
            f"privileged observation missing keys {missing}; available={tuple(priv.keys())}"
        )
    return torch.cat([priv[key] for key in privileged_obs_keys], dim=-1)
