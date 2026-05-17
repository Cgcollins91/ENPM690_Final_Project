"""

Terminal observation helpers for timeout replay rows

File map:

TerminalNextObservation:                Replay next-observation tensors after terminal row substitution
clone_observation_tree:                 Clone nested observation tensors while preserving container shape
terminal_observation_from_info:         Return terminal observation payload from an env info mapping
_timeout_mask_for_tensor:               Handle timeout mask for tensor logic
substitute_timeout_rows:                Return live_next with timeout rows replaced by terminal_next
substitute_terminal_next_observations:  Substitute pre-reset terminal observations on timeout replay rows
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .observations import flatten_policy_obs, flatten_privileged_obs


@dataclass(frozen=True)
class TerminalNextObservation:
    """Replay next-observation tensors after terminal row substitution"""

    next_obs           : torch.Tensor  # next policy observation tensor after the transition step
    next_privileged_obs: torch.Tensor | None  # tensor containing next privileged obs values for batched env rows
    applied            : bool  # boolean value indicating the applied state for terminal next observation


def clone_observation_tree(obs):
    """Clone nested observation tensors while preserving container shape

    Steps:
    - Resolve inputs for `clone_observation_tree` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if torch.is_tensor(obs):
        return obs.clone()
    if isinstance(obs, Mapping):
        return {key: clone_observation_tree(value) for key, value in obs.items()}
    if isinstance(obs, tuple):
        return tuple(clone_observation_tree(value) for value in obs)
    if isinstance(obs, list):
        return [clone_observation_tree(value) for value in obs]
    return obs


def terminal_observation_from_info(info: object) -> object | None:
    """Return terminal observation payload from an env info mapping"""
    if isinstance(info, Mapping):
        return info.get("terminal_observation")
    return None


def _timeout_mask_for_tensor(timeout_flags: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = timeout_flags.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 0:
        mask = mask.reshape(1)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def substitute_timeout_rows(
    live_next    : torch.Tensor,  # Param: tensor input carrying live next values
    terminal_next: torch.Tensor,  # Param: tensor input carrying terminal next values
    timeout_flags: torch.Tensor,  # Param: per-env timeout flags returned by the latest env step
) -> torch.Tensor:
    """Return live_next with timeout rows replaced by terminal_next"""
    if live_next.shape != terminal_next.shape:
        raise ValueError(
            f"terminal tensor shape mismatch: live={tuple(live_next.shape)} "
            f"terminal={tuple(terminal_next.shape)}"
        )
    mask = _timeout_mask_for_tensor(timeout_flags, live_next)
    terminal = terminal_next.to(device=live_next.device, dtype=live_next.dtype)
    return torch.where(mask, terminal, live_next)


def substitute_terminal_next_observations(
    *,
    live_next_obs           : torch.Tensor,  # Param: tensor input carrying live next obs values
    live_next_privileged_obs: torch.Tensor | None,  # Param: tensor input carrying live next privileged obs values
    terminal_obs            : object | None,  # Param: input value used as terminal obs
    timeout_flags           : torch.Tensor,  # Param: per-env timeout flags returned by the latest env step
    obs_keys                : Sequence[str],  # Param: ordered mapping keys used to resolve obs
    privileged_critic       : bool,  # Param: boolean input controlling privileged critic
) -> TerminalNextObservation:
    """Substitute pre-reset terminal observations on timeout replay rows

    Steps:
    - Resolve inputs for `substitute_terminal_next_observations` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if terminal_obs is None or not bool(timeout_flags.any().item()):
        return TerminalNextObservation(
            next_obs=live_next_obs,
            next_privileged_obs=live_next_privileged_obs,
            applied=False,
        )
    if not isinstance(terminal_obs, Mapping):
        raise TypeError("terminal_observation must be a mapping")
    policy_obs = terminal_obs.get("policy")
    if not isinstance(policy_obs, Mapping):
        raise KeyError("terminal_observation is missing policy observations")

    terminal_obs_tensor = flatten_policy_obs(policy_obs, obs_keys)
    next_obs = substitute_timeout_rows(live_next_obs, terminal_obs_tensor, timeout_flags)
    next_privileged_obs = live_next_privileged_obs

    if privileged_critic and live_next_privileged_obs is not None:
        terminal_privileged_tensor = flatten_privileged_obs(terminal_obs)
        if terminal_privileged_tensor is not None:
            next_privileged_obs = substitute_timeout_rows(
                live_next_privileged_obs,
                terminal_privileged_tensor,
                timeout_flags,
            )

    return TerminalNextObservation(
        next_obs=next_obs,
        next_privileged_obs=next_privileged_obs,
        applied=True,
    )
