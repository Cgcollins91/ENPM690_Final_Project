"""

Trainer-side teacher action cache helpers

File map:

episode_step_tensor:                   Return episode step counts as a float tensor
cached_teacher_action:                 Return a cached teacher action when the episode step matches exactly
store_cached_teacher_action:           Store a detached teacher action for the current episode step
clear_cached_teacher_action:           Invalidate cached trainer-side teacher action
get_or_compute_cached_teacher_action:  Return cached teacher action or compute and store it
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def episode_step_tensor(
    episode_step: int | torch.Tensor,  # Param: per-env step count inside the current episode
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> torch.Tensor:
    """Return episode step counts as a float tensor"""
    if torch.is_tensor(episode_step):
        return episode_step.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.full((int(num_envs),), float(episode_step), device=device, dtype=torch.float32)


def cached_teacher_action(
    owner       : Any,  # Param: input value used as owner
    episode_step: int | torch.Tensor,  # Param: per-env step count inside the current episode
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    enabled : bool = True,  # Param: boolean input controlling enabled
) -> torch.Tensor | None:
    """Return a cached teacher action when the episode step matches exactly

    Steps:
    - Resolve inputs for `cached_teacher_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not enabled:
        return None
    action = getattr(owner, "_trainer_cached_teacher_action", None)
    step = getattr(owner, "_trainer_cached_teacher_episode_step", None)
    if not torch.is_tensor(action) or not torch.is_tensor(step):
        return None
    expected_step = episode_step_tensor(episode_step, num_envs=num_envs, device=device)
    if tuple(step.shape) != tuple(expected_step.shape):
        return None
    if int(action.shape[0]) != int(num_envs):
        return None
    if not bool(torch.equal(step.to(device=device), expected_step)):
        return None
    return action.to(device=device).clone()


def store_cached_teacher_action(
    owner       : Any,  # Param: input value used as owner
    episode_step: int | torch.Tensor,  # Param: per-env step count inside the current episode
    action      : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    enabled : bool = True,  # Param: boolean input controlling enabled
) -> None:
    """Store a detached teacher action for the current episode step"""
    if not enabled:
        return
    owner._trainer_cached_teacher_action = action.detach().clone()
    owner._trainer_cached_teacher_episode_step = episode_step_tensor(
        episode_step,
        num_envs=num_envs,
        device=device,
    ).detach().clone()


def clear_cached_teacher_action(owner: Any) -> None:
    """Invalidate cached trainer-side teacher action"""
    owner._trainer_cached_teacher_action = None
    owner._trainer_cached_teacher_episode_step = None


def get_or_compute_cached_teacher_action(
    owner         : Any,  # Param: input value used as owner
    episode_step  : int | torch.Tensor,  # Param: per-env step count inside the current episode
    compute_action: Callable[[], torch.Tensor],  # Param: callback used to compute or fetch compute action
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    enabled : bool = True,  # Param: boolean input controlling enabled
) -> torch.Tensor:
    """Return cached teacher action or compute and store it

    Steps:
    - Resolve inputs for `get_or_compute_cached_teacher_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    cached = cached_teacher_action(
        owner,
        episode_step,
        num_envs=num_envs,
        device=device,
        enabled=enabled,
    )
    if cached is not None:
        return cached
    with torch.no_grad():
        action = compute_action()
    store_cached_teacher_action(
        owner,
        episode_step,
        action,
        num_envs=num_envs,
        device=device,
        enabled=enabled,
    )
    return action
