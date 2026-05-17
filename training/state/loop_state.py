"""

Small rollout loop state helpers

File map:

DoneState:             Done row selection for one vectorized step
done_env_ids:          Return flat done env ids
active_done_env_ids:   Return done env ids filtered by active env mask
summarize_done_state:  Summarize done and active-done state for one vectorized step
active_env_count:      Return count of active env rows
advance_step_count:    Return advanced global step count
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DoneState:
    """Done row selection for one vectorized step"""

    done_ids         : torch.Tensor  # Field: tensor containing done ids values for batched env rows
    active_done_ids  : torch.Tensor  # Field: tensor containing active done ids values for batched env rows
    done_count       : int  # Field: count of done values
    active_done_count: int  # Field: count of active done values
    env0_done        : bool  # Field: boolean value indicating the env0 done state for done state
    env0_active      : bool  # Field: boolean state indicating whether env0 is active


def done_env_ids(done_flags: torch.Tensor) -> torch.Tensor:
    """Return flat done env ids"""
    return torch.nonzero(done_flags.to(dtype=torch.bool), as_tuple=False).squeeze(-1)


def active_done_env_ids(done_flags: torch.Tensor, active_env_mask: torch.Tensor) -> torch.Tensor:
    """Return done env ids filtered by active env mask"""
    ids = done_env_ids(done_flags)
    if ids.numel() == 0:
        return ids
    active = active_env_mask.to(device=done_flags.device, dtype=torch.bool)
    return ids[active[ids]]


def summarize_done_state(done_flags: torch.Tensor, active_env_mask: torch.Tensor) -> DoneState:
    """Summarize done and active-done state for one vectorized step

    Steps:
    - Resolve inputs for `summarize_done_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    done = done_flags.to(dtype=torch.bool).reshape(-1)
    active = active_env_mask.to(device=done.device, dtype=torch.bool).reshape(-1)
    ids = done_env_ids(done)
    active_ids = ids[active[ids]] if ids.numel() > 0 else ids
    return DoneState(
        done_ids=ids,
        active_done_ids=active_ids,
        done_count=int(ids.numel()),
        active_done_count=int(active_ids.numel()),
        env0_done=bool(done[0].item()) if done.numel() > 0 else False,
        env0_active=bool(active[0].item()) if active.numel() > 0 else False,
    )


def active_env_count(active_env_mask: torch.Tensor) -> int:
    """Return count of active env rows"""
    return int(active_env_mask.to(dtype=torch.bool).sum().item())


def advance_step_count(global_step: int, num_added: int) -> int:
    """Return advanced global step count"""
    return int(global_step) + int(num_added)
