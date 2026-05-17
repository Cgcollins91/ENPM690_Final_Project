"""

Native rollout stat and TD update callbacks

File map:

NativeUpdateAgent:            Agent surface needed by native update helpers
NativeRolloutStatUpdate:      Rows used for rollout normalization stats
NativeUpdateRequest:          Inputs for one native TD update call
_active_rows:                 Handle active rows logic
update_native_rollout_stats:  Update agent normalization stats from active rollout rows
run_native_td_update:         Run one native TD update and return a plain dict
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..state.replay_batches import ReplayAddTarget


class NativeUpdateAgent(Protocol):
    """Agent surface needed by native update helpers"""

    def update_obs_stats(self, obs: torch.Tensor) -> None:
        """Update observation normalization stats"""
        ...

    def update_priv_obs_stats(self, priv_obs: torch.Tensor | None) -> None:
        """Update privileged observation normalization stats"""
        ...

    def update_reward_stats(self, reward: torch.Tensor) -> None:
        """Update reward normalization stats"""
        ...

    def update(
        self,
        replay       : ReplayAddTarget,  # Param: replay buffer or replay target used for transition storage
        batch_size   : int,  # Param: number of replay samples required for one update batch
        progress_step: int | None = None,  # Param: step count used for progress step
    ) -> Mapping[str, Any]:
        """Run one TD update"""
        ...


@dataclass(frozen=True)
class NativeRolloutStatUpdate:
    """Rows used for rollout normalization stats"""

    obs_tensor     : torch.Tensor  # policy observation tensor passed to the actor or replay path
    priv_obs_tensor: torch.Tensor | None  # privileged observation tensor passed to critic-side logic
    reward_tensor  : torch.Tensor  # tensor containing reward tensor values for batched env rows
    active_env_mask: torch.Tensor | None = None  # mask selecting env rows that are still active


@dataclass(frozen=True)
class NativeUpdateRequest:
    """Inputs for one native TD update call"""

    agent        : NativeUpdateAgent  # stores agent for native update request
    replay       : ReplayAddTarget  # stores replay for native update request
    batch_size   : int  # number of replay samples used in each update batch
    progress_step: int | None = None  # step count used for progress step scheduling or reporting


def _active_rows(tensor: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if mask is None:
        return tensor
    return tensor[mask.to(device=tensor.device, dtype=torch.bool).reshape(-1)]


def update_native_rollout_stats(
    agent : NativeUpdateAgent,  # Param: TD3 agent whose networks, optimizers, or stats are used
    update: NativeRolloutStatUpdate,  # Param: input value used as update
) -> int:
    """Update agent normalization stats from active rollout rows

    Steps:
    - Resolve inputs for `update_native_rollout_stats` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    obs_rows = _active_rows(update.obs_tensor, update.active_env_mask)
    reward_rows = _active_rows(update.reward_tensor, update.active_env_mask)
    priv_rows = _active_rows(update.priv_obs_tensor, update.active_env_mask)
    if obs_rows is None or reward_rows is None:
        return 0
    row_count = int(obs_rows.shape[0])
    if row_count <= 0:
        return 0
    agent.update_obs_stats(obs_rows)
    agent.update_reward_stats(reward_rows)
    if priv_rows is not None:
        agent.update_priv_obs_stats(priv_rows)
    return row_count


def run_native_td_update(request: NativeUpdateRequest) -> dict[str, Any]:
    """Run one native TD update and return a plain dict"""
    update_info = request.agent.update(
        request.replay,
        int(request.batch_size),
        progress_step=request.progress_step,
    )
    return dict(update_info)
