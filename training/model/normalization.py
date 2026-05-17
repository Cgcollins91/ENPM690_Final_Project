"""

Running normalization helpers for training modules

This module provides helper functions and data structures for tracking running normalization statistics, used by the training loop and agent state

File map:

RunningScalarMoments:               Track running mean and variance for a scalar stream
RunningTensorMoments:               Track running mean and variance for vector observations
reset_obs_stats_for_actor_rollout:  Clear observation stats when switching to actor-driven data
"""

from __future__ import annotations

import math

import torch


class RunningScalarMoments:
    """Track running mean and variance for a scalar stream"""

    def __init__(self, eps_count: float = 1e-4):
        self.count = float(eps_count)
        self.mean = 0.0
        self.var = 1.0

    def update(self, values: torch.Tensor) -> None:
        """Update running state from a value batch

        Steps:
        - Resolve inputs for `update` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """

        flat = values.detach().reshape(-1).float()
        if flat.numel() == 0:
            return
        batch_mean  = float(flat.mean().item())
        batch_var   = float(flat.var(unbiased=False).item()) if flat.numel() > 1 else 0.0
        batch_count = float(flat.numel())

        delta    = batch_mean - self.mean                   # difference between batch mean and current mean
        total    = self.count + batch_count                 # total count after adding the batch
        new_mean = self.mean + delta * batch_count / total  # updated mean
        m_a      = self.var * self.count                    # total variance contribution from existing data
        m_b      = batch_var * batch_count                 # total variance contribution from the new batch
        m2 = m_a + m_b + (delta * delta) * self.count * batch_count / total

        self.mean = new_mean
        self.var = max(m2 / total, 1e-8)
        self.count = total

    @property
    def std(self) -> float:
        """Return guarded running standard deviation"""
        return math.sqrt(max(self.var, 1e-8))

    def state_dict(self) -> dict[str, float]:
        """Serialize runtime state"""
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state: dict) -> None:
        """Restore runtime state"""
        self.count = float(state.get("count", self.count))
        self.mean = float(state.get("mean", self.mean))
        self.var = max(float(state.get("var", self.var)), 1e-8)


class RunningTensorMoments:
    """Track running mean and variance for vector observations"""

    def __init__(self, shape: tuple[int, ...], eps_count: float = 1e-4):
        self.shape = tuple(shape)
        self.count = torch.full(self.shape, float(eps_count), dtype=torch.float32)
        self.mean = torch.zeros(self.shape, dtype=torch.float32)
        self.var = torch.ones(self.shape, dtype=torch.float32)

    def update(self, values: torch.Tensor) -> None:
        """Update running state from a value batch

        Steps:
        - Resolve inputs for `update` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        flat = values.detach().reshape(-1, *self.shape).float().cpu()
        if flat.numel() == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False) if flat.shape[0] > 1 else torch.zeros_like(self.mean)
        batch_count = float(flat.shape[0])
        batch_count_t = torch.full_like(self.count, batch_count)

        delta = batch_mean - self.mean
        total = self.count + batch_count_t
        new_mean = self.mean + delta * (batch_count_t / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count_t
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count_t / total)

        self.mean = new_mean
        self.var = torch.clamp(m2 / total, min=1e-8)
        self.count = total

    def std(self, device: torch.device | str) -> torch.Tensor:
        """Return guarded running standard deviation"""
        return torch.sqrt(self.var.to(device=device).clamp(min=1e-8))

    def mean_tensor(self, device: torch.device | str) -> torch.Tensor:
        """Return running mean on the requested device"""
        return self.mean.to(device=device)

    def rms(self) -> float:
        """Return root mean square statistic"""
        return float(torch.sqrt(self.var.mean()).item())

    def state_dict(self) -> dict[str, object]:
        """Serialize runtime state"""
        return {
            "count": self.count.clone(),
            "mean" : self.mean.clone(),
            "var"  : self.var.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore runtime state"""
        raw_count = state.get("count", self.count)
        if torch.is_tensor(raw_count):
            count_tensor = raw_count.detach().float().cpu()
            if tuple(count_tensor.shape) == self.shape:
                self.count = count_tensor.clone()
            else:
                scalar = float(count_tensor.reshape(-1)[0].item())
                self.count = torch.full(self.shape, scalar, dtype=torch.float32)
        else:
            self.count = torch.full(self.shape, float(raw_count), dtype=torch.float32)
        if "mean" in state:
            self.mean = state["mean"].detach().float().cpu()
        if "var" in state:
            self.var = state["var"].detach().float().cpu().clamp(min=1e-8)


def reset_obs_stats_for_actor_rollout(agent, reason: str) -> None:
    """Clear observation stats when switching to actor-driven data"""
    shape = tuple(agent.obs_stats.mean.shape)
    agent.obs_stats = RunningTensorMoments(shape)
    if getattr(agent, "priv_obs_dim", 0) > 0:
        priv_shape = tuple(agent.priv_obs_stats.mean.shape)
        agent.priv_obs_stats = RunningTensorMoments(priv_shape)
    print(
        f"{reason}: reset obs_stats; observation normalization will rebuild from actor-driven trajectories",
        flush=True,
    )
