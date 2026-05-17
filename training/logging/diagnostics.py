"""

Reward and termination diagnostics read from env manager state

File map:

per_term_rewards:        Return per-reward-term values for one env row
per_term_reward_means:   Return per-reward-term means over env rows
termination_term_flags:  Return current boolean flags for a named termination term
"""

from __future__ import annotations

from typing import Any

import torch


def per_term_rewards(env: Any, env_id: int = 0) -> dict[str, float]:
    """Return per-reward-term values for one env row

    Steps:
    - Resolve inputs for `per_term_rewards` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    manager = getattr(env, "reward_manager", None)
    if manager is None:
        return {}
    names = getattr(manager, "_term_names", None)
    step_reward = getattr(manager, "_step_reward", None)
    if not names or step_reward is None:
        return {}
    try:
        values = step_reward[int(env_id)].detach().cpu().tolist()
    except (IndexError, RuntimeError, TypeError):
        return {}
    return {str(name): float(value) for name, value in zip(names, values)}


def per_term_reward_means(env: Any) -> dict[str, float]:
    """Return per-reward-term means over env rows

    Steps:
    - Resolve inputs for `per_term_reward_means` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    manager = getattr(env, "reward_manager", None)
    if manager is None:
        return {}
    names = getattr(manager, "_term_names", None)
    step_reward = getattr(manager, "_step_reward", None)
    if not names or step_reward is None:
        return {}
    try:
        means = step_reward.mean(dim=0).detach().cpu().tolist()
    except (IndexError, RuntimeError, TypeError):
        return {}
    return {str(name): float(value) for name, value in zip(names, means)}


def termination_term_flags(env: Any, term_name: str) -> torch.Tensor:
    """Return current boolean flags for a named termination term

    Steps:
    - Resolve inputs for `termination_term_flags` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    device = getattr(env, "device", "cpu")
    num_envs = int(getattr(env, "num_envs", 0))
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    active_terms = getattr(manager, "active_terms", ())
    if term_name not in active_terms:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    try:
        flags = manager.get_term(term_name)
    except (KeyError, RuntimeError, AttributeError):
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not torch.is_tensor(flags):
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return flags.to(device=device).bool()
