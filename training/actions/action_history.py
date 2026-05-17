"""

Policy action history and handoff smoothing helpers

File map:

_valid_rows_for_action:                  Handle valid rows for action logic
stash_policy_level_action:               Record policy-level action history for reset-safe rate penalties
clear_policy_level_action_history:       Clear policy-level action validity after reset rows
set_contact_handoff_action_anchor:       Seed post-preroll action smoothing from a teacher action
apply_contact_handoff_action_smoothing:  Clamp abrupt policy jumps during the handoff smoothing window
"""

from __future__ import annotations

import torch


def _valid_rows_for_action(env, action: torch.Tensor, attr_name: str) -> torch.Tensor:
    valid = getattr(env, attr_name, None)
    if not torch.is_tensor(valid) or valid.shape != action.shape[:1]:
        return torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
    return valid.to(device=action.device, dtype=torch.bool)


def stash_policy_level_action(env, policy_level_action: torch.Tensor) -> None:
    """Record policy-level action history for reset-safe rate penalties

    Steps:
    - Resolve inputs for `stash_policy_level_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    action = policy_level_action.detach().clone()
    valid = _valid_rows_for_action(env, action, "_policy_level_action_valid")

    prev_action = getattr(env, "_policy_level_action", None)
    if not torch.is_tensor(prev_action) or prev_action.shape != action.shape:
        prev_action = action.clone()
    else:
        prev_action = prev_action.to(device=action.device, dtype=action.dtype)
        prev_action = torch.where(valid.unsqueeze(-1), prev_action, action)

    env._prev_policy_level_action = prev_action.clone()
    env._policy_level_action = action
    env._policy_level_action_valid = torch.ones_like(valid)


def clear_policy_level_action_history(env, env_ids: torch.Tensor | None = None) -> None:
    """Clear policy-level action validity after reset rows"""
    valid = getattr(env, "_policy_level_action_valid", None)
    if not torch.is_tensor(valid):
        return
    if env_ids is None:
        valid.zero_()
    elif env_ids.numel() > 0:
        valid[env_ids.to(device=valid.device)] = False
    env._policy_level_action_valid = valid


def set_contact_handoff_action_anchor(
    env,                                  # Param: environment or backend object used for runtime calls
    policy_level_action: torch.Tensor,    # Param: tensor input carrying policy level action values
    *,
    smooth_steps: int,  # Param: step count used for smooth steps
    env_ids     : torch.Tensor | None = None,  # Param: tensor input carrying env ids values
) -> bool:
    """Seed post-preroll action smoothing from a teacher action

    Steps:
    - Resolve inputs for `set_contact_handoff_action_anchor` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if int(smooth_steps) <= 0:
        return False
    action = policy_level_action.detach().clone()
    valid = _valid_rows_for_action(env, action, "_contact_handoff_action_valid")
    anchor = getattr(env, "_contact_handoff_action", None)
    if not torch.is_tensor(anchor) or anchor.shape != action.shape:
        anchor = action.clone()
    else:
        anchor = anchor.to(device=action.device, dtype=action.dtype)

    if env_ids is None:
        anchor = action.clone()
        valid = torch.ones_like(valid)
    elif env_ids.numel() > 0:
        ids = env_ids.to(device=action.device)
        anchor[ids] = action[ids]
        valid[ids] = True
    env._contact_handoff_action = anchor
    env._contact_handoff_action_valid = valid
    return True


def apply_contact_handoff_action_smoothing(
    env,                                      # Param: environment or backend object used for runtime calls
    policy_level_action: torch.Tensor,  # Param: tensor input carrying policy level action values
    episode_step       : torch.Tensor,  # Param: per-env step count inside the current episode
    *,
    smooth_steps: int,  # Param: step count used for smooth steps
    max_delta   : float,  # Param: floating-point input for max delta
    active_mask : torch.Tensor | None = None,  # Param: boolean mask selecting active rows
) -> torch.Tensor:
    """Clamp abrupt policy jumps during the handoff smoothing window

    Steps:
    - Resolve inputs for `apply_contact_handoff_action_smoothing` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if int(smooth_steps) <= 0 or float(max_delta) <= 0.0:
        return policy_level_action

    valid = getattr(env, "_contact_handoff_action_valid", None)
    anchor = getattr(env, "_contact_handoff_action", None)
    if (
        not torch.is_tensor(valid)
        or not torch.is_tensor(anchor)
        or anchor.shape != policy_level_action.shape
    ):
        return policy_level_action

    valid = valid.to(device=policy_level_action.device, dtype=torch.bool)
    anchor = anchor.to(device=policy_level_action.device, dtype=policy_level_action.dtype)
    step_t = episode_step.to(device=policy_level_action.device).reshape(-1)
    mask = valid & (step_t < int(smooth_steps))
    if active_mask is not None:
        mask = mask & active_mask.to(device=policy_level_action.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        valid = valid.clone()
        valid[step_t >= int(smooth_steps)] = False
        env._contact_handoff_action_valid = valid
        return policy_level_action

    smoothed = policy_level_action.clone()
    delta = torch.clamp(
        policy_level_action - anchor,
        min=-float(max_delta),
        max=float(max_delta),
    )
    smoothed[mask] = anchor[mask] + delta[mask]

    next_anchor = anchor.clone()
    next_anchor[mask] = smoothed[mask].detach()
    next_valid = valid.clone()
    next_valid[step_t >= int(smooth_steps)] = False
    env._contact_handoff_action = next_anchor
    env._contact_handoff_action_valid = next_valid
    return smoothed
