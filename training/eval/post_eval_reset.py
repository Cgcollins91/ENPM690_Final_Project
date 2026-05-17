"""

Post-eval training reset helpers

File map:

PostEvalResetResult:             State returned after resetting training from eval
advance_next_eval_step:          Advance eval cadence after a completed eval
reset_all_contact_preroll_rows:  Reset all contact pre-roll rows after eval reset
reset_after_eval:                Reset env and trainer-owned state after eval rollout
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..state.episodes import EpisodeState


EnvResetFn = Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]
ClearTouchCountersFn = Callable[[object], object]
FlattenPolicyObsFn = Callable[[Mapping[str, torch.Tensor], tuple[str, ...]], torch.Tensor]
RefreshPrivilegedFn = Callable[[torch.Tensor], object]
FlattenPrivilegedObsFn = Callable[[], torch.Tensor | None]
SampleAllPrerollMaskFn = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class PostEvalResetResult:
    """State returned after resetting training from eval"""

    obs            : Mapping[str, Any]  # policy observation tensor or observation payload for this transition
    info           : Mapping[str, Any]  # auxiliary info mapping returned by the environment or backend
    obs_tensor     : torch.Tensor  # policy observation tensor passed to the actor or replay path
    priv_obs_tensor: torch.Tensor | None  # privileged observation tensor passed to critic-side logic
    next_eval_step : int | None  # step count used for next eval step scheduling or reporting


def advance_next_eval_step(next_eval_step: int | None, eval_every: int | None) -> int | None:
    """Advance eval cadence after a completed eval"""
    if next_eval_step is None or eval_every is None or int(eval_every) <= 0:
        return None
    return int(next_eval_step) + int(eval_every)


def reset_all_contact_preroll_rows(
    *,
    contact_preroll_active        : torch.Tensor,  # Param: mask or boolean input marking contact preroll as active
    contact_preroll_steps         : torch.Tensor,  # Param: step count used for contact preroll steps
    legacy_contact_preroll_enabled: bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled       : bool,  # Param: enables topdown-preroll reset handling
    sample_topdown_preroll_mask   : SampleAllPrerollMaskFn | None = None,  # Param: boolean mask selecting sample topdown preroll rows
) -> None:
    """Reset all contact pre-roll rows after eval reset"""
    contact_preroll_active[:] = False
    if legacy_contact_preroll_enabled:
        contact_preroll_active[:] = True
    elif topdown_preroll_enabled:
        if sample_topdown_preroll_mask is None:
            raise RuntimeError("topdown pre-roll reset requires a sampler")
        contact_preroll_active[:] = sample_topdown_preroll_mask().to(
            device=contact_preroll_active.device,
            dtype=torch.bool,
        )
    contact_preroll_steps.zero_()


def reset_after_eval(
    *,
    env                                : object,  # Param: environment or backend object used for runtime calls
    reset_env_fn                       : EnvResetFn,  # Param: callback used to compute or fetch reset env
    clear_touch_counters_fn            : ClearTouchCountersFn,  # Param: callback used to compute or fetch clear touch counters
    flatten_policy_obs_fn              : FlattenPolicyObsFn,  # Param: callback used to compute or fetch flatten policy obs
    obs_keys                           : tuple[str, ...],  # Param: ordered mapping keys used to resolve obs
    episode_state                      : EpisodeState,  # Param: input value used as episode state
    refresh_privileged_teacher_state_fn: RefreshPrivilegedFn,  # Param: callback used to compute or fetch refresh privileged teacher state
    privileged_critic                  : bool,  # Param: boolean input controlling privileged critic
    flatten_privileged_obs_fn          : FlattenPrivilegedObsFn,  # Param: callback used to compute or fetch flatten privileged obs
    contact_preroll_active             : torch.Tensor,  # Param: mask or boolean input marking contact preroll as active
    contact_preroll_steps              : torch.Tensor,  # Param: step count used for contact preroll steps
    legacy_contact_preroll_enabled     : bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled            : bool,  # Param: enables topdown-preroll reset handling
    sample_topdown_preroll_mask        : SampleAllPrerollMaskFn | None,  # Param: boolean mask selecting sample topdown preroll rows
    next_eval_step                     : int | None,  # Param: next global step that should trigger evaluation, or None when eval is disabled
    eval_every                         : int | None,  # Param: interval controlling how often eval runs
) -> PostEvalResetResult:
    """Reset env and trainer-owned state after eval rollout

    Steps:
    - Resolve inputs for `reset_after_eval` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    obs, info = reset_env_fn()
    clear_touch_counters_fn(env)
    policy_obs = obs.get("policy")
    if not isinstance(policy_obs, Mapping):
        raise RuntimeError("reset obs must contain a policy mapping")
    obs_tensor = flatten_policy_obs_fn(policy_obs, obs_keys)
    episode_state.reset_all_with_new_ids()
    refresh_privileged_teacher_state_fn(obs_tensor)
    if privileged_critic:
        priv_obs_tensor = flatten_privileged_obs_fn()
        if priv_obs_tensor is None:
            raise RuntimeError("--privileged-critic is enabled but reset obs['privileged'] is unavailable")
    else:
        priv_obs_tensor = None
    reset_all_contact_preroll_rows(
        contact_preroll_active=contact_preroll_active,
        contact_preroll_steps=contact_preroll_steps,
        legacy_contact_preroll_enabled=legacy_contact_preroll_enabled,
        topdown_preroll_enabled=topdown_preroll_enabled,
        sample_topdown_preroll_mask=sample_topdown_preroll_mask,
    )
    return PostEvalResetResult(
        obs=obs,
        info=info,
        obs_tensor=obs_tensor,
        priv_obs_tensor=priv_obs_tensor,
        next_eval_step=advance_next_eval_step(next_eval_step, eval_every),
    )
