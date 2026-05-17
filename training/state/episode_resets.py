"""

Done-row episode reset orchestration helpers

File map:

DoneEpisodeResetPlan:           Done rows and active done rows for reset bookkeeping
build_done_episode_reset_plan:  Build done-row reset plan from loop masks
clear_n_step_queues_for_done:   Clear n-step queues for done env ids
reset_contact_preroll_rows:     Reset contact pre-roll state for done rows
apply_done_episode_resets:      Apply monolith done-row reset side effects
"""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Callable

import torch

from .episodes import EpisodeState
from .loop_state import active_done_env_ids, done_env_ids


ClearTouchCountersFn = Callable[[object, torch.Tensor], object]
SamplePrerollMaskFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DoneEpisodeResetPlan:
    """Done rows and active done rows for reset bookkeeping"""

    done_ids                      : torch.Tensor  # Field: tensor containing done ids values for batched env rows
    active_done_ids               : torch.Tensor  # Field: tensor containing active done ids values for batched env rows
    legacy_contact_preroll_enabled: bool  # Field: boolean state indicating whether legacy contact preroll is enabled
    topdown_preroll_enabled       : bool  # Field: boolean state indicating whether topdown preroll is enabled
    env0_done                     : bool = False  # Field: boolean value indicating the env0 done state for done episode reset plan
    env0_active                   : bool = False  # Field: boolean state indicating whether env0 is active

    @property
    def has_done(self) -> bool:
        """Return whether any env row is done"""
        return bool(self.done_ids.numel() > 0)


def build_done_episode_reset_plan(
    done_flags     : torch.Tensor,  # Param: per-env done flags returned by the latest env step
    active_env_mask: torch.Tensor,  # Param: mask selecting env rows still active for collection or reset handling
    *,
    legacy_contact_preroll_enabled: bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled       : bool,  # Param: enables topdown-preroll reset handling
) -> DoneEpisodeResetPlan:
    """Build done-row reset plan from loop masks"""
    done = done_flags.to(dtype=torch.bool).reshape(-1)
    active = active_env_mask.to(device=done.device, dtype=torch.bool).reshape(-1)
    return DoneEpisodeResetPlan(
        done_ids=done_env_ids(done),
        active_done_ids=active_done_env_ids(done, active),
        legacy_contact_preroll_enabled=bool(legacy_contact_preroll_enabled),
        topdown_preroll_enabled=bool(topdown_preroll_enabled),
        env0_done=bool(done[0].item()) if done.numel() > 0 else False,
        env0_active=bool(active[0].item()) if active.numel() > 0 else False,
    )


def clear_n_step_queues_for_done(
    n_step_queues: MutableSequence[object],  # Param: ordered input collection of n step queues entries
    done_ids     : torch.Tensor,  # Param: tensor input carrying done ids values
) -> tuple[int, ...]:
    """Clear n-step queues for done env ids"""
    cleared: list[int] = []
    for env_id in done_ids.to(dtype=torch.long).tolist():
        queue = n_step_queues[int(env_id)]
        if hasattr(queue, "clear"):
            queue.clear()
        cleared.append(int(env_id))
    return tuple(cleared)


def reset_contact_preroll_rows(
    *,
    done_ids                      : torch.Tensor,  # Param: tensor input carrying done ids values
    contact_preroll_active        : torch.Tensor,  # Param: mask or boolean input marking contact preroll as active
    contact_preroll_steps         : torch.Tensor,  # Param: step count used for contact preroll steps
    legacy_contact_preroll_enabled: bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled       : bool,  # Param: enables topdown-preroll reset handling
    sample_topdown_preroll_mask   : SamplePrerollMaskFn | None = None,  # Param: boolean mask selecting sample topdown preroll rows
) -> bool:
    """Reset contact pre-roll state for done rows

    Steps:
    - Resolve inputs for `reset_contact_preroll_rows` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ids = done_ids.to(device=contact_preroll_active.device, dtype=torch.long)
    if ids.numel() == 0:
        return False
    if legacy_contact_preroll_enabled:
        contact_preroll_active[ids] = True
        contact_preroll_steps[ids] = 0
        return True
    if topdown_preroll_enabled:
        if sample_topdown_preroll_mask is None:
            raise RuntimeError("topdown pre-roll reset requires a sampler")
        sampled = sample_topdown_preroll_mask(ids).to(
            device=contact_preroll_active.device,
            dtype=torch.bool,
        )
        contact_preroll_active[ids] = sampled
        contact_preroll_steps[ids] = 0
        return True
    return False


def apply_done_episode_resets(
    *,
    episode_state              : EpisodeState,  # Param: input value used as episode state
    reset_plan                 : DoneEpisodeResetPlan,  # Param: input value used as reset plan
    env                        : object,  # Param: environment or backend object used for runtime calls
    clear_touch_counters_fn    : ClearTouchCountersFn,  # Param: callback used to compute or fetch clear touch counters
    n_step_queues              : MutableSequence[object],  # Param: ordered input collection of n step queues entries
    contact_preroll_active     : torch.Tensor,  # Param: mask or boolean input marking contact preroll as active
    contact_preroll_steps      : torch.Tensor,  # Param: step count used for contact preroll steps
    sample_topdown_preroll_mask: SamplePrerollMaskFn | None = None,  # Param: boolean mask selecting sample topdown preroll rows
) -> tuple[int, ...]:
    """Apply monolith done-row reset side effects

    Steps:
    - Resolve inputs for `apply_done_episode_resets` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not reset_plan.has_done:
        return ()
    episode_state.reset_metrics(reset_plan.done_ids)
    clear_touch_counters_fn(env, reset_plan.done_ids)
    cleared = clear_n_step_queues_for_done(n_step_queues, reset_plan.done_ids)
    reset_contact_preroll_rows(
        done_ids=reset_plan.done_ids,
        contact_preroll_active=contact_preroll_active,
        contact_preroll_steps=contact_preroll_steps,
        legacy_contact_preroll_enabled=reset_plan.legacy_contact_preroll_enabled,
        topdown_preroll_enabled=reset_plan.topdown_preroll_enabled,
        sample_topdown_preroll_mask=sample_topdown_preroll_mask,
    )
    episode_state.assign_new_episode_ids(reset_plan.active_done_ids)
    return cleared
