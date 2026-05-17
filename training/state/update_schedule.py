"""

TD update scheduling and result tracking helpers

File map:

UpdateReadiness:                 Update readiness for one outer env step
UpdateRunResult:                 Latest update metrics captured from update loop
global_step_after_replay_flush:  Return monolith global step after replay flush
actor_loss_metric_present:       Return whether update metrics include a non-nan actor loss
update_readiness:                Resolve whether TD updates should run after one env step
run_update_steps:                Run TD update steps and track latest actor-bearing update
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any


UpdateFn = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class UpdateReadiness:
    """Update readiness for one outer env step"""

    updates_ready: bool  # boolean/tensor readiness state for updates
    should_update: bool  # boolean value indicating the should update state for update readiness
    update_count : int  # count of update values


@dataclass(frozen=True)
class UpdateRunResult:
    """Latest update metrics captured from update loop"""

    update_count          : int  # count of update values
    last_update_info      : Mapping[str, Any] | None  # string last update info value used by update run result
    last_actor_update_info: Mapping[str, Any] | None  # string last actor update info value used by update run result


def global_step_after_replay_flush(transitions_collected: int, num_added: int) -> int:
    """Return monolith global step after replay flush"""
    collected = int(transitions_collected)
    added = int(num_added)
    return collected - 1 if added > 0 else collected


def actor_loss_metric_present(update_info: Mapping[str, Any]) -> bool:
    """Return whether update metrics include a non-nan actor loss"""
    value = update_info.get("actor_loss", math.nan)
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def update_readiness(
    *,
    transitions_collected: int,  # Param: integer input for transitions collected
    start_steps          : int,  # Param: step count used for start steps
    replay_size          : int,  # Param: number of transitions currently available in replay
    batch_size           : int,  # Param: number of replay samples required for one update batch
    num_added            : int,  # Param: number of env transitions added during the current collection step
    updates_per_step     : int,  # Param: step count used for updates per step
) -> UpdateReadiness:
    """Resolve whether TD updates should run after one env step"""
    ready = int(transitions_collected) >= int(start_steps)
    should_update = (
        ready
        and int(replay_size) >= int(batch_size)
        and int(num_added) > 0
        and int(updates_per_step) > 0
    )
    return UpdateReadiness(
        updates_ready=ready,
        should_update=should_update,
        update_count=int(updates_per_step) if should_update else 0,
    )


def run_update_steps(
    *,
    update_fn                 : UpdateFn,  # Param: callback used to compute or fetch update
    update_count              : int,  # Param: count of update
    previous_actor_update_info: Mapping[str, Any] | None = None,  # Param: previously observed actor-bearing update info
) -> UpdateRunResult:
    """Run TD update steps and track latest actor-bearing update

    Steps:
    - Resolve inputs for `run_update_steps` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    last_update_info      : Mapping[str, Any] | None = None
    last_actor_update_info: Mapping[str, Any] | None = previous_actor_update_info
    count = max(0, int(update_count))
    for _ in range(count):
        update_info = update_fn()
        last_update_info = update_info
        if actor_loss_metric_present(update_info):
            last_actor_update_info = update_info
    return UpdateRunResult(
        update_count=count,
        last_update_info=last_update_info,
        last_actor_update_info=last_actor_update_info,
    )
