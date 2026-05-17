"""

Periodic checkpoint save and retention planning

File map:

ScheduledCheckpointJob:           One scheduled checkpoint save request
ScheduledCheckpointPlan:          Scheduled checkpoint jobs and advanced cadence state
build_scheduled_checkpoint_plan:  Build scheduled checkpoint jobs for one training loop step
run_scheduled_checkpoint_plan:    Run scheduled checkpoint saves and best-effort rolling prune
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

from ..state.cadence import (
    advance_periodic_step,
    rolling_checkpoint_name,
    rolling_checkpoint_path,
    rolling_checkpoint_paths_to_prune,
    step_due,
)


SaveScheduledCheckpointFn = Callable[["ScheduledCheckpointJob"], None]
RemovePathFn = Callable[[str], None]


@dataclass(frozen=True)
class ScheduledCheckpointJob:
    """One scheduled checkpoint save request"""

    label         : str  # string label value used by scheduled checkpoint job
    global_step   : int  # training step associated with this record or action
    dest_path     : str | None  # destination path for the file being written or copied
    include_replay: bool  # boolean value indicating the include replay state for scheduled checkpoint job


@dataclass(frozen=True)
class ScheduledCheckpointPlan:
    """Scheduled checkpoint jobs and advanced cadence state"""

    jobs                        : tuple[ScheduledCheckpointJob, ...]  # integer jobs value tracked by scheduled checkpoint plan
    next_checkpoint_step        : int | None  # step count used for next checkpoint step scheduling or reporting
    next_rolling_checkpoint_step: int | None  # step count used for next rolling checkpoint step scheduling or reporting
    prune_paths                 : tuple[str, ...]  # string prune paths value used by scheduled checkpoint plan


def build_scheduled_checkpoint_plan(
    *,
    global_step                 : int,  # Param: current absolute training step
    checkpoint_path             : str,  # Param: base checkpoint path used for scheduled save decisions
    save_replay_in_checkpoint   : bool,  # Param: whether checkpoint saves should include replay-buffer contents
    next_checkpoint_step        : int | None,  # Param: next global step for a regular checkpoint save
    checkpoint_every            : int,  # Param: global-step interval for regular checkpoint saves
    next_rolling_checkpoint_step: int | None,  # Param: next global step for a rolling checkpoint save
    rolling_checkpoint_every    : int,  # Param: global-step interval for rolling checkpoint saves
    rolling_checkpoint_keep     : int,  # Param: maximum number of rolling checkpoints kept after pruning
    existing_checkpoint_names   : tuple[str, ...] = (),  # Param: existing checkpoint filenames used to decide rolling-checkpoint pruning
) -> ScheduledCheckpointPlan:
    """Build scheduled checkpoint jobs for one training loop step

    Steps:
    - Resolve inputs for `build_scheduled_checkpoint_plan` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    jobs: list[ScheduledCheckpointJob] = []
    next_checkpoint = next_checkpoint_step
    next_rolling_checkpoint = next_rolling_checkpoint_step
    prune_paths: tuple[str, ...] = ()

    if step_due(next_checkpoint_step, global_step):
        jobs.append(
            ScheduledCheckpointJob(
                label="checkpoint",
                global_step=int(global_step),
                dest_path=None,
                include_replay=bool(save_replay_in_checkpoint),
            )
        )
        next_checkpoint = advance_periodic_step(next_checkpoint_step, checkpoint_every)

    if step_due(next_rolling_checkpoint_step, global_step):
        step_number = int(global_step) + 1
        rolling_name = rolling_checkpoint_name(step_number)
        jobs.append(
            ScheduledCheckpointJob(
                label="rolling_checkpoint",
                global_step=int(global_step),
                dest_path=rolling_checkpoint_path(checkpoint_path, step_number),
                include_replay=bool(save_replay_in_checkpoint),
            )
        )
        if int(rolling_checkpoint_keep) > 0:
            names_after_save = tuple(existing_checkpoint_names) + (rolling_name,)
            prune_paths = tuple(
                rolling_checkpoint_paths_to_prune(
                    checkpoint_path,
                    names_after_save,
                    int(rolling_checkpoint_keep),
                )
            )
        next_rolling_checkpoint = advance_periodic_step(
            next_rolling_checkpoint_step,
            rolling_checkpoint_every,
        )

    return ScheduledCheckpointPlan(
        jobs=tuple(jobs),
        next_checkpoint_step=next_checkpoint,
        next_rolling_checkpoint_step=next_rolling_checkpoint,
        prune_paths=prune_paths,
    )


def run_scheduled_checkpoint_plan(
    plan              : ScheduledCheckpointPlan,  # Param: precomputed plan object consumed by this helper
    save_checkpoint_fn: SaveScheduledCheckpointFn,  # Param: callback used to compute or fetch save checkpoint
    *,
    remove_path_fn: RemovePathFn = os.remove,       # Param: callback used to compute or fetch remove path
) -> None:
    """Run scheduled checkpoint saves and best-effort rolling prune"""
    for job in plan.jobs:
        save_checkpoint_fn(job)
    for path in plan.prune_paths:
        try:
            remove_path_fn(path)
        except OSError:
            pass
