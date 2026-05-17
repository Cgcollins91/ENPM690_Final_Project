"""

Best eval checkpoint decision helpers

This module provides helper functions and data structures for comparing eval results to the current best,
deciding whether to save a new best checkpoint, and running the save job, used by the evaluation loop

File map:

BestCheckpointJob:               One best-checkpoint save request
BestCheckpointDecision:          Best-checkpoint comparison output for the training loop
build_best_checkpoint_decision:  Build a best-checkpoint save decision from eval summary
run_best_checkpoint_job:         Run an optional best-checkpoint save job
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from ..state.cadence import best_checkpoint_path
from .eval_metrics import (
    EvalComparison,
    EvalTaskKind,
    best_checkpoint_message,
    best_checkpoint_tiebreakers,
    compare_eval_states,
    eval_state_from_summary,
)


SaveBestCheckpointFn = Callable[["BestCheckpointJob"], None]


@dataclass(frozen=True)
class BestCheckpointJob:
    """One best-checkpoint save request"""

    global_step   : int   # Field: training step associated with this record or action
    dest_path     : str   # Field: destination path for the file being written or copied
    include_replay: bool  # Field: boolean value indicating the include replay state for best checkpoint job


@dataclass(frozen=True)
class BestCheckpointDecision:
    """Best-checkpoint comparison output for the training loop"""

    current_state  : dict[str, float | int]  # Field: integer current state value tracked by best checkpoint decision
    next_best_state: Mapping[str, float | int]  # Field: integer next best state value tracked by best checkpoint decision
    comparison     : EvalComparison  # Field: stores comparison for best checkpoint decision
    job            : BestCheckpointJob | None  # Field: integer job value tracked by best checkpoint decision
    message        : str | None  # Field: human-readable status or error detail

    @property
    def is_better(self) -> bool:
        """Return whether current eval became the new best"""
        return bool(self.comparison.is_better)


def build_best_checkpoint_decision(
    *,
    eval_summary             : Mapping[str, object],  # Param: string input for eval summary
    best_eval_state          : Mapping[str, float | int],  # Param: integer input for best eval state
    global_step              : int,  # Param: current absolute training step
    checkpoint_path          : str,  # Param: base checkpoint path used for scheduled save decisions
    task_kind                : EvalTaskKind,  # Param: input value used as task kind
    save_replay_in_checkpoint: bool,  # Param: whether checkpoint saves should include replay-buffer contents
) -> BestCheckpointDecision:
    """Build a best-checkpoint save decision from eval summary

    Steps:
    - Resolve inputs for `build_best_checkpoint_decision` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    current_state = eval_state_from_summary(eval_summary, global_step=int(global_step))
    comparison = compare_eval_states(
        current_state,
        best_eval_state,
        best_checkpoint_tiebreakers(task_kind),
    )
    if not comparison.is_better:
        return BestCheckpointDecision(
            current_state=current_state,
            next_best_state=best_eval_state,
            comparison=comparison,
            job=None,
            message=None,
        )

    dest_path = best_checkpoint_path(checkpoint_path)
    job = BestCheckpointJob(
        global_step=int(global_step),
        dest_path=dest_path,
        include_replay=bool(save_replay_in_checkpoint),
    )
    return BestCheckpointDecision(
        current_state=current_state,
        next_best_state=current_state,
        comparison=comparison,
        job=job,
        message=best_checkpoint_message(
            current_state,
            step=int(global_step),
            task_kind=task_kind,
            best_path=dest_path,
        ),
    )


def run_best_checkpoint_job(
    job               : BestCheckpointJob | None,  # Param: integer input for job
    save_checkpoint_fn: SaveBestCheckpointFn,  # Param: callback used to compute or fetch save checkpoint
) -> bool:
    """Run an optional best-checkpoint save job"""
    if job is None:
        return False
    save_checkpoint_fn(job)
    return True
