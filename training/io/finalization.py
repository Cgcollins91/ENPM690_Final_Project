"""

Final checkpoint and resource cleanup helpers

File map:

FinalCheckpointJob:            One final checkpoint save request
FinalCheckpointResult:         Outcome from best-effort final checkpoint saves
ResourceCloseResult:           Resource close flags for final trainer cleanup
final_checkpoint_global_step:  Return the monolith final checkpoint step
build_final_checkpoint_jobs:   Build final checkpoint save jobs in monolith order
trace_noop:                    Default trace sink for optional finalization tracing
run_final_checkpoint_jobs:     Run final checkpoint jobs while matching monolith best-effort saves
close_training_resources:      Close TensorBoard, env, and simulation app in monolith order
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..logging.tensorboard_setup import close_tensorboard_writer


TraceFn = Callable[[str], None]
SaveCheckpointFn = Callable[["FinalCheckpointJob"], None]


@dataclass(frozen=True)
class FinalCheckpointJob:
    """One final checkpoint save request"""

    label         : str  # string label value used by final checkpoint job
    global_step   : int  # training step associated with this record or action
    dest_path     : str | None  # destination path for the file being written or copied
    include_replay: bool  # boolean value indicating the include replay state for final checkpoint job


@dataclass(frozen=True)
class FinalCheckpointResult:
    """Outcome from best-effort final checkpoint saves"""

    saved_labels: tuple[str, ...]  # string saved labels value used by final checkpoint result
    error       : Exception | None = None  # stores error for final checkpoint result


@dataclass(frozen=True)
class ResourceCloseResult:
    """Resource close flags for final trainer cleanup"""

    tensorboard_closed: bool  # boolean value indicating the tensorboard closed state for resource close result
    env_closed        : bool  # boolean value indicating the env closed state for resource close result
    app_closed        : bool  # boolean value indicating the app closed state for resource close result


def final_checkpoint_global_step(transitions_collected: int) -> int:
    """Return the monolith final checkpoint step"""
    return max(0, int(transitions_collected) - 1)


def build_final_checkpoint_jobs(
    *,
    transitions_collected        : int,  # Param: integer input for transitions collected
    save_replay_in_checkpoint    : bool,  # Param: whether checkpoint saves should include replay-buffer contents
    replay_present               : bool,  # Param: boolean input controlling replay present
    final_handoff_checkpoint_path: str | None = None,  # Param: filesystem path for final handoff checkpoint
) -> tuple[FinalCheckpointJob, ...]:
    """Build final checkpoint save jobs in monolith order

    Steps:
    - Resolve inputs for `build_final_checkpoint_jobs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    global_step = final_checkpoint_global_step(transitions_collected)
    jobs = [
        FinalCheckpointJob(
            label="checkpoint",
            global_step=global_step,
            dest_path=None,
            include_replay=bool(save_replay_in_checkpoint) and bool(replay_present),
        )
    ]
    final_handoff_path = (final_handoff_checkpoint_path or "").strip()
    if final_handoff_path:
        if not replay_present:
            raise RuntimeError("--final-handoff-checkpoint-path requires replay buffer state")
        jobs.append(
            FinalCheckpointJob(
                label="final_handoff_checkpoint",
                global_step=global_step,
                dest_path=final_handoff_path,
                include_replay=True,
            )
        )
    return tuple(jobs)


def trace_noop(message: str) -> None:
    """Default trace sink for optional finalization tracing"""
    del message


def run_final_checkpoint_jobs(
    jobs              : tuple[FinalCheckpointJob, ...],  # Param: integer input for jobs
    save_checkpoint_fn: SaveCheckpointFn,  # Param: callback used to compute or fetch save checkpoint
    *,
    trace_fn: TraceFn = trace_noop,        # Param: callback used to compute or fetch trace
) -> FinalCheckpointResult:
    """Run final checkpoint jobs while matching monolith best-effort saves"""
    saved: list[str] = []
    try:
        for job in jobs:
            save_checkpoint_fn(job)
            saved.append(job.label)
            if job.label == "checkpoint":
                trace_fn("main_finally checkpoint_saved")
            elif job.label == "final_handoff_checkpoint":
                trace_fn("main_finally final_handoff_checkpoint_saved")
    except Exception as exc:
        trace_fn("main_finally checkpoint_save_failed")
        return FinalCheckpointResult(saved_labels=tuple(saved), error=exc)
    return FinalCheckpointResult(saved_labels=tuple(saved), error=None)


def close_training_resources(
    *,
    tensorboard_writer: Any,  # Param: input value used as tensorboard writer
    env               : Any,  # Param: environment or backend object used for runtime calls
    simulation_app    : Any,  # Param: input value used as simulation app
    trace_fn          : TraceFn = trace_noop,  # Param: callback used to compute or fetch trace
) -> ResourceCloseResult:
    """Close TensorBoard, env, and simulation app in monolith order

    Steps:
    - Resolve inputs for `close_training_resources` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    tensorboard_closed = False
    try:
        tensorboard_closed = close_tensorboard_writer(tensorboard_writer)
    except Exception:
        tensorboard_closed = False

    env.close()
    trace_fn("main_finally env_closed")
    simulation_app.close()
    trace_fn("main_finally app_closed")
    return ResourceCloseResult(
        tensorboard_closed=tensorboard_closed,
        env_closed=True,
        app_closed=True,
    )
