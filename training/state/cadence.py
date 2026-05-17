"""

Step cadence helpers for eval and checkpoint scheduling

cadence.py contains logic for calculating the next scheduled steps for evaluation and checkpointing during training,
based on configurable intervals and the number of transitions collected.
It includes utility functions for managing checkpoint paths and pruning old checkpoints according to a retention policy

File map:

CadenceState:                       Next scheduled eval and checkpoint steps
next_periodic_step:                 Return the first scheduled step strictly after current_count
next_eval_step:                     Return the next eval step after a resume or launch
initial_cadence_state:              Build initial step schedules for eval and checkpoints
initial_training_cadence:           Build initial trainer cadence with the public migration name
step_due:                           Return whether global_step has reached a scheduled step
advance_periodic_step:              Advance an existing schedule by one period
checkpoint_directory:               Return the checkpoint directory with script-compatible fallback
checkpoint_dir:                     Return the checkpoint directory using the trainer shorthand
best_checkpoint_path:               Return the best checkpoint path beside the configured checkpoint
rolling_checkpoint_name:            Return the step-tagged rolling checkpoint filename
rolling_checkpoint_path:            Return the step-tagged rolling checkpoint path
is_rolling_checkpoint_name:         Return whether a filename matches the rolling checkpoint convention
rolling_checkpoint_names_to_prune:  Return stale rolling checkpoint names that should be pruned
rolling_checkpoint_paths_to_prune:  Return stale rolling checkpoint paths that should be pruned
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import PurePath


@dataclass(frozen=True)
class CadenceState:
    """Next scheduled eval and checkpoint steps"""

    next_eval_step              : int | None  # step count used for next eval step scheduling or reporting
    next_checkpoint_step        : int | None  # step count used for next checkpoint step scheduling or reporting
    next_rolling_checkpoint_step: int | None  # step count used for next rolling checkpoint step scheduling or reporting
    eval_every                  : int | None = None  # effective interval controlling eval cadence


TrainingCadence = CadenceState


def next_periodic_step(current_count: int, period: int | None) -> int | None:
    """Return the first scheduled step strictly after current_count"""
    if period is None or int(period) <= 0:
        return None
    return ((int(current_count) // int(period)) + 1) * int(period)


def next_eval_step(
    transitions_collected: int,  # Param: integer input for transitions collected
    eval_every           : int | None,  # Param: interval controlling how often eval runs
    *,
    eval_start_steps: int = 0,   # Param: step count used for eval start steps
) -> int | None:
    """Return the next eval step after a resume or launch"""
    scheduled = next_periodic_step(transitions_collected, eval_every)
    if scheduled is None:
        return None
    if int(eval_start_steps) > 0:
        scheduled = max(scheduled, int(eval_start_steps))
    return scheduled


def initial_cadence_state(
    *,
    transitions_collected   : int,  # Param: integer input for transitions collected
    eval_every              : int | None,  # Param: interval controlling how often eval runs
    eval_start_steps        : int,  # Param: step count used for eval start steps
    checkpoint_every        : int,  # Param: global-step interval for regular checkpoint saves
    rolling_checkpoint_every: int,  # Param: global-step interval for rolling checkpoint saves
) -> CadenceState:
    """Build initial step schedules for eval and checkpoints"""
    return CadenceState(
        next_eval_step=next_eval_step(
            transitions_collected,
            eval_every,
            eval_start_steps=eval_start_steps,
        ),
        next_checkpoint_step=next_periodic_step(transitions_collected, checkpoint_every),
        next_rolling_checkpoint_step=next_periodic_step(
            transitions_collected,
            rolling_checkpoint_every,
        ),
        eval_every=eval_every,
    )


def initial_training_cadence(
    *,
    transitions_collected   : int,  # Param: integer input for transitions collected
    eval_every              : int | None,  # Param: interval controlling how often eval runs
    eval_start_steps        : int,  # Param: step count used for eval start steps
    checkpoint_every        : int,  # Param: global-step interval for regular checkpoint saves
    rolling_checkpoint_every: int,  # Param: global-step interval for rolling checkpoint saves
) -> TrainingCadence:
    """Build initial trainer cadence with the public migration name"""
    return initial_cadence_state(
        transitions_collected=transitions_collected,
        eval_every=eval_every,
        eval_start_steps=eval_start_steps,
        checkpoint_every=checkpoint_every,
        rolling_checkpoint_every=rolling_checkpoint_every,
    )


def step_due(next_step: int | None, global_step: int) -> bool:
    """Return whether global_step has reached a scheduled step"""
    return next_step is not None and int(global_step) + 1 >= int(next_step)


def advance_periodic_step(next_step: int | None, period: int | None) -> int | None:
    """Advance an existing schedule by one period"""
    if next_step is None or period is None or int(period) <= 0:
        return None
    return int(next_step) + int(period)


def checkpoint_directory(checkpoint_path: str) -> str:
    """Return the checkpoint directory with script-compatible fallback"""
    directory = os.path.dirname(checkpoint_path)
    return directory or "."


def checkpoint_dir(checkpoint_path: str) -> str:
    """Return the checkpoint directory using the trainer shorthand"""
    return checkpoint_directory(checkpoint_path)


def best_checkpoint_path(checkpoint_path: str) -> str:
    """Return the best checkpoint path beside the configured checkpoint"""
    return os.path.join(checkpoint_directory(checkpoint_path), "best.pt")


def rolling_checkpoint_name(step_number: int) -> str:
    """Return the step-tagged rolling checkpoint filename"""
    return f"step_{int(step_number):06d}.pt"


def rolling_checkpoint_path(checkpoint_path: str, step_number: int) -> str:
    """Return the step-tagged rolling checkpoint path"""
    return os.path.join(checkpoint_directory(checkpoint_path), rolling_checkpoint_name(step_number))


def is_rolling_checkpoint_name(name: str) -> bool:
    """Return whether a filename matches the rolling checkpoint convention"""
    basename = PurePath(name).name
    return basename.startswith("step_") and basename.endswith(".pt")


def rolling_checkpoint_names_to_prune(existing_names: Sequence[str], keep: int) -> list[str]:
    """Return stale rolling checkpoint names that should be pruned"""
    if int(keep) <= 0:
        return []
    rolling = sorted(name for name in existing_names if is_rolling_checkpoint_name(name))
    return rolling[: -int(keep)]


def rolling_checkpoint_paths_to_prune(
    checkpoint_path: str,  # Param: base checkpoint path used for scheduled save decisions
    existing_names : Sequence[str],  # Param: ordered candidate names used to resolve existing
    keep           : int,  # Param: integer input for keep
) -> list[str]:
    """Return stale rolling checkpoint paths that should be pruned"""
    directory = checkpoint_directory(checkpoint_path)
    return [
        os.path.join(directory, name)
        for name in rolling_checkpoint_names_to_prune(existing_names, keep)
    ]
