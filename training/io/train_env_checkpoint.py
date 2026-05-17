"""

Train-env best checkpoint decisions.

File map:

TRAIN_ENV_BEST_COMPARATOR:                 Define train env best comparator constant
TrainEnvBestCheckpointJob:                 One train-env best checkpoint save request
TrainEnvBestCheckpointDecision:            Best-checkpoint decision from train-env summary metrics
_summary_float:                            Read one finite float from a summary row
_summary_block_max:                        Fallback for summaries that only have source-conditioned metrics
train_env_best_state_from_summary:         Extract lift-first train-env state from one topdown summary row
train_env_best_is_better:                  Return whether current train-env stats beat the stored best state
train_env_best_checkpoint_message:         Return the stdout line for a train-env best checkpoint update
build_train_env_best_checkpoint_decision:  Build a train-env best checkpoint decision from one summary row
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from ..geometry.topdown_metrics import SOURCE_BLOCK_NAMES
from ..state.cadence import best_checkpoint_path


TRAIN_ENV_BEST_COMPARATOR: tuple[tuple[str, str, float], ...] = (
    ("lift_max", "higher", 1e-4),
    ("lift_mean", "higher", 1e-5),
    ("success_rate", "higher", 1e-4),
    ("stage_ge2_rate", "higher", 1e-4),
    ("strict_contact_mean", "higher", 1e-4),
    ("contact_mean", "higher", 1e-4),
    ("contact_pose_ready_rate", "higher", 1e-4),
    ("unlock_max", "higher", 1e-4),
    ("contact_palm_dist_mean", "lower", 1e-4),
)


@dataclass(frozen=True)
class TrainEnvBestCheckpointJob:
    """One train-env best checkpoint save request."""

    global_step   : int  # current absolute training step
    dest_path     : str  # destination path for best.pt
    include_replay: bool  # whether to include replay in this checkpoint


@dataclass(frozen=True)
class TrainEnvBestCheckpointDecision:
    """Best-checkpoint decision from train-env summary metrics."""

    current_state  : dict[str, float | int]  # current train-env best state
    next_best_state: Mapping[str, float | int]  # selected next best state
    is_better      : bool  # whether current state improves over previous best
    job            : TrainEnvBestCheckpointJob | None  # optional checkpoint save request
    message        : str | None  # printable progress message for stdout


def _summary_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Read one finite float from a summary row."""
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        return default
    return value


def _summary_block_max(row: Mapping[str, object], suffix: str, default: float = 0.0) -> float:
    """Fallback for summaries that only have source-conditioned metrics."""
    values = [
        _summary_float(row, f"topdown_block_{name}_{suffix}", float("-inf"))
        for name in SOURCE_BLOCK_NAMES
        if f"topdown_block_{name}_{suffix}" in row
    ]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return default
    return max(finite_values)


def train_env_best_state_from_summary(row: Mapping[str, object]) -> dict[str, float | int]:
    """Extract lift-first train-env state from one topdown summary row.

    Steps:
    - Resolve inputs for `train_env_best_state_from_summary` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    lift_max = _summary_float(
        row,
        "topdown_lift_max",
        _summary_block_max(row, "lift_max", 0.0),
    )
    lift_mean = _summary_float(
        row,
        "topdown_lift_mean",
        _summary_block_max(row, "lift_mean", 0.0),
    )
    success_rate = _summary_float(
        row,
        "topdown_success_rate",
        _summary_block_max(row, "success_rate", 0.0),
    )
    strict_contact_mean = _summary_float(
        row,
        "topdown_strict_contact_mean",
        _summary_block_max(row, "strict_contact_mean", 0.0),
    )
    contact_mean = _summary_float(
        row,
        "topdown_contact_mean",
        _summary_block_max(row, "contact_mean", 0.0),
    )
    stage_ge2_rate = _summary_float(row, "topdown_stage_ge2_rate", 0.0)
    contact_pose_ready_rate = _summary_float(row, "topdown_contact_pose_ready_rate", 0.0)
    unlock_max = _summary_float(row, "topdown_finger_unlock_progress_max", 0.0)
    contact_palm_dist_mean = _summary_float(row, "topdown_contact_palm_dist_mean", float("inf"))
    score = (
        1000.0 * lift_max
        + 100.0 * lift_mean
        + 20.0 * success_rate
        + 5.0 * stage_ge2_rate
        + 2.0 * strict_contact_mean
        + contact_mean
        + contact_pose_ready_rate
        + 0.5 * unlock_max
        - contact_palm_dist_mean
    )
    return {
        "global_step"            : int(row.get("global_step", -1)),
        "score"                  : float(score),
        "lift_max"               : lift_max,
        "lift_mean"              : lift_mean,
        "success_rate"           : success_rate,
        "stage_ge2_rate"         : stage_ge2_rate,
        "strict_contact_mean"    : strict_contact_mean,
        "contact_mean"           : contact_mean,
        "contact_pose_ready_rate": contact_pose_ready_rate,
        "unlock_max"             : unlock_max,
        "contact_palm_dist_mean" : contact_palm_dist_mean,
    }


def train_env_best_is_better(
    current: Mapping[str, float | int],
    best   : Mapping[str, float | int] | None,
) -> bool:
    """Return whether current train-env stats beat the stored best state."""
    if best is None:
        return True
    for key, direction, epsilon in TRAIN_ENV_BEST_COMPARATOR:
        current_value = float(current.get(key, 0.0))
        best_value = float(best.get(key, 0.0))
        if direction == "higher":
            if current_value > best_value + epsilon:
                return True
            if current_value < best_value - epsilon:
                return False
        else:
            if current_value < best_value - epsilon:
                return True
            if current_value > best_value + epsilon:
                return False
    return False


def train_env_best_checkpoint_message(state: Mapping[str, float | int], best_path: str) -> str:
    """Return the stdout line for a train-env best checkpoint update."""
    return (
        f"train_best_checkpoint step={int(state['global_step'])} "
        f"score={float(state['score']):.4f} "
        f"lift_max={float(state['lift_max']):.4f} "
        f"lift_mean={float(state['lift_mean']):.4f} "
        f"success={float(state['success_rate']):.3f} "
        f"stage2={float(state['stage_ge2_rate']):.3f} "
        f"strict={float(state['strict_contact_mean']):.3f} "
        f"contact={float(state['contact_mean']):.3f} "
        f"unlock_max={float(state['unlock_max']):.3f} "
        f"palm_dist={float(state['contact_palm_dist_mean']):.4f} "
        f"-> {best_path}"
    )


def build_train_env_best_checkpoint_decision(
    *,
    summary_row               : Mapping[str, object],  # Param: topdown summary row
    best_train_env_state      : Mapping[str, float | int] | None,  # Param: current best state
    checkpoint_path           : str,  # Param: configured latest checkpoint path
    save_replay_in_checkpoint : bool,  # Param: whether best.pt should include replay
) -> TrainEnvBestCheckpointDecision:
    """Build a train-env best checkpoint decision from one summary row.

    Steps:
    - Resolve inputs for `build_train_env_best_checkpoint_decision` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    current_state = train_env_best_state_from_summary(summary_row)
    if not train_env_best_is_better(current_state, best_train_env_state):
        return TrainEnvBestCheckpointDecision(
            current_state=current_state,
            next_best_state=best_train_env_state if best_train_env_state is not None else current_state,
            is_better=False,
            job=None,
            message=None,
        )
    dest_path = best_checkpoint_path(checkpoint_path)
    job = TrainEnvBestCheckpointJob(
        global_step=int(current_state["global_step"]),
        dest_path=dest_path,
        include_replay=bool(save_replay_in_checkpoint),
    )
    return TrainEnvBestCheckpointDecision(
        current_state=current_state,
        next_best_state=current_state,
        is_better=True,
        job=job,
        message=train_env_best_checkpoint_message(current_state, dest_path),
    )
