"""

Import-safe trainer runner scaffold

File map:

TrainingRunResult:               Summary returned by a trainer backend
TrainingBackend:                 Protocol implemented by Isaac-bound runtime backends
UnsupportedIsaacRuntimeBackend:  Default backend used until Isaac runtime binding is migrated
run_training:                    Validate context and dispatch to a training backend
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .context import TrainerRuntimeContext


@dataclass(frozen=True)
class TrainingRunResult:
    """Summary returned by a trainer backend"""

    status     : str  # Field: string status value used by training run result
    global_step: int                  = 0  # Field: training step associated with this record or action
    episode_idx: int                  = 0  # Field: training episode index associated with this record
    metrics    : Mapping[str, object] = field(default_factory=dict)  # Field: named metric values emitted with the result


class TrainingBackend(Protocol):
    """Protocol implemented by Isaac-bound runtime backends"""

    def run(self, context: TrainerRuntimeContext) -> TrainingRunResult:
        """Run training with an explicit context"""


class UnsupportedIsaacRuntimeBackend:
    """Default backend used until Isaac runtime binding is migrated"""

    def run(self, context: TrainerRuntimeContext) -> TrainingRunResult:
        """Raise a clear unsupported-runtime error"""
        raise RuntimeError(
            "refactored training runner requires an Isaac runtime backend; "
            "use training.native_entrypoint or pass an explicit backend"
        )


def run_training(
    context: TrainerRuntimeContext,          # Param: runtime context carrying validated trainer settings
    *,
    backend: TrainingBackend | None = None,  # Param: backend object that performs the runtime operation
) -> TrainingRunResult:
    """Validate context and dispatch to a training backend

    Steps:
    - Resolve inputs for `run_training` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    context.validate_supported()
    active_backend = UnsupportedIsaacRuntimeBackend() if backend is None else backend
    result = active_backend.run(context)
    if not isinstance(result, TrainingRunResult):
        raise TypeError(f"training backend returned {type(result)!r}")
    return result
