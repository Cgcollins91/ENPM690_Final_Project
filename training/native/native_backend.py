"""

Callback-driven native backend orchestration for the refactored trainer

File map:

NativeTrainerState:          Runtime objects owned by a native Isaac training run
NativeTrainerCallbacks:      Callback surface for the native Isaac backend
run_native_isaac_training:   Run native startup loop and finalization callbacks
NativeIsaacTrainingBackend:  TrainingBackend implementation for callback-driven native Isaac runs
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import traceback
from typing import Any

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..core.runner import TrainingRunResult


@dataclass(frozen=True)
class NativeTrainerState:
    """Runtime objects owned by a native Isaac training run"""

    payload : Mapping[str, Any] = field(default_factory=dict)  # Field: string payload value used by native trainer state

    def get(self, key: str, default: Any = None) -> Any:
        """Read one payload value"""
        return self.payload.get(key, default)


@dataclass(frozen=True)
class NativeTrainerCallbacks:
    """Callback surface for the native Isaac backend"""

    startup : Callable[[TrainerRuntimeContext, RuntimeConfigBundle], NativeTrainerState]  # Field: callback used for the startup operation
    run_loop: Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], TrainingRunResult]  # Field: callback used for the run loop operation
    finalize: Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], None] | None = None  # Field: callback used for the finalize operation


def run_native_isaac_training(
    context  : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs  : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    callbacks: NativeTrainerCallbacks,  # Param: input value used as callbacks
) -> TrainingRunResult:
    """Run native startup loop and finalization callbacks"""
    context.validate_supported()
    state = callbacks.startup(context, configs)
    if not isinstance(state, NativeTrainerState):
        raise TypeError(f"native startup returned {type(state)!r}")
    try:
        result = callbacks.run_loop(context, configs, state)
        if not isinstance(result, TrainingRunResult):
            raise TypeError(f"native run_loop returned {type(result)!r}")
        return result
    except BaseException:
        print("native_training_exception_begin", flush=True)
        traceback.print_exc()
        print("native_training_exception_end", flush=True)
        raise
    finally:
        if callbacks.finalize is not None:
            callbacks.finalize(context, configs, state)


class NativeIsaacTrainingBackend:
    """TrainingBackend implementation for callback-driven native Isaac runs"""

    def __init__(
        self,
        *,
        configs  : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        callbacks: NativeTrainerCallbacks,  # Param: input value used as callbacks
    ) -> None:
        self._configs = configs
        self._callbacks = callbacks

    def run(self, context: TrainerRuntimeContext) -> TrainingRunResult:
        """Run native Isaac training through explicit callbacks"""
        return run_native_isaac_training(context, self._configs, self._callbacks)
