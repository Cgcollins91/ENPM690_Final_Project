"""

Native live Isaac TrainingBackend factory

File map:

NativeLiveIsaacBackendOptions:  Options for constructing the live native Isaac backend
NativeLiveIsaacBackend:         TrainingBackend for the live native Isaac callback path
native_live_isaac_backend:      Build the live native Isaac backend
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_backend import run_native_isaac_training
from .native_isaac_hooks import NativeIsaacHookConfig, build_native_isaac_live_hooks_fn
from .native_live import NativeLiveEventCallbacks
from .native_live_backend import (
    NativeLiveTrainerAssembly,
    build_native_live_trainer_callbacks,
)
from .native_loop import NativeLoopOptions
from .native_startup import NativeStartupOptions, build_native_startup_state
from ..core.runner import TrainingRunResult


@dataclass(frozen=True)
class NativeLiveIsaacBackendOptions:
    """Options for constructing the live native Isaac backend"""

    startup_options                : NativeStartupOptions     = NativeStartupOptions()  # stores startup options for native live isaac backend options
    loop_options                   : NativeLoopOptions        = NativeLoopOptions()  # stores loop options for native live isaac backend options
    hook_config                    : NativeIsaacHookConfig    = NativeIsaacHookConfig()  # stores hook config for native live isaac backend options
    event_callbacks                : NativeLiveEventCallbacks = field(default_factory=NativeLiveEventCallbacks)  # stores event callbacks for native live isaac backend options
    topdown_curriculum_obs_contract: bool                     = True  # boolean value indicating the topdown curriculum obs contract state for native live isaac backend options


class NativeLiveIsaacBackend:
    """TrainingBackend for the live native Isaac callback path"""

    def __init__(
        self,
        *,
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        options: NativeLiveIsaacBackendOptions = NativeLiveIsaacBackendOptions(),  # Param: input value used as options
        startup_fn=build_native_startup_state,                                     # Param: callback used to compute or fetch startup
    ) -> None:
        self._configs = configs
        self._options = options
        self._startup_fn = startup_fn

    def run(self, context: TrainerRuntimeContext) -> TrainingRunResult:
        """Run training through native live Isaac callbacks"""
        hooks_fn = build_native_isaac_live_hooks_fn(
            context,
            self._configs,
            hook_config=self._options.hook_config,
        )
        callbacks = build_native_live_trainer_callbacks(
            NativeLiveTrainerAssembly(
                live_hooks_fn=hooks_fn,
                startup_options=self._options.startup_options,
                loop_options=self._options.loop_options,
                event_callbacks=self._options.event_callbacks,
                topdown_curriculum_obs_contract=self._options.topdown_curriculum_obs_contract,
            ),
            startup_fn=self._startup_fn,
        )
        return run_native_isaac_training(context, self._configs, callbacks)


def native_live_isaac_backend(
    *,
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    options: NativeLiveIsaacBackendOptions = NativeLiveIsaacBackendOptions(),  # Param: input value used as options
    startup_fn=build_native_startup_state,                                     # Param: callback used to compute or fetch startup
) -> NativeLiveIsaacBackend:
    """Build the live native Isaac backend"""
    return NativeLiveIsaacBackend(
        configs=configs,
        options=options,
        startup_fn=startup_fn,
    )
