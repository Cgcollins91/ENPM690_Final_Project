"""

Native live backend callback composition

File map:

NativeLiveTrainerAssembly:             Callbacks and options needed by the live native backend
_env_reset_fn:                         Handle env reset fn logic
_supports_keyword:                     Handle supports keyword logic
_state_payload:                        Handle state payload logic
_build_current_handoff_compatibility:  Handle build current handoff compatibility logic
build_native_live_trainer_callbacks:   Build NativeTrainerCallbacks for the live native backend path
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import inspect
import os
from typing import Any

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..io.handoff import build_handoff_compatibility
from .native_backend import NativeTrainerCallbacks, NativeTrainerState
from .native_checkpoint_startup import apply_native_checkpoint_startup
from .native_components import NativeTrainingComponents, create_native_training_components
from .native_finalization import finalize_native_training
from .native_live import (
    NativeLiveEventCallbacks,
    NativeLiveHooks,
    NativeLiveRolloutState,
    build_native_live_loop_callbacks,
)
from .native_loop import NativeLoopOptions, run_native_rollout_loop
from .native_orchestrator import native_loop_state_from_startup
from .native_phase1_startup import apply_native_phase1_startup
from .native_reset import NativeResetRequest, reset_native_env
from .native_startup import NativeStartupOptions, build_native_startup_state
from ..state.cadence import next_periodic_step
from ..state.run_state import TrainingLoopStartupState
from ..core.runner import TrainingRunResult


StartupFn = Callable[..., NativeTrainerState]
ComponentFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle], NativeTrainingComponents]
CheckpointStartupFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainingComponents],
    object,
]
Phase1StartupFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainingComponents],
    object,
]
LiveHooksFn = Callable[[NativeTrainerState, NativeTrainingComponents, NativeLiveRolloutState], NativeLiveHooks]
FinalizeFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], None]


@dataclass(frozen=True)
class NativeLiveTrainerAssembly:
    """Callbacks and options needed by the live native backend"""

    live_hooks_fn                  : LiveHooksFn  # Field: callback used for the live hooks fn operation
    startup_options                : NativeStartupOptions     = NativeStartupOptions()  # Field: stores startup options for native live trainer assembly
    loop_options                   : NativeLoopOptions        = NativeLoopOptions()  # Field: stores loop options for native live trainer assembly
    component_fn                   : ComponentFn              = create_native_training_components  # Field: callback used for the component fn operation
    checkpoint_startup_fn          : CheckpointStartupFn      = apply_native_checkpoint_startup  # Field: callback used for the checkpoint startup fn operation
    phase1_startup_fn              : Phase1StartupFn          = apply_native_phase1_startup  # Field: callback used for the phase1 startup fn operation
    finalize_fn                    : FinalizeFn               = finalize_native_training  # Field: callback used for the finalize fn operation
    event_callbacks                : NativeLiveEventCallbacks = field(default_factory=NativeLiveEventCallbacks)  # Field: stores event callbacks for native live trainer assembly
    topdown_curriculum_obs_contract: bool                     = True  # Field: boolean value indicating the topdown curriculum obs contract state for native live trainer assembly


def _env_reset_fn(startup_state: NativeTrainerState):
    env = startup_state.get("env")
    if env is None or not hasattr(env, "reset"):
        raise RuntimeError("native live startup requires env.reset")
    return env.reset


def _supports_keyword(fn: Callable[..., object], name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return (
        name in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )


def _state_payload(
    startup_state: NativeTrainerState,     # Param: input value used as startup state
    *,
    components        : NativeTrainingComponents,  # Param: input value used as components
    checkpoint_startup: object,  # Param: input value used as checkpoint startup
    phase1_startup    : object,  # Param: input value used as phase1 startup
    loop_state        : TrainingLoopStartupState,  # Param: input value used as loop state
    live_rollout      : NativeLiveRolloutState,  # Param: input value used as live rollout
    reset_info        : object,  # Param: input value used as reset info
    handoff_compatibility: dict[str, object],  # Param: compatibility metadata for replay handoff/checkpoints
) -> NativeTrainerState:
    payload = dict(startup_state.payload)
    payload.update(
        {
            "components"        : components,
            "checkpoint_startup": checkpoint_startup,
            "phase1_startup"    : phase1_startup,
            "loop_state"        : loop_state,
            "live_rollout"      : live_rollout,
            "reset_info"        : reset_info,
            "handoff_compatibility": handoff_compatibility,
        }
    )
    return NativeTrainerState(payload)


def _build_current_handoff_compatibility(context: TrainerRuntimeContext) -> dict[str, object]:
    return build_handoff_compatibility(
        project_root=str(context.env.get("PROJECT_ROOT", os.getcwd())),
        task=context.task,
        td3_backend=context.td3_backend,
        rl_phase_start_steps=int(context.args.get("rl_phase_start_steps", -1) or -1),
        obs_schema_version=context.obs_schema_version,
        obs_keys=tuple(context.obs_keys),
        obs_dim=int(context.dims.obs_dim),
        action_dim=int(context.dims.action_dim),
        priv_obs_dim=int(context.dims.priv_obs_dim),
        policy_action_spec=context.action.policy_action_spec,
        env_action_spec=context.action.env_action_spec,
        args=context.args,
        env=context.env,
    )


def build_native_live_trainer_callbacks(
    assembly: NativeLiveTrainerAssembly,                 # Param: input value used as assembly
    *,
    startup_fn: StartupFn = build_native_startup_state,  # Param: callback used to compute or fetch startup
) -> NativeTrainerCallbacks:
    """Build NativeTrainerCallbacks for the live native backend path"""

    def _startup(context: TrainerRuntimeContext, configs: RuntimeConfigBundle) -> NativeTrainerState:
        """Process for `_startup`

        Steps:
        - Resolve inputs for `_startup` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        startup_state = startup_fn(context, configs, options=assembly.startup_options)
        if not isinstance(startup_state, NativeTrainerState):
            raise TypeError(f"native startup returned {type(startup_state)!r}")
        reset_result = reset_native_env(
            NativeResetRequest(
                env_reset_fn=_env_reset_fn(startup_state),
                privileged_critic=context.dims.priv_obs_dim > 0,
                topdown_curriculum=assembly.topdown_curriculum_obs_contract,
            )
        )
        live_rollout = NativeLiveRolloutState(
            obs=dict(reset_result.obs),
            obs_tensor=reset_result.observation.obs_tensor,
            priv_obs_tensor=reset_result.observation.privileged_obs_tensor,
        )
        components = assembly.component_fn(context, configs)
        handoff_compatibility = _build_current_handoff_compatibility(context)
        if _supports_keyword(assembly.checkpoint_startup_fn, "play_checkpoint"):
            checkpoint_startup = assembly.checkpoint_startup_fn(
                context,
                configs,
                components,
                current_handoff_compatibility=handoff_compatibility,
                handoff_checkpoint=components.checkpoints.handoff,
                play_checkpoint=components.checkpoints.play,
            )
        else:
            checkpoint_startup = assembly.checkpoint_startup_fn(context, configs, components)
        phase1_startup = assembly.phase1_startup_fn(context, configs, components)
        loop_state = native_loop_state_from_startup(context, configs, startup_state)
        transitions = int(getattr(checkpoint_startup, "transitions_collected", 0) or 0)
        if transitions > 0:
            loop_state.transitions_collected = transitions
            loop_state.cadence = replace(
                loop_state.cadence,
                next_checkpoint_step=next_periodic_step(
                    transitions,
                    configs.checkpoint.checkpoint_every,
                ),
                next_rolling_checkpoint_step=next_periodic_step(
                    transitions,
                    configs.checkpoint.rolling_checkpoint_every,
                ),
            )
        loop_state.auto_handoff_loaded = bool(getattr(checkpoint_startup, "auto_handoff_loaded", False))
        loop_state.skip_training_after_handoff_reuse = bool(
            getattr(checkpoint_startup, "skip_training_after_handoff_reuse", False)
        )
        return _state_payload(
            startup_state,
            components=components,
            checkpoint_startup=checkpoint_startup,
            phase1_startup=phase1_startup,
            loop_state=loop_state,
            live_rollout=live_rollout,
            reset_info=reset_result.info,
            handoff_compatibility=handoff_compatibility,
        )

    def _run_loop(
        context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state  : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    ) -> TrainingRunResult:
        """Process for `_run_loop`

        Steps:
        - Resolve inputs for `_run_loop` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        loop_state = state.get("loop_state")
        live_rollout = state.get("live_rollout")
        components = state.get("components")
        if not isinstance(loop_state, TrainingLoopStartupState):
            raise TypeError(f"native state loop_state is {type(loop_state)!r}")
        if not isinstance(live_rollout, NativeLiveRolloutState):
            raise TypeError(f"native state live_rollout is {type(live_rollout)!r}")
        if not isinstance(components, NativeTrainingComponents):
            raise TypeError(f"native state components is {type(components)!r}")
        callbacks = build_native_live_loop_callbacks(
            live_rollout,
            assembly.live_hooks_fn(state, components, live_rollout),
            event_callbacks=assembly.event_callbacks,
        )
        summary = run_native_rollout_loop(
            context,
            configs,
            state,
            loop_state,
            callbacks,
            options=assembly.loop_options,
        )
        return summary.as_result()

    return NativeTrainerCallbacks(
        startup=_startup,
        run_loop=_run_loop,
        finalize=assembly.finalize_fn,
    )
