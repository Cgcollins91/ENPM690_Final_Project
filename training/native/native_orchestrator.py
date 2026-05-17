"""

Native trainer callback assembly

File map:

NativeTrainingAssembly:           Native trainer pieces that can replace the subprocess backend later
_env_attr_int:                    Handle env attr int logic
native_loop_state_from_startup:   Build native rollout state from created Isaac env metadata
_state_with_loop_state:           Handle state with loop state logic
build_native_training_callbacks:  Build native backend callbacks from startup and loop pieces
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_backend import NativeTrainerCallbacks, NativeTrainerState
from .native_components import NativeTrainingComponents, create_native_training_components
from .native_loop import NativeLoopCallbacks, NativeLoopOptions, run_native_rollout_loop
from .native_startup import NativeStartupOptions, build_native_startup_state, close_native_startup_state
from ..state.run_state import TrainingLoopStartupState, initial_training_loop_state
from ..core.runner import TrainingRunResult


StartupFn = Callable[..., NativeTrainerState]
ComponentFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle], Any]
StartupResultFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle, Any], Any]
LoopStateFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], TrainingLoopStartupState]
FinalizeFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], None]


@dataclass(frozen=True)
class NativeTrainingAssembly:
    """Native trainer pieces that can replace the subprocess backend later"""

    loop_callbacks       : NativeLoopCallbacks  # stores loop callbacks for native training assembly
    startup_options      : NativeStartupOptions   = NativeStartupOptions()  # stores startup options for native training assembly
    loop_options         : NativeLoopOptions      = NativeLoopOptions()  # stores loop options for native training assembly
    component_fn         : ComponentFn | None     = None  # callback used for the component fn operation
    checkpoint_startup_fn: StartupResultFn | None = None  # callback used for the checkpoint startup fn operation
    phase1_startup_fn    : StartupResultFn | None = None  # callback used for the phase1 startup fn operation


def _env_attr_int(env: Any, name: str, default: int | None = None) -> int | None:
    value = getattr(env, name, default)
    if value is None:
        return None
    return int(value)


def native_loop_state_from_startup(
    context      : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs      : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    startup_state: NativeTrainerState,  # Param: input value used as startup state
) -> TrainingLoopStartupState:
    """Build native rollout state from created Isaac env metadata

    Steps:
    - Resolve inputs for `native_loop_state_from_startup` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    env = startup_state.get("env")
    max_episode_length = _env_attr_int(env, "max_episode_length")
    if max_episode_length is None:
        max_episode_length = _env_attr_int(env, "max_episode_length_s")
    if max_episode_length is None:
        raise RuntimeError("native loop startup requires env.max_episode_length")

    num_envs = _env_attr_int(env, "num_envs", configs.counts.num_envs)
    device = getattr(env, "device", context.device)
    topdown_preroll_enabled = configs.teacher.topdown_preroll_fraction > 0.0
    return initial_training_loop_state(
        num_envs=num_envs if num_envs is not None else configs.counts.num_envs,
        device=device,
        max_episode_length=max_episode_length,
        transitions_collected=0,
        eval_every=configs.eval.eval_every,
        eval_steps=configs.eval.eval_steps,
        eval_episodes=configs.eval.eval_episodes,
        eval_start_steps=configs.eval.eval_start_steps,
        checkpoint_every=configs.checkpoint.checkpoint_every,
        rolling_checkpoint_every=configs.checkpoint.rolling_checkpoint_every,
        legacy_contact_preroll_enabled=configs.teacher.contact_start_mode == "phase1_terminal",
        topdown_preroll_enabled=topdown_preroll_enabled,
        topdown_preroll_mask=startup_state.get("topdown_preroll_mask"),
        contact_preroll_max_steps=configs.teacher.contact_preroll_max_steps,
        topdown_preroll_max_steps=configs.teacher.topdown_preroll_max_steps,
    )


def _state_with_loop_state(
    startup_state            : NativeTrainerState,  # Param: input value used as startup state
    loop_state               : TrainingLoopStartupState,  # Param: input value used as loop state
    components               : Any = None,  # Param: input value used as components
    checkpoint_startup_result: Any = None,  # Param: input value used as checkpoint startup result
    phase1_startup_result    : Any = None,  # Param: input value used as phase1 startup result
) -> NativeTrainerState:
    """Process for `_state_with_loop_state`

    Steps:
    - Resolve inputs for `_state_with_loop_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    payload = dict(startup_state.payload)
    payload["loop_state"] = loop_state
    if components is not None:
        payload["components"] = components
    if checkpoint_startup_result is not None:
        payload["checkpoint_startup"] = checkpoint_startup_result
    if phase1_startup_result is not None:
        payload["phase1_startup"] = phase1_startup_result
    return NativeTrainerState(payload=payload)


def build_native_training_callbacks(
    assembly: NativeTrainingAssembly,                             # Param: input value used as assembly
    *,
    startup_fn   : StartupFn         = build_native_startup_state,  # Param: callback used to compute or fetch startup
    loop_state_fn: LoopStateFn       = native_loop_state_from_startup,  # Param: callback used to compute or fetch loop state
    finalize_fn  : FinalizeFn | None = None,  # Param: callback used to compute or fetch finalize
) -> NativeTrainerCallbacks:
    """Build native backend callbacks from startup and loop pieces"""

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
        component_fn = create_native_training_components if assembly.component_fn is None else assembly.component_fn
        components = component_fn(context, configs)
        checkpoint_result = (
            assembly.checkpoint_startup_fn(context, configs, components)
            if assembly.checkpoint_startup_fn is not None
            else None
        )
        phase1_result = (
            assembly.phase1_startup_fn(context, configs, components)
            if assembly.phase1_startup_fn is not None
            else None
        )
        loop_payload = dict(startup_state.payload)
        loop_payload["components"] = components
        if checkpoint_result is not None:
            loop_payload["checkpoint_startup"] = checkpoint_result
        if phase1_result is not None:
            loop_payload["phase1_startup"] = phase1_result
        startup_for_loop = NativeTrainerState(payload=loop_payload)
        loop_state = loop_state_fn(context, configs, startup_for_loop)
        transitions = int(getattr(checkpoint_result, "transitions_collected", 0) or 0)
        if transitions > 0:
            loop_state.transitions_collected = transitions
        loop_state.auto_handoff_loaded = bool(getattr(checkpoint_result, "auto_handoff_loaded", False))
        loop_state.skip_training_after_handoff_reuse = bool(
            getattr(checkpoint_result, "skip_training_after_handoff_reuse", False)
        )
        return _state_with_loop_state(
            startup_state,
            loop_state,
            components,
            checkpoint_startup_result=checkpoint_result,
            phase1_startup_result=phase1_result,
        )

    def _run_loop(
        context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state  : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    ) -> TrainingRunResult:
        loop_state = state.get("loop_state")
        if not isinstance(loop_state, TrainingLoopStartupState):
            raise TypeError(f"native state loop_state is {type(loop_state)!r}")
        summary = run_native_rollout_loop(
            context,
            configs,
            state,
            loop_state,
            assembly.loop_callbacks,
            options=assembly.loop_options,
        )
        return summary.as_result()

    def _finalize(
        context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state  : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    ) -> None:
        if finalize_fn is not None:
            finalize_fn(context, configs, state)
            return
        close_native_startup_state(state)

    return NativeTrainerCallbacks(startup=_startup, run_loop=_run_loop, finalize=_finalize)
