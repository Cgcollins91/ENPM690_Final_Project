"""

Native Isaac startup state assembly for the refactored trainer

File map:

NativeStartupOptions:            Runtime startup switches for the native backend
build_native_startup_state:      Seed process state and assemble Isaac runtime objects
close_native_startup_state:      Close env and app objects stored in native startup state
build_native_startup_callbacks:  Build native backend callbacks around standard startup state
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..env.isaac_backend import install_isaac_terminal_observation_patch
from ..env.isaac_startup import IsaacAppStartupResult, IsaacEnvStartupResult, create_isaac_env, launch_isaac_app
from .native_backend import NativeTrainerCallbacks, NativeTrainerState
from ..core.runner import TrainingRunResult
from ..state.seeding import SeedConfig, set_global_seed


SeedFn      = Callable[[int], SeedConfig | Any]
LaunchAppFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle], IsaacAppStartupResult]
CreateEnvFn = Callable[[TrainerRuntimeContext, RuntimeConfigBundle], IsaacEnvStartupResult]
PatchFn     = Callable[[], bool]
RunLoopFn   = Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], TrainingRunResult]
FinalizeFn  = Callable[[TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState], None]


@dataclass(frozen=True)
class NativeStartupOptions:
    """Runtime startup switches for the native backend"""

    launch_app            : bool = True  # boolean value indicating the launch app state for native startup options
    create_env            : bool = True  # boolean value indicating the create env state for native startup options
    install_terminal_patch: bool = True  # boolean value indicating the install terminal patch state for native startup options


def build_native_startup_state(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    *,
    options                  : NativeStartupOptions = NativeStartupOptions(),  # Param: input value used as options
    seed_fn                  : SeedFn               = set_global_seed,  # Param: callback used to compute or fetch seed
    launch_app_fn            : LaunchAppFn          = launch_isaac_app,  # Param: callback used to compute or fetch launch app
    create_env_fn            : CreateEnvFn          = create_isaac_env,  # Param: callback used to compute or fetch create env
    install_terminal_patch_fn: PatchFn              = install_isaac_terminal_observation_patch,  # Param: callback used to compute or fetch install terminal patch
) -> NativeTrainerState:
    """Seed process state and assemble Isaac runtime objects

    Steps:
    - Resolve inputs for `build_native_startup_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    context.validate_supported()
    payload: dict[str, Any] = {
        "seed_config"               : seed_fn(context.seed),
        "terminal_patch_installed"  : False,
        "app_launcher"              : None,
        "simulation_app"            : None,
        "env_cfg"                   : None,
        "env"                       : None,
        "camera_perception_disabled": False,
        "removed_obs_terms"         : (),
    }

    if options.launch_app:
        app_result = launch_app_fn(context, configs)
        payload["app_launcher"] = app_result.app_launcher
        payload["simulation_app"] = app_result.simulation_app

    if options.install_terminal_patch:
        payload["terminal_patch_installed"] = bool(install_terminal_patch_fn())

    if options.create_env:
        env_result = create_env_fn(context, configs)
        payload["env_cfg"] = env_result.env_cfg
        payload["env"] = env_result.env
        payload["camera_perception_disabled"] = env_result.camera_perception_disabled
        payload["removed_obs_terms"] = env_result.removed_obs_terms

    return NativeTrainerState(payload=payload)


def close_native_startup_state(state: NativeTrainerState) -> tuple[str, ...]:
    """Close env and app objects stored in native startup state

    Steps:
    - Resolve inputs for `close_native_startup_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    closed: list[str] = []
    env = state.get("env")
    if env is not None and hasattr(env, "close"):
        env.close()
        closed.append("env")
    simulation_app = state.get("simulation_app")
    if simulation_app is not None and hasattr(simulation_app, "close"):
        simulation_app.close()
        closed.append("simulation_app")
    return tuple(closed)


def build_native_startup_callbacks(
    run_loop: RunLoopFn,                                                         # Param: input value used as run loop
    *,
    options    : NativeStartupOptions              = NativeStartupOptions(),  # Param: input value used as options
    startup_fn : Callable[..., NativeTrainerState] = build_native_startup_state,  # Param: callback used to compute or fetch startup
    finalize_fn: FinalizeFn | None                 = None,  # Param: callback used to compute or fetch finalize
) -> NativeTrainerCallbacks:
    """Build native backend callbacks around standard startup state"""

    def _startup(context: TrainerRuntimeContext, configs: RuntimeConfigBundle) -> NativeTrainerState:
        return startup_fn(context, configs, options=options)

    def _finalize(
        context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state  : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    ) -> None:
        if finalize_fn is not None:
            finalize_fn(context, configs, state)
            return
        close_native_startup_state(state)

    return NativeTrainerCallbacks(startup=_startup, run_loop=run_loop, finalize=_finalize)
