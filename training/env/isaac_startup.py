"""

Runtime-only Isaac app and environment startup helpers

File map:

AppLauncherArgs:             Namespace with dict-style pop support for IsaacLab AppLauncher
IsaacAppStartupResult:       Created Isaac application objects
IsaacEnvStartupResult:       Created Isaac environment and launch metadata
app_launcher_namespace:      Build the minimal AppLauncher namespace from runtime context
launch_isaac_app:            Create the Isaac SimulationApp through AppLauncher
_default_gym_make:           Handle default gym make logic
_configure_env_for_context:  Handle configure env for context logic
_env_mode_tuple:             Handle env mode tuple logic
create_isaac_env:            Create and configure the Isaac task environment
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..core.env_config import TouchEnvModeConfig, configure_touch_env_for_mode, viewport_camera_pose
from .isaac_backend import IsaacRuntimeSymbols, load_isaac_app_launcher_symbol, load_isaac_runtime_symbols


GymMakeFn = Callable[..., Any]
ConfigureEnvFn = Callable[..., TouchEnvModeConfig | tuple[bool, tuple[str, ...]]]


class AppLauncherArgs(SimpleNamespace):
    """Namespace with dict-style pop support for IsaacLab AppLauncher"""

    def __getitem__(self, name: str) -> Any:
        """Read an attribute like a mapping item"""
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        """Set an attribute like a mapping item"""
        setattr(self, name, value)

    def __contains__(self, name: object) -> bool:
        """Return whether an attribute is present"""
        return isinstance(name, str) and hasattr(self, name)

    def get(self, name: str, default: Any = None) -> Any:
        """Read an attribute like mapping get"""
        return getattr(self, name, default)

    def keys(self):
        """Return current argument names"""
        return vars(self).keys()

    def items(self):
        """Return current argument items"""
        return vars(self).items()

    def pop(self, name: str, default: Any = None) -> Any:
        """Pop an attribute value like a mutable argparse dict"""
        if hasattr(self, name):
            value = getattr(self, name)
            delattr(self, name)
            return value
        return default


@dataclass(frozen=True)
class IsaacAppStartupResult:
    """Created Isaac application objects"""

    app_launcher  : Any  # stores app launcher for isaac app startup result
    simulation_app: Any  # stores simulation app for isaac app startup result


@dataclass(frozen=True)
class IsaacEnvStartupResult:
    """Created Isaac environment and launch metadata"""

    env_cfg                   : Any  # stores env cfg for isaac env startup result
    env                       : Any  # environment/backend object used by this runtime helper
    camera_perception_disabled: bool  # boolean value indicating the camera perception disabled state for isaac env startup result
    removed_obs_terms         : tuple[str, ...]  # string removed obs terms value used by isaac env startup result


def app_launcher_namespace(context: TrainerRuntimeContext, configs: RuntimeConfigBundle) -> AppLauncherArgs:
    """Build the minimal AppLauncher namespace from runtime context"""
    return AppLauncherArgs(
        headless=bool(context.args.get("headless", False)),
        enable_cameras=bool(context.args.get("enable_cameras", False)),
        device=context.device,
        task=context.task,
        num_envs=configs.counts.num_envs,
        seed=context.seed,
    )


def launch_isaac_app(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    *,
    app_launcher_cls: type | None = None,  # Param: input value used as app launcher cls
) -> IsaacAppStartupResult:
    """Create the Isaac SimulationApp through AppLauncher"""
    context.validate_supported()
    launcher_cls = load_isaac_app_launcher_symbol() if app_launcher_cls is None else app_launcher_cls
    launcher = launcher_cls(app_launcher_namespace(context, configs))
    return IsaacAppStartupResult(app_launcher=launcher, simulation_app=launcher.app)


def _default_gym_make() -> GymMakeFn:
    import gymnasium as gym

    return gym.make


def _configure_env_for_context(
    env_cfg         : Any,  # Param: input value used as env cfg
    context         : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs         : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    configure_env_fn: ConfigureEnvFn,  # Param: callback used to compute or fetch configure env
) -> TouchEnvModeConfig | tuple[bool, tuple[str, ...]]:
    try:
        return configure_env_fn(
            env_cfg,
            disable_camera_perception=bool(context.args.get("disable_camera_perception", False)),
            arm_controller=configs.teacher.arm_controller,
        )
    except TypeError:
        return configure_env_fn(env_cfg)


def _env_mode_tuple(result: TouchEnvModeConfig | tuple[bool, tuple[str, ...]]) -> tuple[bool, tuple[str, ...]]:
    if isinstance(result, TouchEnvModeConfig):
        return result.camera_perception_disabled, result.removed_obs_terms
    camera_disabled, removed_obs_terms = result
    return bool(camera_disabled), tuple(removed_obs_terms)


def _apply_viewport_camera(env: object, context: TrainerRuntimeContext) -> bool:
    pose = viewport_camera_pose(str(context.args.get("viewport_camera", "overview")))
    if pose is None:
        return False
    sim = getattr(env, "sim", None)
    set_camera_view = getattr(sim, "set_camera_view", None)
    if not callable(set_camera_view):
        return False
    eye, target = pose
    set_camera_view(eye=eye, target=target)
    print(f"viewport_camera name={context.args.get('viewport_camera', 'overview')} eye={eye} target={target}", flush=True)
    return True


def create_isaac_env(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    *,
    symbols         : IsaacRuntimeSymbols | None = None,  # Param: input value used as symbols
    gym_make        : GymMakeFn | None           = None,  # Param: input value used as gym make
    configure_env_fn: ConfigureEnvFn             = configure_touch_env_for_mode,  # Param: callback used to compute or fetch configure env
) -> IsaacEnvStartupResult:
    """Create and configure the Isaac task environment

    Steps:
    - Resolve inputs for `create_isaac_env` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    context.validate_supported()
    runtime_symbols = load_isaac_runtime_symbols() if symbols is None else symbols
    env_cfg = runtime_symbols.parse_env_cfg(
        context.task,
        device=context.device,
        num_envs=configs.counts.num_envs,
    )
    env_cfg.seed = int(context.seed)
    camera_disabled, removed_obs_terms = _env_mode_tuple(
        _configure_env_for_context(env_cfg, context, configs, configure_env_fn)
    )
    make = _default_gym_make() if gym_make is None else gym_make
    env = make(context.task, cfg=env_cfg).unwrapped
    _apply_viewport_camera(env, context)
    return IsaacEnvStartupResult(
        env_cfg=env_cfg,
        env=env,
        camera_perception_disabled=bool(camera_disabled),
        removed_obs_terms=tuple(removed_obs_terms),
    )
