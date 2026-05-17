"""

Isaac-bound backend loading helpers for the refactored trainer

File map:

IsaacRuntimeSymbols:                       Runtime symbols loaded from the Isaac Python environment
ensure_isaacsim_export_available:          Run the local IsaacSim compatibility export hook when available
load_isaac_app_launcher_symbol:            Load AppLauncher before the SimulationApp is created
load_isaac_runtime_symbols:                Load post-SimulationApp Isaac runtime symbols
install_isaac_terminal_observation_patch:  Install the terminal-observation step patch on Isaac envs
IsaacTrainingBackend:                      Backend boundary for the migrated Isaac training loop
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.context import TrainerRuntimeContext
from ..core.runner import TrainingRunResult
from .terminal_patch import install_terminal_observation_patch


@dataclass(frozen=True)
class IsaacRuntimeSymbols:
    """Runtime symbols loaded from the Isaac Python environment"""

    app_launcher                  : type  # stores app launcher for isaac runtime symbols
    manager_based_rl_env          : type  # stores manager based rl env for isaac runtime symbols
    differential_ik_controller    : type  # stores differential ik controller for isaac runtime symbols
    differential_ik_controller_cfg: type  # stores differential ik controller cfg for isaac runtime symbols
    parse_env_cfg                 : Callable[..., object]  # callback used for the parse env cfg operation


def ensure_isaacsim_export_available() -> None:
    """Run the local IsaacSim compatibility export hook when available"""
    try:
        from isaacsim_compat import ensure_isaacsim_simulation_app_export
    except ModuleNotFoundError:
        return
    ensure_isaacsim_simulation_app_export()


def load_isaac_app_launcher_symbol() -> type:
    """Load AppLauncher before the SimulationApp is created"""
    ensure_isaacsim_export_available()
    from isaaclab.app import AppLauncher

    return AppLauncher


def load_isaac_runtime_symbols() -> IsaacRuntimeSymbols:
    """Load post-SimulationApp Isaac runtime symbols"""
    app_launcher = load_isaac_app_launcher_symbol()
    try:
        from isaaclab.controllers.differential_ik import DifferentialIKController
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
        from tasks.utils.parse_cfg import parse_env_cfg
    except ModuleNotFoundError as exc:
        if exc.name == "pxr":
            raise RuntimeError(
                "Isaac runtime symbols that depend on pxr must be loaded after "
                "AppLauncher has created the SimulationApp"
            ) from exc
        raise

    return IsaacRuntimeSymbols(
        app_launcher=app_launcher,
        manager_based_rl_env=ManagerBasedRLEnv,
        differential_ik_controller=DifferentialIKController,
        differential_ik_controller_cfg=DifferentialIKControllerCfg,
        parse_env_cfg=parse_env_cfg,
    )


def install_isaac_terminal_observation_patch() -> bool:
    """Install the terminal-observation step patch on Isaac envs"""
    symbols = load_isaac_runtime_symbols()
    return install_terminal_observation_patch(symbols.manager_based_rl_env)


class IsaacTrainingBackend:
    """Backend boundary for the migrated Isaac training loop"""

    def __init__(
        self,
        run_impl: Callable[[TrainerRuntimeContext], TrainingRunResult] | None = None,  # Param: callback used to compute or fetch run impl
    ) -> None:
        self._run_impl = run_impl

    def run(self, context: TrainerRuntimeContext) -> TrainingRunResult:
        """Run the Isaac backend after validating the standalone task contract

        Steps:
        - Resolve inputs for `run` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        context.validate_supported()
        if self._run_impl is None:
            raise RuntimeError(
                "IsaacTrainingBackend needs a migrated run_impl or native live backend"
            )
        result = self._run_impl(context)
        if not isinstance(result, TrainingRunResult):
            raise TypeError(f"Isaac backend returned {type(result)!r}")
        return result
