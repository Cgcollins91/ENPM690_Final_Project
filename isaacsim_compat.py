"""Compatibility helpers for Isaac Sim namespace-package layouts.

IsaacLab imports ``SimulationApp`` from the top-level ``isaacsim`` package.
Some Isaac Sim 5.x binary installs only expose it via
``isaacsim.simulation_app``. This shim backfills the top-level attribute before
IsaacLab imports it.
"""

from __future__ import annotations


def ensure_isaacsim_simulation_app_export() -> None:
    """Backfill ``isaacsim.SimulationApp`` when only the submodule export exists."""

    try:
        import isaacsim  # type: ignore
    except ModuleNotFoundError:
        return

    if hasattr(isaacsim, "SimulationApp"):
        return

    try:
        from isaacsim.simulation_app import SimulationApp  # type: ignore
    except Exception:
        return

    setattr(isaacsim, "SimulationApp", SimulationApp)
