"""Register the single supported ENPM690 G1 task."""

from __future__ import annotations

import importlib
import os

_SUPPORTED_MODULE = "cgc_topdown_curriculum_g1_29dof_dex3"
_requested = os.environ.get("UNITREE_G1_TASKS_IMPORT_FILTER", _SUPPORTED_MODULE).strip()
_modules = [name.strip() for name in _requested.split(",") if name.strip()]
unsupported = [name for name in _modules if name != _SUPPORTED_MODULE]
if unsupported:
    raise RuntimeError(f"Unsupported task module(s) in standalone ENPM690 repo: {unsupported}")

importlib.import_module(f"{__name__}.{_SUPPORTED_MODULE}")
__all__ = [_SUPPORTED_MODULE]
