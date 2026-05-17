"""

Isaac launch environment helpers for native training tools

File map:

ISAAC_ENV_UNSET_KEYS:           Define isaac env unset keys constant
STALE_CHECKPOINT_ENV_KEYS:      Define stale checkpoint env keys constant
TOPDOWN_TASK_IMPORT_FILTER:     Define topdown task import filter constant
TOPDOWN_G1_TASK_IMPORT_FILTER:  Define topdown g1 task import filter constant
sanitized_isaac_env:            Return an Isaac launch environment with stale Python and checkpoint vars removed
default_training_python:        Return the configured Python executable for training launches
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import sys

from ..model.upstream_fasttd3 import fasttd3_repo_module_path


ISAAC_ENV_UNSET_KEYS = (
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_SHLVL",
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
)
STALE_CHECKPOINT_ENV_KEYS = (
    "ACTOR_INIT_CKPT",
    "RESUME_CKPT",
    "PHASE1_CKPT",
    "HANDOFF_CHECKPOINT_PATH",
    "FINAL_HANDOFF_CHECKPOINT_PATH",
)
TOPDOWN_TASK_IMPORT_FILTER = "tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3"
TOPDOWN_G1_TASK_IMPORT_FILTER = "cgc_topdown_curriculum_g1_29dof_dex3"


def sanitized_isaac_env(
    *,
    project_root: str,
    base_env    : Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an Isaac launch environment with stale Python and checkpoint vars removed"""
    env = dict(os.environ if base_env is None else base_env)
    for key in ISAAC_ENV_UNSET_KEYS:
        env.pop(key, None)
    for key in STALE_CHECKPOINT_ENV_KEYS:
        env.pop(key, None)
    fasttd3_repo = env.get("FASTTD3_REPO", "").strip()
    if fasttd3_repo and not os.path.isfile(fasttd3_repo_module_path(fasttd3_repo)):
        env.pop("FASTTD3_REPO", None)
    env["PROJECT_ROOT"] = project_root
    env["UNITREE_TASKS_IMPORT_FILTER"] = TOPDOWN_TASK_IMPORT_FILTER
    env["UNITREE_G1_TASKS_IMPORT_FILTER"] = TOPDOWN_G1_TASK_IMPORT_FILTER
    return env


def default_training_python(env: Mapping[str, str] | None = None) -> str:
    """Return the configured Python executable for training launches"""
    source = os.environ if env is None else env
    return source.get("ENPM690_PYTHON") or source.get("ISAAC_PYTHON") or sys.executable
