"""No-shell execution engine for the standalone topdown trainer.

This module is the boundary between declarative profiles and long-running
Isaac Sim processes.  Profiles in :mod:`enpm690_final_project.config.profiles`
describe a run as dataclass groups; this engine turns those groups into:

* a sanitized environment dictionary,
* a deterministic command line,
* a manifest that records the exact launch contract, and
* a tee'd ``stdout.log`` process execution.

Two design constraints are intentional:

* The engine strips ambient training-related env vars unless the active
  profile owns them.  Most late-project regressions came from stale shell env
  values silently overriding a profile, so this module fails closed.
* The trainer is launched directly with ``subprocess.Popen`` rather than via a
  shell script.  That keeps replay/manifests portable into Docker, GUI smoke
  tests, and the new standalone repo layout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config.profiles import RunProfile
from .manifest import build_manifest, write_manifest

ENGINE_ENV_KEYS = (
    "PROJECT_ROOT",
    "PYTHONPATH",
    "TENSORBOARD_DIR",
    "UNITREE_TASKS_IMPORT_FILTER",
    "UNITREE_G1_TASKS_IMPORT_FILTER",
)
INFRASTRUCTURE_FAILURE_RC = 90
INFRASTRUCTURE_FAILURE_PATTERNS = (
    "RuntimeError: No CUDA GPUs are available",
    "no CUDA-capable device is detected",
    "No CUDA devices found",
    "Failed to create primary CUDA context",
    "Unable to create PxCudaContextManager",
)

# These keys and prefixes are heuristics for catching env vars that would affect training but are not explicitly owned by the 
# profile.  The danger is that a stale terminal session may still contain a knob from a previous smoke test, which would silently change the behavior of the trainer. 
# By checking for these keys and prefixes, we can alert the user to potential issues before they run the trainer.
TRAINING_OVERRIDE_PREFIXES = (
    "TOPDOWN_",
    "CURRICULUM_",
    "RL_",
    "BC_",
    "TEACHER_",
    "POLICY_",
    "ASSIST_",
    "ADAPTIVE_",
    "CONTACT_",
    "PHASE1_",
    "ACTOR_",
    "CRITIC_",
    "REWARD_",
    "RESET_",
)

# These keys are specific env vars that have been commonly used in training runs and could affect the behavior of the trainer
# if they are set in the environment. By listing them here, we can ensure that they are either owned by the profile or explicitly allowed by the user, rather than being accidentally inherited from a previous run.
TRAINING_OVERRIDE_KEYS = {
    "RUN_DIR",
    "CHECKPOINT_PATH",
    "RESUME_CKPT",
    "ACTOR_INIT_CKPT",
    "PHASE1_CKPT",
    "FASTTD3_REPO",
    "TD3_BACKEND",
    "TASK",
    "DEVICE",
    "SEED",
    "NUM_ENVS",
    "TOTAL_STEPS",
    "START_STEPS",
    "HEADLESS",
    "ENABLE_CAMERAS",
    "DISABLE_CAMERA_PERCEPTION",
}


@dataclass(frozen=True)
class CommandPlan:
    """Fully materialized process plan.

    ``CommandPlan`` is deliberately plain data.  It is safe to print, store in
    a manifest, mutate in smoke-test helpers, or execute later.  The plan does
    not open files or start Isaac Sim until ``run_plan`` is called.
    """

    command   : list[str]
    env       : dict[str, str]
    cwd       : Path
    run_dir   : str
    stdout_log: Path
    profile   : RunProfile


def _runtime_python(profile_env: dict[str, str]) -> str:
    """Resolve the Python interpreter used for the child trainer process."""
    return profile_env.get("ENPM690_PYTHON") or os.environ.get("ENPM690_PYTHON") or sys.executable


def _validate_project_root(project_root: Path) -> Path:
    """Verify that a path points at this standalone project layout."""
    resolved = project_root.resolve()
    required = (
        "run.py",
        "training/native/native_entrypoint.py",
        "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3",
        "assets/robots/g1-29dof-dex3-base-fix-usd",
        "src/enpm690_final_project",
    )
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"project root {resolved} is not this standalone repo; missing: {joined}")
    return resolved


def _external_training_overrides(profile_env: dict[str, str]) -> list[str]:
    """Return inherited env vars that could silently change training behavior.

    The profile may intentionally emit hundreds of ``TOPDOWN_*`` and
    ``CURRICULUM_*`` values.  The danger is the opposite: an old terminal
    session may still contain a knob from a previous smoke test.  This helper
    finds those inherited knobs before they can shadow the profile contract.
    """
    allowed = set(profile_env) | set(ENGINE_ENV_KEYS) | {
        "ENPM690_PYTHON",
        "ENPM690_ALLOW_EXTERNAL_TRAINING_ENV",
        "FASTTD3_REPO",
        "TOPDOWN_PHYSICS_PROFILE",
    }
    offenders: list[str] = []
    for key in os.environ:
        if key in allowed:
            continue
        if key in TRAINING_OVERRIDE_KEYS or key.startswith(TRAINING_OVERRIDE_PREFIXES):
            offenders.append(key)
    return sorted(offenders)


def build_plan(profile: RunProfile, project_root: Path) -> CommandPlan:
    """Build the exact Python command for one deterministic topdown run.

    The ordering here matters:

    1. Validate that ``project_root`` is this standalone repo.
    2. Materialize the profile env before copying ``os.environ``.
    3. Reject stale external training overrides.
    4. Remove Python/venv state that would make a host shell leak into the
       Isaac Sim subprocess.

    The resulting command is suitable for both training and ``--dry-run``
    manifest inspection.
    """

    project_root = _validate_project_root(project_root)
    profile_env = profile.env()
    offenders = _external_training_overrides(profile_env)
    if offenders and os.environ.get("ENPM690_ALLOW_EXTERNAL_TRAINING_ENV") != "1":
        joined = ", ".join(offenders[:20])
        suffix = " ..." if len(offenders) > 20 else ""
        raise ValueError(
            "Refusing to inherit training override env vars not owned by the profile: "
            f"{joined}{suffix}. Unset them or set ENPM690_ALLOW_EXTERNAL_TRAINING_ENV=1."
        )

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    for stale_key in (
        "ACTOR_INIT_CKPT",
        "RESUME_CKPT",
        "PHASE1_CKPT",
        "HANDOFF_CHECKPOINT_PATH",
        "FINAL_HANDOFF_CHECKPOINT_PATH",
    ):
        env.pop(stale_key, None)
    env.update(profile_env)
    env["PROJECT_ROOT"] = str(project_root)
    env["PYTHONPATH"] = f"{project_root / 'src'}:{project_root}"
    env.setdefault("TENSORBOARD_DIR", "")
    env["UNITREE_TASKS_IMPORT_FILTER"] = "tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3"
    env["UNITREE_G1_TASKS_IMPORT_FILTER"] = "cgc_topdown_curriculum_g1_29dof_dex3"

    run_dir = profile_env["RUN_DIR"]
    stdout_log = project_root / run_dir / "stdout.log"
    launcher = [profile.script] if profile.script else ["-m", profile.module]
    command = [
        _runtime_python(profile_env),
        "-u",
        *launcher,
        *profile.app_args(),
        *profile.trainer_args(),
    ]
    return CommandPlan(
        command=command,
        env=env,
        cwd=project_root,
        run_dir=run_dir,
        stdout_log=stdout_log,
        profile=profile,
    )


def manifest_for_plan(plan: CommandPlan) -> dict:
    """Create a deterministic manifest for a command plan.

    Only run-relevant environment keys are included.  This keeps manifests
    compact enough to diff while still recording every knob that can affect
    task geometry, teacher behavior, optimization, or runtime layout.
    """

    profile_env = plan.profile.env()
    public_env = {key: plan.env[key] for key in sorted(profile_env) if key in plan.env}
    for key in sorted(plan.env):
        if (
            key in TRAINING_OVERRIDE_KEYS
            or key.startswith(TRAINING_OVERRIDE_PREFIXES)
            or key == "ENPM690_PYTHON"
        ):
            public_env[key] = plan.env[key]
    for key in ENGINE_ENV_KEYS:
        if key in plan.env:
            public_env[key] = plan.env[key]
    return build_manifest(
        profile_name=plan.profile.name,
        profile_description=plan.profile.description,
        groups=plan.profile.groups,
        command=plan.command,
        env=public_env,
        cwd=plan.cwd,
        run_dir=plan.run_dir,
    )


def _missing_input_checkpoints(plan: CommandPlan) -> list[str]:
    """Return configured checkpoint inputs that do not exist yet."""

    missing: list[str] = []
    for key in ("ACTOR_INIT_CKPT", "RESUME_CKPT", "PHASE1_CKPT"):
        value = plan.env.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = plan.cwd / path
        if not path.is_file():
            missing.append(f"{key}={value}")
    return missing


def run_plan(plan: CommandPlan, manifest_path: Path | None = None) -> int:
    """Run the trainer and tee stdout/stderr to the run's ``stdout.log``.

    This function intentionally streams line-by-line.  Training logs can be
    very large, so callers should inspect them later with bounded ``tail`` or
    targeted ``rg`` commands rather than loading the whole file.
    """

    missing = _missing_input_checkpoints(plan)
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Configured checkpoint input is missing. Run the prerequisite stage "
            f"or update the profile: {joined}"
        )

    plan.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = plan.stdout_log.parent / "manifest.json"
    write_manifest(manifest_path, manifest_for_plan(plan))

    with plan.stdout_log.open("w", encoding="utf-8") as log_file:
        infrastructure_failure = False
        process = subprocess.Popen(
            plan.command,
            cwd=plan.cwd,
            env=plan.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            if any(pattern in line for pattern in INFRASTRUCTURE_FAILURE_PATTERNS):
                infrastructure_failure = True
        rc = process.wait()
        if infrastructure_failure:
            return INFRASTRUCTURE_FAILURE_RC
        return rc
