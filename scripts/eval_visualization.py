#!/usr/bin/env python3
"""Launch GUI visualization for a checkpoint using its training-run manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ISAAC_PYTHON = Path(os.environ.get("ENPM690_PYTHON", sys.executable))
TRAINING_ENV_PREFIXES = (
    "TOPDOWN_",
    "CURRICULUM_",
    "RL_",
    "BC_",
    "TEACHER_",
    "POLICY_",
    "ASSIST_",
    "CONTACT_",
    "PHASE1_",
    "ACTOR_",
    "CRITIC_",
    "REWARD_",
    "RESET_",
)
TRAINING_ENV_KEYS = {
    "RUN_DIR",
    "CHECKPOINT_PATH",
    "RESUME_CKPT",
    "RESUME_REPLAY",
    "RESUME_GLOBAL_STEP",
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
PLAYBACK_CHECKPOINT_ENV_KEYS = {
    "ACTOR_INIT_CKPT",
    "FINAL_HANDOFF_CHECKPOINT_PATH",
    "FORCE_DAGGER_AFTER_RESUME",
    "HANDOFF_CHECKPOINT_PATH",
    "PHASE1_CKPT",
    "RESUME_CKPT",
    "RESET_OBS_STATS_ON_RESUME",
    "RESET_OPTIMIZERS_ON_RESUME",
    "STOP_AFTER_HANDOFF_CHECKPOINT",
}
PLAYBACK_FALSE_ENV_KEYS = {
    "RESUME_GLOBAL_STEP",
    "RESUME_REPLAY",
}


def _replace_arg(command: list[str], flag: str, value: str | int | float) -> None:
    value = str(value)
    if flag in command:
        idx = command.index(flag)
        if idx + 1 >= len(command):
            raise RuntimeError(f"{flag} has no value in command")
        command[idx + 1] = value
    else:
        command.extend([flag, value])


def _remove_arg(command: list[str], flag: str, *, takes_value: bool = True) -> None:
    while flag in command:
        idx = command.index(flag)
        del command[idx : idx + (2 if takes_value and idx + 1 < len(command) else 1)]


def _add_flag(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def _clean_env(parent: dict[str, str]) -> dict[str, str]:
    env = dict(parent)
    for key in list(env):
        if key in TRAINING_ENV_KEYS or key.startswith(TRAINING_ENV_PREFIXES):
            env.pop(key, None)
    for key in list(env):
        if key.startswith("CONDA_") or key in {"VIRTUAL_ENV", "PYTHONHOME"}:
            env.pop(key, None)
    return env


def _scrub_playback_checkpoint_env(env: dict[str, str]) -> None:
    for key in PLAYBACK_CHECKPOINT_ENV_KEYS:
        env.pop(key, None)
    for key in PLAYBACK_FALSE_ENV_KEYS:
        env[key] = "0"


def _parse_env_overrides(items: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise RuntimeError(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise RuntimeError(f"--env has empty key: {item!r}")
        overrides[key] = value
    return overrides


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _load_source_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"source manifest is not an object: {manifest_path}")
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_source_run_dir"] = str(manifest_path.parent)
    return manifest


def _resolve_checkpoint(raw_checkpoint: str, source_run_dir: Path) -> str:
    if not raw_checkpoint:
        return ""
    checkpoint = Path(raw_checkpoint)
    if checkpoint.is_absolute() or checkpoint.exists():
        return str(checkpoint)
    if not checkpoint.is_absolute():
        return str(source_run_dir / checkpoint)
    return str(checkpoint)


def _has_option(command: list[str], option: str) -> bool:
    return option in command or any(item.startswith(f"{option}=") for item in command)


def _ensure_option(command: list[str], option: str, value: str) -> None:
    if not _has_option(command, option):
        command.extend([option, value])


def _ensure_flag(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def _rewrite_legacy_training_command(command: list[str]) -> list[str]:
    """Rewrite saved script launches to the native training module"""
    rewritten = list(command)
    for idx, item in enumerate(tuple(rewritten)):
        normalized = item.replace("\\", "/")
        if normalized.endswith("scripts/topdown_dagger_td3.py") or normalized.endswith("topdown_dagger_td3.py"):
            rewritten[idx:idx + 1] = ["-m", "training.native_entrypoint"]
            _ensure_option(rewritten, "--native-teacher-provider", "env")
            _ensure_flag(rewritten, "--native-contact-attr-parts")
            return rewritten
        if normalized.endswith("scripts/topdown_teacher_dagger_fasttd3.py") or normalized.endswith("topdown_teacher_dagger_fasttd3.py"):
            rewritten[idx:idx + 1] = ["-m", "training.native_entrypoint"]
            _ensure_option(rewritten, "--td3-backend", "upstream_fasttd3")
            _ensure_option(rewritten, "--native-teacher-provider", "env")
            _ensure_flag(rewritten, "--native-contact-attr-parts")
            return rewritten
    return rewritten


def _base_command_env(
    args: argparse.Namespace,
    source_manifest: dict[str, object],
) -> tuple[list[str], dict[str, str]]:
    env = _clean_env(os.environ)
    raw_command = source_manifest.get("command")
    if not isinstance(raw_command, list) or not all(isinstance(item, str) for item in raw_command):
        raise RuntimeError("source manifest does not contain a string-list command")
    command = _rewrite_legacy_training_command(list(raw_command))
    manifest_env = source_manifest.get("env", {})
    if isinstance(manifest_env, dict):
        env.update({str(key): str(value) for key, value in manifest_env.items()})
    if command:
        command[0] = str(Path(args.isaac_python).expanduser())
    return command, env


def build_command(
    args: argparse.Namespace,
    source_run_dir: Path,
    output_dir: Path,
) -> tuple[list[str], dict[str, str], dict[str, object]]:
    source_manifest = _load_source_manifest(source_run_dir)
    checkpoint = _resolve_checkpoint(args.checkpoint, source_run_dir)
    if not checkpoint:
        raise RuntimeError("--checkpoint is required")

    command, env = _base_command_env(args, source_manifest)

    _remove_arg(command, "--headless", takes_value=False)
    _remove_arg(command, "--handoff-checkpoint-path")
    _remove_arg(command, "--final-handoff-checkpoint-path")
    _remove_arg(command, "--resume-checkpoint")
    _remove_arg(command, "--resume-replay", takes_value=False)
    _remove_arg(command, "--resume-global-step", takes_value=False)
    _remove_arg(command, "--actor-init-checkpoint")
    _remove_arg(command, "--stop-after-handoff-checkpoint", takes_value=False)
    _remove_arg(command, "--save-replay-in-checkpoint", takes_value=False)
    _scrub_playback_checkpoint_env(env)

    _add_flag(command, "--play")
    _replace_arg(command, "--log-jsonl", output_dir / "log.jsonl")
    _replace_arg(command, "--num-envs", args.num_envs)
    _replace_arg(command, "--replay-size", args.replay_size)
    _replace_arg(command, "--checkpoint-every", 0)
    _replace_arg(command, "--rolling-checkpoint-every", 0)
    _replace_arg(command, "--eval-steps", args.steps)
    _replace_arg(command, "--eval-episodes", args.episodes)
    _replace_arg(command, "--play-episodes", args.episodes)
    _replace_arg(command, "--sleep", args.sleep)
    _replace_arg(command, "--viewport-camera", args.camera)
    _replace_arg(command, "--tensorboard-dir", "off")

    if args.mode == "teacher":
        teacher_mix = 1.0
        if checkpoint and args.load_checkpoint_in_teacher_mode:
            _replace_arg(command, "--checkpoint-path", checkpoint)
            _remove_arg(command, "--play-skip-checkpoint", takes_value=False)
        else:
            _replace_arg(command, "--checkpoint-path", output_dir / "teacher_only_unused.pt")
            _add_flag(command, "--play-skip-checkpoint")
    elif args.mode == "policy":
        teacher_mix = 0.0
        _replace_arg(command, "--checkpoint-path", checkpoint)
        _remove_arg(command, "--play-skip-checkpoint", takes_value=False)
    else:
        teacher_mix = float(args.teacher_assist_mix)
        _replace_arg(command, "--checkpoint-path", checkpoint)
        _remove_arg(command, "--play-skip-checkpoint", takes_value=False)

    _replace_arg(command, "--eval-teacher-assist-mix", teacher_mix)

    env.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
            "RUN_DIR": str(output_dir),
            "LOG_JSONL": str(output_dir / "log.jsonl"),
            "HEADLESS": "0",
            "ENABLE_CAMERAS": "1",
            "DISABLE_CAMERA_PERCEPTION": "1",
            "NUM_ENVS": str(args.num_envs),
            "EVAL_STEPS": str(args.steps),
            "EVAL_EPISODES": str(args.episodes),
            "REPLAY_SIZE": str(args.replay_size),
            "CHECKPOINT_EVERY": "0",
            "ROLLING_CHECKPOINT_EVERY": "0",
            "SAVE_REPLAY_IN_CHECKPOINT": "0",
            "CHECKPOINT_PATH": str(checkpoint or output_dir / "teacher_only_unused.pt"),
        }
    )
    env.update(_parse_env_overrides(args.env))

    manifest = {
        "mode": args.mode,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "teacher_assist_mix": teacher_mix,
        "run_dir": str(output_dir),
        "source_manifest": source_manifest.get("_manifest_path"),
        "source_run_dir": source_manifest.get("_source_run_dir"),
        "command": command,
        "env_overrides": {
            key: env[key]
            for key in (
                "PROJECT_ROOT",
                "PYTHONPATH",
                "RUN_DIR",
                "LOG_JSONL",
                "CHECKPOINT_PATH",
                "HEADLESS",
                "ENABLE_CAMERAS",
                "DISABLE_CAMERA_PERCEPTION",
                "NUM_ENVS",
                "EVAL_STEPS",
                "EVAL_EPISODES",
            )
        },
    }
    return command, env, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teacher", "policy", "mixed"), default="policy")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to visualize. Relative paths resolve from --run-dir.")
    parser.add_argument(
        "--load-checkpoint-in-teacher-mode",
        action="store_true",
        help="Teacher mode normally skips checkpoint loading; set this to load --checkpoint anyway.",
    )
    parser.add_argument("--teacher-assist-mix", type=float, default=0.5, help="Used only with --mode mixed.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing manifest.json.")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Visualization output directory. Defaults to "
            "<run-dir>/eval_visualization_<checkpoint-stem>."
        ),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replay-size", type=int, default=2048)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--isaac-python", default=str(DEFAULT_ISAAC_PYTHON))
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest and print command without launching Isaac.")
    args = parser.parse_args()

    source_run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    checkpoint_stem = checkpoint.stem or "checkpoint"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else source_run_dir / f"eval_visualization_{checkpoint_stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command, env, manifest = build_command(args, source_run_dir, output_dir)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(_shell_join(command), flush=True)
    if args.dry_run:
        return 0

    with (output_dir / "stdout.log").open("w", encoding="utf-8") as out:
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
