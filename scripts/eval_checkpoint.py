#!/usr/bin/env python3
"""Evaluate one v35 checkpoint from its training manifest.

This is the single-checkpoint companion to ``eval_curriculum_checkpoints.py``.
It reads the source run manifest, swaps in the requested checkpoint, runs the
native trainer in ``--play`` mode, and writes a compact summary under
``--run-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.curriculum.base_v35 import V35_MODULE, V35_SCRIPT, apply_v35_base  # noqa: E402


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
    "RESUME_CHECKPOINT",
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


def _replace_arg(command: list[str], flag: str, value: object) -> None:
    value_s = str(value)
    if flag in command:
        idx = command.index(flag)
        if idx + 1 < len(command) and not command[idx + 1].startswith("--"):
            command[idx + 1] = value_s
        else:
            command.insert(idx + 1, value_s)
    else:
        command.extend([flag, value_s])


def _remove_arg(command: list[str], flag: str, *, takes_value: bool = True) -> None:
    while flag in command:
        idx = command.index(flag)
        del command[idx]
        if takes_value and idx < len(command) and not command[idx].startswith("--"):
            del command[idx]


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


def _manifest_path(args: argparse.Namespace, checkpoint: Path) -> Path:
    if args.manifest:
        path = Path(args.manifest)
        return path / "manifest.json" if path.is_dir() else path
    if args.source_run_dir:
        return Path(args.source_run_dir) / "manifest.json"
    return checkpoint.parent / "manifest.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"manifest is not a JSON object: {path}")
    return manifest


def _resolve_checkpoint(raw_checkpoint: str, manifest_path: Path, source_run_dir: str) -> Path:
    checkpoint = Path(raw_checkpoint)
    if checkpoint.is_absolute() or checkpoint.parent != Path("."):
        return checkpoint
    base = Path(source_run_dir) if source_run_dir else manifest_path.parent
    return base / checkpoint


def _base_command_env(args: argparse.Namespace, manifest: dict[str, Any] | None) -> tuple[list[str], dict[str, str]]:
    env = _clean_env(os.environ)
    if manifest is not None:
        raw_command = manifest.get("command")
        if not isinstance(raw_command, list) or not all(isinstance(item, str) for item in raw_command):
            raise RuntimeError("source manifest does not contain a string-list command")
        command = list(raw_command)
        manifest_env = manifest.get("env", {})
        if isinstance(manifest_env, dict):
            env.update({str(key): str(value) for key, value in manifest_env.items()})
        if command:
            command[0] = str(Path(args.isaac_python).expanduser())
        return command, env

    isaac_python = Path(args.isaac_python).expanduser()
    launcher = [V35_SCRIPT] if V35_SCRIPT else ["-m", V35_MODULE]
    command = [str(isaac_python), "-u", *launcher]
    apply_v35_base(command, env)
    return command, env


def _bounded_jsonl_tail(path: Path, *, max_bytes: int = 1_000_000, max_rows: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()
        data = f.read(max_bytes)
    rows: list[dict[str, Any]] = []
    for line in data.decode("utf-8", errors="replace").splitlines()[-max_rows:]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _safe_float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key not in row:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return default


def _nested_float(row: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    nested = row.get(section)
    if not isinstance(nested, dict):
        return float(default)
    return _safe_float(nested, key, default=default)


def _max_value(rows: list[dict[str, Any]], *extractors, default: float = 0.0) -> float:
    values: list[float] = []
    for row in rows:
        for extractor in extractors:
            try:
                values.append(float(extractor(row)))
            except (TypeError, ValueError):
                continue
    return max(values) if values else float(default)


def _mean_value(rows: list[dict[str, Any]], extractor, default: float = 0.0) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(extractor(row)))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else float(default)


def _summary_from_rollout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_rows = [row for row in rows if row.get("mode") == "topdown_curriculum_summary"]
    if not rollout_rows:
        return {}
    last = rollout_rows[-1]
    best_lift = _max_value(
        rollout_rows,
        lambda row: _safe_float(row, "lift"),
        lambda row: _nested_float(row, "topdown_metrics", "topdown_lift_max"),
        lambda row: _nested_float(row, "lift_stage_metrics", "lift_stage/lift_height_max"),
    )
    return {
        "mode": "rollout_summary",
        "rows": len(rollout_rows),
        "last_global_step": int(_safe_float(last, "global_step", default=0.0)),
        "success_rate": _max_value(
            rollout_rows,
            lambda row: _safe_float(row, "success"),
            lambda row: _nested_float(row, "topdown_metrics", "topdown_success_rate"),
        ),
        "physical_success_rate": 0.0,
        "best_lift": best_lift,
        "median_best_lift": best_lift,
        "median_best_block_disp": _max_value(rollout_rows, lambda row: _safe_float(row, "block_disp")),
        "median_best_stage": _max_value(
            rollout_rows,
            lambda row: _safe_float(row, "stage", default=-1.0),
            lambda row: _nested_float(row, "topdown_metrics", "topdown_stage_mean", default=-1.0),
        ),
        "stage1_rate": _max_value(rollout_rows, lambda row: _nested_float(row, "topdown_metrics", "topdown_stage_ge1_rate")),
        "stage2_rate": _max_value(rollout_rows, lambda row: _nested_float(row, "topdown_metrics", "topdown_stage_ge2_rate")),
        "median_best_strict_contact": _max_value(
            rollout_rows,
            lambda row: _safe_float(row, "strict"),
            lambda row: _nested_float(row, "topdown_metrics", "topdown_strict_contact_mean"),
            lambda row: _nested_float(row, "lift_stage_metrics", "lift_stage/opposed_contact_max"),
        ),
        "median_best_contact": _max_value(
            rollout_rows,
            lambda row: _safe_float(row, "contact"),
            lambda row: _nested_float(row, "topdown_metrics", "topdown_contact_mean"),
        ),
        "reward_last": _safe_float(last, "reward"),
        "reward_mean": _mean_value(rollout_rows, lambda row: _safe_float(row, "reward")),
        "assist_mix_last": _safe_float(last, "assist_mix"),
        "action_source_last": last.get("action_source"),
    }


def _summary_from_log(path: Path) -> dict[str, Any]:
    rows = _bounded_jsonl_tail(path)
    aggregates = [row for row in rows if row.get("mode") == "eval_aggregate"]
    summaries = [row for row in rows if row.get("mode") == "eval_summary"]
    source_rows = aggregates or summaries
    if not source_rows:
        return _summary_from_rollout(rows)
    if aggregates:
        selected = aggregates[-1]
    else:
        selected = {
            "mode": "eval_summary_compact",
            "eval_episodes": len(summaries),
            "eval_success_rate": sum(_safe_float(row, "eval_success") for row in summaries) / max(1, len(summaries)),
            "eval_median_best_lift": max(_safe_float(row, "eval_best_lift") for row in summaries),
            "eval_median_best_block_disp": max(_safe_float(row, "eval_best_block_disp") for row in summaries),
            "eval_median_best_topdown_stage": max(_safe_float(row, "eval_best_topdown_stage", default=-1.0) for row in summaries),
        }
    return {
        "mode": selected.get("mode"),
        "eval_episodes": selected.get("eval_episodes", len(summaries)),
        "success_rate": _safe_float(selected, "eval_success_rate", "success_rate"),
        "physical_success_rate": _safe_float(selected, "eval_physical_success_rate", "physical_success_rate"),
        "median_best_lift": _safe_float(selected, "eval_median_best_lift", "median_best_lift", "eval_best_lift"),
        "best_lift": _safe_float(selected, "eval_best_lift", "median_best_lift", "eval_median_best_lift"),
        "median_best_block_disp": _safe_float(selected, "eval_median_best_block_disp", "median_best_block_disp"),
        "median_best_stage": _safe_float(selected, "eval_median_best_topdown_stage", "median_best_stage", default=-1.0),
        "stage2_rate": _safe_float(selected, "eval_topdown_stage2_episode_rate", "stage2_rate"),
        "median_best_strict_contact": _safe_float(selected, "eval_median_best_strict_light_contact"),
        "median_best_contact": _safe_float(selected, "eval_median_best_contact"),
    }


def _format_metrics_line(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "eval_metrics unavailable"
    return (
        f"eval_metrics mode={metrics.get('mode', 'unknown')} "
        f"success={float(metrics.get('success_rate', 0.0)):.3f} "
        f"stage2={float(metrics.get('stage2_rate', 0.0)):.3f} "
        f"best_lift={float(metrics.get('best_lift', 0.0)):.4f} "
        f"best_stage={float(metrics.get('median_best_stage', -1.0)):.1f} "
        f"strict={float(metrics.get('median_best_strict_contact', 0.0)):.3f} "
        f"contact={float(metrics.get('median_best_contact', 0.0)):.3f} "
        f"disp={float(metrics.get('median_best_block_disp', 0.0)):.4f}"
    )


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_plan(args: argparse.Namespace) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    preliminary_checkpoint = Path(args.checkpoint)
    manifest_path = _manifest_path(args, preliminary_checkpoint)
    manifest = None if args.no_manifest else _load_manifest(manifest_path)
    source_run_dir = ""
    if args.source_run_dir:
        source_run_dir = args.source_run_dir
    elif manifest is not None:
        source_run_dir = str(manifest_path.parent)
    checkpoint = _resolve_checkpoint(args.checkpoint, manifest_path, source_run_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    run_dir = Path(args.run_dir)
    log_jsonl = run_dir / "log.jsonl"
    outer_steps = max(1, ((int(args.episodes) + int(args.num_envs) - 1) // int(args.num_envs)) * int(args.steps))
    total_steps = outer_steps * int(args.num_envs)
    log_every = max(1, min(100, total_steps))
    command, env = _base_command_env(args, manifest)

    _remove_arg(command, "--resume-checkpoint")
    _remove_arg(command, "--resume-replay", takes_value=False)
    _remove_arg(command, "--resume-global-step", takes_value=False)
    _remove_arg(command, "--actor-init-checkpoint")
    _remove_arg(command, "--handoff-checkpoint-path")
    _remove_arg(command, "--final-handoff-checkpoint-path")
    _remove_arg(command, "--stop-after-handoff-checkpoint", takes_value=False)
    _remove_arg(command, "--tensorboard-dir")
    _remove_arg(command, "--save-replay-in-checkpoint", takes_value=False)
    _remove_arg(command, "--adaptive-policy-assist", takes_value=False)

    if args.gui:
        _remove_arg(command, "--headless", takes_value=False)
    else:
        _add_flag(command, "--headless")
    _add_flag(command, "--play")
    _replace_arg(command, "--checkpoint-path", checkpoint)
    _replace_arg(command, "--log-jsonl", log_jsonl)
    _replace_arg(command, "--total-steps", total_steps)
    _replace_arg(command, "--start-steps", 0)
    _replace_arg(command, "--updates-per-step", 0)
    _replace_arg(command, "--num-envs", args.num_envs)
    _replace_arg(command, "--eval-steps", args.steps)
    _replace_arg(command, "--eval-episodes", args.episodes)
    _replace_arg(command, "--play-episodes", args.episodes)
    _replace_arg(command, "--eval-every", 0)
    _replace_arg(command, "--checkpoint-every", 0)
    _replace_arg(command, "--rolling-checkpoint-every", 0)
    _replace_arg(command, "--log-every", log_every)
    _replace_arg(command, "--policy-bc-relabel", 0)
    _replace_arg(command, "--rl-policy-bc-relabel", 0)
    _replace_arg(command, "--policy-assist-mix", 0.0)
    _replace_arg(command, "--policy-assist-mix-floor", 0.0)
    _replace_arg(command, "--policy-assist-arm-mix", 0.0)
    _replace_arg(command, "--policy-assist-arm-mix-floor", 0.0)
    _replace_arg(command, "--policy-assist-finger-mix", 0.0)
    _replace_arg(command, "--policy-assist-finger-mix-floor", 0.0)
    _replace_arg(command, "--teacher-bc-weight", 0.0)
    _replace_arg(command, "--teacher-bc-arm-weight", 0.0)
    _replace_arg(command, "--teacher-bc-finger-weight", 0.0)
    _replace_arg(command, "--bc-only-weight", 0.0)
    _replace_arg(command, "--bc-only-arm-weight", 0.0)
    _replace_arg(command, "--bc-only-finger-weight", 0.0)
    _replace_arg(command, "--exploration-noise", 0.0)
    _replace_arg(command, "--exploration-noise-finger", 0.0)
    _replace_arg(command, "--rl-policy-assist-mix", 0.0)
    _replace_arg(command, "--rl-policy-assist-mix-floor", 0.0)
    _replace_arg(command, "--rl-teacher-bc-weight", 0.0)
    _replace_arg(command, "--rl-teacher-bc-arm-weight", 0.0)
    _replace_arg(command, "--rl-teacher-bc-finger-weight", 0.0)
    _replace_arg(command, "--eval-teacher-assist-mix", args.teacher_assist_mix)
    _replace_arg(command, "--sleep", args.sleep)
    _replace_arg(command, "--tensorboard-dir", "off")

    env.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
            "RUN_DIR": str(run_dir),
            "CHECKPOINT_PATH": str(checkpoint),
            "LOG_JSONL": str(log_jsonl),
            "TOTAL_STEPS": str(total_steps),
            "START_STEPS": "0",
            "UPDATES_PER_STEP": "0",
            "NUM_ENVS": str(args.num_envs),
            "EVAL_STEPS": str(args.steps),
            "EVAL_EPISODES": str(args.episodes),
            "EVAL_EVERY": "0",
            "CHECKPOINT_EVERY": "0",
            "ROLLING_CHECKPOINT_EVERY": "0",
            "SAVE_REPLAY_IN_CHECKPOINT": "0",
            "LOG_EVERY": str(log_every),
            "HEADLESS": "0" if args.gui else "1",
            "ENABLE_CAMERAS": "1" if args.gui else "0",
            "DISABLE_CAMERA_PERCEPTION": "1",
        }
    )
    env.update(_parse_env_overrides(args.env))

    plan = {
        "checkpoint": str(checkpoint),
        "source_manifest": None if manifest is None else str(manifest_path),
        "source_run_dir": source_run_dir or None,
        "run_dir": str(run_dir),
        "steps": int(args.steps),
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "outer_steps": int(outer_steps),
        "total_steps": int(total_steps),
        "log_every": int(log_every),
        "teacher_assist_mix": float(args.teacher_assist_mix),
        "gui": bool(args.gui),
        "command": command,
    }
    return command, env, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate. Basenames resolve inside --source-run-dir.")
    parser.add_argument("--source-run-dir", default="", help="Directory containing manifest.json for this checkpoint.")
    parser.add_argument("--manifest", default="", help="Explicit manifest path or run directory. Defaults to checkpoint-dir/manifest.json.")
    parser.add_argument("--run-dir", default="", help="Defaults to <checkpoint-dir>/eval_<checkpoint-stem>.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--teacher-assist-mix", type=float, default=0.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--isaac-python", default=str(DEFAULT_ISAAC_PYTHON))
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-manifest", action="store_true", help="Build a fresh v35 command instead of replaying a source manifest.")
    parser.add_argument("--stream", action="store_true", help="Also stream trainer stdout to this terminal.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run_dir:
        checkpoint_name = Path(args.checkpoint).stem
        base = Path(args.source_run_dir) if args.source_run_dir else Path(args.checkpoint).parent
        args.run_dir = str(base / f"eval_{checkpoint_name}")
    return args


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    command, env, plan = build_plan(args)
    stdout_log = run_dir / "stdout.log"
    summary_json = run_dir / "summary.json"

    print(_shell_join(command), flush=True)
    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    started = time.time()
    with stdout_log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            out.write(line)
            if args.stream:
                print(line, end="")
        returncode = process.wait()

    summary = {
        "checkpoint": plan["checkpoint"],
        "run_dir": str(run_dir),
        "returncode": int(returncode),
        "duration_s": time.time() - started,
        "metrics": _summary_from_log(run_dir / "log.jsonl"),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (run_dir / "summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(_format_metrics_line(summary["metrics"]), flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
