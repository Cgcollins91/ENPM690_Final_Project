"""Manifest creation for deterministic run identity.

The project has a lot of moving pieces: profile dataclasses, shell/env knobs,
IsaacLab task files, trainer scripts, and optional warm-start checkpoints.
Every serious run gets a manifest so that later GUI smoke tests and
presentations can answer two questions without guesswork:

* What exact command and environment launched this run?
* What source/checkpoint contents did that command depend on?

The manifest is intentionally content-hash based.  A dirty worktree is allowed
while iterating, but the run record will show the dirty status and hash every
source file that was copied into the launch contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {
    ".py",
    ".toml",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".urdf",
    ".usd",
}
# Keep this list project-relative.  Manifests must remain valid inside Docker
# and on another machine with only this repository checked out.
SOURCE_ROOTS = ("src", "scripts", "training", "tasks", "robots", "assets")


def stable_json(data: Any) -> str:
    """Serialize data deterministically for hashing and diffs."""

    return json.dumps(data, indent=2, sort_keys=True)


def digest(data: Any) -> str:
    """Return a short SHA256 digest for a manifest-like object."""

    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for one project file."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    """Run a bounded git query and return its text output."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _source_hashes(cwd: Path) -> dict[str, str]:
    """Return source-file hashes included in deterministic run manifests.

    ``runs/`` and checkpoint blobs are intentionally excluded; they are outputs
    or explicit checkpoint inputs.  This pass covers only human-readable source
    files that define the task, reward, config, trainer, and robot assets.
    """
    hashes: dict[str, str] = {}
    candidates = [cwd / "run.py", cwd / "requirements.txt", cwd / "pyproject.toml", cwd / "README.md"]
    for root_name in SOURCE_ROOTS:
        root = cwd / root_name
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES)
    for path in sorted(set(candidates)):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.exists() and path.is_file():
            hashes[str(path.relative_to(cwd))] = _file_sha256(path)
    return hashes


def _checkpoint_hashes(env: dict[str, str], cwd: Path) -> dict[str, str]:
    """Return hashes for checkpoint inputs referenced by the run environment.

    Only existing input checkpoints are hashed.  Output checkpoint paths like
    ``CHECKPOINT_PATH`` often point at a future file on a fresh run, so missing
    paths are skipped here and validated separately by the launcher when they
    are true prerequisites.
    """
    hashes: dict[str, str] = {}
    for key, value in sorted(env.items()):
        if not key.endswith("CKPT") and "CHECKPOINT" not in key:
            continue
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = cwd / path
        if path.exists() and path.is_file():
            hashes[value] = _file_sha256(path)
    return hashes


def _effective_cli_args(command: list[str]) -> dict[str, Any]:
    """Return a parsed flag map for the exact command line in the manifest."""

    parsed: dict[str, Any] = {}
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        value: Any = True
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            value = command[index + 1]
            index += 1
        existing = parsed.get(key)
        if existing is None:
            parsed[key] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            parsed[key] = [existing, value]
        index += 1
    return parsed


def config_payload(groups: tuple[object, ...]) -> list[dict[str, Any]]:
    """Return dataclass config values in manifest-friendly form."""

    payload: list[dict[str, Any]] = []
    for group in groups:
        if is_dataclass(group):
            payload.append({"type": type(group).__name__, "values": asdict(group)})
        else:
            payload.append({"type": type(group).__name__, "values": repr(group)})
    return payload


def build_manifest(
    *,
    profile_name       : str,
    profile_description: str,
    groups             : tuple[object, ...],
    command            : list[str],
    env                : dict[str, str],
    cwd                : Path,
    run_dir            : str,
) -> dict[str, Any]:
    """Capture the full launch contract needed to reproduce a run.

    The resulting dictionary is stable except for ``created_utc`` and any
    intentionally changed source/checkpoint hash.  ``config_digest`` is the
    quick "did the launch contract change?" identifier; ``source_digest`` is
    the quick "did the code/assets payload change?" identifier.
    """

    config = config_payload(groups)
    source_hashes = _source_hashes(cwd)
    checkpoint_hashes = _checkpoint_hashes(env, cwd)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "description": profile_description,
        "cwd": str(cwd),
        "run_dir": run_dir,
        "command": command,
        "effective_cli_args": _effective_cli_args(command),
        "env": env,
        "config": config,
        "git_commit": _git(["rev-parse", "HEAD"], cwd),
        "git_status_short": _git(["status", "--short"], cwd),
        "source_hashes": source_hashes,
        "source_digest": digest(source_hashes),
        "checkpoint_hashes": checkpoint_hashes,
    }
    manifest["config_digest"] = digest({"config": config, "command": command, "env": env})
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest, creating the parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(manifest) + "\n", encoding="utf-8")
