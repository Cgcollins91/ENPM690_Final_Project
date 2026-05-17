"""Command-line interface for the standalone topdown curriculum launcher.

This is the public entrypoint used by local Python, IsaacSim Python, and the
Docker image.  It intentionally exposes only three operations:

* ``--dry-run``: print/write the manifest without launching Isaac Sim,
* ``--execute``: launch the materialized trainer command, and
* ``--validate-current``: check the legacy "current" profile surface.

Everything else belongs in a named profile.  That keeps command lines short
and makes every real experiment replayable from its manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config.profiles import SUBMISSION_PROFILES
from .manifest import stable_json, write_manifest
from .training_engine import build_plan, manifest_for_plan, run_plan
from .validate import validate_current_equivalence


def _default_project_root() -> Path:
    """Return the repository root inferred from the installed package location."""
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script entry point."""
    parser = argparse.ArgumentParser(description="Run ENPM690 topdown lift training without shell wrappers.")
    parser.add_argument(
        "--profile",
        choices=sorted(SUBMISSION_PROFILES),
        default="teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k",
    )
    parser.add_argument("--project-root", type=Path, default=_default_project_root())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print and optionally write the manifest without training.")
    parser.add_argument("--execute", action="store_true", help="Start the trainer.")
    parser.add_argument("--validate-current", action="store_true", help="Check current-run equivalence surface.")
    return parser.parse_args()


def main() -> int:
    """Parse arguments and run the selected launcher operation.

    ``--dry-run`` is the default behavior unless ``--execute`` is passed.  This
    makes it safe to inspect profiles inside Docker or CI without accidentally
    starting a multi-hour Isaac Sim training job.
    """
    args = parse_args()
    profile = SUBMISSION_PROFILES[args.profile]()
    try:
        plan = build_plan(profile, args.project_root)
        manifest = manifest_for_plan(plan)

        if args.validate_current or args.profile == "dagger_rl_current":
            errors = validate_current_equivalence(plan)
            if errors:
                for error in errors:
                    print(f"validation_error: {error}")
                return 2
            print("validation_ok: current red centered lift launch surface matches expected values")

        if args.manifest is not None:
            write_manifest(args.manifest, manifest)

        if args.dry_run or not args.execute:
            print(stable_json(manifest))
            return 0

        return run_plan(plan, args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"launch_error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
