"""

Module entrypoint for launching the current native training path

File map:

TrainingEntrypointPlan:  Validated plan for the compatibility entrypoint
build_entrypoint_plan:   Build validated context and configs before launching training
_print_entrypoint_help:  Print combined native and trainer CLI help without launching Isaac
main:                    Validate core CLI args and run through the backend interface
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

from .arg_normalization import normalized_training_cli_request
from .cli     import build_static_runtime_context, parse_training_cli
from .context import TrainerRuntimeContext
from .configs import build_runtime_config_bundle
from .configs import RuntimeConfigBundle


@dataclass(frozen=True)
class TrainingEntrypointPlan:
    """Validated plan for the compatibility entrypoint"""

    argv   : tuple[str, ...]  # raw command-line argument list for parser entrypoints
    context: TrainerRuntimeContext  # stores context for training entrypoint plan
    configs: RuntimeConfigBundle  # stores configs for training entrypoint plan


def build_entrypoint_plan(argv: list[str] | tuple[str, ...]) -> TrainingEntrypointPlan:
    """Build validated context and configs before launching training"""
    request = normalized_training_cli_request(parse_training_cli(argv))
    request.validate_supported()
    return TrainingEntrypointPlan(
        argv=request.monolith_argv(),
        context=build_static_runtime_context(request),
        configs=build_runtime_config_bundle(request),
    )


def _print_entrypoint_help() -> None:
    """Print combined native and trainer CLI help without launching Isaac

    Steps:
    - Resolve inputs for `_print_entrypoint_help` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    from .cli import build_training_cli_parser
    from ..native.native_entrypoint import build_native_arg_parser

    print("usage: python -m training [native options] [trainer options]")
    print()
    print("native options:")
    build_native_arg_parser().print_help()
    print()
    print("trainer options:")
    build_training_cli_parser().print_help()


def main(argv: list[str] | None = None) -> int:
    """Validate core CLI args and run through the backend interface

    Steps:
    - Resolve inputs for `main` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if any(item in {"-h", "--help"} for item in raw_argv):
        _print_entrypoint_help()
        return 0
    if "--compat-monolith" in raw_argv:
        print("error: monolith compatibility has been retired; use python -m training", file=sys.stderr)
        return 2

    from ..native.native_entrypoint import main as native_main

    return native_main(list(raw_argv))


if __name__ == "__main__":
    raise SystemExit(main())
