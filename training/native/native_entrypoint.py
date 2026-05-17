"""

Opt-in native Isaac training entrypoint

File map:

NativeEntrypointOptions:                    Native-only switches stripped before parsing trainer args
NativeEntrypointPlan:                       Validated plan for the native Isaac entrypoint
build_native_arg_parser:                    Build parser for native-only launcher flags
split_native_entrypoint_args:               Strip native-only flags from trainer CLI args
build_native_entrypoint_plan:               Build validated native context and runtime configs
zero_teacher_provider_startup_fn:           Wrap a startup function with zero-teacher hook provider methods
native_teacher_source_config_from_plan:     Build native teacher source config from action specs
native_teacher_provider_options_from_plan:  Build env-backed native teacher provider options
native_entrypoint_startup_fn:               Return startup function selected by native entrypoint options
native_entrypoint_event_callbacks:          Build default native entrypoint log and checkpoint callbacks
build_native_entrypoint_backend:            Build the backend selected by native entrypoint options
main:                                       Run native Isaac training through the refactored backend
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import sys

import torch

from ..actions.action_space import FINGER_JOINTS
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..core.entrypoint import build_entrypoint_plan
from .native_contact_teacher_parts import NativeContactTeacherAttrConfig
from .native_backend import NativeTrainerState
from .native_backend_factory import NativeLiveIsaacBackendOptions, native_live_isaac_backend
from .native_events import (
    NativeCheckpointEventCallbacks,
    NativeLogEventCallbacks,
    native_checkpoint_event,
    native_eval_event,
    native_log_event,
)
from .native_isaac_eval import build_native_isaac_eval_event_callbacks
from .native_isaac_hook_provider import NativeIsaacHookProvider
from .native_live import NativeLiveEventCallbacks
from .native_loop import NativeLoopOptions
from .native_startup import NativeStartupOptions, build_native_startup_state
from .native_teacher_sources import NativeTeacherEnvSourceConfig
from .native_teacher_startup import (
    NativeTeacherProviderOptions,
    env_teacher_provider_startup_fn,
)
from ..core.runner import TrainingBackend, run_training


NativeStartupFn = Callable[..., NativeTrainerState]


@dataclass(frozen=True)
class NativeEntrypointOptions:
    """Native-only switches stripped before parsing trainer args"""

    max_outer_steps            : int | None             = None  # step count used for max outer steps scheduling or reporting
    loose_obs_contract         : bool                   = False  # boolean value indicating the loose obs contract state for native entrypoint options
    zero_teacher_smoke         : bool                   = False  # boolean value indicating the zero teacher smoke state for native entrypoint options
    launch_app                 : bool                   = True  # boolean value indicating the launch app state for native entrypoint options
    create_env                 : bool                   = True  # boolean value indicating the create env state for native entrypoint options
    install_terminal_patch     : bool                   = True  # boolean value indicating the install terminal patch state for native entrypoint options
    teacher_provider           : str                    = "none"  # string teacher provider value used by native entrypoint options
    teacher_arm_method_names   : tuple[str, ...] | None = None  # ordered names used to resolve teacher arm method attributes
    teacher_legacy_arm_call    : bool                   = False  # boolean value indicating the teacher legacy arm call state for native entrypoint options
    validate_teacher_arm_action: bool                   = True  # boolean value indicating the validate teacher arm action state for native entrypoint options
    contact_attr_parts         : bool                   = False  # boolean value indicating the contact attr parts state for native entrypoint options
    contact_middle_teacher     : bool                   = False  # boolean value indicating the contact middle teacher state for native entrypoint options
    cache_teacher_action       : bool                   = True  # boolean value indicating the cache teacher action state for native entrypoint options


@dataclass(frozen=True)
class NativeEntrypointPlan:
    """Validated plan for the native Isaac entrypoint"""

    argv   : tuple[str, ...]  # raw command-line argument list for parser entrypoints
    context: TrainerRuntimeContext  # stores context for native entrypoint plan
    configs: RuntimeConfigBundle  # stores configs for native entrypoint plan
    options: NativeEntrypointOptions  # integer options value tracked by native entrypoint plan


def build_native_arg_parser() -> argparse.ArgumentParser:
    """Build parser for native-only launcher flags

    Steps:
    - Resolve inputs for `build_native_arg_parser` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-max-outer-steps", type=int, default=None)
    parser.add_argument("--native-loose-obs-contract", action="store_true")
    parser.add_argument("--native-zero-teacher-smoke", action="store_true")
    parser.add_argument(
        "--native-teacher-provider",
        choices=("none", "env"),
        default="none",
    )
    parser.add_argument("--native-teacher-arm-method", action="append", default=None)
    parser.add_argument("--native-teacher-legacy-arm-call", action="store_true")
    parser.add_argument(
        "--native-validate-teacher-arm-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--native-contact-attr-parts", action="store_true")
    parser.add_argument("--native-contact-middle-teacher", action="store_true")
    parser.add_argument(
        "--native-cache-teacher-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--native-launch-app", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-create-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--native-terminal-patch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def split_native_entrypoint_args(
    argv: list[str] | tuple[str, ...],  # Param: raw command-line arguments forwarded to the parser or subprocess
) -> tuple[tuple[str, ...], NativeEntrypointOptions]:
    """Strip native-only flags from trainer CLI args"""
    native_args, trainer_args = build_native_arg_parser().parse_known_args(tuple(argv))
    return (
        tuple(trainer_args),
        NativeEntrypointOptions(
            max_outer_steps=native_args.native_max_outer_steps,
            loose_obs_contract=bool(native_args.native_loose_obs_contract),
            zero_teacher_smoke=bool(native_args.native_zero_teacher_smoke),
            launch_app=bool(native_args.native_launch_app),
            create_env=bool(native_args.native_create_env),
            install_terminal_patch=bool(native_args.native_terminal_patch),
            teacher_provider=str(native_args.native_teacher_provider),
            teacher_arm_method_names=(
                None
                if native_args.native_teacher_arm_method is None
                else tuple(str(name) for name in native_args.native_teacher_arm_method)
            ),
            teacher_legacy_arm_call=bool(native_args.native_teacher_legacy_arm_call),
            validate_teacher_arm_action=bool(native_args.native_validate_teacher_arm_action),
            contact_attr_parts=bool(native_args.native_contact_attr_parts),
            contact_middle_teacher=bool(native_args.native_contact_middle_teacher),
            cache_teacher_action=bool(native_args.native_cache_teacher_action),
        ),
    )


def build_native_entrypoint_plan(
    argv: list[str] | tuple[str, ...],  # Param: raw command-line arguments forwarded to the parser or subprocess
) -> NativeEntrypointPlan:
    """Build validated native context and runtime configs"""
    trainer_argv, options = split_native_entrypoint_args(argv)
    plan = build_entrypoint_plan(trainer_argv)
    return NativeEntrypointPlan(
        argv=plan.argv,
        context=plan.context,
        configs=plan.configs,
        options=options,
    )


def zero_teacher_provider_startup_fn(
    base_startup_fn: NativeStartupFn = build_native_startup_state,  # Param: callback used to compute or fetch base startup
) -> NativeStartupFn:
    """Wrap a startup function with zero-teacher hook provider methods"""

    def _startup(
        context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        *,
        options: NativeStartupOptions,   # Param: input value used as options
    ) -> NativeTrainerState:
        """Process for `_startup`

        Steps:
        - Resolve inputs for `_startup` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        startup = base_startup_fn(context, configs, options=options)
        env = startup.get("env")
        if env is None:
            raise RuntimeError("native zero-teacher startup did not create env")

        def _teacher_action() -> torch.Tensor:
            return torch.zeros(
                (configs.counts.num_envs, context.dims.action_dim),
                dtype=torch.float32,
                device=context.device,
            )

        payload = dict(startup.payload)
        payload["raw_env"] = env
        payload["env"] = NativeIsaacHookProvider(env=env, teacher_action=_teacher_action)
        return NativeTrainerState(payload)

    return _startup


def native_teacher_source_config_from_plan(plan: NativeEntrypointPlan) -> NativeTeacherEnvSourceConfig:
    """Build native teacher source config from action specs"""
    joint_names = tuple(plan.context.action.env_action_spec.joint_names)
    num_fingers = sum(1 for name in joint_names if name in FINGER_JOINTS)
    num_arm = len(joint_names) - num_fingers
    return NativeTeacherEnvSourceConfig(
        num_arm=num_arm,
        num_fingers=num_fingers,
    )


def native_teacher_provider_options_from_plan(
    plan: NativeEntrypointPlan,  # Param: precomputed plan object consumed by this helper
) -> NativeTeacherProviderOptions:
    """Build env-backed native teacher provider options"""
    source_config = native_teacher_source_config_from_plan(plan)
    contact_attr_config = (
        NativeContactTeacherAttrConfig(
            num_arm=source_config.num_arm,
            num_fingers=source_config.num_fingers,
            max_fraction=max(float(plan.configs.teacher.topdown_contact_teacher_max_fraction), 0.0),
            middle_scale=max(float(plan.configs.teacher.topdown_contact_teacher_middle_scale), 0.0),
            use_middle_teacher=bool(plan.options.contact_middle_teacher),
        )
        if plan.options.contact_attr_parts
        else None
    )
    return NativeTeacherProviderOptions(
        arm_method_names=plan.options.teacher_arm_method_names,
        legacy_arm_call=plan.options.teacher_legacy_arm_call,
        validate_arm_action=plan.options.validate_teacher_arm_action,
        use_contact_attr_parts=plan.options.contact_attr_parts,
        contact_attr_config=contact_attr_config,
        cache_teacher_action=plan.options.cache_teacher_action,
    )


def native_entrypoint_startup_fn(
    plan: NativeEntrypointPlan,                                # Param: precomputed plan object consumed by this helper
    *,
    startup_fn: NativeStartupFn = build_native_startup_state,  # Param: callback used to compute or fetch startup
) -> NativeStartupFn:
    """Return startup function selected by native entrypoint options"""
    if plan.options.zero_teacher_smoke and plan.options.teacher_provider != "none":
        raise RuntimeError("native zero-teacher smoke cannot be combined with native teacher provider")
    if plan.options.zero_teacher_smoke:
        return zero_teacher_provider_startup_fn(startup_fn)
    if plan.options.teacher_provider == "env":
        return env_teacher_provider_startup_fn(
            options=native_teacher_provider_options_from_plan(plan),
            source_config=native_teacher_source_config_from_plan(plan),
            startup_fn=startup_fn,
        )
    return startup_fn


def native_entrypoint_event_callbacks(
    *,
    topdown_curriculum_obs_contract: bool = True,  # Param: boolean input controlling topdown curriculum obs contract
) -> NativeLiveEventCallbacks:
    """Build default native entrypoint log and checkpoint callbacks"""
    return NativeLiveEventCallbacks(
        on_log=native_log_event(NativeLogEventCallbacks(trace_fn=print)),
        on_eval=native_eval_event(
            build_native_isaac_eval_event_callbacks(
                result_fn=None,
                topdown_curriculum_obs_contract=topdown_curriculum_obs_contract,
            )
        ),
        on_checkpoint=native_checkpoint_event(NativeCheckpointEventCallbacks(trace_fn=print)),
    )


def build_native_entrypoint_backend(
    plan: NativeEntrypointPlan,                                # Param: precomputed plan object consumed by this helper
    *,
    startup_fn: NativeStartupFn = build_native_startup_state,  # Param: callback used to compute or fetch startup
) -> TrainingBackend:
    """Build the backend selected by native entrypoint options"""
    actual_startup_fn = native_entrypoint_startup_fn(plan, startup_fn=startup_fn)
    return native_live_isaac_backend(
        configs=plan.configs,
        options=NativeLiveIsaacBackendOptions(
            startup_options=NativeStartupOptions(
                launch_app=plan.options.launch_app,
                create_env=plan.options.create_env,
                install_terminal_patch=plan.options.install_terminal_patch,
            ),
            loop_options=NativeLoopOptions(
                eval_every=plan.configs.eval.eval_every,
                max_outer_steps=plan.options.max_outer_steps,
            ),
            event_callbacks=native_entrypoint_event_callbacks(
                topdown_curriculum_obs_contract=not plan.options.loose_obs_contract,
            ),
            topdown_curriculum_obs_contract=not plan.options.loose_obs_contract,
        ),
        startup_fn=actual_startup_fn,
    )


def main(argv: list[str] | None = None) -> int:
    """Run native Isaac training through the refactored backend

    Steps:
    - Resolve inputs for `main` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if any(item in {"-h", "--help"} for item in raw_argv):
        from ..core.cli import build_training_cli_parser

        print("usage: python -m training.native_entrypoint [native options] [trainer options]")
        print()
        print("native options:")
        build_native_arg_parser().print_help()
        print()
        print("trainer options:")
        build_training_cli_parser().print_help()
        return 0
    plan = build_native_entrypoint_plan(raw_argv)
    backend = build_native_entrypoint_backend(plan)
    result = run_training(plan.context, backend=backend)
    print(
        "native_training_ok "
        f"status={result.status} "
        f"global_step={result.global_step} "
        f"transitions={result.metrics.get('transitions_collected', 0)}"
    )
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
