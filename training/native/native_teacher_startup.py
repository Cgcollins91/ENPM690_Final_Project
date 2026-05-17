"""

Startup factories for env-backed native teacher providers

File map:

NativeTeacherProviderOptions:        Options for installing env-backed native teacher hooks
build_env_teacher_provider_builder:  Build a ProviderBuilder for hook_provider_startup_fn
env_teacher_provider_startup_fn:     Build startup_fn that installs env-backed native teacher hooks
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_contact_teacher_parts import NativeContactTeacherAttrConfig
from .native_hook_startup import ProviderBuilder, hook_provider_startup_fn
from .native_startup import build_native_startup_state
from .native_teacher_arm_backends import (
    TopdownDifferentialIKTeacherArmBackend,
    build_env_teacher_arm_backend,
    env_has_teacher_arm_method,
)
from .native_teacher_sources import (
    ContactPartsFn,
    NativeTeacherEnvSourceConfig,
    build_env_native_teacher_hook_provider,
)


@dataclass(frozen=True)
class NativeTeacherProviderOptions:
    """Options for installing env-backed native teacher hooks"""

    arm_method_names      : tuple[str, ...] | None                = None  # ordered names used to resolve arm method attributes
    legacy_arm_call       : bool                                  = False  # boolean value indicating the legacy arm call state for native teacher provider options
    validate_arm_action   : bool                                  = True  # boolean value indicating the validate arm action state for native teacher provider options
    use_contact_attr_parts: bool                                  = False  # boolean value indicating the use contact attr parts state for native teacher provider options
    contact_attr_config   : NativeContactTeacherAttrConfig | None = None  # stores contact attr config for native teacher provider options
    cache_teacher_action  : bool                                  = True  # boolean value indicating the cache teacher action state for native teacher provider options


def build_env_teacher_provider_builder(
    *,
    options         : NativeTeacherProviderOptions = NativeTeacherProviderOptions(),  # Param: input value used as options
    source_config   : NativeTeacherEnvSourceConfig = NativeTeacherEnvSourceConfig(),  # Param: input value used as source config
    contact_parts_fn: ContactPartsFn | None        = None,  # Param: callback used to compute or fetch contact parts
    arm_backend     : object | None                = None,  # Param: input value used as arm backend
) -> ProviderBuilder:
    """Build a ProviderBuilder for hook_provider_startup_fn"""

    def _provider(env: object, context: TrainerRuntimeContext, configs: RuntimeConfigBundle):
        backend = arm_backend
        if backend is None:
            if env_has_teacher_arm_method(env, options.arm_method_names):
                backend = build_env_teacher_arm_backend(
                    env,
                    method_names=options.arm_method_names,
                    legacy_call=options.legacy_arm_call,
                    validate=options.validate_arm_action,
                    num_arm=source_config.num_arm,
                )
                print("native_teacher_provider arm_backend=env_method", flush=True)
            else:
                backend = TopdownDifferentialIKTeacherArmBackend(
                    env=env,
                    args=context.args,
                    num_arm=source_config.num_arm,
                )
                print("native_teacher_provider arm_backend=topdown_differential_ik", flush=True)
        return build_env_native_teacher_hook_provider(
            env=env,
            context=context,
            configs=configs,
            arm_backend=backend,
            source_config=source_config,
            contact_parts_fn=contact_parts_fn,
            use_contact_attr_parts=options.use_contact_attr_parts,
            contact_attr_config=options.contact_attr_config,
            cache_teacher_action=options.cache_teacher_action,
        )

    return _provider


def env_teacher_provider_startup_fn(
    *,
    options         : NativeTeacherProviderOptions = NativeTeacherProviderOptions(),  # Param: input value used as options
    source_config   : NativeTeacherEnvSourceConfig = NativeTeacherEnvSourceConfig(),  # Param: input value used as source config
    contact_parts_fn: ContactPartsFn | None        = None,  # Param: callback used to compute or fetch contact parts
    arm_backend     : object | None                = None,  # Param: input value used as arm backend
    startup_fn=build_native_startup_state,                                         # Param: callback used to compute or fetch startup
):
    """Build startup_fn that installs env-backed native teacher hooks"""
    provider_builder = build_env_teacher_provider_builder(
        options=options,
        source_config=source_config,
        contact_parts_fn=contact_parts_fn,
        arm_backend=arm_backend,
    )
    return hook_provider_startup_fn(provider_builder, startup_fn=startup_fn)
