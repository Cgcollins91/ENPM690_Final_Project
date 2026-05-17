"""

Startup wrappers for native hook providers

File map:

startup_with_hook_provider:  Run startup and wrap the env with a native hook provider
hook_provider_startup_fn:    Return a startup_fn that installs a native hook provider
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_backend import NativeTrainerState
from .native_isaac_hook_provider import NativeIsaacHookProvider
from .native_startup import NativeStartupOptions, build_native_startup_state


ProviderBuilder = Callable[
    [object, TrainerRuntimeContext, RuntimeConfigBundle],
    NativeIsaacHookProvider,
]
NativeStartupFn = Callable[..., NativeTrainerState]


def startup_with_hook_provider(
    *,
    context         : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs         : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    options         : NativeStartupOptions,  # Param: input value used as options
    provider_builder: ProviderBuilder,  # Param: input value used as provider builder
    startup_fn      : NativeStartupFn = build_native_startup_state,  # Param: callback used to compute or fetch startup
) -> NativeTrainerState:
    """Run startup and wrap the env with a native hook provider

    Steps:
    - Resolve inputs for `startup_with_hook_provider` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    startup = startup_fn(context, configs, options=options)
    env = startup.get("env")
    if env is None:
        raise RuntimeError("native hook startup did not create env")
    provider = provider_builder(env, context, configs)
    has_teacher = callable(getattr(provider, "compute_teacher_action", None))
    print(
        "native_teacher_provider status=installed "
        f"teacher_action_fn={int(has_teacher)} "
        f"provider={type(provider).__name__}",
        flush=True,
    )
    payload = dict(startup.payload)
    payload["raw_env"] = env
    payload["env"] = provider
    return NativeTrainerState(payload)


def hook_provider_startup_fn(
    provider_builder: ProviderBuilder,                         # Param: input value used as provider builder
    *,
    startup_fn: NativeStartupFn = build_native_startup_state,  # Param: callback used to compute or fetch startup
) -> NativeStartupFn:
    """Return a startup_fn that installs a native hook provider"""

    def _startup(
        context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        *,
        options: NativeStartupOptions,   # Param: input value used as options
    ) -> NativeTrainerState:
        return startup_with_hook_provider(
            context=context,
            configs=configs,
            options=options,
            provider_builder=provider_builder,
            startup_fn=startup_fn,
        )

    return _startup
