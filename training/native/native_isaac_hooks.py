"""

Isaac env hook factory for native live training

File map:

NativeIsaacHookConfig:             Attribute names used to discover Isaac live hooks
_owners:                           Handle owners logic
_get_from_owner:                   Handle get from owner logic
_find_value:                       Handle find value logic
_call_or_value:                    Handle call or value logic
_tensor_value:                     Handle tensor value logic
_robot_from_env:                   Handle robot from env logic
_callable_hook:                    Handle callable hook logic
_mask_fn:                          Handle mask fn logic
_preroll_action_fn:                Handle preroll action fn logic
_existing_names_fn:                Handle existing names fn logic
_action_assembly_config:           Handle action assembly config logic
build_native_isaac_live_hooks_fn:  Return a NativeLiveTrainerAssembly hook factory for Isaac envs
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..actions.action_space import get_action_mapping
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_backend import NativeTrainerState
from .native_components import NativeTrainingComponents
from .native_live import NativeLiveHooks, NativeLiveRolloutState
from .native_step import NativeEnvActionAssemblyConfig


HookCallable = Callable[..., Any]


@dataclass(frozen=True)
class NativeIsaacHookConfig:
    """Attribute names used to discover Isaac live hooks"""

    teacher_action_names       : tuple[str, ...] = (                                                  # ordered names used to resolve teacher action attributes
        "teacher_action_fn",
        "compute_teacher_action",
        "get_teacher_action",
    )
    assemble_action_names      : tuple[str, ...] = (                                                  # ordered names used to resolve assemble action attributes
        "assemble_env_action_fn",
        "assemble_env_action",
    )
    arm_reduced_action_names   : tuple[str, ...] = (                                                  # ordered names used to resolve arm reduced action attributes
        "arm_reduced_action_fn",
        "current_arm_reduced_action",
        "_teacher_arm_reduced_action",
    )
    mapped_indices_names     : tuple[str, ...] = ("mapped_indices", "_mapped_indices")  # source names for mapped action-column indices
    mapped_scales_names      : tuple[str, ...] = ("mapped_scales", "_mapped_scales")  # source names for mapped action-column scales
    preroll_action_names     : tuple[str, ...] = ("preroll_action_fn", "current_preroll_action")  # ordered names used to resolve preroll action attributes
    preroll_mask_names       : tuple[str, ...] = ("preroll_mask_fn", "current_preroll_mask")  # ordered names used to resolve preroll mask attributes
    active_env_mask_names    : tuple[str, ...] = ("active_env_mask_fn", "current_active_env_mask")  # ordered names used to resolve active env mask attributes
    existing_checkpoint_names: tuple[str, ...] = (  # ordered names used to resolve existing checkpoint attributes
        "existing_checkpoint_names_fn",
        "existing_checkpoint_names",
    )
    use_action_assembly_config : bool = True                                                          # boolean value indicating the use action assembly config state for native isaac hook config


def _owners(state: NativeTrainerState) -> tuple[object, ...]:
    env = state.get("env")
    return tuple(owner for owner in (state.payload, env) if owner is not None)


def _get_from_owner(owner: object, name: str) -> object | None:
    if isinstance(owner, dict):
        return owner.get(name)
    return getattr(owner, name, None)


def _find_value(owners: Sequence[object], names: Sequence[str]) -> object | None:
    for owner in owners:
        for name in names:
            value = _get_from_owner(owner, name)
            if value is not None:
                return value
    return None


def _call_or_value(value: object, runtime: NativeLiveRolloutState | None = None) -> object:
    if callable(value):
        try:
            return value(runtime) if runtime is not None else value()
        except TypeError:
            return value()
    return value


def _tensor_value(value: object, runtime: NativeLiveRolloutState | None = None) -> torch.Tensor | None:
    resolved = _call_or_value(value, runtime)
    return resolved if torch.is_tensor(resolved) else None


def _robot_from_env(env: object) -> object | None:
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    try:
        return scene["robot"]
    except Exception:
        return None


def _callable_hook(owners: Sequence[object], names: Sequence[str]) -> HookCallable | None:
    value = _find_value(owners, names)
    return value if callable(value) else None


def _mask_fn(owners: Sequence[object], names: Sequence[str]):
    value = _find_value(owners, names)
    if value is None:
        return None

    def _mask(runtime: NativeLiveRolloutState) -> torch.Tensor | None:
        return _tensor_value(value, runtime)

    return _mask


def _preroll_action_fn(owners: Sequence[object], names: Sequence[str]):
    value = _find_value(owners, names)
    if value is None:
        return None

    def _action(runtime: NativeLiveRolloutState) -> torch.Tensor | None:
        return _tensor_value(value, runtime)

    return _action


def _existing_names_fn(owners: Sequence[object], names: Sequence[str]):
    value = _find_value(owners, names)
    if value is None:
        return None

    def _names() -> tuple[str, ...]:
        resolved = _call_or_value(value)
        if resolved is None:
            return ()
        return tuple(str(name) for name in resolved)

    return _names


def _action_assembly_config(
    *,
    owners     : Sequence[object],  # Param: ordered input collection of owners entries
    components : NativeTrainingComponents,  # Param: input value used as components
    context    : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs    : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    hook_config: NativeIsaacHookConfig,  # Param: input value used as hook config
):
    if not hook_config.use_action_assembly_config:
        return None
    cached_mapping: tuple[torch.Tensor, torch.Tensor] | None = None

    def _config() -> NativeEnvActionAssemblyConfig:
        """Process for `_config`

        Steps:
        - Resolve inputs for `_config` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        nonlocal cached_mapping
        env = owners[-1] if owners else None
        arm_value = _find_value(owners, hook_config.arm_reduced_action_names)
        mapped_indices = _tensor_value(_find_value(owners, hook_config.mapped_indices_names))
        mapped_scales = _tensor_value(_find_value(owners, hook_config.mapped_scales_names))
        if (mapped_indices is None or mapped_scales is None) and env is not None:
            robot = _robot_from_env(env)
            if robot is not None:
                if cached_mapping is None:
                    cached_mapping = get_action_mapping(
                        robot,
                        context.device,
                        context.action.env_action_spec,
                    )
                mapped_indices, mapped_scales = cached_mapping
        return NativeEnvActionAssemblyConfig(
            gate_config=components.td3_config.gate_config,
            arm_controller=configs.teacher.arm_controller,
            finger_action_mode=configs.teacher.finger_action_mode,
            arm_reduced_action=_tensor_value(arm_value),
            env=env,
            mapped_indices=mapped_indices,
            mapped_scales=mapped_scales,
            finger_delta_scale=configs.teacher.finger_delta_scale,
        )

    return _config


def build_native_isaac_live_hooks_fn(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    *,
    hook_config: NativeIsaacHookConfig = NativeIsaacHookConfig(),  # Param: input value used as hook config
):
    """Return a NativeLiveTrainerAssembly hook factory for Isaac envs"""

    def _hooks(
        state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
        components: NativeTrainingComponents,  # Param: input value used as components
        runtime   : NativeLiveRolloutState,  # Param: input value used as runtime
    ) -> NativeLiveHooks:
        """Process for `_hooks`

        Steps:
        - Resolve inputs for `_hooks` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        del runtime
        env = state.get("env")
        if env is None or not hasattr(env, "step"):
            raise RuntimeError("native Isaac live hooks require env.step")
        owners = _owners(state)
        return NativeLiveHooks(
            env_step_fn=env.step,
            teacher_action_fn=_callable_hook(owners, hook_config.teacher_action_names),
            assemble_env_action_fn=_callable_hook(owners, hook_config.assemble_action_names),
            action_assembly_fn=_action_assembly_config(
                owners=owners,
                components=components,
                context=context,
                configs=configs,
                hook_config=hook_config,
            ),
            preroll_action_fn=_preroll_action_fn(owners, hook_config.preroll_action_names),
            preroll_mask_fn=_mask_fn(owners, hook_config.preroll_mask_names),
            active_env_mask_fn=_mask_fn(owners, hook_config.active_env_mask_names),
            existing_checkpoint_names_fn=_existing_names_fn(
                owners,
                hook_config.existing_checkpoint_names,
            ),
        )

    return _hooks
