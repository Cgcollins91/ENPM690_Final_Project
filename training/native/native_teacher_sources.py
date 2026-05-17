"""

Env-backed sources for native teacher hooks

File map:

NativeTeacherEnvSourceConfig:            Conventional env source names for native teacher hooks
_find_value:                             Handle find value logic
_tensor_from_value:                      Handle tensor from value logic
_robot_from_env:                         Handle robot from env logic
context_action_mapping_sources:          Build lazy action mapping sources from the runtime context and env robot
env_tensor_source:                       Return a callable tensor source backed by env attrs
env_episode_step_source:                 Return an episode-step source backed by conventional env attrs
contact_parts_source_from_fn:            Build a contact-parts source from an injected stateful teacher function
contact_parts_source_from_env_attrs:     Build a contact-parts source from env contact-teacher attrs
native_teacher_config_from_runtime:      Build native teacher mode config from runtime config bundle
build_env_native_teacher_hook_provider:  Build a native teacher provider from env attrs and injected backends
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..actions.action_space import compute_teacher_finger_reduced_in_current_mode, get_action_mapping
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from .native_contact_teacher_parts import (
    NativeContactTeacherAttrConfig,
    contact_teacher_parts_from_env_attrs,
)
from .native_isaac_hook_provider import NativeIsaacHookProvider
from .native_teacher import NativeTeacherConfig
from .native_teacher_hooks import build_native_teacher_hook_provider
from ..teacher.teacher_actions import TopdownContactTeacherParts, topdown_contact_teacher_parts_from_tuple
from ..teacher.teacher_arm_controller import TeacherArmBackend


ContactPartsFn = Callable[[object, torch.Tensor, torch.Tensor, int | torch.Tensor], object]


@dataclass(frozen=True)
class NativeTeacherEnvSourceConfig:
    """Conventional env source names for native teacher hooks"""

    mapped_indices_names  : tuple[str, ...] = ("mapped_indices", "_mapped_indices")  # Field: source names for mapped action-column indices
    mapped_scales_names   : tuple[str, ...] = ("mapped_scales", "_mapped_scales")  # Field: source names for mapped action-column scales
    stage_names           : tuple[str, ...] = ("_topdown_stage", "topdown_stage")  # Field: ordered names used to resolve stage attributes
    closure_fraction_names: tuple[str, ...] = (  # Field: ordered names used to resolve closure fraction attributes
        "_topdown_contact_teacher_closure_fraction",
        "topdown_contact_teacher_closure_fraction",
    )
    contact_parts_names    : tuple[str, ...] = (                                       # Field: ordered names used to resolve contact parts attributes
        "native_contact_teacher_parts",
        "topdown_contact_teacher_parts",
        "contact_teacher_parts_fn",
    )
    episode_step_names: tuple[str, ...] = ("episode_length_buf", "episode_step")  # Field: ordered names used to resolve episode step attributes
    num_arm           : int             = 6  # Field: number of arm action dimensions in the active layout
    num_fingers       : int             = 7  # Field: number of finger action dimensions in the active layout


def _find_value(owner: object, names: Sequence[str]) -> object | None:
    for name in names:
        if isinstance(owner, dict):
            value = owner.get(name)
        else:
            value = getattr(owner, name, None)
        if value is not None:
            return value
    return None


def _tensor_from_value(value: object, *, label: str) -> torch.Tensor:
    resolved = value() if callable(value) else value
    if not torch.is_tensor(resolved):
        raise TypeError(f"{label} must resolve to a tensor, got {type(resolved)!r}")
    return resolved


def _robot_from_env(env: object) -> object:
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("native teacher env has no mapped action attrs and no scene robot")
    try:
        return scene["robot"]
    except Exception as exc:
        raise RuntimeError("native teacher env has no mapped action attrs and no scene['robot']") from exc


def context_action_mapping_sources(
    *,
    env    : object,  # Param: environment or backend object used for runtime calls
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """Build lazy action mapping sources from the runtime context and env robot."""
    cached: tuple[torch.Tensor, torch.Tensor] | None = None

    def _mapping() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal cached
        if cached is None:
            cached = get_action_mapping(
                _robot_from_env(env),
                context.device,
                context.action.env_action_spec,
            )
        return cached

    return lambda: _mapping()[0], lambda: _mapping()[1]


def env_tensor_source(
    env  : object,  # Param: environment or backend object used for runtime calls
    names: Sequence[str],  # Param: ordered candidate names used during lookup
    *,
    label   : str,  # Param: string input for label
    required: bool = True,  # Param: boolean input controlling required
) -> Callable[[], torch.Tensor | None]:
    """Return a callable tensor source backed by env attrs"""

    def _source() -> torch.Tensor | None:
        value = _find_value(env, names)
        if value is None:
            if required:
                raise RuntimeError(f"native teacher env is missing {label}: {tuple(names)}")
            return None
        return _tensor_from_value(value, label=label)

    return _source


def env_episode_step_source(
    env  : object,  # Param: environment or backend object used for runtime calls
    names: Sequence[str],  # Param: ordered candidate names used during lookup
) -> Callable[[], int | torch.Tensor | None]:
    """Return an episode-step source backed by conventional env attrs"""

    def _source() -> int | torch.Tensor | None:
        value = _find_value(env, names)
        return value() if callable(value) else value

    return _source


def contact_parts_source_from_fn(
    *,
    env             : object,  # Param: environment or backend object used for runtime calls
    mapped_indices  : Callable[[], torch.Tensor],  # Param: callback used to compute or fetch mapped indices
    mapped_scales   : Callable[[], torch.Tensor],  # Param: callback used to compute or fetch mapped scales
    episode_step    : Callable[[], int | torch.Tensor | None],  # Param: per-env step count inside the current episode
    contact_parts_fn: ContactPartsFn,  # Param: callback used to compute or fetch contact parts
) -> Callable[[], TopdownContactTeacherParts]:
    """Build a contact-parts source from an injected stateful teacher function"""

    def _source() -> TopdownContactTeacherParts:
        """Process for `_source`

        Steps:
        - Resolve inputs for `_source` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        step = episode_step()
        if step is None:
            raise RuntimeError("native contact teacher source needs episode_step")
        try:
            values = contact_parts_fn(env, mapped_indices(), mapped_scales(), step)
        except TypeError:
            values = contact_parts_fn(mapped_indices(), mapped_scales(), step)
        if isinstance(values, TopdownContactTeacherParts):
            return values
        if not isinstance(values, tuple) or len(values) != 6:
            raise TypeError(
                "native contact teacher parts function must return "
                f"TopdownContactTeacherParts or 6-tuple, got {type(values)!r}"
            )
        return topdown_contact_teacher_parts_from_tuple(values)

    return _source


def contact_parts_source_from_env_attrs(
    *,
    env           : object,  # Param: environment or backend object used for runtime calls
    mapped_indices: Callable[[], torch.Tensor],  # Param: callback used to compute or fetch mapped indices
    mapped_scales : Callable[[], torch.Tensor],  # Param: callback used to compute or fetch mapped scales
    configs       : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    attr_config   : NativeContactTeacherAttrConfig | None = None,  # Param: input value used as attr config
) -> Callable[[], TopdownContactTeacherParts]:
    """Build a contact-parts source from env contact-teacher attrs"""

    def _source() -> TopdownContactTeacherParts:
        return contact_teacher_parts_from_env_attrs(
            env=env,
            mapped_indices=mapped_indices(),
            mapped_scales=mapped_scales(),
            configs=configs,
            attr_config=attr_config,
        )

    return _source


def native_teacher_config_from_runtime(configs: RuntimeConfigBundle) -> NativeTeacherConfig:
    """Build native teacher mode config from runtime config bundle"""
    return NativeTeacherConfig(
        topdown_contact_teacher_enabled=bool(configs.teacher.topdown_contact_teacher),
        topdown_curriculum_task=True,
    )


def build_env_native_teacher_hook_provider(
    *,
    env                   : object,  # Param: environment or backend object used for runtime calls
    context               : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs               : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    arm_backend           : TeacherArmBackend,  # Param: input value used as arm backend
    source_config         : NativeTeacherEnvSourceConfig                     = NativeTeacherEnvSourceConfig(),  # Param: input value used as source config
    contact_parts_fn      : ContactPartsFn | None                            = None,  # Param: callback used to compute or fetch contact parts
    use_contact_attr_parts: bool                                             = False,  # Param: boolean input selecting whether contact attr parts is used
    contact_attr_config   : NativeContactTeacherAttrConfig | None            = None,  # Param: input value used as contact attr config
    cache_teacher_action  : bool                                             = False,  # Param: boolean input controlling cache teacher action
    mapped_indices        : torch.Tensor | Callable[[], torch.Tensor] | None = None,  # Param: callback used to compute or fetch mapped indices
    mapped_scales         : torch.Tensor | Callable[[], torch.Tensor] | None = None,  # Param: callback used to compute or fetch mapped scales
    closure_fraction      : torch.Tensor | Callable[[], torch.Tensor] | None = None,  # Param: callback used to compute or fetch closure fraction
) -> NativeIsaacHookProvider:
    """Build a native teacher provider from env attrs and injected backends

    Steps:
    - Resolve inputs for `build_env_native_teacher_hook_provider` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    context_mapped_indices, context_mapped_scales = context_action_mapping_sources(
        env=env,
        context=context,
    )
    mapped_indices_source = (
        mapped_indices
        if mapped_indices is not None
        else (
            env_tensor_source(env, source_config.mapped_indices_names, label="mapped_indices", required=False)
            if _find_value(env, source_config.mapped_indices_names) is not None
            else context_mapped_indices
        )
    )
    mapped_scales_source = (
        mapped_scales
        if mapped_scales is not None
        else (
            env_tensor_source(env, source_config.mapped_scales_names, label="mapped_scales", required=False)
            if _find_value(env, source_config.mapped_scales_names) is not None
            else context_mapped_scales
        )
    )
    stage_source = env_tensor_source(
        env,
        source_config.stage_names,
        label="topdown_stage",
        required=False,
    )
    episode_step_source = env_episode_step_source(env, source_config.episode_step_names)
    env_contact_parts_fn = _find_value(env, source_config.contact_parts_names)
    if contact_parts_fn is None and callable(env_contact_parts_fn):
        contact_parts_fn = env_contact_parts_fn
    if contact_parts_fn is not None:
        contact_source = contact_parts_source_from_fn(
            env=env,
            mapped_indices=lambda: _tensor_from_value(mapped_indices_source, label="mapped_indices"),
            mapped_scales=lambda: _tensor_from_value(mapped_scales_source, label="mapped_scales"),
            episode_step=episode_step_source,
            contact_parts_fn=contact_parts_fn,
        )
    elif use_contact_attr_parts:
        contact_source = contact_parts_source_from_env_attrs(
            env=env,
            mapped_indices=lambda: _tensor_from_value(mapped_indices_source, label="mapped_indices"),
            mapped_scales=lambda: _tensor_from_value(mapped_scales_source, label="mapped_scales"),
            configs=configs,
            attr_config=contact_attr_config,
        )
    else:
        contact_source = None
    closure_source = closure_fraction
    if closure_source is None and contact_source is None:
        closure_source = env_tensor_source(
            env,
            source_config.closure_fraction_names,
            label="closure_fraction",
        )

    def _finger_in_current_mode(fraction: torch.Tensor) -> torch.Tensor:
        return compute_teacher_finger_reduced_in_current_mode(
            env,
            _tensor_from_value(mapped_indices_source, label="mapped_indices"),
            _tensor_from_value(mapped_scales_source, label="mapped_scales"),
            fraction,
            num_arm=source_config.num_arm,
            num_fingers=source_config.num_fingers,
            finger_action_mode=configs.teacher.finger_action_mode,
            finger_delta_scale=configs.teacher.finger_delta_scale,
        )

    return build_native_teacher_hook_provider(
        env=env,
        mapped_indices=mapped_indices_source,
        mapped_scales=mapped_scales_source,
        closure_fraction=closure_source,
        compute_finger_in_current_mode=_finger_in_current_mode,
        arm_backend=arm_backend,
        config=native_teacher_config_from_runtime(configs),
        contact_parts=contact_source,
        stage=stage_source,
        episode_step=episode_step_source,
        cache_enabled=bool(cache_teacher_action),
    )
