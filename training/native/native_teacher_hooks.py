"""

Native teacher hook provider factory

File map:

NativeTeacherHookState:              Latest native teacher action captured for env action assembly
clear_native_teacher_hook_state:     Clear cached native teacher action state
_cache_hit:                          Handle cache hit logic
_store_cache:                        Handle store cache logic
_resolve_tensor:                     Handle resolve tensor logic
_resolve_optional_tensor:            Handle resolve optional tensor logic
_resolve_contact_parts:              Handle resolve contact parts logic
_resolve_episode_step:               Handle resolve episode step logic
build_native_teacher_hook_provider:  Build hook provider backed by the refactored native teacher
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .native_isaac_hook_provider import NativeIsaacHookProvider
from .native_teacher import (
    FingerModeFn,
    NativeTeacherAction,
    NativeTeacherConfig,
    NativeTeacherRequest,
    select_native_teacher_action,
)
from ..teacher.teacher_actions import TopdownContactTeacherParts
from ..teacher.teacher_arm_controller import TeacherArmBackend
from ..teacher.teacher_cache import episode_step_tensor


TensorSource = torch.Tensor | Callable[[], torch.Tensor]
OptionalTensorValueSource = TensorSource | None
OptionalTensorSource = torch.Tensor | Callable[[], torch.Tensor | None] | None
ContactPartsSource = TopdownContactTeacherParts | Callable[[], TopdownContactTeacherParts | None] | None
EpisodeStepSource = int | torch.Tensor | Callable[[], int | torch.Tensor | None] | None


@dataclass
class NativeTeacherHookState:
    """Latest native teacher action captured for env action assembly"""

    last_action        : NativeTeacherAction | None = None  # Field: stores last action for native teacher hook state
    cached_action      : NativeTeacherAction | None = None  # Field: stores cached action for native teacher hook state
    cached_episode_step: torch.Tensor | None        = None  # Field: step count used for cached episode step scheduling or reporting


def clear_native_teacher_hook_state(state: NativeTeacherHookState) -> None:
    """Clear cached native teacher action state"""
    state.last_action = None
    state.cached_action = None
    state.cached_episode_step = None


def _cache_hit(
    state       : NativeTeacherHookState,  # Param: mutable or immutable runtime state read by this helper
    episode_step: int | torch.Tensor | None,  # Param: per-env step count inside the current episode
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> NativeTeacherAction | None:
    """Process for `_cache_hit`

    Steps:
    - Resolve inputs for `_cache_hit` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if episode_step is None or state.cached_action is None or state.cached_episode_step is None:
        return None
    expected = episode_step_tensor(episode_step, num_envs=num_envs, device=device)
    if tuple(state.cached_episode_step.shape) != tuple(expected.shape):
        return None
    if not bool(torch.equal(state.cached_episode_step.to(device=device), expected)):
        return None
    return state.cached_action


def _store_cache(
    state       : NativeTeacherHookState,  # Param: mutable or immutable runtime state read by this helper
    episode_step: int | torch.Tensor | None,  # Param: per-env step count inside the current episode
    action      : NativeTeacherAction,  # Param: action tensor applied to the environment or stored in replay
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> None:
    if episode_step is None:
        return
    state.cached_action = action
    state.cached_episode_step = episode_step_tensor(
        episode_step,
        num_envs=num_envs,
        device=device,
    ).detach().clone()


def _resolve_tensor(source: TensorSource) -> torch.Tensor:
    value = source() if callable(source) else source
    if not torch.is_tensor(value):
        raise TypeError(f"native teacher tensor source returned {type(value)!r}")
    return value


def _resolve_optional_tensor(source: OptionalTensorSource) -> torch.Tensor | None:
    """Process for `_resolve_optional_tensor`

    Steps:
    - Resolve inputs for `_resolve_optional_tensor` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if source is None:
        return None
    value = source() if callable(source) else source
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"native teacher optional tensor source returned {type(value)!r}")
    return value


def _resolve_contact_parts(source: ContactPartsSource) -> TopdownContactTeacherParts | None:
    return source() if callable(source) else source


def _resolve_episode_step(source: EpisodeStepSource) -> int | torch.Tensor | None:
    return source() if callable(source) else source


def build_native_teacher_hook_provider(
    *,
    env                           : object,  # Param: environment or backend object used for runtime calls
    mapped_indices                : TensorSource,  # Param: input value used as mapped indices
    mapped_scales                 : TensorSource,  # Param: input value used as mapped scales
    closure_fraction              : OptionalTensorValueSource,  # Param: input value used as closure fraction
    compute_finger_in_current_mode: FingerModeFn,  # Param: mode string selecting the compute finger in current behavior
    arm_backend                   : TeacherArmBackend,  # Param: input value used as arm backend
    config                        : NativeTeacherConfig,  # Param: configuration object used by this helper
    contact_parts                 : ContactPartsSource            = None,  # Param: input value used as contact parts
    stage                         : OptionalTensorSource          = None,  # Param: input value used as stage
    episode_step                  : EpisodeStepSource             = None,  # Param: per-env step count inside the current episode
    state                         : NativeTeacherHookState | None = None,  # Param: mutable or immutable runtime state read by this helper
    cache_enabled                 : bool                          = False,  # Param: boolean input enabling cache
) -> NativeIsaacHookProvider:
    """Build hook provider backed by the refactored native teacher

    Steps:
    - Resolve inputs for `build_native_teacher_hook_provider` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    hook_state = NativeTeacherHookState() if state is None else state

    def _compute_action(step_value: int | torch.Tensor | None) -> NativeTeacherAction:
        parts = _resolve_contact_parts(contact_parts)
        closure = _resolve_tensor(closure_fraction) if closure_fraction is not None else (
            None if parts is None else parts.closure_fraction
        )
        if closure is None:
            raise RuntimeError("native teacher closure_fraction is not configured")
        return select_native_teacher_action(
            NativeTeacherRequest(
                env=env,
                mapped_indices=_resolve_tensor(mapped_indices),
                mapped_scales=_resolve_tensor(mapped_scales),
                closure_fraction=closure,
                compute_finger_in_current_mode=compute_finger_in_current_mode,
                contact_parts=parts,
                stage=_resolve_optional_tensor(stage),
                episode_step=step_value,
            ),
            config=config,
            arm_backend=arm_backend,
        )

    def _teacher_action() -> torch.Tensor:
        """Process for `_teacher_action`

        Steps:
        - Resolve inputs for `_teacher_action` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        step_value = _resolve_episode_step(episode_step)
        if bool(cache_enabled):
            cached = _cache_hit(
                hook_state,
                step_value,
                num_envs=int(getattr(env, "num_envs")),
                device=getattr(env, "device", "cpu"),
            )
            if cached is not None:
                hook_state.last_action = cached
                return cached.action.clone()
        action = _compute_action(step_value)
        hook_state.last_action = action
        if bool(cache_enabled):
            _store_cache(
                hook_state,
                step_value,
                action,
                num_envs=int(getattr(env, "num_envs")),
                device=getattr(env, "device", "cpu"),
            )
        return action.action

    def _arm_reduced_action() -> torch.Tensor | None:
        return None if hook_state.last_action is None else hook_state.last_action.arm_action

    return NativeIsaacHookProvider(
        env=env,
        teacher_action=_teacher_action,
        arm_reduced_action=_arm_reduced_action,
        mapped_indices=_resolve_tensor(mapped_indices),
        mapped_scales=_resolve_tensor(mapped_scales),
        reset_callback=lambda: clear_native_teacher_hook_state(hook_state),
    )
