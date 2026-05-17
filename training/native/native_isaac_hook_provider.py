"""

Provider object for native Isaac live hook methods

File map:

_resolve_tensor:          Handle resolve tensor logic
_resolve_names:           Handle resolve names logic
NativeIsaacHookProvider:  Env wrapper exposing the default hook names used by native_isaac_hooks
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

from .native_live import NativeLiveRolloutState


TensorFn = Callable[[], torch.Tensor]
RuntimeTensorFn = Callable[[NativeLiveRolloutState], torch.Tensor]
NamesFn = Callable[[], Iterable[str]]
ResetCallback = Callable[[], None]


def _resolve_tensor(value: torch.Tensor | TensorFn | RuntimeTensorFn | None, runtime=None) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    try:
        out = value(runtime) if runtime is not None else value()
    except TypeError:
        out = value()
    return out if torch.is_tensor(out) else None


def _resolve_names(value: Iterable[str] | NamesFn | None) -> tuple[str, ...]:
    if value is None:
        return ()
    resolved = value() if callable(value) else value
    return tuple(str(name) for name in resolved)


@dataclass
class NativeIsaacHookProvider:
    """Env wrapper exposing the default hook names used by native_isaac_hooks"""

    env               : Any  # Field: environment/backend object used by this runtime helper
    teacher_action    : torch.Tensor | TensorFn | None        = None  # Field: teacher action tensor used for override or behavior cloning
    arm_reduced_action: torch.Tensor | TensorFn | None        = None  # Field: tensor containing arm reduced action values for batched env rows
    mapped_indices    : torch.Tensor | None                   = None  # Field: column indices used to map between action layouts
    mapped_scales     : torch.Tensor | None                   = None  # Field: scales applied while mapping action columns
    preroll_action    : torch.Tensor | RuntimeTensorFn | None = None  # Field: tensor containing preroll action values for batched env rows
    preroll_mask      : torch.Tensor | RuntimeTensorFn | None = None  # Field: boolean mask selecting preroll rows for native isaac hook provider
    active_env_mask   : torch.Tensor | RuntimeTensorFn | None = None  # Field: mask selecting env rows that are still active
    checkpoint_names  : Iterable[str] | NamesFn | None        = None  # Field: ordered names used to resolve checkpoint attributes
    reset_callback    : ResetCallback | None                  = None  # Field: stores reset callback for native isaac hook provider

    def __getattr__(self, name: str):
        """Delegate missing attributes to the wrapped env"""
        return getattr(self.env, name)

    def reset(self):
        """Delegate reset to the wrapped env"""
        if self.reset_callback is not None:
            self.reset_callback()
        return self.env.reset()

    def step(self, action: torch.Tensor):
        """Delegate step to the wrapped env"""
        return self.env.step(action)

    def compute_teacher_action(self) -> torch.Tensor:
        """Return the current reduced teacher action"""
        action = _resolve_tensor(self.teacher_action)
        if action is None:
            raise RuntimeError("NativeIsaacHookProvider teacher_action is not configured")
        return action

    def current_arm_reduced_action(self) -> torch.Tensor | None:
        """Return the current IK arm reduced action"""
        return _resolve_tensor(self.arm_reduced_action)

    def current_preroll_action(self, runtime: NativeLiveRolloutState) -> torch.Tensor | None:
        """Return current preroll action rows"""
        return _resolve_tensor(self.preroll_action, runtime)

    def current_preroll_mask(self, runtime: NativeLiveRolloutState) -> torch.Tensor | None:
        """Return current preroll mask rows"""
        return _resolve_tensor(self.preroll_mask, runtime)

    def current_active_env_mask(self, runtime: NativeLiveRolloutState | None = None) -> torch.Tensor | None:
        """Return active rows used for rollout stats"""
        return _resolve_tensor(self.active_env_mask, runtime)

    def clear_cached_teacher_action(self) -> None:
        """Clear provider-owned teacher cache"""
        if self.reset_callback is not None:
            self.reset_callback()

    def existing_checkpoint_names(self) -> tuple[str, ...]:
        """Return existing checkpoint names for rolling retention"""
        return _resolve_names(self.checkpoint_names)
