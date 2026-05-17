"""

Explicit trainer context objects for the refactored runtime

File map:

TrainerPaths:           Filesystem paths used by one training run
TrainerDimensions:      Resolved observation and action dimensions
TrainerActionContext:   Resolved action specs and controller mode
TrainerRuntimeContext:  Import-safe runtime context for trainer modules
namespace_to_dict:      Convert argparse or simple namespace objects to plain dicts This is used to convert parsed CLI args and config namespaces into plain dicts for easier manipulation and normalization in the runtime context, without needing to import argparse or other libraries that may be used by the legacy monolith trainer
mapping_get_bool:       Read a bool-like key from a mapping
mapping_get_int:        Read an int-like key from a mapping
mapping_get_float:      Read a float-like key from a mapping
build_action_context:   Build action context with normalized scalar fields
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from types import SimpleNamespace

from ..actions.action_space import ReducedActionSpec
from .runtime import SUPPORTED_TOPDOWN_TASK


@dataclass(frozen=True)
class TrainerPaths:
    """Filesystem paths used by one training run"""

    checkpoint_path: str  # Field: checkpoint file path used for load/save operations
    log_jsonl      : str  # Field: JSONL log path or enablement flag for structured logging
    tensorboard_dir: str | None = None  # Field: filesystem location for tensorboard dir

    @property
    def checkpoint_dir(self) -> str:
        """Return checkpoint directory with script-compatible fallback"""
        return os.path.dirname(self.checkpoint_path) or "."


@dataclass(frozen=True)
class TrainerDimensions:
    """Resolved observation and action dimensions"""

    obs_dim        : int  # Field: width of the policy observation vector
    action_dim     : int  # Field: width of the policy action vector
    full_action_dim: int  # Field: integer full action dim value tracked by trainer dimensions
    priv_obs_dim   : int = 0  # Field: width of the privileged observation vector


@dataclass(frozen=True)
class TrainerActionContext:
    """Resolved action specs and controller mode"""

    arm_controller    : str                # Field: string arm controller value used by trainer action context
    finger_action_mode: str                # Field: configured interpretation of finger action columns
    finger_delta_scale: float              # Field: scale applied to finger delta action columns
    policy_action_spec: ReducedActionSpec  # Field: action layout spec expected by the policy output
    env_action_spec   : ReducedActionSpec  # Field: action layout spec expected by the environment


@dataclass(frozen=True)
class TrainerRuntimeContext:
    """Import-safe runtime context for trainer modules"""

    task              : str                     # Field: string task value used by trainer runtime context
    td3_backend       : str                     # Field: string td3 backend value used by trainer runtime context
    seed              : int                     # Field: integer seed value tracked by trainer runtime context
    device            : str                     # Field: torch device where tensor fields should live
    paths             : TrainerPaths            # Field: filesystem path for paths
    dims              : TrainerDimensions       # Field: stores dims for trainer runtime context
    action            : TrainerActionContext    # Field: environment action tensor selected for the step
    obs_schema_version: int                                                 # Field: integer obs schema version value tracked by trainer runtime context
    obs_keys          : tuple[str, ...]                                     # Field: ordered keys used to resolve obs values
    env               : Mapping[str, str]    = field(default_factory=dict)  # Field: environment/backend object used by this runtime helper
    args              : Mapping[str, object] = field(default_factory=dict)  # Field: parsed CLI/config arguments passed into this helper

    def validate_supported(self) -> None:
        """Raise if this context is outside the standalone trainer contract"""
        if self.task != SUPPORTED_TOPDOWN_TASK:
            raise RuntimeError(
                "ENPM690 standalone trainer supports only "
                f"{SUPPORTED_TOPDOWN_TASK}; got {self.task!r}"
            )


def namespace_to_dict(namespace: object) -> dict[str, object]:
    """
    Convert argparse or simple namespace objects to plain dicts
    This is used to convert parsed CLI args and config namespaces into plain dicts for easier manipulation and normalization in the runtime context,
    without needing to import argparse or other libraries that may be used by the legacy monolith trainer.
    """

    if isinstance(namespace, Mapping):
        return dict(namespace)

    if isinstance(namespace, SimpleNamespace):
        return dict(vars(namespace))
    if hasattr(namespace, "__dict__"):
        return dict(vars(namespace))
    raise TypeError(f"cannot convert namespace of type {type(namespace)!r}")


def mapping_get_bool(mapping: Mapping[str, object], key: str, default: bool = False) -> bool:
    """Read a bool-like key from a mapping"""
    value = mapping.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def mapping_get_int(mapping: Mapping[str, object], key: str, default: int = 0) -> int:
    """Read an int-like key from a mapping"""
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def mapping_get_float(mapping: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Read a float-like key from a mapping"""
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def build_action_context(
    *,
    arm_controller    : str,  # Param: string input for arm controller
    finger_action_mode: str,  # Param: mode string selecting the finger action behavior
    finger_delta_scale: float,  # Param: multiplier applied to finger delta
    policy_action_spec: ReducedActionSpec,  # Param: input value used as policy action spec
    env_action_spec   : ReducedActionSpec,  # Param: input value used as env action spec
) -> TrainerActionContext:
    """Build action context with normalized scalar fields"""
    return TrainerActionContext(
        arm_controller=str(arm_controller),
        finger_action_mode=str(finger_action_mode),
        finger_delta_scale=float(finger_delta_scale),
        policy_action_spec=policy_action_spec,
        env_action_spec=env_action_spec,
    )
