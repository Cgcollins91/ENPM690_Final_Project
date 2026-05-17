"""

JSON row construction helpers for rollout and eval logging

File map:

LIFT_NOISE_KEYS:              Define lift noise keys constant
add_prefixed_fields:          Add mapping values with a key prefix
drop_row_keys:                Drop keys from a row when present
drop_lift_noise_keys:         Drop lift-only fields for non-lift task rows
tensor_attr_value:            Read one scalar tensor attr value from an owner
add_tensor_attr_fields:       Add scalar tensor attr fields from specs
add_vec3_tensor_attr_fields:  Add x y z fields from a vec3 tensor attr when present
finite_or_nan:                Return a finite float or nan
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import torch


LIFT_NOISE_KEYS = (
    "lift_height",
    "block_lift_height",
    "best_lift_this_episode",
    "best_block_lift_height_this_episode",
    "best_lift_with_strict_contact_this_episode",
    "episode_physical_success",
    "off_table",
    "block_drift",
    "terminal_off_table_inferred",
    "terminal_block_drift_inferred",
    "post_reset_lift_height",
)


def add_prefixed_fields(row: dict[str, object], prefix: str, values: Mapping[str, object]) -> dict[str, object]:
    """Add mapping values with a key prefix"""
    for key, value in values.items():
        row[f"{prefix}{key}"] = value
    return row


def drop_row_keys(row: dict[str, object], keys: Iterable[str]) -> dict[str, object]:
    """Drop keys from a row when present"""
    for key in keys:
        row.pop(key, None)
    return row


def drop_lift_noise_keys(row: dict[str, object]) -> dict[str, object]:
    """Drop lift-only fields for non-lift task rows"""
    return drop_row_keys(row, LIFT_NOISE_KEYS)


def tensor_attr_value(
    owner    : object,  # Param: input value used as owner
    attr_name: str,  # Param: string input for attr name
    env_id   : int,  # Param: integer input for env id
    default  : object,  # Param: fallback value used when the input omits or rejects a setting
    *,
    value_type: str = "float",  # Param: string input for value type
) -> object:
    """Read one scalar tensor attr value from an owner

    Steps:
    - Resolve inputs for `tensor_attr_value` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    attr = getattr(owner, attr_name, None)
    if not torch.is_tensor(attr) or attr.numel() <= int(env_id):
        return default
    value = attr.reshape(attr.shape[0], -1)[int(env_id), 0].item()
    if value_type == "bool":
        return bool(value)
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    raise ValueError(f"unknown tensor attr value_type: {value_type!r}")


def add_tensor_attr_fields(
    row   : dict[str, object],  # Param: string input for row
    owner : object,  # Param: input value used as owner
    env_id: int,  # Param: integer input for env id
    specs : Iterable[tuple[str, str, object, str]],  # Param: string input for specs
) -> dict[str, object]:
    """Add scalar tensor attr fields from specs"""
    for key, attr_name, default, value_type in specs:
        row[key] = tensor_attr_value(owner, attr_name, env_id, default, value_type=value_type)
    return row


def add_vec3_tensor_attr_fields(
    row      : dict[str, object],  # Param: string input for row
    owner    : object,  # Param: input value used as owner
    env_id   : int,  # Param: integer input for env id
    prefix   : str,  # Param: string input for prefix
    attr_name: str,  # Param: string input for attr name
) -> dict[str, object]:
    """Add x y z fields from a vec3 tensor attr when present

    Steps:
    - Resolve inputs for `add_vec3_tensor_attr_fields` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    attr = getattr(owner, attr_name, None)
    if not torch.is_tensor(attr) or attr.dim() < 2 or attr.shape[0] <= int(env_id):
        return row
    vec = attr[int(env_id)].detach().reshape(-1)
    if vec.numel() < 3:
        return row
    row[f"{prefix}_x"] = float(vec[0].item())
    row[f"{prefix}_y"] = float(vec[1].item())
    row[f"{prefix}_z"] = float(vec[2].item())
    return row


def finite_or_nan(value: object) -> float:
    """Return a finite float or nan"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan
