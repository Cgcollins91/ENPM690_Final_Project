"""Shared config helpers.

The rule for this package is that every public config object stays small.  Each
dataclass is capped at 25 fields so one file never becomes another hidden shell
wrapper with hundreds of unstructured knobs.
"""

from __future__ import annotations  # keeps annotations lazy for forward references

from dataclasses import fields, is_dataclass  # imports dataclass helpers used by config groups
from pathlib import Path  # imports filesystem path support for config values
from typing import Any  # imports typing helpers used by config annotations


FIELD_LIMIT = 30  # caps dataclass config groups before they become too broad


def bool01(value: bool) -> str:  # defines the bool01 helper
    """Return the legacy environment representation for a boolean."""

    return "1" if value else "0"  # returns the computed value


def value(value: Any) -> str:  # defines the value helper
    """Convert config values to stable CLI/env strings."""

    if isinstance(value, bool):  # Checks whether isinstance(value, bool)
        return bool01(value)  # returns boolean value as legacy 0 or 1 text
    if isinstance(value, Path):  # Checks whether isinstance(value, path)
        return str(value)  # returns value converted to text
    return str(value)  # returns value converted to text


def clean_dict(items: dict[str, Any]) -> dict[str, str]:  # defines the clean dict helper
    """Drop unset values and stringify everything else."""

    return {key: value(raw) for key, raw in items.items() if raw is not None}  # returns the computed value


def add_arg(args: list[str], name: str, raw: Any) -> None:  # defines the add arg helper
    """Append a `--name value` pair for scalar trainer arguments."""

    args.extend([name, value(raw)])  # appends these trainer CLI tokens


def add_flag(args: list[str], enabled: bool, name: str) -> None:  # defines the add flag helper
    """Append a boolean CLI flag only when enabled."""

    if enabled:  # Checks whether enabled
        args.append(name)  # appends one trainer CLI token


def assert_field_limits(configs: tuple[object, ...], limit: int = FIELD_LIMIT) -> None:  # defines the assert field limits helper
    """Fail fast if a config group becomes too broad to understand."""

    offenders: list[str] = []  # Collects config groups that exceed the field limit
    for cfg in configs:  # iterates over configured values
        if not is_dataclass(cfg):  # Checks whether not is dataclass(cfg)
            continue  # skips this item and continues validation
        count = len(fields(cfg))  # Counts fields in the current dataclass config group
        if count > limit:  # Checks whether count > limit
            offenders.append(f"{type(cfg).__name__}={count}")  # appends the computed value to the collection
    if offenders:  # Checks whether offenders
        joined = ", ".join(offenders)  # Formats all field-limit offenders for the error message
        raise ValueError(f"config field limit exceeded ({limit}): {joined}")  # raises an error for invalid config state
