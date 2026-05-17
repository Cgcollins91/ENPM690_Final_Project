"""

JSONL and scalar logging helpers for trainer outputs

File map:

to_jsonable:         Convert common trainer values into JSON-safe objects
jsonl_line:          Serialize one row as a JSONL line
write_jsonl_row:     Write one JSONL row and optionally flush
scalar_log_items:    Return finite scalar metrics suitable for add_scalar
add_scalar_mapping:  Add finite scalar metrics to a TensorBoard-like writer
"""

from __future__ import annotations

from collections.abc import Mapping
import io
import json
import math
from typing import Any, TextIO

import torch


def to_jsonable(value: Any) -> Any:
    """Convert common trainer values into JSON-safe objects

    Steps:
    - Resolve inputs for `to_jsonable` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.reshape(-1)[0].item()
        return detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def jsonl_line(row: Mapping[str, Any]) -> str:
    """Serialize one row as a JSONL line"""
    return json.dumps(to_jsonable(row)) + "\n"


def write_jsonl_row(log_file: TextIO | io.StringIO, row: Mapping[str, Any], *, flush: bool = False) -> None:
    """Write one JSONL row and optionally flush"""
    log_file.write(jsonl_line(row))
    if flush:
        log_file.flush()


def scalar_log_items(prefix: str, metrics: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Return finite scalar metrics suitable for add_scalar"""
    items: list[tuple[str, float]] = []
    base = prefix.rstrip("/")
    for key, value in metrics.items():
        if isinstance(value, bool):
            items.append((f"{base}/{key}", float(int(value))))
            continue
        if isinstance(value, (int, float)):
            scalar = float(value)
            if not math.isnan(scalar):
                items.append((f"{base}/{key}", scalar))
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            scalar = float(value.detach().reshape(-1)[0].item())
            if not math.isnan(scalar):
                items.append((f"{base}/{key}", scalar))
    return items


def add_scalar_mapping(writer: Any, prefix: str, metrics: Mapping[str, Any], global_step: int) -> int:
    """Add finite scalar metrics to a TensorBoard-like writer"""
    count = 0
    for name, value in scalar_log_items(prefix, metrics):
        writer.add_scalar(name, value, int(global_step))
        count += 1
    return count
