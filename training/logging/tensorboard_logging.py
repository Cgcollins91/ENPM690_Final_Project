"""

TensorBoard scalar event helpers

File map:

ScalarEvent:                One TensorBoard scalar write
TRAIN_ENV_TENSOR_STATS:     Define train env tensor stats constant
finite_scalar_events:       Build finite scalar events from a mapping
update_info_events:         Build update and actor_update scalar events
reward_term_events:         Build reward-term scalar events
topdown_metric_events:      Build topdown aggregate scalar events with compact names
active_metric_mask:         Return active mask or all rows when no active rows remain
masked_tensor_stat_events:  Build scalar events for masked tensor statistics
train_env_metric_events:    Build train_env tensor stat events used by the rollout loop
write_scalar_events:        Write scalar events to a TensorBoard-like writer
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class ScalarEvent:
    """One TensorBoard scalar write"""

    name       : str  # Field: string name value used by scalar event
    value      : float  # Field: floating-point value value used by scalar event
    global_step: int  # Field: training step associated with this record or action


TRAIN_ENV_TENSOR_STATS = (
    ("contact", "contact", ("mean", "max")),
    ("both_contact", "both_contact", ("mean", "max")),
    ("fingertip_contact", "fingertip_contact", ("mean", "max")),
    ("hand_contact", "hand_contact", ("mean", "max")),
    ("thumb_contact", "thumb_contact", ("mean", "max")),
    ("index_contact", "index_contact", ("mean", "max")),
    ("thumb_contact_force_N", "thumb_contact_force_N", ("mean", "max")),
    ("index_contact_force_N", "index_contact_force_N", ("mean", "max")),
    ("strict_light_contact", "strict_light_contact", ("mean", "max")),
    ("lift", "lift", ("mean", "max")),
    ("finger_unlock", "finger_unlock", ("mean", "max")),
    ("curl", "curl", ("mean", "max")),
    ("opposed_face", "opposed_face", ("mean", "max")),
    ("align_face_dist", "align_face_dist", ("mean", "min")),
    ("align_angle", "align_angle", ("mean", "min")),
    ("phase1_ready", "phase1_ready", ("rate",)),
    ("success", "success", ("rate",)),
)


def finite_scalar_events(
    prefix     : str,  # Param: string input for prefix
    metrics    : Mapping[str, Any],  # Param: metric mapping emitted with the result or log row
    global_step: int,  # Param: current absolute training step
) -> tuple[ScalarEvent, ...]:
    """Build finite scalar events from a mapping"""
    base = prefix.rstrip("/")
    events: list[ScalarEvent] = []
    for key, value in metrics.items():
        scalar: float | None = None
        if isinstance(value, bool):
            scalar = float(int(value))
        elif isinstance(value, (int, float)):
            scalar = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            scalar = float(value.detach().reshape(-1)[0].item())
        if scalar is not None and not math.isnan(scalar):
            events.append(ScalarEvent(f"{base}/{key}", scalar, int(global_step)))
    return tuple(events)


def update_info_events(
    update_info: Mapping[str, Any] | None,               # Param: string input for update info
    *,
    actor_update_info: Mapping[str, Any] | None = None,  # Param: string input for actor update info
    global_step      : int,  # Param: current absolute training step
) -> tuple[ScalarEvent, ...]:
    """Build update and actor_update scalar events"""
    events: list[ScalarEvent] = []
    if update_info is not None:
        events.extend(finite_scalar_events("update", update_info, global_step))
    if actor_update_info is not None:
        events.extend(finite_scalar_events("actor_update", actor_update_info, global_step))
    return tuple(events)


def reward_term_events(
    term_means: Mapping[str, float],  # Param: floating-point input for term means
    *,
    global_step: int,                 # Param: current absolute training step
) -> tuple[ScalarEvent, ...]:
    """Build reward-term scalar events"""
    return finite_scalar_events("rterm", term_means, global_step)


def topdown_metric_events(
    topdown_metrics: Mapping[str, float],  # Param: floating-point input for topdown metrics
    *,
    global_step: int,                      # Param: current absolute training step
) -> tuple[ScalarEvent, ...]:
    """Build topdown aggregate scalar events with compact names"""
    events: list[ScalarEvent] = []
    for name, value in topdown_metrics.items():
        scalar = float(value)
        if math.isnan(scalar):
            continue
        compact_name = str(name).removeprefix("topdown_")
        events.append(ScalarEvent(f"topdown/{compact_name}", scalar, int(global_step)))
    return tuple(events)


def active_metric_mask(active_env_mask: torch.Tensor) -> torch.Tensor:
    """Return active mask or all rows when no active rows remain"""
    mask = active_env_mask.to(dtype=torch.bool)
    if bool(mask.any().item()):
        return mask
    return torch.ones_like(mask, dtype=torch.bool)


def masked_tensor_stat_events(
    prefix : str,  # Param: string input for prefix
    tensors: Mapping[str, torch.Tensor],  # Param: tensor input carrying tensors values
    *,
    mask       : torch.Tensor,  # Param: boolean mask selecting mask rows
    global_step: int,  # Param: current absolute training step
    stats      : Sequence[str] = ("mean", "max"),  # Param: string input for stats
) -> tuple[ScalarEvent, ...]:
    """Build scalar events for masked tensor statistics

    Steps:
    - Resolve inputs for `masked_tensor_stat_events` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    metric_mask = mask.to(dtype=torch.bool)
    events: list[ScalarEvent] = []
    base = prefix.rstrip("/")
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        values = tensor.detach()
        if values.shape[:1] == metric_mask.shape[:1]:
            values = values[metric_mask]
        if values.numel() == 0:
            continue
        flat = values.to(dtype=torch.float32).reshape(-1)
        for stat in stats:
            if stat == "mean":
                value = float(flat.mean().item())
            elif stat == "max":
                value = float(flat.max().item())
            elif stat == "min":
                value = float(flat.min().item())
            else:
                raise ValueError(f"unknown tensor stat: {stat!r}")
            if not math.isnan(value):
                events.append(ScalarEvent(f"{base}/{name}_{stat}", value, int(global_step)))
    return tuple(events)


def train_env_metric_events(
    tensors: Mapping[str, torch.Tensor],  # Param: tensor input carrying tensors values
    *,
    active_env_mask: torch.Tensor,  # Param: mask selecting env rows still active for collection or reset handling
    global_step    : int,  # Param: current absolute training step
) -> tuple[ScalarEvent, ...]:
    """Build train_env tensor stat events used by the rollout loop"""
    metric_mask = active_metric_mask(active_env_mask)
    events: list[ScalarEvent] = []
    for tensor_key, event_name, stats in TRAIN_ENV_TENSOR_STATS:
        tensor = tensors.get(tensor_key)
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        values = tensor.detach()
        if values.shape[:1] == metric_mask.shape[:1]:
            values = values[metric_mask]
        if values.numel() == 0:
            continue
        flat = values.to(dtype=torch.float32).reshape(-1)
        for stat in stats:
            if stat in ("mean", "rate"):
                value = float(flat.mean().item())
            elif stat == "max":
                value = float(flat.max().item())
            elif stat == "min":
                value = float(flat.min().item())
            else:
                raise ValueError(f"unknown train env stat: {stat!r}")
            if not math.isnan(value):
                events.append(ScalarEvent(f"train_env/{event_name}_{stat}", value, int(global_step)))
    return tuple(events)


def write_scalar_events(writer: Any, events: Sequence[ScalarEvent]) -> int:
    """Write scalar events to a TensorBoard-like writer"""
    if writer is None:
        return 0
    count = 0
    for event in events:
        writer.add_scalar(event.name, event.value, event.global_step)
        count += 1
    return count
