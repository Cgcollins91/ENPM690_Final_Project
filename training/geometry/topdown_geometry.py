"""

Import-safe topdown target geometry helpers

File map:

YawAxisSelection:                 Normalized yaw-axis mode and world XY axis
normalize_topdown_yaw_axis_mode:  Normalize configured topdown yaw-axis mode
block_axis_xy_from_quat:          Return a normalized block-local X or Y axis projected to world XY
select_topdown_yaw_axis:          Select world yaw axis including block-x and block-y modes
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .geometry import quat_wxyz_to_matrix


@dataclass(frozen=True)
class YawAxisSelection:
    """Normalized yaw-axis mode and world XY axis"""

    mode      : str  # string mode value used by yaw axis selection
    axis_world: torch.Tensor  # tensor containing axis world values for batched env rows


def normalize_topdown_yaw_axis_mode(axis_mode: str | None) -> str:
    """Normalize configured topdown yaw-axis mode

    Steps:
    - Resolve inputs for `normalize_topdown_yaw_axis_mode` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    mode = (axis_mode or "block_y").strip().lower()
    if mode in {"", "block", "block_y", "block-y", "grip", "grip_axis", "grip-axis"}:
        return "block_y"
    if mode in {"block_x", "block-x"}:
        return "block_x"
    if mode in {"+y", "y", "world_y", "world-y"}:
        return "world_y"
    if mode in {"-y", "neg_y", "negative_y"}:
        return "neg_y"
    if mode in {"+x", "x", "world_x", "world-x"}:
        return "world_x"
    if mode in {"-x", "neg_x", "negative_x"}:
        return "neg_x"
    raise RuntimeError(f"unsupported TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS={mode!r}")


def block_axis_xy_from_quat(
    block_quat_wxyz: torch.Tensor,     # Param: tensor input carrying block quat wxyz values
    *,
    axis_index : int,  # Param: index selecting the axis entry
    fallback_xy: tuple[float, float],  # Param: floating-point input for fallback xy
) -> torch.Tensor:
    """Return a normalized block-local X or Y axis projected to world XY

    Steps:
    - Resolve inputs for `block_axis_xy_from_quat` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    matrix = quat_wxyz_to_matrix(block_quat_wxyz)
    axis = matrix[..., :, int(axis_index)]
    axis = axis.clone()
    axis[..., 2] = 0.0
    norm = torch.linalg.norm(axis[..., :2], dim=-1, keepdim=True)
    fallback = torch.tensor(
        (float(fallback_xy[0]), float(fallback_xy[1]), 0.0),
        device=axis.device,
        dtype=axis.dtype,
    )
    axis = torch.where(norm > 1.0e-6, axis, fallback.expand_as(axis))
    return F.normalize(axis, dim=-1)


def select_topdown_yaw_axis(
    axis_mode: str | None,          # Param: mode string selecting the axis behavior
    *,
    block_quat_wxyz: torch.Tensor,  # Param: tensor input carrying block quat wxyz values
) -> YawAxisSelection:
    """Select world yaw axis including block-x and block-y modes

    Steps:
    - Resolve inputs for `select_topdown_yaw_axis` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    normalized = normalize_topdown_yaw_axis_mode(axis_mode)
    device = block_quat_wxyz.device
    dtype = block_quat_wxyz.dtype
    num_rows = int(block_quat_wxyz.shape[0])
    if normalized == "block_y":
        axis = block_axis_xy_from_quat(block_quat_wxyz, axis_index=1, fallback_xy=(0.0, 1.0))
    elif normalized == "block_x":
        axis = block_axis_xy_from_quat(block_quat_wxyz, axis_index=0, fallback_xy=(1.0, 0.0))
    elif normalized == "world_y":
        axis = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype).view(1, 3).expand(num_rows, -1)
    elif normalized == "neg_y":
        axis = torch.tensor((0.0, -1.0, 0.0), device=device, dtype=dtype).view(1, 3).expand(num_rows, -1)
    elif normalized == "world_x":
        axis = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype).view(1, 3).expand(num_rows, -1)
    elif normalized == "neg_x":
        axis = torch.tensor((-1.0, 0.0, 0.0), device=device, dtype=dtype).view(1, 3).expand(num_rows, -1)
    else:
        raise RuntimeError(f"unsupported normalized yaw axis mode={normalized!r}")
    return YawAxisSelection(mode=normalized, axis_world=axis)
