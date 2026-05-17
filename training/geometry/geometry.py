"""

Geometry and joint-selection helpers for trainer modules

File map:

quat_wxyz_to_matrix:      Convert wxyz quaternions to rotation matrices
parse_joint_name_list:    Parse a comma-separated joint name list
joint_selection_weights:  Return a 0 or 1 weight vector for selected joint names
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices

    Steps:
    - Resolve inputs for `quat_wxyz_to_matrix` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def parse_joint_name_list(raw: str | None, default: str) -> tuple[str, ...]:
    """Parse a comma-separated joint name list"""
    source = default if raw is None or raw == "" else raw
    return tuple(name.strip() for name in source.split(",") if name.strip())


def joint_selection_weights(
    joint_names   : Sequence[str],  # Param: ordered candidate names used to resolve joint
    selected_names: Sequence[str],  # Param: ordered candidate names used to resolve selected
    *,
    device: torch.device | str = "cpu",  # Param: torch device where tensors are read or allocated
    dtype : torch.dtype        = torch.float32,  # Param: torch dtype used when converting or allocating tensors
) -> torch.Tensor:
    """Return a 0 or 1 weight vector for selected joint names"""
    lowered = {name.lower() for name in selected_names}
    if "all" in lowered:
        return torch.ones(len(joint_names), device=device, dtype=dtype)
    selected = set(selected_names)
    return torch.tensor(
        [1.0 if joint_name in selected else 0.0 for joint_name in joint_names],
        device=device,
        dtype=dtype,
    )
