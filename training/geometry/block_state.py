"""

Block state diagnostic adapters


block_angular_velocity_magnitude:  Return per-env block angular velocity magnitude
block_tilt_deg:                    Return per-env active block tilt in degrees
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def block_angular_velocity_magnitude(
    env,                                                                                  # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task            : bool,  # Param: boolean input controlling topdown curriculum task
    topdown_block_angular_velocity_norm: Callable[[object], torch.Tensor] | None = None,  # Param: callback used to compute or fetch topdown block angular velocity norm
) -> torch.Tensor:
    """Return per-env block angular velocity magnitude"""
    if topdown_curriculum_task:
        if topdown_block_angular_velocity_norm is None:
            raise RuntimeError("topdown block angular velocity function is required")
        return topdown_block_angular_velocity_norm(env)
    obj = env.scene["object"]
    return torch.linalg.norm(obj.data.root_ang_vel_w[:, :3], dim=1)


def block_tilt_deg(
    env,                                                                           # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task     : bool,  # Param: boolean input controlling topdown curriculum task
    topdown_block_tilt_angle_rad: Callable[[object], torch.Tensor] | None = None,  # Param: callback used to compute or fetch topdown block tilt angle rad
) -> torch.Tensor:
    """Return per-env active block tilt in degrees"""
    if topdown_curriculum_task:
        if topdown_block_tilt_angle_rad is None:
            raise RuntimeError("topdown block tilt function is required")
        return torch.rad2deg(topdown_block_tilt_angle_rad(env))
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
