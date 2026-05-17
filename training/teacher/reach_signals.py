"""

Phase-1 reach signal helpers

This module provides helper functions for computing Phase-1 reach signals and diagnostics, used by the v35 teacher.

File map:

phase1_reach_ready_mask:      Return rows inside the Phase-1 reach shell
phase1_reach_signal_tensors:  Return Phase-1 palm distance orientation error and readiness mask
phase1_reach_signals:         Return scalar Phase-1 reach diagnostics for one env
"""

from __future__ import annotations

import torch


def phase1_reach_ready_mask(
    *,
    palm_dist           : torch.Tensor,  # Param: tensor input carrying palm dist values
    height_err          : torch.Tensor,  # Param: tensor input carrying height err values
    orient_rad          : torch.Tensor,  # Param: tensor input carrying orient rad values
    palm_tolerance      : float,  # Param: tolerance allowed for palm
    height_tolerance    : float,  # Param: tolerance allowed for height
    orient_tolerance_rad: float,  # Param: floating-point input for orient tolerance rad
) -> torch.Tensor:
    """Return rows inside the Phase-1 reach shell"""
    return (
        (palm_dist <= float(palm_tolerance))
        & (height_err.to(device=palm_dist.device) <= float(height_tolerance))
        & (orient_rad.to(device=palm_dist.device) <= float(orient_tolerance_rad))
    )


def phase1_reach_signal_tensors(
    *,
    palm_dist           : torch.Tensor,  # Param: tensor input carrying palm dist values
    height_err          : torch.Tensor,  # Param: tensor input carrying height err values
    orient_rad          : torch.Tensor,  # Param: tensor input carrying orient rad values
    palm_tolerance      : float,  # Param: tolerance allowed for palm
    height_tolerance    : float,  # Param: tolerance allowed for height
    orient_tolerance_rad: float,  # Param: floating-point input for orient tolerance rad
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Phase-1 palm distance orientation error and readiness mask"""
    ready = phase1_reach_ready_mask(
        palm_dist=palm_dist,
        height_err=height_err,
        orient_rad=orient_rad,
        palm_tolerance=palm_tolerance,
        height_tolerance=height_tolerance,
        orient_tolerance_rad=orient_tolerance_rad,
    )
    return palm_dist, orient_rad, ready


def phase1_reach_signals(
    *,
    palm_dist : torch.Tensor,  # Param: tensor input carrying palm dist values
    orient_rad: torch.Tensor,  # Param: tensor input carrying orient rad values
    ready     : torch.Tensor,  # Param: tensor input carrying ready values
    env_id    : int = 0,  # Param: integer input for env id
) -> tuple[float, float, bool]:
    """Return scalar Phase-1 reach diagnostics for one env"""
    try:
        idx = int(env_id)
        return (
            float(palm_dist[idx].item()),
            float(torch.rad2deg(orient_rad[idx]).item()),
            bool(ready[idx].item()),
        )
    except (IndexError, AttributeError, ValueError, RuntimeError):
        return float("inf"), float("inf"), False
