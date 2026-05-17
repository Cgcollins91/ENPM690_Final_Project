"""

Pure curriculum gate selectors for topdown stage logic

File map:

FingerXYZGateConfig:          Distance thresholds for block-center xyz finger gate
FingerCenterGateConfig:       Finger center readiness gate selection
resolve_xyz_gate_thresholds:  Resolve xyz gate thresholds with nonnegative overrides
xyz_gate_from_max_error:      Return xyz close gate from weighted fingertip max error
finger_center_ready_mask:     Return finger-center live mask using legacy residuals or xyz gate
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FingerXYZGateConfig:
    """Distance thresholds for block-center xyz finger gate"""

    start_m : float = 0.085  # Field: floating-point start m value used by finger x y z gate config
    full_m  : float = 0.025  # Field: floating-point full m value used by finger x y z gate config
    linear  : bool  = False  # Field: boolean value indicating the linear state for finger x y z gate config
    min_gate: float = 1.0e-6  # Field: floating-point min gate value used by finger x y z gate config


@dataclass(frozen=True)
class FingerCenterGateConfig:
    """Finger center readiness gate selection"""

    use_xyz_gate       : bool  = False  # Field: boolean value indicating the use xyz gate state for finger center gate config
    xyz_gate_min       : float = 1.0e-6  # Field: floating-point xyz gate min value used by finger center gate config
    tip_xy_max         : float = 0.0  # Field: floating-point tip xy max value used by finger center gate config
    max_tip_xy_max     : float = 0.0  # Field: floating-point max tip xy max value used by finger center gate config
    tip_z_max          : float = 0.075  # Field: floating-point tip z max value used by finger center gate config
    align_angle_max_deg: float = 15.0  # Field: floating-point align angle max deg value used by finger center gate config
    align_err_max      : float = 0.0  # Field: floating-point align err max value used by finger center gate config


def resolve_xyz_gate_thresholds(
    *,
    default_start_m : float,  # Param: floating-point input for default start m
    default_full_m  : float,  # Param: floating-point input for default full m
    start_m_override: float | None = None,  # Param: floating-point input for start m override
    full_m_override : float | None = None,  # Param: floating-point input for full m override
) -> tuple[float, float]:
    """Resolve xyz gate thresholds with nonnegative overrides"""
    start = (
        float(start_m_override)
        if start_m_override is not None and float(start_m_override) >= 0.0
        else float(default_start_m)
    )
    full = (
        float(full_m_override)
        if full_m_override is not None and float(full_m_override) >= 0.0
        else float(default_full_m)
    )
    return max(start, 0.0), max(full, 0.0)


def xyz_gate_from_max_error(
    max_err: torch.Tensor,  # Param: tensor input carrying max err values
    config : FingerXYZGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Return xyz close gate from weighted fingertip max error"""
    start_m, full_m = resolve_xyz_gate_thresholds(
        default_start_m=config.start_m,
        default_full_m=config.full_m,
    )
    if start_m > full_m + 1.0e-6:
        linear_gate = torch.clamp((start_m - max_err) / (start_m - full_m), 0.0, 1.0)
    else:
        linear_gate = (max_err <= full_m).to(dtype=torch.float32)
    if bool(config.linear):
        return linear_gate
    return (max_err <= start_m).to(dtype=torch.float32)


def finger_center_ready_mask(
    *,
    in_stage_2           : torch.Tensor,  # Param: tensor input carrying in stage 2 values
    contact_pose_ready   : torch.Tensor,  # Param: mask or boolean input marking contact pose as ready
    center_xy            : torch.Tensor,  # Param: tensor input carrying center xy values
    center_max_xy        : torch.Tensor,  # Param: tensor input carrying center max xy values
    center_z             : torch.Tensor,  # Param: tensor input carrying center z values
    center_angle_deg     : torch.Tensor,  # Param: tensor input carrying center angle deg values
    align_err            : torch.Tensor,  # Param: tensor input carrying align err values
    xyz_gate             : torch.Tensor,  # Param: tensor input carrying xyz gate values
    requires_contact_pose: bool,  # Param: boolean input controlling requires contact pose
    config               : FingerCenterGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Return finger-center live mask using legacy residuals or xyz gate

    Steps:
    - Resolve inputs for `finger_center_ready_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ready = in_stage_2.to(dtype=torch.bool)
    if bool(requires_contact_pose):
        ready = ready & contact_pose_ready.to(device=ready.device, dtype=torch.bool)
    if config.use_xyz_gate:
        ready = ready & (xyz_gate.to(device=ready.device, dtype=torch.float32) > float(config.xyz_gate_min))
    else:
        if float(config.tip_xy_max) > 0.0:
            ready = ready & (center_xy.to(device=ready.device) <= float(config.tip_xy_max))
        if float(config.max_tip_xy_max) > 0.0:
            ready = ready & (center_max_xy.to(device=ready.device) <= float(config.max_tip_xy_max))
        if float(config.tip_z_max) > 0.0:
            ready = ready & (center_z.to(device=ready.device) <= float(config.tip_z_max))
    if float(config.align_angle_max_deg) > 0.0:
        ready = ready & (center_angle_deg.to(device=ready.device) <= float(config.align_angle_max_deg))
    if float(config.align_err_max) > 0.0:
        ready = ready & (align_err.to(device=ready.device) <= float(config.align_err_max))
    return ready
