"""

Import-safe IK servo tensor solvers

File map:

ServoDelta:                  Joint delta plus before and after residual diagnostics
damped_least_squares_delta:  Solve one batched damped least-squares joint correction
align_line_z_delta:          Solve scalar thumb-index line-Z alignment correction
planar_align_delta:          Solve planar thumb-index relative alignment correction
accept_nonworsening_delta:   Reject active corrections that make residuals too much worse
clamp_to_soft_limits:        Clamp joint positions to soft limits
tip_jacobian_joint_weights:  Return joint weights for direct fingertip DLS corrections
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ServoDelta:
    """Joint delta plus before and after residual diagnostics"""

    delta_q   : torch.Tensor  # Field: per-joint IK increment applied to the current joint positions
    err_before: torch.Tensor  # Field: tensor containing err before values for batched env rows
    err_after : torch.Tensor  # Field: tensor containing err after values for batched env rows
    active    : torch.Tensor  # Field: whether this configuration or runtime path is active


def damped_least_squares_delta(
    *,
    jacobian      : torch.Tensor,  # Param: tensor input carrying jacobian values
    error         : torch.Tensor,  # Param: tensor input carrying error values
    damping       : float,  # Param: damped least-squares stabilizer for the IK solve
    max_joint_step: float,  # Param: maximum per-joint IK update before soft-limit clamping
    active        : torch.Tensor | None = None,  # Param: tensor input carrying active values
    joint_weight  : torch.Tensor | None = None,  # Param: weight applied to joint
) -> ServoDelta:
    """Solve one batched damped least-squares joint correction

    Steps:
    - Resolve inputs for `damped_least_squares_delta` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    jac = jacobian
    err = error
    if joint_weight is not None:
        weight = joint_weight.to(device=jac.device, dtype=jac.dtype).view(1, 1, -1)
        jac = jac * weight
    jj_t = torch.bmm(jac, jac.transpose(1, 2))
    eye = torch.eye(jj_t.shape[-1], dtype=jac.dtype, device=jac.device).unsqueeze(0)
    rhs = err.unsqueeze(-1)
    try:
        solved = torch.linalg.solve(jj_t + eye * (float(damping) * float(damping)), rhs)
        delta_q = torch.bmm(jac.transpose(1, 2), solved).squeeze(-1)
    except RuntimeError:
        delta_q = torch.zeros((jac.shape[0], jac.shape[-1]), dtype=jac.dtype, device=jac.device)
    if joint_weight is not None:
        delta_q = delta_q * joint_weight.to(device=jac.device, dtype=jac.dtype).view(1, -1)
    if float(max_joint_step) > 0.0:
        delta_q = torch.clamp(delta_q, min=-float(max_joint_step), max=float(max_joint_step))
    if active is None:
        active_mask = torch.ones(jac.shape[0], dtype=torch.bool, device=jac.device)
    else:
        active_mask = active.to(device=jac.device, dtype=torch.bool)
        delta_q = torch.where(active_mask.unsqueeze(-1), delta_q, torch.zeros_like(delta_q))
    err_before = torch.linalg.norm(err, dim=1)
    err_after_vec = err - torch.einsum("nij,nj->ni", jacobian, delta_q)
    err_after = torch.linalg.norm(err_after_vec, dim=1)
    return ServoDelta(delta_q=delta_q, err_before=err_before, err_after=err_after, active=active_mask)


def align_line_z_delta(
    *,
    line_z        : torch.Tensor,  # Param: tensor input carrying line z values
    line_jacobian : torch.Tensor,  # Param: tensor input carrying line jacobian values
    active        : torch.Tensor,  # Param: tensor input carrying active values
    gain          : float,  # Param: floating-point input for gain
    max_dz        : float,  # Param: floating-point input for max dz
    damping       : float,  # Param: damped least-squares stabilizer for the IK solve
    max_joint_step: float,  # Param: maximum per-joint IK update before soft-limit clamping
) -> ServoDelta:
    """Solve scalar thumb-index line-Z alignment correction"""
    desired_dz = torch.clamp(-float(gain) * line_z, min=-float(max_dz), max=float(max_dz))
    err = desired_dz.unsqueeze(-1)
    jac = line_jacobian.unsqueeze(1)
    return damped_least_squares_delta(
        jacobian=jac,
        error=err,
        damping=damping,
        max_joint_step=max_joint_step,
        active=active,
    )


def planar_align_delta(
    *,
    rel_err_xy     : torch.Tensor,  # Param: tensor input carrying rel err xy values
    rel_jacobian_xy: torch.Tensor,  # Param: tensor input carrying rel jacobian xy values
    active         : torch.Tensor,  # Param: tensor input carrying active values
    gain           : float,  # Param: floating-point input for gain
    max_xy         : float,  # Param: floating-point input for max xy
    damping        : float,  # Param: damped least-squares stabilizer for the IK solve
    max_joint_step : float,  # Param: maximum per-joint IK update before soft-limit clamping
    joint_weight   : torch.Tensor | None = None,  # Param: weight applied to joint
) -> ServoDelta:
    """Solve planar thumb-index relative alignment correction

    Steps:
    - Resolve inputs for `planar_align_delta` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    desired_xy = rel_err_xy * float(gain)
    desired_norm = torch.linalg.norm(desired_xy, dim=1, keepdim=True)
    desired_xy = desired_xy * torch.clamp(float(max_xy) / desired_norm.clamp_min(1.0e-6), max=1.0)
    desired_xy = torch.where(active.to(device=rel_err_xy.device, dtype=torch.bool).unsqueeze(-1), desired_xy, torch.zeros_like(desired_xy))
    return damped_least_squares_delta(
        jacobian=rel_jacobian_xy,
        error=desired_xy,
        damping=damping,
        max_joint_step=max_joint_step,
        active=active,
        joint_weight=joint_weight,
    )


def accept_nonworsening_delta(
    *,
    delta_q   : torch.Tensor,  # Param: tensor input carrying delta q values
    err_before: torch.Tensor,  # Param: tensor input carrying err before values
    err_after : torch.Tensor,  # Param: tensor input carrying err after values
    active    : torch.Tensor,  # Param: tensor input carrying active values
    max_worse : float = 0.0,  # Param: floating-point input for max worse
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reject active corrections that make residuals too much worse"""
    accepted = active.to(device=delta_q.device, dtype=torch.bool) & (err_after <= err_before + float(max_worse))
    delta = torch.where(accepted.unsqueeze(-1), delta_q, torch.zeros_like(delta_q))
    after = torch.where(accepted, err_after, err_before)
    return delta, after, accepted


def clamp_to_soft_limits(
    joint_pos: torch.Tensor,  # Param: current joint-position tensor used as the IK starting point
    lower    : torch.Tensor,  # Param: tensor input carrying lower values
    upper    : torch.Tensor,  # Param: tensor input carrying upper values
) -> torch.Tensor:
    """Clamp joint positions to soft limits"""
    return torch.clamp(joint_pos, min=lower, max=upper)


def tip_jacobian_joint_weights(
    joint_names: Sequence[str],          # Param: ordered candidate names used to resolve joint
    *,
    spec  : str                = "base",  # Param: string input for spec
    device: torch.device | str = "cpu",  # Param: torch device where tensors are read or allocated
    dtype : torch.dtype        = torch.float32,  # Param: torch dtype used when converting or allocating tensors
) -> torch.Tensor:
    """Return joint weights for direct fingertip DLS corrections"""
    cleaned = str(spec).strip()
    lowered = cleaned.lower()
    if lowered in ("", "all"):
        enabled = [True for _ in joint_names]
    elif lowered == "base":
        enabled = [
            joint_name not in {"right_wrist_roll_joint", "waist_yaw_joint"}
            for joint_name in joint_names
        ]
    else:
        names = {name.strip() for name in cleaned.split(",") if name.strip()}
        enabled = [joint_name in names for joint_name in joint_names]
    return torch.tensor(enabled, dtype=dtype, device=device)
