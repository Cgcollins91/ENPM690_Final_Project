"""

Topdown task-space IK tensor solver

This module provides a batched task-space IK solver for top-down hand control, along with related utilities f
or computing per-env IK weight scales based on the current curriculum stage and for scoring proposed IK solutions based
on predicted residuals at the thumb and index fingertips. The task-space IK solver uses a damped least-squares approach
to solve for joint updates that reduce the center-position, span, and orientation errors of the hand, with optional nullspace
posture regularization to pull joints toward a default pose. The pocket-sweep search uses the Jacobians from the task-space
IK solver to efficiently evaluate proposed joint updates in the neighborhood of the current joint positions, allowing the
policy to explore alternative configurations that may yield better predicted outcomes in task space. The pocket-sweep search
can be configured to apply different step sizes and weightings to different joints, allowing it to focus on joints that have
a larger impact on the thumb and index positions, which are critical for successful grasping. By incorporating task-space IK a
nd pocket-sweep search into the training loop, the policy can learn to predict joint positions that achieve the desired hand pose more
effectively, improving convergence and final performance on the top-down grasping task.

File map:

TaskSpaceIKResult:            Solved task-space IK state and residual diagnostics
TaskSpaceIKWeightScales:      Per-env task-space IK weight scales and lift mode mask
_as_row_weight:               Handle as row weight logic
task_space_ik_weight_scales:  Resolve prehover and lift-latched IK weight scales
solve_topdown_task_space_ik:  Solve weighted center span and orientation IK with optional posture nullspace
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskSpaceIKResult:
    """Solved task-space IK state and residual diagnostics"""

    joint_pos_des   : torch.Tensor  # desired joint positions after the IK step and soft-limit clamp
    delta_q         : torch.Tensor  # per-joint IK increment applied to the current joint positions
    center_err      : torch.Tensor  # norm of the center-position error before the IK update
    center_err_after: torch.Tensor  # predicted center-position error norm after the IK update
    span_err_z      : torch.Tensor  # absolute Z component of the span error before the IK update
    span_err_z_after: torch.Tensor  # predicted absolute Z span error after the IK update
    orient_err      : torch.Tensor  # norm of the orientation error before the IK update
    orient_err_after: torch.Tensor  # predicted orientation error norm after the IK update


@dataclass(frozen=True)
class TaskSpaceIKWeightScales:
    """Per-env task-space IK weight scales and lift mode mask"""

    span_scale        : torch.Tensor  # per-env multiplier for span-error rows in the task-space IK solve
    orient_scale      : torch.Tensor  # per-env multiplier for orientation-error rows in the task-space IK solve
    posture_weight    : torch.Tensor  # nullspace weight pulling joints back toward the default posture
    lift_position_only: torch.Tensor  # mask for lift rows that should solve position only


def _as_row_weight(
    weight: torch.Tensor,  # Param: tensor input carrying weight values
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device,  # Param: torch device where tensors are read or allocated
    dtype   : torch.dtype,  # Param: torch dtype used when converting or allocating tensors
) -> torch.Tensor:
    value = weight.to(device=device, dtype=dtype)
    if value.ndim == 1:
        return value.view(1, -1).expand(num_envs, -1)
    return value


def task_space_ik_weight_scales(
    *,
    prehover_active           : torch.Tensor,  # Param: mask selecting env rows currently in prehover mode
    lift_latched              : torch.Tensor,  # Param: mask selecting env rows with lift latch active
    prehover_span_scale       : float,  # Param: span-error multiplier used while prehover is active
    prehover_orientation_scale: float,  # Param: orientation-error multiplier used while prehover is active
    lift_span_scale           : float = -1.0,  # Param: span-error multiplier used after lift latch when enabled
    lift_orientation_scale    : float = -1.0,  # Param: orientation-error multiplier used after lift latch when enabled
    base_posture_weight       : float = 0.0,  # Param: default nullspace posture weight for IK rows
    prehover_posture_weight   : float = -1.0,  # Param: posture weight override used while prehover is active
    lift_posture_weight       : float = -1.0,  # Param: posture weight override used after lift latch
) -> TaskSpaceIKWeightScales:
    """Resolve prehover and lift-latched IK weight scales

    Steps:
    - Resolve inputs for `task_space_ik_weight_scales` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    device = prehover_active.device
    dtype = torch.float32
    prehover = prehover_active.to(device=device, dtype=torch.bool).reshape(-1)
    latched = lift_latched.to(device=device, dtype=torch.bool).reshape(-1)
    span = torch.where(
        prehover,
        torch.full_like(prehover, float(prehover_span_scale), dtype=dtype),
        torch.ones_like(prehover, dtype=dtype),
    )
    orient = torch.where(
        prehover,
        torch.full_like(prehover, float(prehover_orientation_scale), dtype=dtype),
        torch.ones_like(prehover, dtype=dtype),
    )
    if float(lift_span_scale) >= 0.0:
        span = torch.where(latched, torch.full_like(span, max(float(lift_span_scale), 0.0)), span)
    if float(lift_orientation_scale) >= 0.0:
        orient = torch.where(
            latched,
            torch.full_like(orient, max(float(lift_orientation_scale), 0.0)),
            orient,
        )
    posture = torch.full_like(span, float(base_posture_weight))
    if float(prehover_posture_weight) >= 0.0:
        posture = torch.where(prehover, torch.full_like(posture, float(prehover_posture_weight)), posture)
    if float(lift_posture_weight) >= 0.0:
        posture = torch.where(latched, torch.full_like(posture, float(lift_posture_weight)), posture)
    position_only = latched & (span <= 1.0e-6) & (orient <= 1.0e-6)
    return TaskSpaceIKWeightScales(
        span_scale=span,
        orient_scale=orient,
        posture_weight=posture,
        lift_position_only=position_only,
    )


def solve_topdown_task_space_ik(
    *,
    joint_pos             : torch.Tensor,  # Param: current joint-position tensor used as the IK starting point
    default_joint_pos     : torch.Tensor,  # Param: default joint positions used by the IK posture term
    soft_limits           : torch.Tensor,  # Param: per-joint lower and upper limits used to clamp IK output
    center_err            : torch.Tensor,  # Param: task-space center-position error before the IK step
    span_err              : torch.Tensor,  # Param: task-space finger-span error before the IK step
    orient_err            : torch.Tensor,  # Param: task-space orientation error before the IK step
    center_jacobian       : torch.Tensor,  # Param: Jacobian rows mapping joints to center-position error
    span_jacobian         : torch.Tensor,  # Param: Jacobian rows mapping joints to span error
    orient_jacobian       : torch.Tensor,  # Param: Jacobian rows mapping joints to orientation error
    center_weight         : torch.Tensor,  # Param: per-task weights applied to center-position error rows
    span_weight           : torch.Tensor,  # Param: per-task weights applied to span-error rows
    orient_weight         : torch.Tensor,  # Param: per-task weights applied to orientation-error rows
    damping               : float,  # Param: damped least-squares stabilizer for the IK solve
    max_joint_step        : float,  # Param: maximum per-joint IK update before soft-limit clamping
    posture_weight        : float               = 0.0,  # Param: nullspace weight pulling joints toward default_joint_pos
    posture_weight_per_env: torch.Tensor | None = None,  # Param: per-env override for the IK nullspace posture weight
) -> TaskSpaceIKResult:
    """Solve weighted center span and orientation IK with optional posture nullspace

    Steps:
    - Resolve inputs for `solve_topdown_task_space_ik` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    dtype = joint_pos.dtype
    device = joint_pos.device
    num_envs = joint_pos.shape[0]
    center_w = _as_row_weight(center_weight, num_envs=num_envs, device=device, dtype=dtype)
    span_w = _as_row_weight(span_weight, num_envs=num_envs, device=device, dtype=dtype)
    orient_w = _as_row_weight(orient_weight, num_envs=num_envs, device=device, dtype=dtype)

    err_stack = torch.cat(
        [
            center_err.to(device=device, dtype=dtype) * center_w,
            span_err.to(device=device, dtype=dtype) * span_w,
            orient_err.to(device=device, dtype=dtype) * orient_w,
        ],
        dim=1,
    )
    jac_stack = torch.cat(
        [
            center_jacobian.to(device=device, dtype=dtype) * center_w.unsqueeze(-1),
            span_jacobian.to(device=device, dtype=dtype) * span_w.unsqueeze(-1),
            orient_jacobian.to(device=device, dtype=dtype) * orient_w.unsqueeze(-1),
        ],
        dim=1,
    )

    jj_t     = torch.bmm(jac_stack, jac_stack.transpose(1, 2))
    eye_task = torch.eye(jj_t.shape[-1], dtype=dtype, device=device).unsqueeze(0)
    rhs      = err_stack.unsqueeze(-1)
    try:
        system = jj_t + eye_task * (float(damping) * float(damping))
        solved = torch.linalg.solve(system, rhs)
        pinv_rhs = eye_task.expand(system.shape[0], -1, -1)
        j_pinv = torch.bmm(jac_stack.transpose(1, 2), torch.linalg.solve(system, pinv_rhs))
        delta_q = torch.bmm(jac_stack.transpose(1, 2), solved).squeeze(-1)
    except RuntimeError:
        j_pinv = torch.zeros(
            (num_envs, joint_pos.shape[1], jac_stack.shape[1]),
            dtype=dtype,
            device=device,
        )
        delta_q = torch.zeros_like(joint_pos)

    if posture_weight_per_env is not None:
        posture_weight_vec = posture_weight_per_env.to(device=device, dtype=dtype).view(num_envs, 1)
    else:
        posture_weight_vec = torch.full((num_envs, 1), float(posture_weight), dtype=dtype, device=device)
    if bool((posture_weight_vec > 0.0).any().item()):
        eye_joint = torch.eye(joint_pos.shape[1], dtype=dtype, device=device).unsqueeze(0)
        null_projector = eye_joint - torch.bmm(j_pinv, jac_stack)
        posture_delta = (default_joint_pos.to(device=device, dtype=dtype) - joint_pos) * posture_weight_vec
        delta_q = delta_q + torch.bmm(null_projector, posture_delta.unsqueeze(-1)).squeeze(-1)

    if float(max_joint_step) > 0.0:
        delta_q = torch.clamp(delta_q, min=-float(max_joint_step), max=float(max_joint_step))
    joint_pos_des = torch.clamp(joint_pos + delta_q, min=soft_limits[..., 0], max=soft_limits[..., 1])

    pred_center_err = center_err.to(device=device, dtype=dtype) - torch.einsum("nij,nj->ni", center_jacobian, delta_q)
    pred_span_err = span_err.to(device=device, dtype=dtype) - torch.einsum("nij,nj->ni", span_jacobian, delta_q)
    pred_orient_err = orient_err.to(device=device, dtype=dtype) - torch.einsum("nij,nj->ni", orient_jacobian, delta_q)
    return TaskSpaceIKResult(
        joint_pos_des=joint_pos_des,
        delta_q=delta_q,
        center_err=torch.linalg.norm(center_err.to(device=device, dtype=dtype), dim=1),
        center_err_after=torch.linalg.norm(pred_center_err, dim=1),
        span_err_z=torch.abs(span_err.to(device=device, dtype=dtype)[:, 2]),
        span_err_z_after=torch.abs(pred_span_err[:, 2]),
        orient_err=torch.linalg.norm(orient_err.to(device=device, dtype=dtype), dim=1),
        orient_err_after=torch.linalg.norm(pred_orient_err, dim=1),
    )
