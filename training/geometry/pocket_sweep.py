"""

Prehold pocket-sweep joint search helpers


These functions implement a local search around the current IK solution to find a nearby joint configuration that improves the
predicted thumb and index positions. The search is performed by iterating through each joint and applying small positive and
negative increments, then scoring the resulting configurations to select the best one. This can help the policy find better local minima
in the IK solution space and improve convergence, especially in the early stages of training when the policy may not have
learned to predict accurate joint positions yet. The pocket-sweep search is designed to be lightweight and can be applied
selectively based on configurable conditions, such as only applying it to certain joints or only applying it when the
predicted thumb and index positions are far from their targets.

File map:

PocketSweepResult:    Best pocket-sweep joint proposal and score diagnostics
pocket_sweep_score:   Score predicted thumb and index target residuals
pocket_sweep_search:  Search local one-joint pocket-sweep candidates
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class PocketSweepResult:
    """Best pocket-sweep joint proposal and score diagnostics"""

    joint_pos_des: torch.Tensor  # desired joint positions after the IK step and soft-limit clamp
    delta_q      : torch.Tensor  # per-joint IK increment applied to the current joint positions
    score_before : torch.Tensor  # tensor containing score before values for batched env rows
    score_after  : torch.Tensor  # tensor containing score after values for batched env rows
    active       : torch.Tensor  # whether this configuration or runtime path is active


def pocket_sweep_score(
    *,
    q             : torch.Tensor,  # Param: tensor input carrying q values
    joint_pos     : torch.Tensor,  # Param: current joint-position tensor used as the IK starting point
    thumb_pos     : torch.Tensor,  # Param: tensor input carrying thumb pos values
    index_pos     : torch.Tensor,  # Param: tensor input carrying index pos values
    thumb_target  : torch.Tensor,  # Param: target value for thumb
    index_target  : torch.Tensor,  # Param: target value for index
    thumb_jacobian: torch.Tensor,  # Param: tensor input carrying thumb jacobian values
    index_jacobian: torch.Tensor,  # Param: tensor input carrying index jacobian values
    z_weight      : float,  # Param: weight applied to z
) -> torch.Tensor:
    """Score predicted thumb and index target residuals

    Steps:
    - Resolve inputs for `pocket_sweep_score` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    dq = q - joint_pos
    pred_thumb = thumb_pos + torch.einsum("nij,nj->ni", thumb_jacobian, dq)
    pred_index = index_pos + torch.einsum("nij,nj->ni", index_jacobian, dq)
    thumb_err = thumb_target - pred_thumb
    index_err = index_target - pred_index
    thumb_xy = torch.sum(thumb_err[:, :2] * thumb_err[:, :2], dim=1)
    index_xy = torch.sum(index_err[:, :2] * index_err[:, :2], dim=1)
    thumb_z = thumb_err[:, 2] * thumb_err[:, 2]
    index_z = index_err[:, 2] * index_err[:, 2]
    return thumb_xy + index_xy + max(float(z_weight), 0.0) * (thumb_z + index_z)


def pocket_sweep_search(
    *,
    joint_pos     : torch.Tensor,     # Param: current joint-position tensor used as the IK starting point
    joint_pos_des : torch.Tensor,     # Param: tensor input carrying joint pos des values
    soft_limits   : torch.Tensor,     # Param: per-joint lower and upper limits used to clamp IK output
    max_step      : torch.Tensor,     # Param: step count used for max step
    active        : torch.Tensor,     # Param: tensor input carrying active values
    thumb_pos     : torch.Tensor,     # Param: tensor input carrying thumb pos values
    index_pos     : torch.Tensor,     # Param: tensor input carrying index pos values
    thumb_target  : torch.Tensor,     # Param: target value for thumb
    index_target  : torch.Tensor,     # Param: target value for index
    thumb_jacobian: torch.Tensor,     # Param: tensor input carrying thumb jacobian values
    index_jacobian: torch.Tensor,     # Param: tensor input carrying index jacobian values
    step_radians  : Sequence[float],  # Param: floating-point input for step radians
    joint_enabled : torch.Tensor,     # Param: boolean input enabling joint
    z_weight      : float = 0.25,     # Param: weight applied to z
    iters         : int   = 2,        # Param: integer input for iters
) -> PocketSweepResult:
    """Search local one-joint pocket-sweep candidates

    Steps:
    - Resolve inputs for `pocket_sweep_search` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    lower = torch.maximum(soft_limits[..., 0], joint_pos - max_step.to(device=joint_pos.device, dtype=joint_pos.dtype))
    upper = torch.minimum(soft_limits[..., 1], joint_pos + max_step.to(device=joint_pos.device, dtype=joint_pos.dtype))
    best_q = torch.clamp(joint_pos_des, min=lower, max=upper)
    active_mask = active.to(device=joint_pos.device, dtype=torch.bool)
    enabled = joint_enabled.to(device=joint_pos.device, dtype=torch.bool)
    score_before = pocket_sweep_score(
        q=best_q,
        joint_pos=joint_pos,
        thumb_pos=thumb_pos,
        index_pos=index_pos,
        thumb_target=thumb_target,
        index_target=index_target,
        thumb_jacobian=thumb_jacobian,
        index_jacobian=index_jacobian,
        z_weight=z_weight,
    )
    best_score = score_before

    steps = tuple(abs(float(step)) for step in step_radians if abs(float(step)) > 0.0)
    for _ in range(max(int(iters), 1)):
        for step_rad in steps:
            for joint_i in range(best_q.shape[1]):
                if not bool(enabled[joint_i].item()):
                    continue
                for sign in (-1.0, 1.0):
                    candidate = best_q.clone()
                    candidate[:, joint_i] = candidate[:, joint_i] + sign * step_rad
                    candidate = torch.clamp(candidate, min=lower, max=upper)
                    candidate_score = pocket_sweep_score(
                        q=candidate,
                        joint_pos=joint_pos,
                        thumb_pos=thumb_pos,
                        index_pos=index_pos,
                        thumb_target=thumb_target,
                        index_target=index_target,
                        thumb_jacobian=thumb_jacobian,
                        index_jacobian=index_jacobian,
                        z_weight=z_weight,
                    )
                    improved = active_mask & (candidate_score < best_score)
                    if bool(improved.any().item()):
                        best_q = torch.where(improved.unsqueeze(-1), candidate, best_q)
                        best_score = torch.where(improved, candidate_score, best_score)

    delta_q = best_q - torch.clamp(joint_pos_des, min=lower, max=upper)
    joint_out = torch.where(active_mask.unsqueeze(-1), best_q, joint_pos_des)
    return PocketSweepResult(
        joint_pos_des=joint_out,
        delta_q=delta_q,
        score_before=score_before,
        score_after=best_score,
        active=active_mask,
    )
