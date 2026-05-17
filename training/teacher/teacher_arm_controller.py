"""

Teacher arm controller orchestration boundary

File map:

TeacherArmRequest:            Inputs needed to compute teacher arm reduced actions
TeacherArmBackend:            Backend surface for Isaac-bound teacher arm IK
compute_teacher_arm_reduced:  Dispatch teacher arm action computation to an explicit backend
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TeacherArmRequest:
    """Inputs needed to compute teacher arm reduced actions"""

    env                      : object  # environment/backend object used by this runtime helper
    mapped_indices           : torch.Tensor  # column indices used to map between action layouts
    mapped_scales            : torch.Tensor  # scales applied while mapping action columns
    closure_fraction         : float | torch.Tensor        = 0.0  # normalized finger-closure progress fraction
    episode_step             : int | torch.Tensor | None   = None  # per-env step count inside the current episode
    topdown_contact_descent  : float | torch.Tensor | None = None  # tensor containing topdown contact descent values for batched env rows
    topdown_contact_xy_offset: torch.Tensor | None         = None  # tensor containing topdown contact xy offset values for batched env rows
    topdown_contact_inward   : float | torch.Tensor | None = None  # tensor containing topdown contact inward values for batched env rows
    topdown_contact_tip_servo: torch.Tensor | None         = None  # tensor containing topdown contact tip servo values for batched env rows


class TeacherArmBackend(Protocol):
    """Backend surface for Isaac-bound teacher arm IK"""

    def compute_teacher_arm_reduced(self, request: TeacherArmRequest) -> torch.Tensor:
        """Return reduced arm actions for the request"""
        ...


def compute_teacher_arm_reduced(
    request: TeacherArmRequest,  # Param: normalized request object passed into this helper
    *,
    backend: TeacherArmBackend,  # Param: backend object that performs the runtime operation
) -> torch.Tensor:
    """Dispatch teacher arm action computation to an explicit backend"""
    result = backend.compute_teacher_arm_reduced(request)
    if not torch.is_tensor(result):
        raise TypeError(f"teacher arm backend returned {type(result)!r}")
    return result.clamp(-1.0, 1.0)
