"""

Native closed-loop teacher action orchestration

File map:

NativeTeacherConfig:           Teacher action mode switches
NativeTeacherRequest:          Inputs for one native teacher action
NativeTeacherAction:           Reduced teacher action and named components
select_native_teacher_action:  Build one closed-loop native teacher reduced action
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..teacher.teacher_actions import (
    TopdownContactTeacherParts,
    apply_stage2_teacher_gate,
    assemble_teacher_reduced_action,
    teacher_finger_action,
)
from ..teacher.teacher_arm_controller import TeacherArmBackend, TeacherArmRequest, compute_teacher_arm_reduced


FingerModeFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class NativeTeacherConfig:
    """Teacher action mode switches"""

    topdown_contact_teacher_enabled: bool  # boolean state indicating whether topdown contact teacher is enabled
    topdown_curriculum_task        : bool = True  # boolean value indicating the topdown curriculum task state for native teacher config


@dataclass(frozen=True)
class NativeTeacherRequest:
    """Inputs for one native teacher action"""

    env                           : object  # environment/backend object used by this runtime helper
    mapped_indices                : torch.Tensor  # column indices used to map between action layouts
    mapped_scales                 : torch.Tensor  # scales applied while mapping action columns
    closure_fraction              : torch.Tensor  # normalized finger-closure progress fraction
    compute_finger_in_current_mode: FingerModeFn  # stores compute finger in current mode for native teacher request
    contact_parts                 : TopdownContactTeacherParts | None = None  # stores contact parts for native teacher request
    stage                         : torch.Tensor | None               = None  # tensor containing stage values for batched env rows
    episode_step                  : int | torch.Tensor | None         = None  # per-env step count inside the current episode


@dataclass(frozen=True)
class NativeTeacherAction:
    """Reduced teacher action and named components"""

    action          : torch.Tensor  # environment action tensor selected for the step
    arm_action      : torch.Tensor  # tensor containing arm action values for batched env rows
    finger_action   : torch.Tensor  # tensor containing finger action values for batched env rows
    closure_fraction: torch.Tensor  # normalized finger-closure progress fraction
    descent         : torch.Tensor | None  # tensor containing descent values for batched env rows


def select_native_teacher_action(
    request: NativeTeacherRequest,   # Param: normalized request object passed into this helper
    *,
    config     : NativeTeacherConfig,  # Param: configuration object used by this helper
    arm_backend: TeacherArmBackend,  # Param: input value used as arm backend
) -> NativeTeacherAction:
    """Build one closed-loop native teacher reduced action

    Steps:
    - Resolve inputs for `select_native_teacher_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    descent = request.contact_parts.descent if request.contact_parts is not None else None
    if bool(config.topdown_contact_teacher_enabled) and request.contact_parts is not None:
        # The contact teacher owns its own readiness/geometry gates.  Applying the
        # generic Stage-2 gate here suppresses the very descent/closure that
        # should create the Stage-2 transition.
        closure = request.contact_parts.closure_fraction
        gated_descent = descent
    else:
        closure, gated_descent = apply_stage2_teacher_gate(
            closure_fraction=request.closure_fraction,
            descent=descent,
            stage=request.stage,
            topdown_curriculum_task=config.topdown_curriculum_task,
        )
    finger_action = teacher_finger_action(
        topdown_contact_teacher_enabled=config.topdown_contact_teacher_enabled,
        contact_parts=request.contact_parts,
        closure_fraction=closure,
        compute_in_current_mode=request.compute_finger_in_current_mode,
    )
    arm_action = compute_teacher_arm_reduced(
        TeacherArmRequest(
            env=request.env,
            mapped_indices=request.mapped_indices,
            mapped_scales=request.mapped_scales,
            closure_fraction=closure,
            episode_step=request.episode_step,
            topdown_contact_descent=gated_descent,
            topdown_contact_xy_offset=(
                None if request.contact_parts is None else request.contact_parts.xy_offset
            ),
            topdown_contact_inward=(
                None if request.contact_parts is None else request.contact_parts.inward
            ),
            topdown_contact_tip_servo=(
                None if request.contact_parts is None else request.contact_parts.tip_servo
            ),
        ),
        backend=arm_backend,
    )
    return NativeTeacherAction(
        action=assemble_teacher_reduced_action(arm_action=arm_action, finger_action=finger_action),
        arm_action=arm_action,
        finger_action=finger_action,
        closure_fraction=closure,
        descent=gated_descent,
    )
