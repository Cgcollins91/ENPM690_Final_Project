"""

Teacher finger-closure schedule helpers

File map:

TouchTeacherLatchState:              Updated teacher contact latch state
smoothstep01:                        Apply smoothstep shaping to values in 0 to 1
teacher_time_closure_fraction:       Return time-based teacher closure fraction
teacher_topdown_closure_proximity:   Return topdown closure proximity from palm and orientation errors
light_contact_closure_gate:          Return light-contact teacher close gate
teacher_closure_fraction_from_mode:  Combine closure fractions for distance or time mode
lift_hold_closure_fraction:          Return partially closed lift-only teacher fraction
update_touch_teacher_latch:          Update teacher closure latch from contact and scheduled fraction
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TouchTeacherLatchState:
    """Updated teacher contact latch state"""

    fraction          : torch.Tensor  # Field: tensor containing fraction values for batched env rows
    latched           : torch.Tensor  # Field: per-env latch mask or aggregate latch state
    hold_fraction     : torch.Tensor  # Field: tensor containing hold fraction values for batched env rows
    first_contact_step: torch.Tensor  # Field: step count used for first contact step scheduling or reporting
    latch_step        : torch.Tensor  # Field: step count used for latch step scheduling or reporting


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    """Apply smoothstep shaping to values in 0 to 1"""
    clipped = value.clamp(0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def teacher_time_closure_fraction(
    episode_step: torch.Tensor,   # Param: per-env step count inside the current episode
    *,
    finger_curl_start   : float,  # Param: floating-point input for finger curl start
    finger_curl_duration: float,  # Param: floating-point input for finger curl duration
) -> torch.Tensor:
    """Return time-based teacher closure fraction"""
    step = episode_step.to(dtype=torch.float32)
    if float(finger_curl_duration) <= 0.0:
        return (step >= float(finger_curl_start)).to(dtype=torch.float32)
    progress = (step - float(finger_curl_start)) / float(finger_curl_duration)
    return progress.clamp(0.0, 1.0)


def teacher_topdown_closure_proximity(
    *,
    palm_dist       : torch.Tensor,  # Param: tensor input carrying palm dist values
    orient_rad      : torch.Tensor,  # Param: tensor input carrying orient rad values
    palm_outer      : float = 0.16,  # Param: floating-point input for palm outer
    palm_inner      : float = 0.09,  # Param: floating-point input for palm inner
    orient_outer_deg: float = 45.0,  # Param: floating-point input for orient outer deg
    orient_inner_deg: float = 25.0,  # Param: floating-point input for orient inner deg
) -> torch.Tensor:
    """Return topdown closure proximity from palm and orientation errors

    Steps:
    - Resolve inputs for `teacher_topdown_closure_proximity` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    palm_span = max(float(palm_outer) - float(palm_inner), 1.0e-6)
    palm_progress = torch.clamp((float(palm_outer) - palm_dist) / palm_span, 0.0, 1.0)
    orient_outer = math.radians(float(orient_outer_deg))
    orient_inner = math.radians(float(orient_inner_deg))
    orient_span = max(orient_outer - orient_inner, 1.0e-6)
    orient_progress = torch.clamp((orient_outer - orient_rad) / orient_span, 0.0, 1.0)
    return palm_progress * orient_progress


def light_contact_closure_gate(
    *,
    palm_dist      : torch.Tensor,  # Param: tensor input carrying palm dist values
    height_err     : torch.Tensor,  # Param: tensor input carrying height err values
    orient_rad     : torch.Tensor,  # Param: tensor input carrying orient rad values
    thumb_face_dist: torch.Tensor,  # Param: tensor input carrying thumb face dist values
    index_face_dist: torch.Tensor,  # Param: tensor input carrying index face dist values
    align_err      : torch.Tensor,  # Param: tensor input carrying align err values
    line_angle_rad : torch.Tensor,  # Param: tensor input carrying line angle rad values
    opposed_gate   : torch.Tensor,  # Param: tensor input carrying opposed gate values
    max_palm_dist  : float = 0.07,  # Param: floating-point input for max palm dist
    max_height_err : float = 0.05,  # Param: floating-point input for max height err
    max_orient_deg : float = 22.0,  # Param: floating-point input for max orient deg
    max_face_dist  : float = 0.10,  # Param: floating-point input for max face dist
    max_align_dist : float = 0.17,  # Param: floating-point input for max align dist
    max_line_deg   : float = 25.0,  # Param: floating-point input for max line deg
    opposed_min    : float = 0.5,  # Param: floating-point input for opposed min
) -> torch.Tensor:
    """Return light-contact teacher close gate"""
    face_err = torch.maximum(thumb_face_dist, index_face_dist)
    ready = (
        (palm_dist <= float(max_palm_dist))
        & (height_err <= float(max_height_err))
        & (orient_rad <= math.radians(float(max_orient_deg)))
        & (face_err <= float(max_face_dist))
        & (align_err <= float(max_align_dist))
        & (line_angle_rad <= math.radians(float(max_line_deg)))
        & (opposed_gate >= float(opposed_min))
    )
    return ready.to(dtype=torch.float32)


def teacher_closure_fraction_from_mode(
    *,
    mode              : str,  # Param: string input for mode
    time_fraction     : torch.Tensor,  # Param: tensor input carrying time fraction values
    proximity_fraction: torch.Tensor,  # Param: tensor input carrying proximity fraction values
) -> torch.Tensor:
    """Combine closure fractions for distance or time mode"""
    if mode == "distance":
        fraction = proximity_fraction
    else:
        fraction = time_fraction * (0.35 + 0.65 * proximity_fraction)
    return smoothstep01(fraction)


def lift_hold_closure_fraction(time_fraction: torch.Tensor) -> torch.Tensor:
    """Return partially closed lift-only teacher fraction"""
    return torch.clamp(0.55 + 0.45 * time_fraction, 0.0, 1.0)


def update_touch_teacher_latch(
    *,
    episode_step          : torch.Tensor,  # Param: per-env step count inside the current episode
    fraction              : torch.Tensor,  # Param: tensor input carrying fraction values
    contact_now           : torch.Tensor,  # Param: tensor input carrying contact now values
    latched               : torch.Tensor,  # Param: tensor input carrying latched values
    hold_fraction         : torch.Tensor,  # Param: tensor input carrying hold fraction values
    first_contact_step    : torch.Tensor,  # Param: step count used for first contact step
    latch_step            : torch.Tensor,  # Param: step count used for latch step
    hand_contact_threshold: float,  # Param: cutoff used when evaluating hand contact
    min_fraction_for_latch: float        = 0.30,  # Param: floating-point input for min fraction for latch
    hold_fraction_cap     : float | None = None,  # Param: floating-point input for hold fraction cap
) -> TouchTeacherLatchState:
    """Update teacher closure latch from contact and scheduled fraction

    Steps:
    - Resolve inputs for `update_touch_teacher_latch` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    step = episode_step.to(dtype=torch.float32)
    latched_next = latched.to(device=fraction.device, dtype=torch.bool).clone()
    hold_next = hold_fraction.to(device=fraction.device, dtype=fraction.dtype).clone()
    first_next = first_contact_step.to(device=fraction.device, dtype=fraction.dtype).clone()
    latch_next = latch_step.to(device=fraction.device, dtype=fraction.dtype).clone()

    reset_mask = step <= 1.0
    if bool(reset_mask.any().item()):
        latched_next[reset_mask] = False
        hold_next[reset_mask] = 0.0
        first_next[reset_mask] = -1.0
        latch_next[reset_mask] = -1.0

    first_contact = (first_next < 0.0) & (contact_now.to(device=fraction.device) > float(hand_contact_threshold))
    first_next[first_contact] = step[first_contact]

    newly_latched = (
        (~latched_next)
        & (contact_now.to(device=fraction.device) > float(hand_contact_threshold))
        & (fraction >= float(min_fraction_for_latch))
    )
    latched_fraction = fraction[newly_latched]
    if hold_fraction_cap is not None:
        latched_fraction = torch.clamp(latched_fraction, max=float(hold_fraction_cap))
    hold_next[newly_latched] = latched_fraction
    latched_next[newly_latched] = True
    latch_next[newly_latched] = step[newly_latched]
    out_fraction = torch.where(latched_next, hold_next, fraction)
    return TouchTeacherLatchState(
        fraction=out_fraction,
        latched=latched_next,
        hold_fraction=hold_next,
        first_contact_step=first_next,
        latch_step=latch_next,
    )
