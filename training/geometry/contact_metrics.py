"""

Contact metric helpers with explicit tensor inputs

This module provides helper functions for computing contact metrics from explicit tensor inputs, used by the teacher provider
and rollout diagnostics

back_contact_strength:             computes back-finger contact strength from index and middle finger contact values
both_contact_strength:             computes opposed contact strength from thumb and back-finger contact values
primary_contact_strength:          computes the contact strength used for task success checks based on the current task
contact_strength_tensors:          computes the primary both-tip any-tip and hand contact tensors for diagnostics and success checks
strict_contact_gate:               computes a strict contact gate for rollout diagnostics based on the current task and contact sensor configuration
finger_contact_diagnostic_tensors: computes a set of diagnostic tensors for finger contact analysis including smooth contact raw values and a strict gate
contact_success_strength_for_task: computes the contact strength threshold used for success checks based on the current task
curl_closure_scale_for_task:       computes the closure scale used for signed finger curl based on the current task
target_link_distances:             computes the distances from thumb and index to their respective targets for diagnostics
target_delta_tensors:              computes the world-frame deltas from thumb and index to their respective targets for diagnostics or teacher features
zero_target_delta_tensors:         produces zero tensors for target deltas when no face targets are used
tip_diagnostic_error:              computes a worst-case fingertip error tensor for diagnostics based on whether topdown grip targets are used
grasp_signal_values:               extracts scalar grasp-related signals from tensors for logging
pregrasp_link_errors:              extracts scalar thumb and index errors from tensors for logging
align_angle_deg:                   converts a line angle from radians to degrees for diagnostics



DEFAULT_CONTACT_SUCCESS_STRENGTH:        Define default contact success strength constant
DEFAULT_LIGHT_CONTACT_SUCCESS_STRENGTH:  Define default light contact success strength constant
DEFAULT_CURL_SUCCESS_THRESHOLD:          Define default curl success threshold constant
DEFAULT_FINGER_CURL_CLOSURE_SCALE:       Define default finger curl closure scale constant
back_contact_strength:                   Return active back-finger contact strength
both_contact_strength:                   Return thumb plus back-finger opposed contact strength
primary_contact_strength:                Return contact strength used by current task objective
contact_strength_tensors:                Return primary both-tip any-tip and hand contact tensors
strict_contact_gate:                     Return strict contact gate for rollout diagnostics
finger_contact_diagnostic_tensors:       Return smooth contact raw force and strict gate diagnostics
contact_success_strength_for_task:       Return normalized contact threshold for current task
curl_closure_scale_for_task:             Return closure scale used for signed finger curl
target_link_distances:                   Return thumb and index distances to face targets
target_delta_tensors:                    Return target minus link world-frame deltas
zero_target_delta_tensors:               Return zero deltas for tasks without face targets
tip_diagnostic_error:                    Return worst fingertip target error for diagnostics
scalar_at:                               Read one finite scalar from a tensor with fallback
grasp_signal_values:                     Return force contact lift and displacement scalars for logs
pregrasp_link_errors:                    Return thumb and index target errors for one env
align_angle_deg:                         Return fingertip line elevation angle in degrees
"""


from __future__ import annotations

import math

import torch


DEFAULT_CONTACT_SUCCESS_STRENGTH       = 0.35 
DEFAULT_LIGHT_CONTACT_SUCCESS_STRENGTH = 0.08
DEFAULT_CURL_SUCCESS_THRESHOLD         = 2.20
DEFAULT_FINGER_CURL_CLOSURE_SCALE    = 6.00


def back_contact_strength(
    index : torch.Tensor,  # Param: tensor input carrying index values
    middle: torch.Tensor | None = None,  # Param: tensor input carrying middle values
    *,
    use_middle_back: bool = False,       # Param: boolean input selecting whether middle back is used
) -> torch.Tensor:
    """Return active back-finger contact strength"""
    if use_middle_back and middle is not None:
        return torch.maximum(index, middle.to(device=index.device, dtype=index.dtype))
    return index


def both_contact_strength(thumb: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
    """Return thumb plus back-finger opposed contact strength"""
    return torch.minimum(thumb, back.to(device=thumb.device, dtype=thumb.dtype))


def primary_contact_strength(
    *,
    both              : torch.Tensor,  # Param: tensor input carrying both values
    hand              : torch.Tensor,  # Param: tensor input carrying hand values
    grasp_contact_task: bool,  # Param: boolean input controlling grasp contact task
    topdown_lift_task : bool,  # Param: boolean input controlling topdown lift task
) -> torch.Tensor:
    """Return contact strength used by current task objective"""
    if grasp_contact_task or topdown_lift_task:
        return both
    return hand.to(device=both.device, dtype=both.dtype)


def contact_strength_tensors(
    *,
    thumb             : torch.Tensor,                    # Param: tensor input carrying thumb values
    index             : torch.Tensor,                    # Param: tensor input carrying index values
    any_tip           : torch.Tensor,                    # Param: tensor input carrying any tip values
    hand              : torch.Tensor,                    # Param: tensor input carrying hand values
    middle            : torch.Tensor | None = None,      # Param: tensor input carrying middle values
    grasp_contact_task: bool                = False,     # Param: boolean input controlling grasp contact task
    topdown_lift_task : bool                = False,     # Param: boolean input controlling topdown lift task
    use_middle_back   : bool                = False,     # Param: boolean input selecting whether middle back is used
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return primary both-tip any-tip and hand contact tensors"""
    back = back_contact_strength(index, middle, use_middle_back=use_middle_back)
    both = both_contact_strength(thumb, back)
    primary = primary_contact_strength(
        both=both,
        hand=hand,
        grasp_contact_task=grasp_contact_task,
        topdown_lift_task=topdown_lift_task,
    )
    return primary, both, any_tip.to(device=thumb.device, dtype=thumb.dtype), hand.to(device=thumb.device, dtype=thumb.dtype)


def strict_contact_gate(
    *,
    both                     : torch.Tensor,  # Param: tensor input carrying both values
    use_topdown_contact_chain: bool,  # Param: boolean input selecting whether topdown contact chain is used
    grasp_light_contact_task : bool,  # Param: boolean input controlling grasp light contact task
    opposed_contact          : torch.Tensor | None = None,  # Param: tensor input carrying opposed contact values
    light_opposed_contact    : torch.Tensor | None = None,  # Param: tensor input carrying light opposed contact values
    opposite_face            : torch.Tensor | None = None,  # Param: tensor input carrying opposite face values
) -> torch.Tensor:
    """Return strict contact gate for rollout diagnostics"""
    if use_topdown_contact_chain:
        if opposed_contact is None:
            raise ValueError("opposed_contact is required for topdown contact chain")
        return opposed_contact.to(device=both.device, dtype=both.dtype)
    if grasp_light_contact_task:
        if light_opposed_contact is None:
            raise ValueError("light_opposed_contact is required for light contact task")
        return light_opposed_contact.to(device=both.device, dtype=both.dtype)
    if opposite_face is None:
        raise ValueError("opposite_face is required for legacy strict contact")
    return both * opposite_face.to(device=both.device, dtype=both.dtype)


def finger_contact_diagnostic_tensors(
    *,
    thumb                    : torch.Tensor,  # Param: tensor input carrying thumb values
    index                    : torch.Tensor,  # Param: tensor input carrying index values
    thumb_raw                : torch.Tensor,  # Param: tensor input carrying thumb raw values
    index_raw                : torch.Tensor,  # Param: tensor input carrying index raw values
    use_topdown_contact_chain: bool,  # Param: boolean input selecting whether topdown contact chain is used
    grasp_light_contact_task : bool,  # Param: boolean input controlling grasp light contact task
    middle                   : torch.Tensor | None = None,  # Param: tensor input carrying middle values
    use_middle_back          : bool                = False,  # Param: boolean input selecting whether middle back is used
    opposed_contact          : torch.Tensor | None = None,  # Param: tensor input carrying opposed contact values
    light_opposed_contact    : torch.Tensor | None = None,  # Param: tensor input carrying light opposed contact values
    opposite_face            : torch.Tensor | None = None,  # Param: tensor input carrying opposite face values
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return smooth contact raw force and strict gate diagnostics"""
    back = back_contact_strength(index, middle, use_middle_back=use_middle_back)
    both = both_contact_strength(thumb, back)
    strict = strict_contact_gate(
        both=both,
        use_topdown_contact_chain=use_topdown_contact_chain,
        grasp_light_contact_task=grasp_light_contact_task,
        opposed_contact=opposed_contact,
        light_opposed_contact=light_opposed_contact,
        opposite_face=opposite_face,
    )
    return (
        thumb,
        index.to(device=thumb.device, dtype=thumb.dtype),
        thumb_raw.to(device=thumb.device, dtype=thumb.dtype),
        index_raw.to(device=thumb.device, dtype=thumb.dtype),
        strict,
    )


def contact_success_strength_for_task(
    *,
    grasp_light_contact_task      : bool,                                            # Param: boolean input controlling grasp light contact task
    contact_success_strength      : float = DEFAULT_CONTACT_SUCCESS_STRENGTH,        # Param: floating-point input for contact success strength
    light_contact_success_strength: float = DEFAULT_LIGHT_CONTACT_SUCCESS_STRENGTH,  # Param: floating-point input for light contact success strength
) -> float:
    """Return normalized contact threshold for current task"""
    if grasp_light_contact_task:
        return float(light_contact_success_strength)
    return float(contact_success_strength)


def curl_closure_scale_for_task(
    *,
    grasp_contact_task       : bool,  # Param: boolean input controlling grasp contact task
    curl_success_threshold   : float = DEFAULT_CURL_SUCCESS_THRESHOLD,  # Param: cutoff used when evaluating curl success
    finger_curl_closure_scale: float = DEFAULT_FINGER_CURL_CLOSURE_SCALE,  # Param: multiplier applied to finger curl closure
) -> float:
    """Return closure scale used for signed finger curl"""
    if grasp_contact_task:
        return float(curl_success_threshold)
    return float(finger_curl_closure_scale)


def target_link_distances(
    *,
    thumb_pos   : torch.Tensor,  # Param: tensor input carrying thumb pos values
    index_pos   : torch.Tensor,  # Param: tensor input carrying index pos values
    thumb_target: torch.Tensor,  # Param: target value for thumb
    index_target: torch.Tensor,  # Param: target value for index
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return thumb and index distances to face targets"""
    return (
        torch.linalg.norm(thumb_pos - thumb_target.to(device=thumb_pos.device, dtype=thumb_pos.dtype), dim=-1),
        torch.linalg.norm(index_pos - index_target.to(device=index_pos.device, dtype=index_pos.dtype), dim=-1),
    )


def target_delta_tensors(
    *,
    thumb_pos   : torch.Tensor,  # Param: tensor input carrying thumb pos values
    index_pos   : torch.Tensor,  # Param: tensor input carrying index pos values
    thumb_target: torch.Tensor,  # Param: target value for thumb
    index_target: torch.Tensor,  # Param: target value for index
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target minus link world-frame deltas"""
    return (
        thumb_target.to(device=thumb_pos.device, dtype=thumb_pos.dtype) - thumb_pos,
        index_target.to(device=index_pos.device, dtype=index_pos.dtype) - index_pos,
    )


def zero_target_delta_tensors(
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype   : torch.dtype = torch.float32,  # Param: torch dtype used when converting or allocating tensors
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero deltas for tasks without face targets"""
    zero = torch.zeros((int(num_envs), 3), device=device, dtype=dtype)
    return zero, zero.clone()


def tip_diagnostic_error(
    *,
    thumb_error              : torch.Tensor,  # Param: tensor input carrying thumb error values
    index_error              : torch.Tensor,  # Param: tensor input carrying index error values
    fallback_error           : torch.Tensor | None = None,  # Param: tensor input carrying fallback error values
    uses_topdown_grip_targets: bool                = True,  # Param: target values for uses topdown grip
) -> torch.Tensor:
    """Return worst fingertip target error for diagnostics"""
    if uses_topdown_grip_targets:
        return torch.maximum(thumb_error, index_error.to(device=thumb_error.device, dtype=thumb_error.dtype))
    if fallback_error is None:
        raise ValueError("fallback_error is required when topdown grip targets are disabled")
    return fallback_error.to(device=thumb_error.device, dtype=thumb_error.dtype)


def scalar_at(
    tensor: torch.Tensor | None,  # Param: tensor input carrying tensor values
    env_id: int,  # Param: integer input for env id
    *,
    default: float = 0.0,         # Param: fallback value used when the input omits or rejects a setting
) -> float:
    """Read one finite scalar from a tensor with fallback"""
    if tensor is None:
        return float(default)
    try:
        value = float(tensor[int(env_id)].item())
    except (IndexError, RuntimeError, TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def grasp_signal_values(
    *,
    total_force     : torch.Tensor | None,  # Param: tensor input carrying total force values
    contact_strength: torch.Tensor | None,  # Param: tensor input carrying contact strength values
    lift_height     : torch.Tensor | None,  # Param: tensor input carrying lift height values
    displacement    : torch.Tensor | None,  # Param: tensor input carrying displacement values
    env_id          : int = 0,  # Param: integer input for env id
) -> tuple[float, float, float, float]:
    """Return force contact lift and displacement scalars for logs"""
    return (
        scalar_at(total_force, env_id),
        scalar_at(contact_strength, env_id),
        scalar_at(lift_height, env_id),
        scalar_at(displacement, env_id),
    )


def pregrasp_link_errors(
    thumb_error: torch.Tensor | None,  # Param: tensor input carrying thumb error values
    index_error: torch.Tensor | None,  # Param: tensor input carrying index error values
    *,
    env_id: int = 0,                   # Param: integer input for env id
) -> tuple[float, float]:
    """Return thumb and index target errors for one env"""
    return scalar_at(thumb_error, env_id), scalar_at(index_error, env_id)


def align_angle_deg(line_angle_rad: torch.Tensor) -> torch.Tensor:
    """Return fingertip line elevation angle in degrees"""
    return torch.rad2deg(line_angle_rad)
