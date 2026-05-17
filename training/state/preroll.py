"""

Pre-roll readiness and release predicate helpers

File map:

contact_phase1_terminal_start_enabled:  Return whether Phase 1 terminal contact start is active
topdown_curriculum_preroll_enabled:     Return whether topdown curriculum pre-roll is active
sample_preroll_mask:                    Sample pre-roll rows for a vectorized env batch
topdown_preroll_ready:                  Return topdown pre-roll release readiness
contact_phase1_align_ready:             Return Phase 1.5 alignment readiness from explicit tensors
contact_touch_ready:                    Return contact pre-roll touch readiness for one mode
update_touch_phase_latch:               Update touch-phase latch from alignment readiness
contact_preroll_release_mask:           Return contact pre-roll release rows
"""

from __future__ import annotations

import math

import torch


def contact_phase1_terminal_start_enabled(*, contact_family_task: bool, contact_start_mode: str) -> bool:
    """Return whether Phase 1 terminal contact start is active"""
    return bool(contact_family_task) and str(contact_start_mode) == "phase1_terminal"


def topdown_curriculum_preroll_enabled(*, topdown_curriculum_task: bool, topdown_preroll_fraction: float) -> bool:
    """Return whether topdown curriculum pre-roll is active"""
    return bool(topdown_curriculum_task) and float(topdown_preroll_fraction) > 0.0


def sample_preroll_mask(
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    fraction: float,  # Param: floating-point input for fraction
    env_ids : torch.Tensor | None = None,  # Param: tensor input carrying env ids values
) -> torch.Tensor:
    """Sample pre-roll rows for a vectorized env batch

    Steps:
    - Resolve inputs for `sample_preroll_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    sample_fraction = max(0.0, min(1.0, float(fraction)))
    shape = (int(num_envs),) if env_ids is None else (int(env_ids.shape[0]),)
    if sample_fraction <= 0.0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    if sample_fraction >= 1.0:
        return torch.ones(shape, dtype=torch.bool, device=device)
    return torch.rand(shape, device=device) < sample_fraction


def topdown_preroll_ready(
    *,
    shell_now         : torch.Tensor,  # Param: tensor input carrying shell now values
    contact_pose_ready: torch.Tensor,  # Param: mask or boolean input marking contact pose as ready
    unlock            : torch.Tensor,  # Param: tensor input carrying unlock values
    unlock_progress   : float,  # Param: floating-point input for unlock progress
) -> torch.Tensor:
    """Return topdown pre-roll release readiness"""
    return (
        shell_now.to(dtype=torch.bool)
        & contact_pose_ready.to(device=shell_now.device, dtype=torch.bool)
        & (unlock.to(device=shell_now.device, dtype=torch.float32) >= float(unlock_progress))
    )


def contact_phase1_align_ready(
    *,
    palm_dist            : torch.Tensor,  # Param: tensor input carrying palm dist values
    height_err           : torch.Tensor,  # Param: tensor input carrying height err values
    orient_rad           : torch.Tensor,  # Param: tensor input carrying orient rad values
    unlock_gate          : torch.Tensor,  # Param: tensor input carrying unlock gate values
    palm_tolerance       : float,  # Param: tolerance allowed for palm
    height_tolerance     : float,  # Param: tolerance allowed for height
    orient_deg           : float,  # Param: floating-point input for orient deg
    unlock_gate_threshold: float,  # Param: cutoff used when evaluating unlock gate
    align_face_error     : torch.Tensor | None = None,  # Param: tensor input carrying align face error values
    opposed_gate         : torch.Tensor | None = None,  # Param: tensor input carrying opposed gate values
    align_face_tolerance : float               = 0.0,  # Param: tolerance allowed for align face
    opposed_threshold    : float               = 0.0,  # Param: cutoff used when evaluating opposed
) -> torch.Tensor:
    """Return Phase 1.5 alignment readiness from explicit tensors

    Steps:
    - Resolve inputs for `contact_phase1_align_ready` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    palm = palm_dist.to(dtype=torch.float32)
    height = height_err.to(device=palm.device, dtype=torch.float32)
    orient = orient_rad.to(device=palm.device, dtype=torch.float32)
    unlock = unlock_gate.to(device=palm.device, dtype=torch.float32)
    geom_ready = (
        (palm <= float(palm_tolerance))
        & (height <= float(height_tolerance))
        & (orient <= math.radians(float(orient_deg)))
    )
    loose_geom_ready = (
        (palm <= max(float(palm_tolerance), 0.10))
        & (height <= max(float(height_tolerance), 0.06))
        & (orient <= math.radians(max(float(orient_deg), 30.0)))
    )
    release_ready = geom_ready | ((unlock >= float(unlock_gate_threshold)) & loose_geom_ready)
    if float(align_face_tolerance) > 0.0:
        if align_face_error is None or opposed_gate is None:
            release_ready = torch.zeros_like(release_ready, dtype=torch.bool)
        else:
            align_ready = (
                align_face_error.to(device=palm.device, dtype=torch.float32) <= float(align_face_tolerance)
            ) & (opposed_gate.to(device=palm.device, dtype=torch.float32) >= float(opposed_threshold))
            release_ready = release_ready & align_ready
    return release_ready


def contact_touch_ready(
    *,
    mode          : str,  # Param: string input for mode
    any_contact   : torch.Tensor,  # Param: tensor input carrying any contact values
    both_contact  : torch.Tensor,  # Param: tensor input carrying both contact values
    strict_contact: torch.Tensor,  # Param: tensor input carrying strict contact values
    threshold     : float,  # Param: cutoff used by the comparison or gate
) -> torch.Tensor:
    """Return contact pre-roll touch readiness for one mode

    Steps:
    - Resolve inputs for `contact_touch_ready` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if mode == "off":
        return torch.ones_like(any_contact, dtype=torch.bool)
    if mode == "any":
        return any_contact >= float(threshold)
    if mode == "both":
        return both_contact >= float(threshold)
    if mode == "strict":
        return strict_contact >= float(threshold)
    raise RuntimeError(f"unsupported contact_preroll_touch_mode={mode!r}")


def update_touch_phase_latch(
    current_latch: torch.Tensor,  # Param: tensor input carrying current latch values
    align_ready  : torch.Tensor | None = None,  # Param: mask or boolean input marking align as ready
) -> torch.Tensor:
    """Update touch-phase latch from alignment readiness"""
    latch = current_latch.to(dtype=torch.bool)
    if align_ready is None:
        return latch
    return latch | align_ready.to(device=latch.device, dtype=torch.bool)


def contact_preroll_release_mask(
    *,
    align_ready       : torch.Tensor,  # Param: mask or boolean input marking align as ready
    touch_mode_enabled: bool,  # Param: boolean input enabling touch mode
    touch_phase_latch : torch.Tensor,  # Param: tensor input carrying touch phase latch values
    touch_ready       : torch.Tensor,  # Param: mask or boolean input marking touch as ready
) -> torch.Tensor:
    """Return contact pre-roll release rows"""
    if not bool(touch_mode_enabled):
        return align_ready.to(dtype=torch.bool)
    return touch_phase_latch.to(dtype=torch.bool) & touch_ready.to(
        device=touch_phase_latch.device,
        dtype=torch.bool,
    )
