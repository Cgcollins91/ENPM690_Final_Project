"""

Contact pre-roll env-state adapters

This module contains helper functions for calculating masks related to the contact pre-roll phase of the topdown red-block curriculum,
such as determining which environments have reached the align-ready condition, which have reached the touch-ready condition based on the
configured touch mode, and which are ready to hand off from pre-roll based on the align and touch conditions.

This is not currently used in v35, but is intended for use in future curriculum phases that involve contact pre-roll,
and serves as a reference for how to implement similar env-state adapters for other curriculum conditions or phases.

File map:

contact_preroll_touch_mode_enabled:       Return whether contact pre-roll touch mode is enabled
contact_phase1_preroll_align_ready_mask:  Return env rows that reached the Phase 1.5 handoff shell
contact_phase1_preroll_touch_ready_mask:  Return env rows that reached the configured touch condition
contact_phase1_preroll_touch_phase_mask:  Return and update the contact pre-roll touch-phase latch
contact_phase1_preroll_release_mask:      Return env rows ready to hand off from contact pre-roll
"""

from __future__ import annotations

import torch

from .preroll import (
    contact_phase1_align_ready,
    contact_preroll_release_mask,
    contact_touch_ready,
    update_touch_phase_latch,
)


def contact_preroll_touch_mode_enabled(mode: str) -> bool:
    """Return whether contact pre-roll touch mode is enabled"""
    return str(mode) != "off"


def contact_phase1_preroll_align_ready_mask(
    env,                                           # Param: environment or backend object used for runtime calls
    *,
    enabled              : bool,  # Param: boolean input controlling enabled
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
    """Return env rows that reached the Phase 1.5 handoff shell"""
    if not bool(enabled):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return contact_phase1_align_ready(
        palm_dist=palm_dist,
        height_err=height_err,
        orient_rad=orient_rad,
        unlock_gate=unlock_gate,
        palm_tolerance=palm_tolerance,
        height_tolerance=height_tolerance,
        orient_deg=orient_deg,
        unlock_gate_threshold=unlock_gate_threshold,
        align_face_error=align_face_error,
        opposed_gate=opposed_gate,
        align_face_tolerance=align_face_tolerance,
        opposed_threshold=opposed_threshold,
    ).to(device=env.device, dtype=torch.bool)


def contact_phase1_preroll_touch_ready_mask(
    env,                           # Param: environment or backend object used for runtime calls
    *,
    mode          : str,  # Param: string input for mode
    any_contact   : torch.Tensor,  # Param: tensor input carrying any contact values
    both_contact  : torch.Tensor,  # Param: tensor input carrying both contact values
    strict_contact: torch.Tensor,  # Param: tensor input carrying strict contact values
    threshold     : float,  # Param: cutoff used by the comparison or gate
) -> torch.Tensor:
    """Return env rows that reached the configured touch condition"""
    ready = contact_touch_ready(
        mode=mode,
        any_contact=any_contact,
        both_contact=both_contact,
        strict_contact=strict_contact,
        threshold=threshold,
    )
    return ready.to(device=env.device, dtype=torch.bool)


def contact_phase1_preroll_touch_phase_mask(
    env,                                                      # Param: environment or backend object used for runtime calls
    align_ready: torch.Tensor | None = None,                  # Param: mask or boolean input marking align as ready
    *,
    attr_name: str = "_contact_preroll_touch_phase_latched",  # Param: string input for attr name
) -> torch.Tensor:
    """Return and update the contact pre-roll touch-phase latch

    Steps:
    - Resolve inputs for `contact_phase1_preroll_touch_phase_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    latched = getattr(env, attr_name, None)
    if not torch.is_tensor(latched) or latched.shape[0] != env.num_envs:
        latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        latched = latched.to(device=env.device, dtype=torch.bool)
    latched = update_touch_phase_latch(latched, align_ready)
    if align_ready is not None:
        setattr(env, attr_name, latched)
    return latched


def contact_phase1_preroll_release_mask(
    env,                                                      # Param: environment or backend object used for runtime calls
    *,
    align_ready: torch.Tensor,  # Param: mask or boolean input marking align as ready
    touch_ready: torch.Tensor,  # Param: mask or boolean input marking touch as ready
    touch_mode : str,  # Param: mode string selecting the touch behavior
    attr_name  : str = "_contact_preroll_touch_phase_latched",  # Param: string input for attr name
) -> torch.Tensor:
    """Return env rows ready to hand off from contact pre-roll"""
    if not contact_preroll_touch_mode_enabled(touch_mode):
        return align_ready.to(device=env.device, dtype=torch.bool)
    touch_phase = contact_phase1_preroll_touch_phase_mask(
        env,
        align_ready,
        attr_name=attr_name,
    )
    return contact_preroll_release_mask(
        align_ready=align_ready,
        touch_mode_enabled=True,
        touch_phase_latch=touch_phase,
        touch_ready=touch_ready,
    ).to(device=env.device, dtype=torch.bool)
