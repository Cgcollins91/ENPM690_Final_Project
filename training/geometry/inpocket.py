"""

In-pocket arm attenuation and hold helpers

These functions implement the live gating, latch state management, and action modification needed to support
in-pocket arm attenuation and hold features. The in-pocket live gate is computed from explicit tensors representing
various distance and alignment metrics, and can be configured with multiple thresholds and conditions. The latch state is
updated based on the live gate and episode length to create a debounced active mask that can be used to selectively apply arm
attenuation or hold. The arm hold function captures the current arm joint positions when the freeze conditions are met and holds
them in place by overriding the corresponding action dimensions, while allowing finger dimensions to remain trainable. These helpers
enable more stable training in scenarios where the hand is close to the pocket by reducing arm movement and encouraging the policy to
focus on learning finger control.

File map:

InPocketThresholds:              Thresholds for in-pocket live gate predicates
InPocketLatchState:              Updated in-pocket latch state tensors
ArmHoldState:                    Updated in-pocket arm hold state tensors
inpocket_arm_hold_enabled:       Return whether in-pocket arm hold is enabled
inpocket_live_gate:              Return live in-pocket gate from explicit tensors
update_inpocket_latch:           Update debounced in-pocket latch state
inpocket_active_mask:            Update owner latched in-pocket state and return active rows
contact_center_freeze_ready:     Return in-pocket rows centered enough to freeze arm targets
apply_inpocket_arm_attenuation:  Scale arm dimensions for active in-pocket rows
current_arm_reduced_action:      Convert current arm joint positions into reduced-action coordinates
apply_inpocket_arm_hold:         Apply in-pocket arm hold while leaving finger dimensions trainable
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InPocketThresholds:
    """Thresholds for in-pocket live gate predicates"""

    palm_max                 : float = 0.06  # floating-point palm max value used by in pocket thresholds
    palm_z_max               : float = 0.06  # floating-point palm z max value used by in pocket thresholds
    align_max                : float = 0.22  # floating-point align max value used by in pocket thresholds
    align_angle_max_deg      : float = 10.0  # floating-point align angle max deg value used by in pocket thresholds
    tip_max                  : float = 0.14  # floating-point tip max value used by in pocket thresholds
    tip_xy_max               : float = 0.08  # floating-point tip xy max value used by in pocket thresholds
    tip_z_max                : float = 0.06  # floating-point tip z max value used by in pocket thresholds
    require_wrist_yaw_release: bool  = True  # boolean value indicating the require wrist yaw release state for in pocket thresholds
    require_stage2           : bool  = True  # boolean value indicating the require stage2 state for in pocket thresholds


@dataclass(frozen=True)
class InPocketLatchState:
    """Updated in-pocket latch state tensors"""

    latched   : torch.Tensor  # per-env latch mask or aggregate latch state
    hold_count: torch.Tensor  # count of hold values
    reset_mask: torch.Tensor  # boolean mask selecting reset rows for in pocket latch state


@dataclass(frozen=True)
class ArmHoldState:
    """Updated in-pocket arm hold state tensors"""

    action      : torch.Tensor  # environment action tensor selected for the step
    held        : torch.Tensor  # tensor containing held values for batched env rows
    valid       : torch.Tensor  # tensor containing valid values for batched env rows
    frozen      : torch.Tensor  # tensor containing frozen values for batched env rows
    lift_release: torch.Tensor  # tensor containing lift release values for batched env rows


def inpocket_arm_hold_enabled(value: object = False) -> bool:
    """Return whether in-pocket arm hold is enabled"""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def inpocket_live_gate(
    *,
    palm_dist      : torch.Tensor,  # Param: tensor input carrying palm dist values
    palm_height    : torch.Tensor,  # Param: tensor input carrying palm height values
    align_error    : torch.Tensor,  # Param: tensor input carrying align error values
    thresholds     : InPocketThresholds,  # Param: input value used as thresholds
    align_angle_deg: torch.Tensor | None = None,  # Param: tensor input carrying align angle deg values
    thumb_err      : torch.Tensor | None = None,  # Param: tensor input carrying thumb err values
    index_err      : torch.Tensor | None = None,  # Param: tensor input carrying index err values
    thumb_delta    : torch.Tensor | None = None,  # Param: tensor input carrying thumb delta values
    index_delta    : torch.Tensor | None = None,  # Param: tensor input carrying index delta values
    yaw_locked     : torch.Tensor | None = None,  # Param: tensor input carrying yaw locked values
    stage          : torch.Tensor | None = None,  # Param: tensor input carrying stage values
) -> torch.Tensor:
    """Return live in-pocket gate from explicit tensors

    Steps:
    - Resolve inputs for `inpocket_live_gate` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    device = palm_dist.device
    gate = (
        (palm_dist <= float(thresholds.palm_max))
        & (palm_height.to(device=device) <= float(thresholds.palm_z_max))
        & (align_error.to(device=device) <= float(thresholds.align_max))
    )
    if thresholds.align_angle_max_deg > 0.0 and align_angle_deg is not None:
        gate = gate & (align_angle_deg.to(device=device) <= float(thresholds.align_angle_max_deg))
    if thresholds.tip_max > 0.0 and thumb_err is not None and index_err is not None:
        gate = gate & (thumb_err.to(device=device) <= thresholds.tip_max)
        gate = gate & (index_err.to(device=device) <= thresholds.tip_max)
    if thresholds.tip_xy_max > 0.0 and thumb_delta is not None and index_delta is not None:
        thumb_xy = torch.linalg.norm(thumb_delta.to(device=device)[:, :2], dim=-1)
        index_xy = torch.linalg.norm(index_delta.to(device=device)[:, :2], dim=-1)
        gate = gate & (thumb_xy <= thresholds.tip_xy_max) & (index_xy <= thresholds.tip_xy_max)
    if thresholds.tip_z_max > 0.0 and thumb_delta is not None and index_delta is not None:
        gate = gate & (torch.abs(thumb_delta.to(device=device)[:, 2]) <= thresholds.tip_z_max)
        gate = gate & (torch.abs(index_delta.to(device=device)[:, 2]) <= thresholds.tip_z_max)
    if thresholds.require_wrist_yaw_release and yaw_locked is not None:
        gate = gate & (~yaw_locked.to(device=device, dtype=torch.bool))
    if thresholds.require_stage2 and stage is not None:
        gate = gate & (stage.to(device=device) >= 2)
    return gate


def update_inpocket_latch(
    *,
    live_gate              : torch.Tensor,  # Param: tensor input carrying live gate values
    episode_length         : torch.Tensor | None,  # Param: tensor input carrying episode length values
    previous_episode_length: torch.Tensor | None,  # Param: tensor input carrying previous episode length values
    previous_latched       : torch.Tensor | None,  # Param: tensor input carrying previous latched values
    previous_hold_count    : torch.Tensor | None,  # Param: count of previous hold
    latch_enabled          : bool = True,  # Param: boolean input enabling latch
    hold_steps             : int  = 0,  # Param: step count used for hold steps
) -> InPocketLatchState:
    """Update debounced in-pocket latch state

    Steps:
    - Resolve inputs for `update_inpocket_latch` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not bool(latch_enabled) or episode_length is None:
        zeros = torch.zeros_like(live_gate, dtype=torch.bool)
        return InPocketLatchState(live_gate.to(dtype=torch.bool), torch.zeros_like(live_gate, dtype=torch.long), zeros)

    ep_len = episode_length.to(device=live_gate.device)
    prev_ep = ep_len if previous_episode_length is None else previous_episode_length.to(device=live_gate.device)
    latched = (
        torch.zeros_like(live_gate, dtype=torch.bool)
        if previous_latched is None
        else previous_latched.to(device=live_gate.device, dtype=torch.bool).clone()
    )
    hold_count = (
        torch.zeros_like(ep_len, dtype=torch.long)
        if previous_hold_count is None
        else previous_hold_count.to(device=live_gate.device, dtype=torch.long).clone()
    )
    reset_mask = (ep_len <= 1) | (ep_len < prev_ep)
    if bool(reset_mask.any().item()):
        latched[reset_mask] = False
        hold_count[reset_mask] = 0
    if int(hold_steps) > 0:
        hold_count = torch.where(
            live_gate,
            hold_count + 1,
            torch.zeros_like(hold_count),
        )
        latched = latched | (hold_count >= int(hold_steps))
    else:
        latched = latched | live_gate
    return InPocketLatchState(latched=latched, hold_count=hold_count, reset_mask=reset_mask)


def inpocket_active_mask(
    owner,                                # Param: input value used as owner
    *,
    live_gate     : torch.Tensor,  # Param: tensor input carrying live gate values
    episode_length: torch.Tensor | None,  # Param: tensor input carrying episode length values
    latch_enabled : bool = True,  # Param: boolean input enabling latch
    hold_steps    : int  = 0,  # Param: step count used for hold steps
) -> torch.Tensor:
    """Update owner latched in-pocket state and return active rows

    Steps:
    - Resolve inputs for `inpocket_active_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not bool(latch_enabled) or episode_length is None:
        return live_gate.to(dtype=torch.bool)
    ep_len = episode_length.to(device=live_gate.device)
    previous_episode_length = getattr(owner, "_inpocket_arm_prev_eplen", None)
    previous_latched = getattr(owner, "_inpocket_arm_latched", None)
    previous_hold_count = getattr(owner, "_inpocket_arm_hold_count", None)
    state = update_inpocket_latch(
        live_gate=live_gate.to(dtype=torch.bool),
        episode_length=ep_len,
        previous_episode_length=previous_episode_length,
        previous_latched=previous_latched,
        previous_hold_count=previous_hold_count,
        latch_enabled=latch_enabled,
        hold_steps=hold_steps,
    )
    if bool(state.reset_mask.any().item()):
        for attr_name in ("_inpocket_arm_hold_valid", "_inpocket_arm_hold_frozen"):
            value = getattr(owner, attr_name, None)
            if torch.is_tensor(value) and value.shape == live_gate.shape:
                value = value.to(device=live_gate.device, dtype=torch.bool).clone()
                value[state.reset_mask] = False
                setattr(owner, attr_name, value)
        held = getattr(owner, "_inpocket_arm_hold_action", None)
        if torch.is_tensor(held) and held.ndim == 2 and held.shape[0] == live_gate.shape[0]:
            held = held.to(device=live_gate.device).clone()
            held[state.reset_mask] = 0.0
            owner._inpocket_arm_hold_action = held
    owner._inpocket_arm_latched = state.latched.detach().clone()
    owner._inpocket_arm_prev_eplen = ep_len.detach().clone()
    owner._inpocket_arm_hold_count = state.hold_count.detach().clone()
    return owner._inpocket_arm_latched


def contact_center_freeze_ready(
    *,
    active                 : torch.Tensor,  # Param: tensor input carrying active values
    require_finger_center  : bool                = False,  # Param: boolean input controlling require finger center
    finger_center_gate     : torch.Tensor | None = None,  # Param: tensor input carrying finger center gate values
    require_contact_center : bool                = False,  # Param: boolean input controlling require contact center
    contact_teacher_enabled: bool                = False,  # Param: boolean input enabling contact teacher
    thumb_contact          : torch.Tensor | None = None,  # Param: tensor input carrying thumb contact values
    back_contact           : torch.Tensor | None = None,  # Param: tensor input carrying back contact values
    thumb_delta            : torch.Tensor | None = None,  # Param: tensor input carrying thumb delta values
    index_delta            : torch.Tensor | None = None,  # Param: tensor input carrying index delta values
    align_angle_deg        : torch.Tensor | None = None,  # Param: tensor input carrying align angle deg values
    contact_threshold      : float               = 0.30,  # Param: cutoff used when evaluating contact
    tip_xy_max             : float               = 0.015,  # Param: floating-point input for tip xy max
    tip_z_max              : float               = 0.060,  # Param: floating-point input for tip z max
    align_angle_max_deg    : float               = 8.0,  # Param: floating-point input for align angle max deg
) -> torch.Tensor:
    """Return in-pocket rows centered enough to freeze arm targets

    Steps:
    - Resolve inputs for `contact_center_freeze_ready` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ready = active.to(dtype=torch.bool)
    if bool(require_finger_center):
        if finger_center_gate is None:
            return torch.zeros_like(ready)
        return ready & finger_center_gate.to(device=ready.device, dtype=torch.bool)
    if not bool(require_contact_center) or not bool(contact_teacher_enabled):
        return ready
    if thumb_contact is None or back_contact is None or thumb_delta is None or index_delta is None:
        return ready
    thumb_delta_t = thumb_delta.to(device=ready.device, dtype=torch.float32)
    index_delta_t = index_delta.to(device=ready.device, dtype=torch.float32)
    thumb_xy = torch.linalg.norm(thumb_delta_t[:, :2], dim=-1)
    index_xy = torch.linalg.norm(index_delta_t[:, :2], dim=-1)
    centered = (
        ready
        & (thumb_contact.to(device=ready.device, dtype=torch.float32) >= float(contact_threshold))
        & (back_contact.to(device=ready.device, dtype=torch.float32) >= float(contact_threshold))
        & (thumb_xy <= max(float(tip_xy_max), 0.0))
        & (index_xy <= max(float(tip_xy_max), 0.0))
    )
    if float(tip_z_max) > 0.0:
        centered = centered & (torch.abs(thumb_delta_t[:, 2]) <= float(tip_z_max))
        centered = centered & (torch.abs(index_delta_t[:, 2]) <= float(tip_z_max))
    if float(align_angle_max_deg) > 0.0 and align_angle_deg is not None:
        centered = centered & (
            align_angle_deg.to(device=ready.device, dtype=torch.float32) <= float(align_angle_max_deg)
        )
    return centered


def apply_inpocket_arm_attenuation(
    action: torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    active: torch.Tensor,  # Param: tensor input carrying active values
    *,
    num_arm  : int,  # Param: number of arm action dimensions in the active layout
    arm_scale: float,  # Param: multiplier applied to arm
) -> torch.Tensor:
    """Scale arm dimensions for active in-pocket rows

    Steps:
    - Resolve inputs for `apply_inpocket_arm_attenuation` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if float(arm_scale) >= 1.0 or action.shape[-1] < int(num_arm):
        return action
    if not bool(active.any().item()):
        return action
    out = action.clone()
    out[active.to(device=action.device, dtype=torch.bool), : int(num_arm)] = (
        action[active.to(device=action.device, dtype=torch.bool), : int(num_arm)] * float(arm_scale)
    )
    return out.clamp(-1.0, 1.0)


def current_arm_reduced_action(
    joint_pos        : torch.Tensor,  # Param: current joint-position tensor used as the IK starting point
    default_joint_pos: torch.Tensor,  # Param: default joint positions used by the IK posture term
    arm_joint_ids    : torch.Tensor,  # Param: tensor input carrying arm joint ids values
    arm_scales       : torch.Tensor,  # Param: tensor input carrying arm scales values
) -> torch.Tensor:
    """Convert current arm joint positions into reduced-action coordinates"""
    ids = arm_joint_ids.to(device=joint_pos.device, dtype=torch.long)
    scales = arm_scales.to(device=joint_pos.device, dtype=joint_pos.dtype).unsqueeze(0).clamp_min(1.0e-6)
    return ((joint_pos[:, ids] - default_joint_pos[:, ids]) / scales).clamp(-1.0, 1.0)


def apply_inpocket_arm_hold(
    action: torch.Tensor,               # Param: action tensor applied to the environment or stored in replay
    *,
    active               : torch.Tensor,  # Param: tensor input carrying active values
    freeze_ready         : torch.Tensor,  # Param: mask or boolean input marking freeze as ready
    held                 : torch.Tensor,  # Param: tensor input carrying held values
    valid                : torch.Tensor,  # Param: tensor input carrying valid values
    frozen               : torch.Tensor,  # Param: tensor input carrying frozen values
    current_arm          : torch.Tensor,  # Param: tensor input carrying current arm values
    lift_latched         : torch.Tensor | None,  # Param: mask selecting env rows with lift latch active
    release_on_lift_latch: bool,  # Param: boolean input controlling release on lift latch
    num_arm              : int,  # Param: number of arm action dimensions in the active layout
) -> ArmHoldState:
    """Apply in-pocket arm hold while leaving finger dimensions trainable

    Steps:
    - Resolve inputs for `apply_inpocket_arm_hold` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    active_b = active.to(device=action.device, dtype=torch.bool)
    freeze_b = freeze_ready.to(device=action.device, dtype=torch.bool)
    held_next = held.to(device=action.device, dtype=action.dtype).clone()
    valid_next = valid.to(device=action.device, dtype=torch.bool) | active_b
    frozen_next = frozen.to(device=action.device, dtype=torch.bool).clone()

    capture = active_b & freeze_b & (~frozen_next)
    if bool(capture.any().item()):
        held_next[capture] = current_arm.to(device=action.device, dtype=action.dtype)[capture]
        frozen_next[capture] = True

    lift_release = torch.zeros_like(active_b)
    if bool(release_on_lift_latch) and lift_latched is not None:
        lift_release = active_b & frozen_next & lift_latched.to(device=action.device, dtype=torch.bool)
    hold_mask = active_b & frozen_next & (~lift_release)
    out = action.clone()
    out[hold_mask, : int(num_arm)] = held_next[hold_mask]
    return ArmHoldState(
        action=out.clamp(-1.0, 1.0),
        held=held_next.detach().clone(),
        valid=valid_next.detach().clone(),
        frozen=frozen_next.detach().clone(),
        lift_release=lift_release.detach().clone(),
    )
