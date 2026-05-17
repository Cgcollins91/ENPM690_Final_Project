"""

Progress-line stage bit formatters

File map:

TopdownStageBitInputs:          Scalar inputs for topdown progress stage bits
GraspAlignStageBitInputs:       Scalar inputs for grasp-align stage bits
ContactStageBitInputs:          Scalar inputs for contact-family stage bits
tensor_env_float:               Read one float from an env tensor attr
tensor_env_int:                 Read one int from an env tensor attr
tensor_env_bit:                 Read one bool bit from an env tensor attr
_has_env_tensors:               Handle has env tensors logic
_append_hold_bits:              Handle append hold bits logic
_append_finger_center_bits:     Handle append finger center bits logic
_append_success_bits:           Handle append success bits logic
_append_contact_teacher_bits:   Handle append contact teacher bits logic
format_topdown_stage_bits:      Format topdown curriculum progress-line stage bits
format_topdown_done_bits:       Format topdown done bits
format_grasp_align_stage_bits:  Format grasp-align progress-line stage bits
format_contact_stage_bits:      Format contact-family progress-line stage bits
format_phase1_progress_bits:    Format phase-1 progress bits
format_grasp_align_done_bits:   Format grasp-align done bits
format_contact_done_bits:       Format contact-family done bits
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopdownStageBitInputs:
    """Scalar inputs for topdown progress stage bits"""

    topdown_stage            : int  # Field: current topdown curriculum stage per environment
    best_topdown_stage       : int  # Field: highest topdown curriculum stage reached so far
    reach_hold               : int  # Field: integer reach hold value tracked by topdown stage bit inputs
    align_hold               : int  # Field: integer align hold value tracked by topdown stage bit inputs
    contact_pose_ready       : int  # Field: boolean/tensor readiness state for contact pose
    contact_pose_hold        : int  # Field: integer contact pose hold value tracked by topdown stage bit inputs
    contact_pose_shell       : int  # Field: integer contact pose shell value tracked by topdown stage bit inputs
    contact_palm_dist        : float  # Field: floating-point contact palm dist value used by topdown stage bit inputs
    contact_palm_height      : float  # Field: floating-point contact palm height value used by topdown stage bit inputs
    stage2_age               : int  # Field: integer stage2 age value tracked by topdown stage bit inputs
    unlock_progress          : float  # Field: floating-point unlock progress value used by topdown stage bit inputs
    effective_unlock_progress: float  # Field: floating-point effective unlock progress value used by topdown stage bit inputs
    finger_arm_hold_gate     : float  # Field: floating-point finger arm hold gate value used by topdown stage bit inputs
    prehold_servo            : float  # Field: floating-point prehold servo value used by topdown stage bit inputs
    align_line_z             : float  # Field: floating-point align line z value used by topdown stage bit inputs
    align_servo_q            : float  # Field: floating-point align servo q value used by topdown stage bit inputs
    align_servo_active       : int  # Field: boolean state indicating whether align servo is active
    pocket_sweep_q           : float  # Field: floating-point pocket sweep q value used by topdown stage bit inputs
    pocket_score_before      : float  # Field: floating-point pocket score before value used by topdown stage bit inputs
    pocket_score_after       : float  # Field: floating-point pocket score after value used by topdown stage bit inputs
    stage_ge1_rate           : float  # Field: floating-point stage ge1 rate value used by topdown stage bit inputs
    stage_ge2_rate           : float  # Field: floating-point stage ge2 rate value used by topdown stage bit inputs
    contact                  : float  # Field: floating-point contact value used by topdown stage bit inputs
    strict_contact           : float  # Field: floating-point strict contact value used by topdown stage bit inputs
    thumb_contact            : float  # Field: contact strength observed at the thumb side
    index_contact            : float  # Field: contact strength observed at the index-finger side
    align_face               : float  # Field: block face selected for alignment scoring
    align_angle              : float  # Field: alignment angle value used by topdown/contact metrics
    opposed_face             : float  # Field: block face opposite the active contact/alignment face
    lift                     : float  # Field: floating-point lift value used by topdown stage bit inputs
    block_disp               : float  # Field: block displacement value used by metrics or summaries


@dataclass(frozen=True)
class GraspAlignStageBitInputs:
    """Scalar inputs for grasp-align stage bits"""

    contact     : float  # Field: floating-point contact value used by grasp align stage bit inputs
    curl        : float  # Field: floating-point curl value used by grasp align stage bit inputs
    pinch_curl  : float  # Field: floating-point pinch curl value used by grasp align stage bit inputs
    opposed_face: float  # Field: block face opposite the active contact/alignment face
    align_face  : float  # Field: block face selected for alignment scoring
    align_angle : float  # Field: alignment angle value used by topdown/contact metrics
    hand_force  : float  # Field: aggregate hand/contact force used for diagnostics or gates
    block_disp  : float  # Field: block displacement value used by metrics or summaries


@dataclass(frozen=True)
class ContactStageBitInputs:
    """Scalar inputs for contact-family stage bits"""

    contact          : float  # Field: floating-point contact value used by contact stage bit inputs
    both_contact     : float  # Field: floating-point both contact value used by contact stage bit inputs
    strict_contact   : float  # Field: floating-point strict contact value used by contact stage bit inputs
    fingertip_contact: float  # Field: floating-point fingertip contact value used by contact stage bit inputs
    thumb_contact    : float  # Field: contact strength observed at the thumb side
    index_contact    : float  # Field: contact strength observed at the index-finger side
    thumb_contact_raw: float  # Field: floating-point thumb contact raw value used by contact stage bit inputs
    index_contact_raw: float  # Field: floating-point index contact raw value used by contact stage bit inputs
    hand_contact     : float  # Field: floating-point hand contact value used by contact stage bit inputs
    lift             : float  # Field: floating-point lift value used by contact stage bit inputs
    curl             : float  # Field: floating-point curl value used by contact stage bit inputs
    pinch_curl       : float  # Field: floating-point pinch curl value used by contact stage bit inputs
    opposed_face     : float  # Field: block face opposite the active contact/alignment face
    align_face       : float  # Field: block face selected for alignment scoring
    align_angle      : float  # Field: alignment angle value used by topdown/contact metrics
    hand_force       : float  # Field: aggregate hand/contact force used for diagnostics or gates
    block_disp       : float  # Field: block displacement value used by metrics or summaries


def tensor_env_float(owner: object, attr_name: str, env_id: int, default: float = 0.0) -> float:
    """Read one float from an env tensor attr"""
    attr = getattr(owner, attr_name, None)
    if torch.is_tensor(attr) and attr.numel() > int(env_id):
        return float(attr.reshape(-1)[int(env_id)].item())
    return float(default)


def tensor_env_int(owner: object, attr_name: str, env_id: int, default: int = 0) -> int:
    """Read one int from an env tensor attr"""
    attr = getattr(owner, attr_name, None)
    if torch.is_tensor(attr) and attr.numel() > int(env_id):
        return int(attr.reshape(-1)[int(env_id)].item())
    return int(default)


def tensor_env_bit(owner: object, attr_name: str, env_id: int, default: int = 0) -> int:
    """Read one bool bit from an env tensor attr"""
    attr = getattr(owner, attr_name, None)
    if torch.is_tensor(attr) and attr.numel() > int(env_id):
        return int(bool(attr.reshape(-1)[int(env_id)].item()))
    return int(default)


def _has_env_tensors(owner: object, attr_names: tuple[str, ...], env_id: int) -> bool:
    for attr_name in attr_names:
        attr = getattr(owner, attr_name, None)
        if not torch.is_tensor(attr) or attr.numel() <= int(env_id):
            return False
    return True


def _append_hold_bits(bits: str, env: object, log_env_id: int) -> str:
    return (
        bits
        + f"arm_hold_live={tensor_env_bit(env, '_inpocket_arm_hold_live_gate', log_env_id)} "
        + f"arm_hold={tensor_env_bit(env, '_inpocket_arm_hold_active', log_env_id)} "
        + f"arm_hold_valid={tensor_env_bit(env, '_inpocket_arm_hold_valid', log_env_id)} "
        + f"arm_freeze_ready={tensor_env_bit(env, '_inpocket_arm_hold_freeze_ready', log_env_id)} "
        + f"arm_frozen={tensor_env_bit(env, '_inpocket_arm_hold_frozen', log_env_id)} "
    )


def _append_finger_center_bits(bits: str, env: object, log_env_id: int) -> str:
    required = (
        "_topdown_finger_center_live",
        "_topdown_finger_center_ready",
        "_topdown_finger_center_xy_err",
        "_topdown_finger_center_max_xy_err",
        "_topdown_finger_center_z_err",
        "_topdown_finger_center_align_angle_deg",
        "_topdown_finger_center_hold",
        "_topdown_raw_finger_unlock_progress",
    )
    if not _has_env_tensors(env, required, log_env_id):
        return bits
    return (
        bits
        + f"raw_unlock={tensor_env_float(env, '_topdown_raw_finger_unlock_progress', log_env_id):.2f} "
        + f"finger_cent_live={tensor_env_bit(env, '_topdown_finger_center_live', log_env_id)} "
        + f"finger_cent={tensor_env_bit(env, '_topdown_finger_center_ready', log_env_id)} "
        + f"finger_cent_xy={tensor_env_float(env, '_topdown_finger_center_xy_err', log_env_id):.3f} "
        + f"finger_cent_max_xy={tensor_env_float(env, '_topdown_finger_center_max_xy_err', log_env_id):.3f} "
        + f"finger_cent_z={tensor_env_float(env, '_topdown_finger_center_z_err', log_env_id):.3f} "
        + f"finger_cent_ang={tensor_env_float(env, '_topdown_finger_center_align_angle_deg', log_env_id):.1f} "
        + f"finger_cent_hold={tensor_env_int(env, '_topdown_finger_center_hold', log_env_id)} "
    )


def _append_success_bits(bits: str, env: object, log_env_id: int) -> str:
    required = (
        "_topdown_light_contact_success_base",
        "_topdown_success_center_ready",
        "_topdown_success_center_xy_err",
        "_topdown_success_center_align_angle_deg",
        "_topdown_success_hold",
    )
    if not _has_env_tensors(env, required, log_env_id):
        return bits
    return (
        bits
        + f"succ_base={tensor_env_bit(env, '_topdown_light_contact_success_base', log_env_id)} "
        + f"succ_cent={tensor_env_bit(env, '_topdown_success_center_ready', log_env_id)} "
        + f"succ_xy={tensor_env_float(env, '_topdown_success_center_xy_err', log_env_id):.3f} "
        + f"succ_ang={tensor_env_float(env, '_topdown_success_center_align_angle_deg', log_env_id):.1f} "
        + f"succ_hold={tensor_env_int(env, '_topdown_success_hold', log_env_id)} "
    )


def _append_contact_teacher_bits(bits: str, env: object, log_env_id: int) -> str:
    required = (
        "_topdown_contact_teacher_thumb_fraction",
        "_topdown_contact_teacher_index_fraction",
        "_topdown_contact_teacher_thumb_latched",
        "_topdown_contact_teacher_index_latched",
        "_topdown_contact_teacher_thumb_hold_fraction",
        "_topdown_contact_teacher_index_hold_fraction",
        "_topdown_contact_teacher_descent_z",
        "_topdown_contact_teacher_descent_z_need",
        "_topdown_contact_teacher_descent_closure_gate",
        "_topdown_contact_teacher_inward_m",
        "_topdown_contact_teacher_tip_servo_m",
        "_topdown_contact_teacher_precenter_servo_m",
        "_topdown_contact_teacher_precenter_active",
        "_topdown_contact_teacher_center_servo_m",
        "_topdown_contact_teacher_center_servo_active",
        "_topdown_contact_teacher_center_err_xy",
        "_topdown_contact_teacher_ready",
        "_topdown_contact_teacher_finger_ready",
        "_topdown_contact_teacher_center_gate",
        "_topdown_contact_teacher_finger_close_gate",
        "_topdown_contact_teacher_wrist_yaw_release_gate",
        "_topdown_contact_teacher_descent_ready",
        "_topdown_contact_teacher_descent_start_gate",
        "_topdown_contact_teacher_arm_hold_unlock_fallback",
    )
    if not _has_env_tensors(env, required, log_env_id):
        return bits
    return (
        bits
        + f"tct_ready={tensor_env_bit(env, '_topdown_contact_teacher_ready', log_env_id)} "
        + f"tct_finger={tensor_env_bit(env, '_topdown_contact_teacher_finger_ready', log_env_id)} "
        + f"tct_cent_gate={tensor_env_bit(env, '_topdown_contact_teacher_center_gate', log_env_id)} "
        + f"tct_close_gate={tensor_env_float(env, '_topdown_contact_teacher_finger_close_gate', log_env_id):.2f} "
        + f"tct_wrist={tensor_env_bit(env, '_topdown_contact_teacher_wrist_yaw_release_gate', log_env_id)} "
        + f"tct_desc_ready={tensor_env_bit(env, '_topdown_contact_teacher_descent_ready', log_env_id)} "
        + f"tct_start={tensor_env_bit(env, '_topdown_contact_teacher_descent_start_gate', log_env_id)} "
        + f"tct_holdfb={tensor_env_bit(env, '_topdown_contact_teacher_arm_hold_unlock_fallback', log_env_id)} "
        + f"tct_thumb={tensor_env_float(env, '_topdown_contact_teacher_thumb_fraction', log_env_id):.2f} "
        + f"tct_idx={tensor_env_float(env, '_topdown_contact_teacher_index_fraction', log_env_id):.2f} "
        + f"tct_latch={tensor_env_bit(env, '_topdown_contact_teacher_thumb_latched', log_env_id)}/"
        + f"{tensor_env_bit(env, '_topdown_contact_teacher_index_latched', log_env_id)} "
        + f"tct_hold={tensor_env_float(env, '_topdown_contact_teacher_thumb_hold_fraction', log_env_id):.2f}/"
        + f"{tensor_env_float(env, '_topdown_contact_teacher_index_hold_fraction', log_env_id):.2f} "
        + f"tct_z={tensor_env_float(env, '_topdown_contact_teacher_descent_z', log_env_id):.3f} "
        + f"tct_zneed={tensor_env_float(env, '_topdown_contact_teacher_descent_z_need', log_env_id):.2f} "
        + f"tct_zclose={tensor_env_float(env, '_topdown_contact_teacher_descent_closure_gate', log_env_id):.2f} "
        + f"tct_in={tensor_env_float(env, '_topdown_contact_teacher_inward_m', log_env_id):.3f} "
        + f"tct_servo={tensor_env_float(env, '_topdown_contact_teacher_tip_servo_m', log_env_id):.3f} "
        + f"tct_precent={tensor_env_float(env, '_topdown_contact_teacher_precenter_servo_m', log_env_id):.3f} "
        + f"tct_precent_on={tensor_env_bit(env, '_topdown_contact_teacher_precenter_active', log_env_id)} "
        + f"tct_center={tensor_env_float(env, '_topdown_contact_teacher_center_servo_m', log_env_id):.3f} "
        + f"tct_cent_on={tensor_env_bit(env, '_topdown_contact_teacher_center_servo_active', log_env_id)} "
        + f"tct_cent_err={tensor_env_float(env, '_topdown_contact_teacher_center_err_xy', log_env_id):.3f} "
    )


def format_topdown_stage_bits(
    values: TopdownStageBitInputs,            # Param: input value used as values
    *,
    env                      : object | None = None,  # Param: environment or backend object used for runtime calls
    log_env_id               : int           = 0,  # Param: integer input for log env id
    inpocket_arm_hold_enabled: bool          = False,  # Param: boolean input enabling inpocket arm hold
    contact_teacher_enabled  : bool          = False,  # Param: boolean input enabling contact teacher
) -> str:
    """Format topdown curriculum progress-line stage bits

    Steps:
    - Resolve inputs for `format_topdown_stage_bits` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    bits = (
        f"stage={int(values.topdown_stage)} best_stage={int(values.best_topdown_stage)} "
        f"reach_hold={int(values.reach_hold)} align_hold={int(values.align_hold)} "
        f"pose_ready={int(values.contact_pose_ready)} pose_hold={int(values.contact_pose_hold)} "
        f"pose_shell={int(values.contact_pose_shell)} "
        f"palm_c={float(values.contact_palm_dist):.3f} "
        f"palm_ch={float(values.contact_palm_height):.3f} "
        f"s2_age={int(values.stage2_age)} unlock_prog={float(values.unlock_progress):.2f} "
        f"eff_unlock={float(values.effective_unlock_progress):.2f} "
        f"finger_hold_gate={float(values.finger_arm_hold_gate):.0f} "
        f"prehold_servo={float(values.prehold_servo):.3f} "
        f"align_line_z={float(values.align_line_z):.3f} "
        f"align_servo_q={float(values.align_servo_q):.3f} "
        f"align_servo_on={int(values.align_servo_active)} "
        f"pocket_sweep_q={float(values.pocket_sweep_q):.3f} "
        f"pocket_score={float(values.pocket_score_before):.4f}->{float(values.pocket_score_after):.4f} "
        f"s1_rate={float(values.stage_ge1_rate):.2f} "
        f"s2_rate={float(values.stage_ge2_rate):.2f} "
        f"contact={float(values.contact):.3f} strict={float(values.strict_contact):.3f} "
        f"thumb_c={float(values.thumb_contact):.3f} idx_c={float(values.index_contact):.3f} "
        f"align={float(values.align_face):.3f} align_angle={float(values.align_angle):.1f} "
        f"opp={float(values.opposed_face):.3f} lift={float(values.lift):.3f} "
        f"blk_disp={float(values.block_disp):.3f} "
    )
    if env is None:
        return bits
    bits += (
        f"lift_latch={tensor_env_bit(env, '_arm_lift_latched', log_env_id)} "
        f"lift_cnt={tensor_env_int(env, '_arm_lift_contact_counter', log_env_id)} "
        f"lift_prog={tensor_env_float(env, '_teacher_ik_topdown_lift_progress', log_env_id):.2f} "
        f"zblend={tensor_env_float(env, '_teacher_ik_topdown_nominal_z_blend_progress', log_env_id):.2f} "
        f"xyfix={tensor_env_float(env, '_teacher_ik_topdown_block_xy_stabilizer_m', log_env_id):.3f} "
        f"hold_rel={tensor_env_bit(env, '_inpocket_arm_hold_lift_release', log_env_id)} "
    )
    if inpocket_arm_hold_enabled:
        bits = _append_hold_bits(bits, env, log_env_id)
    if torch.is_tensor(getattr(env, "_teacher_ik_position_only", None)):
        bits += f"ik_posonly={tensor_env_bit(env, '_teacher_ik_position_only', log_env_id)} "
    bits = _append_finger_center_bits(bits, env, log_env_id)
    bits = _append_success_bits(bits, env, log_env_id)
    if contact_teacher_enabled:
        bits = _append_contact_teacher_bits(bits, env, log_env_id)
    return bits


def format_topdown_done_bits(*, success: bool, block_drift: bool, done: bool) -> str:
    """Format topdown done bits"""
    return f"success={int(success)} block_drift={int(block_drift)} done={int(done)} "


def format_grasp_align_stage_bits(values: GraspAlignStageBitInputs) -> str:
    """Format grasp-align progress-line stage bits"""
    return (
        f"contact={float(values.contact):.3f} curl={float(values.curl):.3f} "
        f"pinch={float(values.pinch_curl):.3f} opp={float(values.opposed_face):.3f} "
        f"align={float(values.align_face):.3f} align_angle={float(values.align_angle):.1f} "
        f"hand_N={float(values.hand_force):.2f} blk_disp={float(values.block_disp):.3f} "
    )


def format_contact_stage_bits(
    values: ContactStageBitInputs,  # Param: input value used as values
    *,
    env       : object | None = None,  # Param: environment or backend object used for runtime calls
    log_env_id: int           = 0,  # Param: integer input for log env id
) -> str:
    """Format contact-family progress-line stage bits"""
    bits = (
        f"contact={float(values.contact):.3f} both={float(values.both_contact):.3f} "
        f"strict={float(values.strict_contact):.3f} any={float(values.fingertip_contact):.3f} "
        f"thumb_c={float(values.thumb_contact):.3f} idx_c={float(values.index_contact):.3f} "
        f"thumb_N={float(values.thumb_contact_raw):.2f} idx_N={float(values.index_contact_raw):.2f} "
        f"hand={float(values.hand_contact):.3f} "
        f"lift={float(values.lift):.3f} curl={float(values.curl):.3f} "
        f"pinch={float(values.pinch_curl):.3f} opp={float(values.opposed_face):.3f} "
        f"align={float(values.align_face):.3f} align_angle={float(values.align_angle):.1f} "
        f"hand_N={float(values.hand_force):.2f} blk_disp={float(values.block_disp):.3f} "
    )
    if env is None:
        return bits
    return (
        bits
        + f"lift_latch={tensor_env_bit(env, '_arm_lift_latched', log_env_id)} "
        + f"lift_prog={tensor_env_float(env, '_teacher_ik_topdown_lift_progress', log_env_id):.2f} "
        + f"hold_rel={tensor_env_bit(env, '_inpocket_arm_hold_lift_release', log_env_id)} "
    )


def format_phase1_progress_bits(*, phase1_ready: bool, finger_unlock: float) -> str:
    """Format phase-1 progress bits"""
    return f"phase1={int(phase1_ready)} unlock={float(finger_unlock):.2f} "


def format_grasp_align_done_bits(*, success: bool, done: bool) -> str:
    """Format grasp-align done bits"""
    return f"success={int(success)} done={int(done)} "


def format_contact_done_bits(
    *,
    success            : bool,  # Param: boolean input controlling success
    phase15_shell_drift: bool,  # Param: boolean input controlling phase15 shell drift
    off_table          : bool,  # Param: boolean input controlling off table
    block_drift        : bool,  # Param: boolean input controlling block drift
    done               : bool,  # Param: done flag or tensor for the transition
) -> str:
    """Format contact-family done bits"""
    return (
        f"success={int(success)} shell_drift={int(phase15_shell_drift)} "
        f"off_table={int(off_table)} block_drift={int(block_drift)} done={int(done)} "
    )
