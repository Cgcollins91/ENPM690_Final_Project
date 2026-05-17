"""Task, curriculum-gate, and success configuration for red centered lift.

These dataclasses are the stable, reviewable surface for environment variables
consumed by the IsaacLab task.  The trainer still reads legacy
``TOPDOWN_*``/``CURRICULUM_*`` knobs because that is how the late-project
debugging loop evolved, but new runs should be expressed through these typed
groups instead of ad-hoc shell exports.

Conceptually this file owns *what the task considers ready/successful*:

* ``TaskIdentity``: block size, source pose, physics identity, episode length.
* ``StageGateConfig``: reach/align/contact stage latches.
* ``FingerCenteringConfig``: the pre-curl pocket contract.
* ``LiftSuccessConfig``: physical lift and drift criteria.
* ``ContactPoseFallbackConfig``: permissive shells that keep the teacher from
  starving contact attempts while still preventing obvious bad closures.
"""

from __future__ import annotations  # keeps annotations lazy for forward references

import os  # imports access to parent process environment variables
from dataclasses import dataclass  # imports dataclass helpers used by config groups

from .base import bool01, clean_dict  # imports shared env and CLI conversion helpers


@dataclass(frozen=True)  # makes the following config group immutable
class TaskIdentity:  # defines the task identity config group
    """The single task kept for the final project.

    Geometry/physics values here define the actual simulated world.  They are
    intentionally separated from reward weights and teacher parameters so a run
    manifest makes it obvious whether a change altered the task or just the
    controller trying to solve it.
    """

    task                    : str   = "Isaac-Topdown-Curriculum-G129-Dex3-Joint"  # Selects the Isaac Lab task name to launch
    lift_task               : bool  = True  # Controls whether the environment uses the lift-task reward and termination path
    wrap_table              : bool  = True  # Controls whether the table is wrapped by the topdown task setup
    source_pose_mode        : str   = "red"  # Selects which source pose layout the task loads
    keep_distractors_visible: bool  = False  # Controls whether unused distractor blocks remain visible
    dynamic_block           : bool  = True  # Controls whether dynamic block is enabled
    dynamic_distractors     : bool  = False  # Controls whether distractor blocks are simulated dynamically
    block_size              : float = 0.08  # Sets the target block edge length in meters
    block_mass              : float = 0.25  # Sets the target block mass in kilograms
    block_static_friction   : float = 10.0  # Sets static friction for the target block material
    block_dynamic_friction  : float = 1.5  # Sets dynamic friction for the target block material
    contact_offset          : float = 0.002  # Sets the collision contact generation distance
    block_jitter_x          : float = 0.025  # Sets the block jitter X distance in meters
    block_jitter_y          : float = 0.025  # Sets the block jitter Y distance in meters
    episode_length_s        : float = 6.0  # Sets maximum episode duration in seconds
    hover_above_block_top   : float = 0.05  # Sets the nominal palm hover clearance above the block top

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group.

        When ``TOPDOWN_PHYSICS_PROFILE`` is set in the calling process's
        environment to a recognized profile name, the material/solver physics
        env vars the profile controls are SUPPRESSED from emission (so the
        profile's defaults apply in the subprocess) and
        ``TOPDOWN_PHYSICS_PROFILE`` is propagated explicitly. Unknown or
        unset profile values preserve the historical full per-knob emission.
        ``TOPDOWN_BLOCK_SIZE`` remains task geometry, not a profile default:
        profile runs still need to preserve 6cm/8cm curriculum variants.
        """
        from enpm690_final_project.config.physics_profile import VALID_PROFILES  # imports config dependencies from enpm690_final_projectconfigphysics_profile

        profile = os.environ.get("TOPDOWN_PHYSICS_PROFILE", "").strip().lower()  # Reads the requested physics profile from the parent environment
        use_profile = profile in VALID_PROFILES and profile != "default"  # Tracks whether a non-default profile should own physics defaults

        base: dict[str, object] = {  # Collects env vars before unset values are removed
            "TASK": self.task,  # Exports TASK from the task setting
            "TOPDOWN_LIFT_TASK": bool01(self.lift_task),  # Exports TOPDOWN_LIFT_TASK as legacy 0 or 1 from the lift task setting
            "TOPDOWN_WRAP_TABLE": bool01(self.wrap_table),  # Exports TOPDOWN_WRAP_TABLE as legacy 0 or 1 from the wrap table setting
            "TOPDOWN_SOURCE_POSE_MODE": self.source_pose_mode,  # Exports TOPDOWN_SOURCE_POSE_MODE from the source pose mode setting
            "TOPDOWN_KEEP_DISTRACTORS_VISIBLE": bool01(self.keep_distractors_visible),  # Exports TOPDOWN_KEEP_DISTRACTORS_VISIBLE as legacy 0 or 1 from the keep distractors visible setting
            "TOPDOWN_DYNAMIC_DISTRACTORS": bool01(self.dynamic_distractors),  # Exports TOPDOWN_DYNAMIC_DISTRACTORS as legacy 0 or 1 from the dynamic distractors setting
            "TOPDOWN_BLOCK_JITTER_X": self.block_jitter_x,  # Exports TOPDOWN_BLOCK_JITTER_X from the block jitter X setting
            "TOPDOWN_BLOCK_JITTER_Y": self.block_jitter_y,  # Exports TOPDOWN_BLOCK_JITTER_Y from the block jitter Y setting
            "TOPDOWN_EPISODE_LENGTH_S": self.episode_length_s,  # Exports TOPDOWN_EPISODE_LENGTH_S from the episode length s setting
            "CURRICULUM_TOPDOWN_HOVER_ABOVE_BLOCK_TOP": self.hover_above_block_top,  # Exports CURRICULUM_TOPDOWN_HOVER_ABOVE_BLOCK_TOP from the hover above block top setting
        }  # closes the current expression

        if use_profile:  # Checks whether use profile
            base["TOPDOWN_PHYSICS_PROFILE"] = profile  # stores the resolved value in the mapping
            base["TOPDOWN_BLOCK_SIZE"] = self.block_size  # stores the resolved value in the mapping
        else:  # handles the fallback branch
            base.update(  # merges override values into the current mapping
                {  # opens a nested expression
                    "TOPDOWN_DYNAMIC_BLOCK": bool01(self.dynamic_block),  # Exports TOPDOWN_DYNAMIC_BLOCK as legacy 0 or 1 from the dynamic block setting
                    "TOPDOWN_BLOCK_SIZE": self.block_size,  # Exports TOPDOWN_BLOCK_SIZE from the block size setting
                    "TOPDOWN_BLOCK_MASS": self.block_mass,  # Exports TOPDOWN_BLOCK_MASS from the block mass setting
                    "TOPDOWN_BLOCK_STATIC_FRICTION": self.block_static_friction,  # Exports TOPDOWN_BLOCK_STATIC_FRICTION from the block static friction setting
                    "TOPDOWN_BLOCK_DYNAMIC_FRICTION": self.block_dynamic_friction,  # Exports TOPDOWN_BLOCK_DYNAMIC_FRICTION from the block dynamic friction setting
                    "TOPDOWN_CONTACT_OFFSET": self.contact_offset,  # Exports TOPDOWN_CONTACT_OFFSET from the contact offset setting
                }  # closes the current expression
            )  # closes the current expression

        return clean_dict(base)  # returns env vars after dropping unset values

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        return ["--task", self.task]  # returns the computed value


@dataclass(frozen=True)  # makes the following config group immutable
class StageGateConfig:  # defines the stage gate config group
    """Latch gates for reach, alignment, and contact-pose entry.

    Stage gates are coarse curriculum phase boundaries, not final quality
    metrics.  In particular, ``success_palm_orient_max_deg`` is the contact
    pose shell used downstream by success/contact predicates; if it is tighter
    than what the IK can physically reach, the contact teacher never activates.
    """

    reach_hold_steps          : int   = 5  # Sets consecutive steps required for reach
    align_hold_steps          : int   = 5  # Sets consecutive steps required for align
    stage1_palm_dist_max      : float = 0.2  # Sets the maximum allowed stage1 palm dist
    stage1_palm_height_max    : float = 0.14  # Sets the maximum allowed stage1 palm height
    stage1_palm_orient_max_deg: float = 45  # Sets the stage1 palm orient maximum deg angular threshold
    stage1_palm_yaw_max_deg   : float = 65  # Sets the stage1 palm yaw maximum deg angular threshold
    stage1_align_err_max      : float = 0.45  # Sets the maximum allowed stage1 align err
    stage1_line_angle_max_deg : float = 65  # Sets the stage1 line angle maximum deg angular threshold
    stage1_opposed_gate_min   : float = 0.0  # Sets the minimum required stage1 opposed gate
    stage1_no_contact_max     : float = 0.08  # Sets the maximum allowed stage1 no contact
    stage2_palm_dist_max      : float = 0.12  # Sets the maximum allowed stage2 palm dist
    stage2_palm_height_max    : float = 0.06  # Sets the maximum allowed stage2 palm height
    stage2_palm_orient_max_deg: float = 35  # Sets the stage2 palm orient maximum deg angular threshold
    stage2_palm_yaw_max_deg   : float = 45  # Sets the stage2 palm yaw maximum deg angular threshold
    stage2_align_err_max      : float = 0.24  # Sets the maximum allowed stage2 align err
    stage2_line_angle_max_deg : float = 40  # Sets the stage2 line angle maximum deg angular threshold
    stage2_opposed_gate_min   : float = 0.50  # Sets the minimum required stage2 opposed gate
    stage2_no_contact_max     : float = 2.01  # Sets the maximum allowed stage2 no contact
    # Success/contact-pose shell - distinct from stage 2 entry If a profile
    # relaxes stage2_palm_orient_max_deg, this should usually be relaxed in
    # tandem or contact_pose_ready never latches
    success_palm_orient_max_deg: float = 35  # Sets the success palm orient maximum deg angular threshold
    success_palm_yaw_max_deg   : float = 35  # Sets the success palm yaw maximum deg angular threshold
    success_align_err_max      : float = 0.20  # Sets the maximum allowed success align err
    success_opposed_gate_min   : float = 0.50  # Sets the minimum required success opposed gate

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "CURRICULUM_REACH_HOLD_STEPS": self.reach_hold_steps,  # Exports CURRICULUM_REACH_HOLD_STEPS from the reach hold steps setting
                "CURRICULUM_ALIGN_HOLD_STEPS": self.align_hold_steps,  # Exports CURRICULUM_ALIGN_HOLD_STEPS from the align hold steps setting
                "CURRICULUM_STAGE1_PALM_DIST_MAX": self.stage1_palm_dist_max,  # Exports CURRICULUM_STAGE1_PALM_DIST_MAX from the stage1 palm distance maximum setting
                "CURRICULUM_STAGE1_PALM_HEIGHT_MAX": self.stage1_palm_height_max,  # Exports CURRICULUM_STAGE1_PALM_HEIGHT_MAX from the stage1 palm height maximum setting
                "CURRICULUM_STAGE1_PALM_ORIENT_MAX_DEG": self.stage1_palm_orient_max_deg,  # Exports CURRICULUM_STAGE1_PALM_ORIENT_MAX_DEG from the stage1 palm orient maximum deg setting
                "CURRICULUM_STAGE1_PALM_YAW_MAX_DEG": self.stage1_palm_yaw_max_deg,  # Exports CURRICULUM_STAGE1_PALM_YAW_MAX_DEG from the stage1 palm yaw maximum deg setting
                "CURRICULUM_STAGE1_ALIGN_ERR_MAX": self.stage1_align_err_max,  # Exports CURRICULUM_STAGE1_ALIGN_ERR_MAX from the stage1 align error maximum setting
                "CURRICULUM_STAGE1_LINE_ANGLE_MAX_DEG": self.stage1_line_angle_max_deg,  # Exports CURRICULUM_STAGE1_LINE_ANGLE_MAX_DEG from the stage1 line angle maximum deg setting
                "CURRICULUM_STAGE1_OPPOSED_GATE_MIN": self.stage1_opposed_gate_min,  # Exports CURRICULUM_STAGE1_OPPOSED_GATE_MIN from the stage1 opposed gate minimum setting
                "CURRICULUM_STAGE1_NO_CONTACT_MAX": self.stage1_no_contact_max,  # Exports CURRICULUM_STAGE1_NO_CONTACT_MAX from the stage1 no contact maximum setting
                "CURRICULUM_STAGE2_PALM_DIST_MAX": self.stage2_palm_dist_max,  # Exports CURRICULUM_STAGE2_PALM_DIST_MAX from the stage2 palm distance maximum setting
                "CURRICULUM_STAGE2_PALM_HEIGHT_MAX": self.stage2_palm_height_max,  # Exports CURRICULUM_STAGE2_PALM_HEIGHT_MAX from the stage2 palm height maximum setting
                "CURRICULUM_STAGE2_PALM_ORIENT_MAX_DEG": self.stage2_palm_orient_max_deg,  # Exports CURRICULUM_STAGE2_PALM_ORIENT_MAX_DEG from the stage2 palm orient maximum deg setting
                "CURRICULUM_STAGE2_PALM_YAW_MAX_DEG": self.stage2_palm_yaw_max_deg,  # Exports CURRICULUM_STAGE2_PALM_YAW_MAX_DEG from the stage2 palm yaw maximum deg setting
                "CURRICULUM_STAGE2_ALIGN_ERR_MAX": self.stage2_align_err_max,  # Exports CURRICULUM_STAGE2_ALIGN_ERR_MAX from the stage2 align error maximum setting
                "CURRICULUM_STAGE2_LINE_ANGLE_MAX_DEG": self.stage2_line_angle_max_deg,  # Exports CURRICULUM_STAGE2_LINE_ANGLE_MAX_DEG from the stage2 line angle maximum deg setting
                "CURRICULUM_STAGE2_OPPOSED_GATE_MIN": self.stage2_opposed_gate_min,  # Exports CURRICULUM_STAGE2_OPPOSED_GATE_MIN from the stage2 opposed gate minimum setting
                "CURRICULUM_STAGE2_NO_CONTACT_MAX": self.stage2_no_contact_max,  # Exports CURRICULUM_STAGE2_NO_CONTACT_MAX from the stage2 no contact maximum setting
                "CURRICULUM_SUCCESS_PALM_ORIENT_MAX_DEG": self.success_palm_orient_max_deg,  # Exports CURRICULUM_SUCCESS_PALM_ORIENT_MAX_DEG from the success palm orient maximum deg setting
                "CURRICULUM_SUCCESS_PALM_YAW_MAX_DEG": self.success_palm_yaw_max_deg,  # Exports CURRICULUM_SUCCESS_PALM_YAW_MAX_DEG from the success palm yaw maximum deg setting
                "CURRICULUM_SUCCESS_ALIGN_ERR_MAX": self.success_align_err_max,  # Exports CURRICULUM_SUCCESS_ALIGN_ERR_MAX from the success align error maximum setting
                "CURRICULUM_SUCCESS_OPPOSED_GATE_MIN": self.success_opposed_gate_min,  # Exports CURRICULUM_SUCCESS_OPPOSED_GATE_MIN from the success opposed gate minimum setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class FingerCenteringConfig:  # defines the finger centering config group
    """Finger-centering gate that controls finger unlock.

    This group encodes the "do not curl until the open hand is centered over
    the pocket" rule.  It is deliberately independent from the 20 percent
    preload used by some teacher profiles; preload is a finger starting pose,
    while this gate decides when active closure/descent is allowed.
    """

    requires_center          : bool         = True  # Controls whether requires center is enabled
    latch                    : bool         = True  # Controls whether latch is enabled
    hold_steps               : int          = 2  # Sets the number of steps for hold
    unlock_ramp_steps        : int          = 60  # Sets the number of steps for unlock ramp
    align_angle_max_deg      : float        = 15  # Sets the align angle maximum deg angular threshold
    align_error_max          : float        = 0.20  # Sets the maximum allowed align error
    three_finger_centering   : bool         = True  # Controls whether three finger centering is enabled
    back_finger_spread_offset: float        = 0.020  # Sets the back finger spread offset distance in meters
    requires_contact_pose    : bool         = True  # Controls whether requires contact pose is enabled
    use_xyz_gate             : bool         = False  # Controls whether use XYZ gate is enabled
    xyz_gate_min             : float | None = None  # Sets the minimum required XYZ gate
    stage2_xyz_gate_start_m  : float | None = None  # Sets the stage2 XYZ gate start m distance in meters
    stage2_xyz_gate_full_m   : float | None = None  # Sets the stage2 XYZ gate full m distance in meters
    tip_xy_max               : float        = 0.025  # Sets the maximum allowed tip XY
    max_tip_xy_max           : float        = 0.080  # Sets the maximum allowed max tip XY
    tip_z_max                : float        = 0.075  # Sets the maximum allowed tip Z
    face_top_margin          : float        = 0.012  # Sets the face top margin config value
    face_half_extent         : float        = 0.025  # Sets the face half extent config value

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "CURRICULUM_FINGER_UNLOCK_REQUIRES_CENTER": bool01(self.requires_center),  # Exports CURRICULUM_FINGER_UNLOCK_REQUIRES_CENTER as legacy 0 or 1 from the requires center setting
                "CURRICULUM_FINGER_CENTER_LATCH": bool01(self.latch),  # Exports CURRICULUM_FINGER_CENTER_LATCH as legacy 0 or 1 from the latch setting
                "CURRICULUM_FINGER_CENTER_HOLD_STEPS": self.hold_steps,  # Exports CURRICULUM_FINGER_CENTER_HOLD_STEPS from the hold steps setting
                "CURRICULUM_FINGER_UNLOCK_RAMP_STEPS": self.unlock_ramp_steps,  # Exports CURRICULUM_FINGER_UNLOCK_RAMP_STEPS from the unlock ramp steps setting
                "CURRICULUM_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG": self.align_angle_max_deg,  # Exports CURRICULUM_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG from the align angle maximum deg setting
                "CURRICULUM_FINGER_CENTER_ALIGN_ERR_MAX": self.align_error_max,  # Exports CURRICULUM_FINGER_CENTER_ALIGN_ERR_MAX from the align error maximum setting
                "CURRICULUM_THREE_FINGER_CENTERING": bool01(self.three_finger_centering),  # Exports CURRICULUM_THREE_FINGER_CENTERING as legacy 0 or 1 from the three finger centering setting
                "CURRICULUM_BACK_FINGER_SPREAD_OFFSET": self.back_finger_spread_offset,  # Exports CURRICULUM_BACK_FINGER_SPREAD_OFFSET from the back finger spread offset setting
                "CURRICULUM_FINGER_CENTER_REQUIRES_CONTACT_POSE": bool01(self.requires_contact_pose),  # Exports CURRICULUM_FINGER_CENTER_REQUIRES_CONTACT_POSE as legacy 0 or 1 from the requires contact pose setting
                "CURRICULUM_FINGER_CENTER_USE_XYZ_GATE": bool01(self.use_xyz_gate),  # Exports CURRICULUM_FINGER_CENTER_USE_XYZ_GATE as legacy 0 or 1 from the use XYZ gate setting
                "CURRICULUM_FINGER_CENTER_XYZ_GATE_MIN": self.xyz_gate_min,  # Exports CURRICULUM_FINGER_CENTER_XYZ_GATE_MIN from the XYZ gate minimum setting
                "CURRICULUM_STAGE2_FINGER_XYZ_GATE_START_M": self.stage2_xyz_gate_start_m,  # Exports CURRICULUM_STAGE2_FINGER_XYZ_GATE_START_M from the stage2 XYZ gate start m setting
                "CURRICULUM_STAGE2_FINGER_XYZ_GATE_FULL_M": self.stage2_xyz_gate_full_m,  # Exports CURRICULUM_STAGE2_FINGER_XYZ_GATE_FULL_M from the stage2 XYZ gate full m setting
                "CURRICULUM_FINGER_CENTER_TIP_XY_MAX": self.tip_xy_max,  # Exports CURRICULUM_FINGER_CENTER_TIP_XY_MAX from the tip XY maximum setting
                "CURRICULUM_FINGER_CENTER_MAX_TIP_XY_MAX": self.max_tip_xy_max,  # Exports CURRICULUM_FINGER_CENTER_MAX_TIP_XY_MAX from the max tip XY maximum setting
                "CURRICULUM_FINGER_CENTER_TIP_Z_MAX": self.tip_z_max,  # Exports CURRICULUM_FINGER_CENTER_TIP_Z_MAX from the tip Z maximum setting
                "CURRICULUM_FINGER_FACE_TOP_MARGIN": self.face_top_margin,  # Exports CURRICULUM_FINGER_FACE_TOP_MARGIN from the face top margin setting
                "CURRICULUM_FACE_HALF_EXTENT": self.face_half_extent,  # Exports CURRICULUM_FACE_HALF_EXTENT from the face half extent setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class LiftSuccessConfig:  # defines the lift success config group
    """Centered upright lift criteria and drop termination thresholds.

    These thresholds are both metrics and training feedback.  Loosening drift
    limits can make an MVP lift demonstrable; tightening them defines the next
    optimization target for RL after the teacher reliably forms a grasp.
    """

    latch_hold_steps                  : int          = 3  # Sets consecutive steps required for latch
    latch_contact_threshold           : float        = 0.30  # Sets threshold for latch contact threshold
    latch_opposed_face_min            : float        = 0.50  # Sets the minimum required latch opposed face
    latch_requires_center             : bool         = False  # Controls whether latch requires center is enabled
    latch_requires_center_live        : bool         = False  # Controls whether latch requires center live is enabled
    latch_requires_descent_z_min      : float        = 0.0  # Sets the minimum required latch requires descent Z
    grip_settle_steps                 : float | None = None  # Sets the number of steps for grip settle
    success_hold_steps                : int          = 15  # Sets consecutive steps required for success
    success_height                    : float        = 0.050  # Sets the success height distance in meters
    success_mode                      : str          = "gated"  # Selects the success mode behavior
    success_contact_min               : float        = 0.30  # Sets the minimum required success contact
    success_xy_drift_max              : float        = 0.050  # Sets the maximum allowed success XY drift
    terminate_drop_from_max           : float        = 0.025  # Sets the maximum allowed terminate drop from
    terminate_drop_min_peak           : float        = 0.035  # Sets the terminate drop minimum peak config value
    terminate_drop_hold_steps         : int          = 2  # Sets consecutive steps required for terminate drop
    block_drift_threshold             : float        = 0.12  # Sets threshold for block drift threshold
    lift_requires_grip                : bool         = True  # Controls whether lift requires grip is enabled
    block_xy_vel_soft_cap             : float        = 0.5  # Sets the block XY velocity soft cap config value
    success_center_tip_xy_max         : float        = 0.015  # Sets the maximum allowed success center tip XY
    success_center_tip_z_max          : float        = 0.060  # Sets the maximum allowed success center tip Z
    success_center_align_angle_max_deg: float        = 8.0  # Sets the success center align angle maximum deg angular threshold
    success_center_hold_steps         : int          = 0  # Sets consecutive steps required for success center

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TOPDOWN_LIFT_LATCH_HOLD_STEPS": self.latch_hold_steps,  # Exports TOPDOWN_LIFT_LATCH_HOLD_STEPS from the latch hold steps setting
                "TOPDOWN_LIFT_LATCH_CONTACT_THRESHOLD": self.latch_contact_threshold,  # Exports TOPDOWN_LIFT_LATCH_CONTACT_THRESHOLD from the latch contact threshold setting
                "TOPDOWN_LIFT_LATCH_OPPOSED_FACE_MIN": self.latch_opposed_face_min,  # Exports TOPDOWN_LIFT_LATCH_OPPOSED_FACE_MIN from the latch opposed face minimum setting
                "TOPDOWN_LIFT_LATCH_REQUIRES_CENTER": bool01(self.latch_requires_center),  # Exports TOPDOWN_LIFT_LATCH_REQUIRES_CENTER as legacy 0 or 1 from the latch requires center setting
                "TOPDOWN_LIFT_LATCH_REQUIRES_CENTER_LIVE": bool01(  # Starts env export expression for TOPDOWN_LIFT_LATCH_REQUIRES_CENTER_LIVE
                    self.latch_requires_center_live  # Passes the latch requires center live setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_LIFT_LATCH_REQUIRES_DESCENT_Z_MIN": self.latch_requires_descent_z_min,  # Exports TOPDOWN_LIFT_LATCH_REQUIRES_DESCENT_Z_MIN from the latch requires descent Z minimum setting
                "TOPDOWN_LIFT_GRIP_SETTLE_STEPS": self.grip_settle_steps,  # Exports TOPDOWN_LIFT_GRIP_SETTLE_STEPS from the grip settle steps setting
                "TOPDOWN_LIFT_SUCCESS_HOLD_STEPS": self.success_hold_steps,  # Exports TOPDOWN_LIFT_SUCCESS_HOLD_STEPS from the success hold steps setting
                "TOPDOWN_LIFT_SUCCESS_HEIGHT": self.success_height,  # Exports TOPDOWN_LIFT_SUCCESS_HEIGHT from the success height setting
                "TOPDOWN_LIFT_SUCCESS_MODE": self.success_mode,  # Exports TOPDOWN_LIFT_SUCCESS_MODE from the success mode setting
                "TOPDOWN_LIFT_SUCCESS_CONTACT_MIN": self.success_contact_min,  # Exports TOPDOWN_LIFT_SUCCESS_CONTACT_MIN from the success contact minimum setting
                "TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX": self.success_xy_drift_max,  # Exports TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX from the success XY drift maximum setting
                "TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX": self.terminate_drop_from_max,  # Exports TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX from the terminate drop from maximum setting
                "TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK": self.terminate_drop_min_peak,  # Exports TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK from the terminate drop minimum peak setting
                "TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS": self.terminate_drop_hold_steps,  # Exports TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS from the terminate drop hold steps setting
                "CURRICULUM_BLOCK_DRIFT_THRESHOLD": self.block_drift_threshold,  # Exports CURRICULUM_BLOCK_DRIFT_THRESHOLD from the block drift threshold setting
                "CURRICULUM_LIFT_HEIGHT_PROGRESS_REQUIRES_GRIP": bool01(self.lift_requires_grip),  # Exports CURRICULUM_LIFT_HEIGHT_PROGRESS_REQUIRES_GRIP as legacy 0 or 1 from the lift requires grip setting
                "CURRICULUM_BLOCK_XY_VEL_SOFT_CAP": self.block_xy_vel_soft_cap,  # Exports CURRICULUM_BLOCK_XY_VEL_SOFT_CAP from the block XY velocity soft cap setting
                "CURRICULUM_SUCCESS_CENTER_TIP_XY_MAX": self.success_center_tip_xy_max,  # Exports CURRICULUM_SUCCESS_CENTER_TIP_XY_MAX from the success center tip XY maximum setting
                "CURRICULUM_SUCCESS_CENTER_TIP_Z_MAX": self.success_center_tip_z_max,  # Exports CURRICULUM_SUCCESS_CENTER_TIP_Z_MAX from the success center tip Z maximum setting
                "CURRICULUM_SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG": self.success_center_align_angle_max_deg,  # Exports CURRICULUM_SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG from the success center align angle maximum deg setting
                "CURRICULUM_SUCCESS_CENTER_HOLD_STEPS": self.success_center_hold_steps,  # Exports CURRICULUM_SUCCESS_CENTER_HOLD_STEPS from the success center hold steps setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class ArmHoldCenteringConfig:  # defines the arm hold centering config group
    """Extra gates for when the in-pocket arm target is allowed to freeze."""

    latch_hold_steps              : int   = 4  # Sets consecutive steps required for latch
    freeze_requires_finger_center : bool  = True  # Controls whether freeze requires finger center is enabled
    freeze_requires_contact_center: bool  = False  # Controls whether freeze requires contact center is enabled
    center_tip_xy_max             : float = 0.020  # Sets the maximum allowed center tip XY
    center_tip_z_max              : float = 0.045  # Sets the maximum allowed center tip Z
    center_align_angle_max_deg    : float = 12.0  # Sets the center align angle maximum deg angular threshold

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "INPOCKET_LATCH_HOLD_STEPS": self.latch_hold_steps,  # Exports INPOCKET_LATCH_HOLD_STEPS from the latch hold steps setting
                "INPOCKET_ARM_HOLD_FREEZE_REQUIRES_FINGER_CENTER": bool01(  # Starts env export expression for INPOCKET_ARM_HOLD_FREEZE_REQUIRES_FINGER_CENTER
                    self.freeze_requires_finger_center  # Passes the freeze requires finger center setting into the surrounding call
                ),  # closes the current expression
                "INPOCKET_ARM_HOLD_FREEZE_REQUIRES_CONTACT_CENTER": bool01(  # Starts env export expression for INPOCKET_ARM_HOLD_FREEZE_REQUIRES_CONTACT_CENTER
                    self.freeze_requires_contact_center  # Passes the freeze requires contact center setting into the surrounding call
                ),  # closes the current expression
                "INPOCKET_ARM_HOLD_CENTER_TIP_XY_MAX": self.center_tip_xy_max,  # Exports INPOCKET_ARM_HOLD_CENTER_TIP_XY_MAX from the center tip XY maximum setting
                "INPOCKET_ARM_HOLD_CENTER_TIP_Z_MAX": self.center_tip_z_max,  # Exports INPOCKET_ARM_HOLD_CENTER_TIP_Z_MAX from the center tip Z maximum setting
                "INPOCKET_ARM_HOLD_CENTER_ALIGN_ANGLE_MAX_DEG": self.center_align_angle_max_deg,  # Exports INPOCKET_ARM_HOLD_CENTER_ALIGN_ANGLE_MAX_DEG from the center align angle maximum deg setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class ContactPoseFallbackConfig:  # defines the contact pose fallback config group
    """Loose fallback shell used after stage-two entry."""

    fallback_steps          : int          = 20  # Sets the number of steps for fallback
    align_err_max           : float        = 0.35  # Sets the maximum allowed align err
    palm_dist_max           : float        = 0.14  # Sets the maximum allowed palm dist
    palm_height_max         : float        = 0.09  # Sets the maximum allowed palm height
    palm_orient_max_deg     : float        = 100.0  # Sets the palm orient maximum deg angular threshold
    palm_yaw_max_deg        : float        = 110.0  # Sets the palm yaw maximum deg angular threshold
    opposed_gate_min        : float        = 0.0  # Sets the minimum required opposed gate
    contact_force_threshold : float        = 0.05  # Sets threshold for contact force threshold
    contact_force_saturation: float        = 1.5  # Sets saturation level for contact force saturation
    contact_pose_hold_steps : int | None   = None  # Sets consecutive steps required for contact pose
    block_disp_max          : float | None = None  # Sets the maximum allowed block disp
    contact_lift_max        : float | None = None  # Sets the maximum allowed contact lift

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_STEPS": self.fallback_steps,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_STEPS from the fallback steps setting
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_ALIGN_ERR_MAX": self.align_err_max,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_ALIGN_ERR_MAX from the align error maximum setting
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_DIST_MAX": self.palm_dist_max,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_DIST_MAX from the palm distance maximum setting
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_HEIGHT_MAX": self.palm_height_max,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_HEIGHT_MAX from the palm height maximum setting
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_ORIENT_MAX_DEG": self.palm_orient_max_deg,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_ORIENT_MAX_DEG from the palm orient maximum deg setting
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_YAW_MAX_DEG": self.palm_yaw_max_deg,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_YAW_MAX_DEG from the palm yaw maximum deg setting
                "CURRICULUM_CONTACT_POSE_READY_FALLBACK_OPPOSED_GATE_MIN": self.opposed_gate_min,  # Exports CURRICULUM_CONTACT_POSE_READY_FALLBACK_OPPOSED_GATE_MIN from the opposed gate minimum setting
                "CURRICULUM_CONTACT_POSE_HOLD_STEPS": self.contact_pose_hold_steps,  # Exports CURRICULUM_CONTACT_POSE_HOLD_STEPS from the contact pose hold steps setting
                "CURRICULUM_CONTACT_BLOCK_DISP_MAX": self.block_disp_max,  # Exports CURRICULUM_CONTACT_BLOCK_DISP_MAX from the block disp maximum setting
                "CURRICULUM_CONTACT_LIFT_MAX": self.contact_lift_max,  # Exports CURRICULUM_CONTACT_LIFT_MAX from the contact lift maximum setting
                "TOPDOWN_CONTACT_FORCE_THRESHOLD": self.contact_force_threshold,  # Exports TOPDOWN_CONTACT_FORCE_THRESHOLD from the contact force threshold setting
                "TOPDOWN_CONTACT_FORCE_SATURATION": self.contact_force_saturation,  # Exports TOPDOWN_CONTACT_FORCE_SATURATION from the contact force saturation setting
            }  # closes the current expression
        )  # closes the current expression
