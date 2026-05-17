"""IK teacher and action-surface configuration.

This file owns *how demonstrations are produced*.  The final-project pipeline
uses a scripted IK/contact teacher to create behavior-cloning targets, then
lets TD3 refine the same action surface.  Keeping the teacher knobs separate
from task gates makes it easier to tell whether a run changed the environment,
the teacher, or the RL optimizer.

Important naming convention:

* "prehold" means the open-hand positioning phase before intentional closure.
* "task-space IK" means the stacked DLS solve over grasp-center/span/palm
  angular tasks.
* "contact teacher" means the finger closure/descent/lift finite-state logic.
"""

from __future__ import annotations  # keeps annotations lazy for forward references

from dataclasses import dataclass  # imports dataclass helpers used by config groups

from .base import add_arg, add_flag, bool01, clean_dict  # imports shared env and CLI conversion helpers


@dataclass(frozen=True)  # makes the following config group immutable
class TeacherProfile:  # defines the teacher profile config group
    """The high-level teacher/controller contract.

    This group toggles the major surfaces: policy-vs-IK arm source, whether the
    contact teacher is active, closure rate/fraction, and how assisted actions
    are exposed to behavior cloning.  Fine geometric details live in the
    prehold/task-space/lift groups below.
    """

    arm_controller                      : str          = "policy"  # Selects whether arm actions come from policy or scripted control
    teacher_arm_source                  : str          = "ik"  # Selects which teacher source supplies arm targets
    assist_noise_clean_bc_target        : bool         = True  # Keeps BC targets clean while noise is applied to assisted actions
    bc_target_includes_inpocket_arm_hold: bool         = True  # Controls whether BC labels include the in-pocket arm hold action
    contact_teacher                     : bool         = True  # Enables the scripted contact teacher state machine
    close_rate                          : float        = 0.010  # Sets the finger closure increment applied by the contact teacher
    start_fraction                      : float        = 0.08  # Sets the initial finger closure fraction before scripted closing
    middle_scale                        : float        = 1.0  # Scales middle-finger contribution in three-finger contact logic
    bypass_unlock                       : bool         = False  # Allows the contact teacher to skip the normal finger unlock gate
    drive_until_lift_latch              : bool         = True  # Keeps the contact teacher active until lift latch occurs
    drive_after_lift_latch              : bool         = True  # Keeps the contact teacher active after lift latch occurs
    prehold_ik_position_only            : bool         = True  # Uses position-only IK during the prehold phase
    prehold_align_angle_servo           : bool         = True  # Enables the prehold fingertip-line angle servo
    inpocket_arm_hold                   : bool         = True  # Holds the arm target once the hand reaches the in-pocket pose
    finger_unlock_requires_arm_hold     : bool         = False  # Requires the arm hold gate before active finger unlock
    finger_unlock_min                   : float | None = None  # Sets the minimum unlock fraction for scripted finger motion
    finger_arm_hold_fallback            : bool | None  = None  # Controls fallback arm holding when finger unlock state is ambiguous
    finger_requires_center              : bool | None  = None  # Requires finger-centering before scripted closure begins

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "ARM_CONTROLLER": self.arm_controller,  # Exports ARM_CONTROLLER from the arm controller setting
                "TEACHER_ARM_SOURCE": self.teacher_arm_source,  # Exports TEACHER_ARM_SOURCE from the teacher arm source setting
                "ASSIST_NOISE_CLEAN_BC_TARGET": bool01(self.assist_noise_clean_bc_target),  # Exports ASSIST_NOISE_CLEAN_BC_TARGET as legacy 0 or 1 from the assist noise clean BC target setting
                "BC_TARGET_INCLUDES_INPOCKET_ARM_HOLD": bool01(self.bc_target_includes_inpocket_arm_hold),  # Exports BC_TARGET_INCLUDES_INPOCKET_ARM_HOLD as legacy 0 or 1 from the BC target includes inpocket arm hold setting
                "TOPDOWN_CONTACT_TEACHER": bool01(self.contact_teacher),  # Exports TOPDOWN_CONTACT_TEACHER as legacy 0 or 1 from the contact teacher setting
                "TOPDOWN_CONTACT_TEACHER_CLOSE_RATE": self.close_rate,  # Exports TOPDOWN_CONTACT_TEACHER_CLOSE_RATE from the close rate setting
                "TOPDOWN_CONTACT_TEACHER_START_FRACTION": self.start_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_START_FRACTION from the start fraction setting
                "TOPDOWN_CONTACT_TEACHER_MIDDLE_SCALE": self.middle_scale,  # Exports TOPDOWN_CONTACT_TEACHER_MIDDLE_SCALE from the middle scale setting
                "TOPDOWN_CONTACT_TEACHER_BYPASS_UNLOCK": bool01(self.bypass_unlock),  # Exports TOPDOWN_CONTACT_TEACHER_BYPASS_UNLOCK as legacy 0 or 1 from the bypass unlock setting
                "TOPDOWN_CONTACT_TEACHER_DRIVE_UNTIL_LIFT_LATCH": bool01(self.drive_until_lift_latch),  # Exports TOPDOWN_CONTACT_TEACHER_DRIVE_UNTIL_LIFT_LATCH as legacy 0 or 1 from the drive until lift latch setting
                "TOPDOWN_CONTACT_TEACHER_DRIVE_AFTER_LIFT_LATCH": bool01(self.drive_after_lift_latch),  # Exports TOPDOWN_CONTACT_TEACHER_DRIVE_AFTER_LIFT_LATCH as legacy 0 or 1 from the drive after lift latch setting
                "TOPDOWN_PREHOLD_IK_POSITION_ONLY": bool01(self.prehold_ik_position_only),  # Exports TOPDOWN_PREHOLD_IK_POSITION_ONLY as legacy 0 or 1 from the prehold IK position only setting
                "TOPDOWN_PREHOLD_ALIGN_ANGLE_SERVO": bool01(self.prehold_align_angle_servo),  # Exports TOPDOWN_PREHOLD_ALIGN_ANGLE_SERVO as legacy 0 or 1 from the prehold align angle servo setting
                "INPOCKET_ARM_HOLD": bool01(self.inpocket_arm_hold),  # Exports INPOCKET_ARM_HOLD as legacy 0 or 1 from the inpocket arm hold setting
                "FINGER_UNLOCK_REQUIRES_ARM_HOLD": bool01(self.finger_unlock_requires_arm_hold),  # Exports FINGER_UNLOCK_REQUIRES_ARM_HOLD as legacy 0 or 1 from the finger unlock requires arm hold setting
                "TOPDOWN_CONTACT_TEACHER_FINGER_UNLOCK_MIN": self.finger_unlock_min,  # Exports TOPDOWN_CONTACT_TEACHER_FINGER_UNLOCK_MIN from the finger unlock minimum setting
                "TOPDOWN_CONTACT_TEACHER_FINGER_ARM_HOLD_FALLBACK": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_FINGER_ARM_HOLD_FALLBACK
                    bool01(self.finger_arm_hold_fallback)  # Converts the finger arm hold fallback setting to legacy 0 or 1 text
                    if self.finger_arm_hold_fallback is not None  # Checks whether optional finger arm hold fallback override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_FINGER_REQUIRES_CENTER": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_FINGER_REQUIRES_CENTER
                    bool01(self.finger_requires_center)  # Converts the finger requires center setting to legacy 0 or 1 text
                    if self.finger_requires_center is not None  # Checks whether optional finger requires center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        add_arg(args, "--arm-controller", self.arm_controller)  # adds a scalar trainer CLI option
        add_arg(args, "--teacher-arm-source", self.teacher_arm_source)  # adds a scalar trainer CLI option
        add_arg(args, "--assist-noise-clean-bc-target", int(self.assist_noise_clean_bc_target))  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-close-rate", self.close_rate)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-start-fraction", self.start_fraction)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-middle-scale", self.middle_scale)  # adds a scalar trainer CLI option
        add_flag(args, self.contact_teacher, "--topdown-contact-teacher")  # adds the trainer CLI flag when enabled
        add_flag(args, self.bypass_unlock, "--topdown-contact-teacher-bypass-unlock")  # adds the trainer CLI flag when enabled
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class TeacherPreholdConfig:  # defines the teacher prehold config group
    """Closed-loop IK corrections that move the open hand into the topdown pocket.

    These are legacy-compatible servos layered around the main palm IK.  They
    are intentionally small/gated: the goal is to remove centimeter-scale
    centering and line-angle errors before descent, not to replace the main
    task-space solve or fight the contact teacher after closure starts.
    """

    with_contact_teacher       : bool  = True  # Controls whether with contact teacher is enabled
    tip_jacobian_ik            : bool  = False  # Controls whether tip jacobian IK is enabled
    tip_jacobian_stage_min     : int   = 1  # Sets the first stage where tip jacobian applies
    tip_jacobian_gain          : float = 1.0  # Sets servo gain for tip jacobian
    tip_jacobian_damping       : float = 0.05  # Sets damping for tip jacobian
    tip_jacobian_max_joint_step: float = 0.035  # Sets the tip jacobian maximum joint step config value
    tip_jacobian_xy_weight     : float = 1.0  # Sets optimization weight for tip jacobian XY
    tip_jacobian_z_weight      : float = 1.25  # Sets optimization weight for tip jacobian Z
    tip_jacobian_use_middle    : bool  = True  # Controls whether tip jacobian use middle is enabled
    tip_jacobian_joints        : str   = "base"  # Selects the tip jacobian joints behavior
    align_angle_stage_min      : int   = 1  # Sets the first stage where align angle applies
    align_angle_gain           : float = 0.75  # Sets the align angle gain angular threshold
    align_angle_max_dz         : float = 0.025  # Sets the align angle maximum dz angular threshold
    align_angle_max_joint_step : float = 0.08  # Sets the align angle maximum joint step angular threshold
    planar_align_servo         : bool  = True  # Controls whether planar align servo is enabled
    planar_align_stage_min     : int   = 1  # Sets the first stage where planar align applies
    planar_align_gain          : float = 0.75  # Sets servo gain for planar align
    planar_align_max_xy        : float = 0.050  # Sets the planar align maximum XY config value
    planar_align_max_joint_step: float = 0.08  # Sets the planar align maximum joint step config value
    ik_tip_servo_stage_min     : int   = 1  # Sets the first stage where IK tip servo applies
    ik_tip_servo_gain          : float = 0.65  # Sets servo gain for IK tip servo
    ik_tip_servo_max_m         : float = 0.080  # Sets the IK tip servo maximum m distance in meters
    pocket_sweep               : bool  = True  # Controls whether pocket sweep is enabled
    pocket_sweep_stage_min     : int   = 2  # Sets the first stage where pocket sweep applies
    pocket_sweep_iters         : int   = 1  # Sets iteration count for pocket sweep

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TOPDOWN_PREHOLD_IK_WITH_CONTACT_TEACHER": bool01(self.with_contact_teacher),  # Exports TOPDOWN_PREHOLD_IK_WITH_CONTACT_TEACHER as legacy 0 or 1 from the with contact teacher setting
                "TOPDOWN_TIP_JACOBIAN_IK": bool01(self.tip_jacobian_ik),  # Exports TOPDOWN_TIP_JACOBIAN_IK as legacy 0 or 1 from the tip jacobian IK setting
                "TOPDOWN_TIP_JACOBIAN_IK_STAGE_MIN": self.tip_jacobian_stage_min,  # Exports TOPDOWN_TIP_JACOBIAN_IK_STAGE_MIN from the tip jacobian stage minimum setting
                "TOPDOWN_TIP_JACOBIAN_IK_GAIN": self.tip_jacobian_gain,  # Exports TOPDOWN_TIP_JACOBIAN_IK_GAIN from the tip jacobian gain setting
                "TOPDOWN_TIP_JACOBIAN_IK_DAMPING": self.tip_jacobian_damping,  # Exports TOPDOWN_TIP_JACOBIAN_IK_DAMPING from the tip jacobian damping setting
                "TOPDOWN_TIP_JACOBIAN_IK_MAX_JOINT_STEP": self.tip_jacobian_max_joint_step,  # Exports TOPDOWN_TIP_JACOBIAN_IK_MAX_JOINT_STEP from the tip jacobian maximum joint step setting
                "TOPDOWN_TIP_JACOBIAN_IK_XY_WEIGHT": self.tip_jacobian_xy_weight,  # Exports TOPDOWN_TIP_JACOBIAN_IK_XY_WEIGHT from the tip jacobian XY weight setting
                "TOPDOWN_TIP_JACOBIAN_IK_Z_WEIGHT": self.tip_jacobian_z_weight,  # Exports TOPDOWN_TIP_JACOBIAN_IK_Z_WEIGHT from the tip jacobian Z weight setting
                "TOPDOWN_TIP_JACOBIAN_IK_USE_MIDDLE": bool01(self.tip_jacobian_use_middle),  # Exports TOPDOWN_TIP_JACOBIAN_IK_USE_MIDDLE as legacy 0 or 1 from the tip jacobian use middle setting
                "TOPDOWN_TIP_JACOBIAN_IK_JOINTS": self.tip_jacobian_joints,  # Exports TOPDOWN_TIP_JACOBIAN_IK_JOINTS from the tip jacobian joints setting
                "TOPDOWN_PREHOLD_ALIGN_ANGLE_STAGE_MIN": self.align_angle_stage_min,  # Exports TOPDOWN_PREHOLD_ALIGN_ANGLE_STAGE_MIN from the align angle stage minimum setting
                "TOPDOWN_PREHOLD_ALIGN_ANGLE_GAIN": self.align_angle_gain,  # Exports TOPDOWN_PREHOLD_ALIGN_ANGLE_GAIN from the align angle gain setting
                "TOPDOWN_PREHOLD_ALIGN_ANGLE_MAX_DZ": self.align_angle_max_dz,  # Exports TOPDOWN_PREHOLD_ALIGN_ANGLE_MAX_DZ from the align angle maximum dz setting
                "TOPDOWN_PREHOLD_ALIGN_ANGLE_MAX_JOINT_STEP": self.align_angle_max_joint_step,  # Exports TOPDOWN_PREHOLD_ALIGN_ANGLE_MAX_JOINT_STEP from the align angle maximum joint step setting
                "TOPDOWN_PREHOLD_PLANAR_ALIGN_SERVO": bool01(self.planar_align_servo),  # Exports TOPDOWN_PREHOLD_PLANAR_ALIGN_SERVO as legacy 0 or 1 from the planar align servo setting
                "TOPDOWN_PREHOLD_PLANAR_ALIGN_STAGE_MIN": self.planar_align_stage_min,  # Exports TOPDOWN_PREHOLD_PLANAR_ALIGN_STAGE_MIN from the planar align stage minimum setting
                "TOPDOWN_PREHOLD_PLANAR_ALIGN_GAIN": self.planar_align_gain,  # Exports TOPDOWN_PREHOLD_PLANAR_ALIGN_GAIN from the planar align gain setting
                "TOPDOWN_PREHOLD_PLANAR_ALIGN_MAX_XY": self.planar_align_max_xy,  # Exports TOPDOWN_PREHOLD_PLANAR_ALIGN_MAX_XY from the planar align maximum XY setting
                "TOPDOWN_PREHOLD_PLANAR_ALIGN_MAX_JOINT_STEP": self.planar_align_max_joint_step,  # Exports TOPDOWN_PREHOLD_PLANAR_ALIGN_MAX_JOINT_STEP from the planar align maximum joint step setting
                "TOPDOWN_PREHOLD_IK_TIP_SERVO_STAGE_MIN": self.ik_tip_servo_stage_min,  # Exports TOPDOWN_PREHOLD_IK_TIP_SERVO_STAGE_MIN from the IK tip servo stage minimum setting
                "TOPDOWN_PREHOLD_IK_TIP_SERVO_GAIN": self.ik_tip_servo_gain,  # Exports TOPDOWN_PREHOLD_IK_TIP_SERVO_GAIN from the IK tip servo gain setting
                "TOPDOWN_PREHOLD_IK_TIP_SERVO_MAX_M": self.ik_tip_servo_max_m,  # Exports TOPDOWN_PREHOLD_IK_TIP_SERVO_MAX_M from the IK tip servo maximum m setting
                "TOPDOWN_POCKET_SWEEP": bool01(self.pocket_sweep),  # Exports TOPDOWN_POCKET_SWEEP as legacy 0 or 1 from the pocket sweep setting
                "TOPDOWN_POCKET_SWEEP_STAGE_MIN": self.pocket_sweep_stage_min,  # Exports TOPDOWN_POCKET_SWEEP_STAGE_MIN from the pocket sweep stage minimum setting
                "TOPDOWN_POCKET_SWEEP_ITERS": self.pocket_sweep_iters,  # Exports TOPDOWN_POCKET_SWEEP_ITERS from the pocket sweep iterations setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class TeacherPreholdAdvancedConfig:  # defines the teacher prehold advanced config group
    """Narrow extra knobs for IK experiments without widening the base config.

    Most fields are optional.  ``None`` means "do not emit an environment
    override" so older profiles keep their historical behavior.  This is where
    fragile experiment knobs live: palm basis choice, face-axis convention,
    xyz-front close gate scaling, and which joints a prehold servo may touch.
    """

    tip_jacobian_respect_arm_hold        : bool | None  = None  # Controls whether tip jacobian respect arm hold is enabled
    tip_jacobian_accept_worse            : bool | None  = None  # Controls whether tip jacobian accept worse is enabled
    tip_jacobian_max_worse_m             : float | None = None  # Sets the tip jacobian maximum worse m distance in meters
    tip_jacobian_z_requires_center       : bool | None  = None  # Controls whether tip jacobian Z requires center is enabled
    tip_jacobian_mode                    : str | None   = None  # Selects the tip jacobian mode behavior
    tip_jacobian_center_xy_weight        : float | None = None  # Sets optimization weight for tip jacobian center XY
    tip_jacobian_center_z_weight         : float | None = None  # Sets optimization weight for tip jacobian center Z
    tip_jacobian_span_xy_weight          : float | None = None  # Sets optimization weight for tip jacobian span XY
    tip_jacobian_span_z_weight           : float | None = None  # Sets optimization weight for tip jacobian span Z
    tip_jacobian_center_z_requires_center: bool | None  = None  # Controls whether tip jacobian center Z requires center is enabled
    tip_servo_z_requires_center          : bool | None  = None  # Controls whether tip servo Z requires center is enabled
    position_only_stage_min              : int | None   = None  # Sets the first stage where position only applies
    align_angle_joints                   : str | None   = None  # Sets the align angle joints angular threshold
    planar_align_joints                  : str | None   = None  # Selects the planar align joints behavior
    target_palm_basis                    : str | None   = None  # Selects the target palm basis behavior
    target_grip_finger_model             : str | None   = None  # Selects the target grip finger model behavior
    target_palm_yaw_world_axis           : str | None   = None  # Sets the target palm yaw world axis angular threshold
    grip_face_axis                       : str | None   = None  # Selects the grip face axis behavior
    target_palm_position_mode            : str | None   = None  # Selects the target palm position mode behavior
    palm_local_grip_offset_mode          : str | None   = None  # Sets the palm local grip offset mode distance in meters
    finger_close_gate_mode               : str | None   = None  # Selects the finger close gate mode behavior
    finger_xyz_gate_start_m              : float | None = None  # Sets the finger XYZ gate start m distance in meters
    finger_xyz_gate_full_m               : float | None = None  # Sets the finger XYZ gate full m distance in meters
    finger_xyz_gate_linear               : bool | None  = None  # Controls whether finger XYZ gate linear is enabled
    finger_front_face_tolerance_m        : float | None = None  # Sets the finger front face tolerance m distance in meters
    topdown_palm_offset_x                : float | None = None  # Sets the topdown palm offset X distance in meters
    topdown_palm_offset_y                : float | None = None  # Sets the topdown palm offset Y distance in meters

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TOPDOWN_TIP_JACOBIAN_IK_RESPECT_ARM_HOLD": (  # Starts env export expression for TOPDOWN_TIP_JACOBIAN_IK_RESPECT_ARM_HOLD
                    bool01(self.tip_jacobian_respect_arm_hold)  # Converts the tip jacobian respect arm hold setting to legacy 0 or 1 text
                    if self.tip_jacobian_respect_arm_hold is not None  # Checks whether optional tip jacobian respect arm hold override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_TIP_JACOBIAN_IK_ACCEPT_WORSE": (  # Starts env export expression for TOPDOWN_TIP_JACOBIAN_IK_ACCEPT_WORSE
                    bool01(self.tip_jacobian_accept_worse)  # Converts the tip jacobian accept worse setting to legacy 0 or 1 text
                    if self.tip_jacobian_accept_worse is not None  # Checks whether optional tip jacobian accept worse override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_TIP_JACOBIAN_IK_MAX_WORSE_M": self.tip_jacobian_max_worse_m,  # Exports TOPDOWN_TIP_JACOBIAN_IK_MAX_WORSE_M from the tip jacobian maximum worse m setting
                "TOPDOWN_TIP_JACOBIAN_IK_Z_REQUIRES_CENTER": (  # Starts env export expression for TOPDOWN_TIP_JACOBIAN_IK_Z_REQUIRES_CENTER
                    bool01(self.tip_jacobian_z_requires_center)  # Converts the tip jacobian Z requires center setting to legacy 0 or 1 text
                    if self.tip_jacobian_z_requires_center is not None  # Checks whether optional tip jacobian Z requires center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_TIP_JACOBIAN_IK_MODE": self.tip_jacobian_mode,  # Exports TOPDOWN_TIP_JACOBIAN_IK_MODE from the tip jacobian mode setting
                "TOPDOWN_TIP_JACOBIAN_IK_CENTER_XY_WEIGHT": self.tip_jacobian_center_xy_weight,  # Exports TOPDOWN_TIP_JACOBIAN_IK_CENTER_XY_WEIGHT from the tip jacobian center XY weight setting
                "TOPDOWN_TIP_JACOBIAN_IK_CENTER_Z_WEIGHT": self.tip_jacobian_center_z_weight,  # Exports TOPDOWN_TIP_JACOBIAN_IK_CENTER_Z_WEIGHT from the tip jacobian center Z weight setting
                "TOPDOWN_TIP_JACOBIAN_IK_SPAN_XY_WEIGHT": self.tip_jacobian_span_xy_weight,  # Exports TOPDOWN_TIP_JACOBIAN_IK_SPAN_XY_WEIGHT from the tip jacobian span XY weight setting
                "TOPDOWN_TIP_JACOBIAN_IK_SPAN_Z_WEIGHT": self.tip_jacobian_span_z_weight,  # Exports TOPDOWN_TIP_JACOBIAN_IK_SPAN_Z_WEIGHT from the tip jacobian span Z weight setting
                "TOPDOWN_TIP_JACOBIAN_IK_CENTER_Z_REQUIRES_CENTER": (  # Starts env export expression for TOPDOWN_TIP_JACOBIAN_IK_CENTER_Z_REQUIRES_CENTER
                    bool01(self.tip_jacobian_center_z_requires_center)  # Converts the tip jacobian center Z requires center setting to legacy 0 or 1 text
                    if self.tip_jacobian_center_z_requires_center is not None  # Checks whether optional tip jacobian center Z requires center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_PREHOLD_IK_TIP_SERVO_Z_REQUIRES_CENTER": (  # Starts env export expression for TOPDOWN_PREHOLD_IK_TIP_SERVO_Z_REQUIRES_CENTER
                    bool01(self.tip_servo_z_requires_center)  # Converts the tip servo Z requires center setting to legacy 0 or 1 text
                    if self.tip_servo_z_requires_center is not None  # Checks whether optional tip servo Z requires center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_PREHOLD_IK_POSITION_ONLY_STAGE_MIN": self.position_only_stage_min,  # Exports TOPDOWN_PREHOLD_IK_POSITION_ONLY_STAGE_MIN from the position only stage minimum setting
                "TOPDOWN_PREHOLD_ALIGN_ANGLE_JOINTS": self.align_angle_joints,  # Exports TOPDOWN_PREHOLD_ALIGN_ANGLE_JOINTS from the align angle joints setting
                "TOPDOWN_PREHOLD_PLANAR_ALIGN_JOINTS": self.planar_align_joints,  # Exports TOPDOWN_PREHOLD_PLANAR_ALIGN_JOINTS from the planar align joints setting
                "TOPDOWN_TARGET_PALM_BASIS": self.target_palm_basis,  # Exports TOPDOWN_TARGET_PALM_BASIS from the target palm basis setting
                "TOPDOWN_GRIP_FINGER_MODEL": self.target_grip_finger_model,  # Exports TOPDOWN_GRIP_FINGER_MODEL from the target grip finger model setting
                "TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS": self.target_palm_yaw_world_axis,  # Exports TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS from the target palm yaw world axis setting
                "TOPDOWN_GRIP_FACE_AXIS": self.grip_face_axis,  # Exports TOPDOWN_GRIP_FACE_AXIS from the grip face axis setting
                "TOPDOWN_TARGET_PALM_POSITION_MODE": self.target_palm_position_mode,  # Exports TOPDOWN_TARGET_PALM_POSITION_MODE from the target palm position mode setting
                "TOPDOWN_PALM_LOCAL_GRIP_OFFSET_MODE": self.palm_local_grip_offset_mode,  # Exports TOPDOWN_PALM_LOCAL_GRIP_OFFSET_MODE from the palm local grip offset mode setting
                "TOPDOWN_FINGER_CLOSE_GATE_MODE": self.finger_close_gate_mode,  # Exports TOPDOWN_FINGER_CLOSE_GATE_MODE from the finger close gate mode setting
                "TOPDOWN_FINGER_XYZ_GATE_START_M": self.finger_xyz_gate_start_m,  # Exports TOPDOWN_FINGER_XYZ_GATE_START_M from the finger XYZ gate start m setting
                "TOPDOWN_FINGER_XYZ_GATE_FULL_M": self.finger_xyz_gate_full_m,  # Exports TOPDOWN_FINGER_XYZ_GATE_FULL_M from the finger XYZ gate full m setting
                "TOPDOWN_FINGER_XYZ_GATE_LINEAR": (  # Starts env export expression for TOPDOWN_FINGER_XYZ_GATE_LINEAR
                    bool01(self.finger_xyz_gate_linear)  # Converts the finger XYZ gate linear setting to legacy 0 or 1 text
                    if self.finger_xyz_gate_linear is not None  # Checks whether optional finger XYZ gate linear override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_FINGER_FRONT_FACE_TOLERANCE_M": self.finger_front_face_tolerance_m,  # Exports TOPDOWN_FINGER_FRONT_FACE_TOLERANCE_M from the finger front face tolerance m setting
                "TEACHER_TOPDOWN_PALM_OFFSET_X": self.topdown_palm_offset_x,  # Exports TEACHER_TOPDOWN_PALM_OFFSET_X from the topdown palm offset X setting
                "TEACHER_TOPDOWN_PALM_OFFSET_Y": self.topdown_palm_offset_y,  # Exports TEACHER_TOPDOWN_PALM_OFFSET_Y from the topdown palm offset Y setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class TaskSpaceIKConfig:  # defines the task space IK config group
    """Constrained grasp-frame IK knobs for the topdown teacher.

    The DLS system stacks several feature tasks with different units:

    * grip-center position controls block-centered approach,
    * span vector controls thumb/index face alignment,
    * angular palm rows gently control orientation,
    * posture weight keeps the arm near a stable "L" rather than waving.

    The weights here therefore define a *priority blend*, not reward weights.
    Small orientation/posture values can be better than strict palm-down when
    strict orientation would make the wrist collide with the block/camera.
    """

    enabled                           : bool         = False  # Controls whether enabled is enabled
    direct_grip_center                : bool         = False  # Controls whether direct grip center is enabled
    grip_offset_live_start_fraction   : float | None = None  # Sets the grip offset live start fraction distance in meters
    grip_offset_live_full_fraction    : float | None = None  # Sets the grip offset live full fraction distance in meters
    grip_offset_blend_requires_descent: bool | None  = None  # Controls whether grip offset blend requires descent is enabled
    center_xy_weight                  : float        = 4.0  # Sets optimization weight for center XY
    center_z_weight                   : float        = 1.0  # Sets optimization weight for center Z
    span_xy_weight                    : float        = 1.0  # Sets optimization weight for span XY
    span_z_weight                     : float        = 4.0  # Sets optimization weight for span Z
    drop_weight                       : float        = 2.0  # Sets optimization weight for drop
    orientation_weight                : float | None = None  # Sets the orientation weight angular threshold
    orientation_sign                  : float        = 1.0  # Sets the orientation sign angular threshold
    posture_weight                    : float        = 0.04  # Sets optimization weight for posture
    prehover_span_scale               : float | None = None  # Sets scale factor for prehover span
    prehover_orientation_scale        : float | None = None  # Sets the prehover orientation scale angular threshold
    prehover_posture_weight           : float | None = None  # Sets optimization weight for prehover posture
    lift_span_scale                   : float | None = None  # Sets scale factor for lift span
    lift_orientation_scale            : float | None = None  # Sets the lift orientation scale angular threshold
    lift_posture_weight               : float | None = None  # Sets optimization weight for lift posture
    damping                           : float        = 0.045  # Sets the damping config value
    max_joint_step                    : float        = 0.075  # Sets the max joint step config value

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TOPDOWN_TASK_SPACE_IK": bool01(self.enabled),  # Exports TOPDOWN_TASK_SPACE_IK as legacy 0 or 1 from the enabled setting
                "TOPDOWN_TASK_SPACE_IK_DIRECT_GRIP_CENTER": bool01(self.direct_grip_center),  # Exports TOPDOWN_TASK_SPACE_IK_DIRECT_GRIP_CENTER as legacy 0 or 1 from the direct grip center setting
                "TOPDOWN_PALM_LOCAL_GRIP_OFFSET_LIVE_START_FRACTION": self.grip_offset_live_start_fraction,  # Exports TOPDOWN_PALM_LOCAL_GRIP_OFFSET_LIVE_START_FRACTION from the grip offset live start fraction setting
                "TOPDOWN_PALM_LOCAL_GRIP_OFFSET_LIVE_FULL_FRACTION": self.grip_offset_live_full_fraction,  # Exports TOPDOWN_PALM_LOCAL_GRIP_OFFSET_LIVE_FULL_FRACTION from the grip offset live full fraction setting
                "TOPDOWN_PALM_LOCAL_GRIP_OFFSET_BLEND_REQUIRES_DESCENT": (  # Starts env export expression for TOPDOWN_PALM_LOCAL_GRIP_OFFSET_BLEND_REQUIRES_DESCENT
                    bool01(self.grip_offset_blend_requires_descent)  # Converts the grip offset blend requires descent setting to legacy 0 or 1 text
                    if self.grip_offset_blend_requires_descent is not None  # Checks whether optional grip offset blend requires descent override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_TASK_SPACE_IK_CENTER_XY_WEIGHT": self.center_xy_weight,  # Exports TOPDOWN_TASK_SPACE_IK_CENTER_XY_WEIGHT from the center XY weight setting
                "TOPDOWN_TASK_SPACE_IK_CENTER_Z_WEIGHT": self.center_z_weight,  # Exports TOPDOWN_TASK_SPACE_IK_CENTER_Z_WEIGHT from the center Z weight setting
                "TOPDOWN_TASK_SPACE_IK_SPAN_XY_WEIGHT": self.span_xy_weight,  # Exports TOPDOWN_TASK_SPACE_IK_SPAN_XY_WEIGHT from the span XY weight setting
                "TOPDOWN_TASK_SPACE_IK_SPAN_Z_WEIGHT": self.span_z_weight,  # Exports TOPDOWN_TASK_SPACE_IK_SPAN_Z_WEIGHT from the span Z weight setting
                "TOPDOWN_TASK_SPACE_IK_DROP_WEIGHT": self.drop_weight,  # Exports TOPDOWN_TASK_SPACE_IK_DROP_WEIGHT from the drop weight setting
                "TOPDOWN_TASK_SPACE_IK_ORIENTATION_WEIGHT": self.orientation_weight,  # Exports TOPDOWN_TASK_SPACE_IK_ORIENTATION_WEIGHT from the orientation weight setting
                "TOPDOWN_TASK_SPACE_IK_ORIENTATION_SIGN": self.orientation_sign,  # Exports TOPDOWN_TASK_SPACE_IK_ORIENTATION_SIGN from the orientation sign setting
                "TOPDOWN_TASK_SPACE_IK_POSTURE_WEIGHT": self.posture_weight,  # Exports TOPDOWN_TASK_SPACE_IK_POSTURE_WEIGHT from the posture weight setting
                "TOPDOWN_TASK_SPACE_IK_PREHOVER_SPAN_SCALE": self.prehover_span_scale,  # Exports TOPDOWN_TASK_SPACE_IK_PREHOVER_SPAN_SCALE from the prehover span scale setting
                "TOPDOWN_TASK_SPACE_IK_PREHOVER_ORIENTATION_SCALE": (  # Starts env export expression for TOPDOWN_TASK_SPACE_IK_PREHOVER_ORIENTATION_SCALE
                    self.prehover_orientation_scale  # Passes the prehover orientation scale setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_TASK_SPACE_IK_PREHOVER_POSTURE_WEIGHT": (  # Starts env export expression for TOPDOWN_TASK_SPACE_IK_PREHOVER_POSTURE_WEIGHT
                    self.prehover_posture_weight  # Passes the prehover posture weight setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_TASK_SPACE_IK_LIFT_SPAN_SCALE": self.lift_span_scale,  # Exports TOPDOWN_TASK_SPACE_IK_LIFT_SPAN_SCALE from the lift span scale setting
                "TOPDOWN_TASK_SPACE_IK_LIFT_ORIENTATION_SCALE": self.lift_orientation_scale,  # Exports TOPDOWN_TASK_SPACE_IK_LIFT_ORIENTATION_SCALE from the lift orientation scale setting
                "TOPDOWN_TASK_SPACE_IK_LIFT_POSTURE_WEIGHT": self.lift_posture_weight,  # Exports TOPDOWN_TASK_SPACE_IK_LIFT_POSTURE_WEIGHT from the lift posture weight setting
                "TOPDOWN_TASK_SPACE_IK_DAMPING": self.damping,  # Exports TOPDOWN_TASK_SPACE_IK_DAMPING from the damping setting
                "TOPDOWN_TASK_SPACE_IK_MAX_JOINT_STEP": self.max_joint_step,  # Exports TOPDOWN_TASK_SPACE_IK_MAX_JOINT_STEP from the max joint step setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class TeacherLiftConfig:  # defines the teacher lift config group
    """Teacher details that are only needed once lift physics is enabled."""

    teacher_lift_z                    : float        = 0.05  # Sets the teacher lift Z config value
    teacher_lift_ramp_steps           : int          = 60  # Sets the number of steps for teacher lift ramp
    max_fraction                      : float        = 0.85  # Sets the max fraction config value
    descent_z                         : float        = 0.045  # Sets the descent Z config value
    hold_on_contact                   : bool         = True  # Controls whether hold on contact is enabled
    hold_extra_fraction               : float        = 0.08  # Sets the hold extra fraction config value
    hold_max_fraction                 : float        = 0.85  # Sets the hold maximum fraction config value
    prelift_squeeze_fraction          : float        = -1.0  # Sets the prelift squeeze fraction config value
    lift_squeeze_fraction             : float        = 0.85  # Sets the lift squeeze fraction config value
    freeze_finger_fraction_at_latch   : bool | None  = None  # Controls whether freeze finger fraction at latch is enabled
    lift_finger_freeze_extra_fraction : float | None = None  # Sets the lift finger freeze extra fraction config value
    lift_finger_freeze_max_fraction   : float | None = None  # Sets the lift finger freeze maximum fraction config value
    missing_contact_extra_descent     : float        = 0.020  # Sets the missing contact extra descent config value
    inward_m                          : float        = 0.035  # Sets the inward m distance in meters
    missing_contact_extra_inward      : float        = 0.020  # Sets the missing contact extra inward config value
    descent_tip_z_target              : float        = 0.004  # Sets the descent tip Z target config value
    descent_requires_wrist_yaw_release: bool         = False  # Controls whether descent requires wrist yaw release is enabled
    descent_requires_center           : bool         = False  # Controls whether descent requires center is enabled
    descent_uses_center_ready         : bool | None  = None  # Controls whether descent uses center ready is enabled
    inward_requires_center            : bool         = False  # Controls whether inward requires center is enabled
    inward_vertical_only              : bool         = False  # Controls whether inward vertical only is enabled
    vertical_drop_lock_xy             : bool         = False  # Controls whether vertical drop lock XY is enabled
    descent_tip_servo_xy_max_m        : float | None = None  # Sets the descent tip servo XY maximum m distance in meters
    descent_min_closure_fraction      : float | None = None  # Sets the descent minimum closure fraction config value
    descent_full_closure_fraction     : float | None = None  # Sets the descent full closure fraction config value
    descent_uses_z_need               : bool | None  = None  # Controls whether descent uses Z need is enabled
    pre_descent_hover_height_max      : float | None = None  # Sets the maximum allowed pre descent hover height

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TEACHER_LIFT_Z": self.teacher_lift_z,  # Exports TEACHER_LIFT_Z from the teacher lift Z setting
                "TEACHER_LIFT_RAMP_STEPS": self.teacher_lift_ramp_steps,  # Exports TEACHER_LIFT_RAMP_STEPS from the teacher lift ramp steps setting
                "TOPDOWN_CONTACT_TEACHER_MAX_FRACTION": self.max_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_MAX_FRACTION from the max fraction setting
                "TOPDOWN_CONTACT_TEACHER_DESCENT_Z": self.descent_z,  # Exports TOPDOWN_CONTACT_TEACHER_DESCENT_Z from the descent Z setting
                "TOPDOWN_CONTACT_TEACHER_HOLD_ON_CONTACT": bool01(self.hold_on_contact),  # Exports TOPDOWN_CONTACT_TEACHER_HOLD_ON_CONTACT as legacy 0 or 1 from the hold on contact setting
                "TOPDOWN_CONTACT_TEACHER_HOLD_EXTRA_FRACTION": self.hold_extra_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_HOLD_EXTRA_FRACTION from the hold extra fraction setting
                "TOPDOWN_CONTACT_TEACHER_HOLD_MAX_FRACTION": self.hold_max_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_HOLD_MAX_FRACTION from the hold maximum fraction setting
                "TOPDOWN_CONTACT_TEACHER_PRELIFT_SQUEEZE_FRACTION": self.prelift_squeeze_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_PRELIFT_SQUEEZE_FRACTION from the prelift squeeze fraction setting
                "TOPDOWN_LIFT_SQUEEZE_FRACTION": self.lift_squeeze_fraction,  # Exports TOPDOWN_LIFT_SQUEEZE_FRACTION from the lift squeeze fraction setting
                "TOPDOWN_LIFT_FREEZE_FINGER_FRACTION_AT_LATCH": (  # Starts env export expression for TOPDOWN_LIFT_FREEZE_FINGER_FRACTION_AT_LATCH
                    bool01(self.freeze_finger_fraction_at_latch)  # Converts the freeze finger fraction at latch setting to legacy 0 or 1 text
                    if self.freeze_finger_fraction_at_latch is not None  # Checks whether optional freeze finger fraction at latch override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_LIFT_FINGER_FREEZE_EXTRA_FRACTION": self.lift_finger_freeze_extra_fraction,  # Exports TOPDOWN_LIFT_FINGER_FREEZE_EXTRA_FRACTION from the lift finger freeze extra fraction setting
                "TOPDOWN_LIFT_FINGER_FREEZE_MAX_FRACTION": self.lift_finger_freeze_max_fraction,  # Exports TOPDOWN_LIFT_FINGER_FREEZE_MAX_FRACTION from the lift finger freeze maximum fraction setting
                "TOPDOWN_CONTACT_TEACHER_MISSING_CONTACT_EXTRA_DESCENT": self.missing_contact_extra_descent,  # Exports TOPDOWN_CONTACT_TEACHER_MISSING_CONTACT_EXTRA_DESCENT from the missing contact extra descent setting
                "TOPDOWN_CONTACT_TEACHER_INWARD_M": self.inward_m,  # Exports TOPDOWN_CONTACT_TEACHER_INWARD_M from the inward m setting
                "TOPDOWN_CONTACT_TEACHER_MISSING_CONTACT_EXTRA_INWARD": self.missing_contact_extra_inward,  # Exports TOPDOWN_CONTACT_TEACHER_MISSING_CONTACT_EXTRA_INWARD from the missing contact extra inward setting
                "TOPDOWN_CONTACT_TEACHER_DESCENT_TIP_Z_TARGET": self.descent_tip_z_target,  # Exports TOPDOWN_CONTACT_TEACHER_DESCENT_TIP_Z_TARGET from the descent tip Z target setting
                "TOPDOWN_CONTACT_TEACHER_DESCENT_REQUIRES_WRIST_YAW_RELEASE": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_DESCENT_REQUIRES_WRIST_YAW_RELEASE
                    self.descent_requires_wrist_yaw_release  # Passes the descent requires wrist yaw release setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_DESCENT_REQUIRES_CENTER": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_DESCENT_REQUIRES_CENTER
                    self.descent_requires_center  # Passes the descent requires center setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_DESCENT_USES_CENTER_READY": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_DESCENT_USES_CENTER_READY
                    bool01(self.descent_uses_center_ready)  # Converts the descent uses center ready setting to legacy 0 or 1 text
                    if self.descent_uses_center_ready is not None  # Checks whether optional descent uses center ready override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_INWARD_REQUIRES_CENTER": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_INWARD_REQUIRES_CENTER
                    self.inward_requires_center  # Passes the inward requires center setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_INWARD_VERTICAL_ONLY": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_INWARD_VERTICAL_ONLY
                    self.inward_vertical_only  # Passes the inward vertical only setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_VERTICAL_DROP_LOCK_XY": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_VERTICAL_DROP_LOCK_XY
                    self.vertical_drop_lock_xy  # Passes the vertical drop lock XY setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_DESCENT_TIP_SERVO_XY_MAX_M": self.descent_tip_servo_xy_max_m,  # Exports TOPDOWN_CONTACT_TEACHER_DESCENT_TIP_SERVO_XY_MAX_M from the descent tip servo XY maximum m setting
                "TOPDOWN_CONTACT_TEACHER_DESCENT_MIN_CLOSURE_FRACTION": self.descent_min_closure_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_DESCENT_MIN_CLOSURE_FRACTION from the descent minimum closure fraction setting
                "TOPDOWN_CONTACT_TEACHER_DESCENT_FULL_CLOSURE_FRACTION": self.descent_full_closure_fraction,  # Exports TOPDOWN_CONTACT_TEACHER_DESCENT_FULL_CLOSURE_FRACTION from the descent full closure fraction setting
                "TOPDOWN_CONTACT_TEACHER_DESCENT_USES_Z_NEED": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_DESCENT_USES_Z_NEED
                    bool01(self.descent_uses_z_need)  # Converts the descent uses Z need setting to legacy 0 or 1 text
                    if self.descent_uses_z_need is not None  # Checks whether optional descent uses Z need override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_PRE_DESCENT_HOVER_HEIGHT_MAX": self.pre_descent_hover_height_max,  # Exports TOPDOWN_CONTACT_TEACHER_PRE_DESCENT_HOVER_HEIGHT_MAX from the pre descent hover height maximum setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        add_arg(args, "--teacher-lift-z", self.teacher_lift_z)  # adds a scalar trainer CLI option
        add_arg(args, "--teacher-lift-ramp-steps", self.teacher_lift_ramp_steps)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-max-fraction", self.max_fraction)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-descent-z", self.descent_z)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-missing-contact-extra-descent", self.missing_contact_extra_descent)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-inward-m", self.inward_m)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-contact-teacher-missing-contact-extra-inward", self.missing_contact_extra_inward)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class ActionSurfaceConfig:  # defines the action surface config group
    """Action-space and non-visual runtime mode."""

    include_wrist_roll               : bool  = True  # Controls whether include wrist roll is enabled
    include_waist_yaw                : bool  = False  # Controls whether include waist yaw is enabled
    waist_yaw_action_scale           : float = 1.0  # Sets the waist yaw action scale angular threshold
    arm_action_scale_profile         : str   = "topdown"  # Sets the arm action scale profile config value
    observation_normalization        : bool  = True  # Controls whether observation normalization is enabled
    reward_normalization             : bool  = False  # Controls whether reward normalization is enabled
    privileged_critic                : bool  = True  # Controls whether privileged critic is enabled
    soft_policy_arm_assist           : bool  = False  # Controls whether soft policy arm assist is enabled
    finger_action_mode               : str   = "absolute"  # Selects the finger action mode behavior
    finger_delta_scale               : float = 0.05  # Sets scale factor for finger delta
    contact_finger_close_cap         : float = 0.20  # Sets the contact finger close cap config value
    contact_teacher_hold_fraction_cap: float = 0.15  # Sets the contact teacher hold fraction cap config value
    contact_start_mode               : str   = "reset"  # Selects the contact start mode behavior
    topdown_preroll_fraction         : float = 0.0  # Sets the topdown preroll fraction config value
    topdown_preroll_max_steps        : int   = 250  # Sets the number of steps for topdown preroll maximum
    topdown_preroll_unlock_progress  : float = 0.0  # Sets the topdown preroll unlock progress config value
    contact_preroll_touch_mode       : str   = "off"  # Selects the contact preroll touch mode behavior
    contact_preroll_ik_descend_z     : float = 0.06  # Sets the contact preroll IK descend Z config value
    mirror_middle_to_index           : bool  = False  # Controls whether mirror middle to index is enabled
    actor_q_action_gate_mode         : str   = "env"  # Selects the actor Q action gate mode behavior
    actor_bc_action_gate_mode        : str   = "env"  # Selects the actor BC action gate mode behavior

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "INCLUDE_WRIST_ROLL": bool01(self.include_wrist_roll),  # Exports INCLUDE_WRIST_ROLL as legacy 0 or 1 from the include wrist roll setting
                "INCLUDE_WAIST_YAW": bool01(self.include_waist_yaw),  # Exports INCLUDE_WAIST_YAW as legacy 0 or 1 from the include waist yaw setting
                "WAIST_YAW_ACTION_SCALE": self.waist_yaw_action_scale,  # Exports WAIST_YAW_ACTION_SCALE from the waist yaw action scale setting
                "ARM_ACTION_SCALE_PROFILE": self.arm_action_scale_profile,  # Exports ARM_ACTION_SCALE_PROFILE from the arm action scale profile setting
                "OBSERVATION_NORMALIZATION": bool01(self.observation_normalization),  # Exports OBSERVATION_NORMALIZATION as legacy 0 or 1 from the observation normalization setting
                "REWARD_NORMALIZATION": bool01(self.reward_normalization),  # Exports REWARD_NORMALIZATION as legacy 0 or 1 from the reward normalization setting
                "PRIVILEGED_CRITIC": bool01(self.privileged_critic),  # Exports PRIVILEGED_CRITIC as legacy 0 or 1 from the privileged critic setting
                "SOFT_POLICY_ARM_ASSIST": bool01(self.soft_policy_arm_assist),  # Exports SOFT_POLICY_ARM_ASSIST as legacy 0 or 1 from the soft policy arm assist setting
                "FINGER_ACTION_MODE": self.finger_action_mode,  # Exports FINGER_ACTION_MODE from the finger action mode setting
                "FINGER_DELTA_SCALE": self.finger_delta_scale,  # Exports FINGER_DELTA_SCALE from the finger delta scale setting
                "CONTACT_FINGER_CLOSE_CAP": self.contact_finger_close_cap,  # Exports CONTACT_FINGER_CLOSE_CAP from the contact finger close cap setting
                "CONTACT_TEACHER_HOLD_FRACTION_CAP": self.contact_teacher_hold_fraction_cap,  # Exports CONTACT_TEACHER_HOLD_FRACTION_CAP from the contact teacher hold fraction cap setting
                "CONTACT_START_MODE": self.contact_start_mode,  # Exports CONTACT_START_MODE from the contact start mode setting
                "TOPDOWN_PREROLL_FRACTION": self.topdown_preroll_fraction,  # Exports TOPDOWN_PREROLL_FRACTION from the topdown preroll fraction setting
                "TOPDOWN_PREROLL_MAX_STEPS": self.topdown_preroll_max_steps,  # Exports TOPDOWN_PREROLL_MAX_STEPS from the topdown preroll maximum steps setting
                "TOPDOWN_PREROLL_UNLOCK_PROGRESS": self.topdown_preroll_unlock_progress,  # Exports TOPDOWN_PREROLL_UNLOCK_PROGRESS from the topdown preroll unlock progress setting
                "CONTACT_PREROLL_TOUCH_MODE": self.contact_preroll_touch_mode,  # Exports CONTACT_PREROLL_TOUCH_MODE from the contact preroll touch mode setting
                "CONTACT_PREROLL_IK_DESCEND_Z": self.contact_preroll_ik_descend_z,  # Exports CONTACT_PREROLL_IK_DESCEND_Z from the contact preroll IK descend Z setting
                "TOPDOWN_MIRROR_MIDDLE_TO_INDEX": bool01(self.mirror_middle_to_index),  # Exports TOPDOWN_MIRROR_MIDDLE_TO_INDEX as legacy 0 or 1 from the mirror middle to index setting
                "ACTOR_Q_ACTION_GATE_MODE": self.actor_q_action_gate_mode,  # Exports ACTOR_Q_ACTION_GATE_MODE from the actor Q action gate mode setting
                "ACTOR_BC_ACTION_GATE_MODE": self.actor_bc_action_gate_mode,  # Exports ACTOR_BC_ACTION_GATE_MODE from the actor BC action gate mode setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        add_arg(args, "--arm-action-scale-profile", self.arm_action_scale_profile)  # adds a scalar trainer CLI option
        add_arg(args, "--finger-action-mode", self.finger_action_mode)  # adds a scalar trainer CLI option
        add_arg(args, "--finger-delta-scale", self.finger_delta_scale)  # adds a scalar trainer CLI option
        add_arg(args, "--contact-finger-close-cap", self.contact_finger_close_cap)  # adds a scalar trainer CLI option
        add_arg(args, "--contact-teacher-hold-fraction-cap", self.contact_teacher_hold_fraction_cap)  # adds a scalar trainer CLI option
        add_arg(args, "--contact-start-mode", self.contact_start_mode)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-preroll-fraction", self.topdown_preroll_fraction)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-preroll-max-steps", self.topdown_preroll_max_steps)  # adds a scalar trainer CLI option
        add_arg(args, "--topdown-preroll-unlock-progress", self.topdown_preroll_unlock_progress)  # adds a scalar trainer CLI option
        add_arg(args, "--contact-preroll-touch-mode", self.contact_preroll_touch_mode)  # adds a scalar trainer CLI option
        add_arg(args, "--contact-preroll-ik-descend-z", self.contact_preroll_ik_descend_z)  # adds a scalar trainer CLI option
        add_flag(args, self.mirror_middle_to_index, "--topdown-mirror-middle-to-index")  # adds the trainer CLI flag when enabled
        add_arg(args, "--actor-q-action-gate-mode", self.actor_q_action_gate_mode)  # adds a scalar trainer CLI option
        add_arg(args, "--actor-bc-action-gate-mode", self.actor_bc_action_gate_mode)  # adds a scalar trainer CLI option
        add_flag(args, self.include_wrist_roll, "--include-wrist-roll")  # adds the trainer CLI flag when enabled
        if self.include_waist_yaw:  # Checks whether include waist yaw
            args.append("--include-waist-yaw")  # appends one trainer CLI token
            add_arg(args, "--waist-yaw-action-scale", self.waist_yaw_action_scale)  # adds a scalar trainer CLI option
        add_flag(args, self.observation_normalization, "--observation-normalization")  # adds the trainer CLI flag when enabled
        add_flag(args, self.reward_normalization, "--reward-normalization")  # adds the trainer CLI flag when enabled
        add_flag(args, self.privileged_critic, "--privileged-critic")  # adds the trainer CLI flag when enabled
        add_flag(args, self.soft_policy_arm_assist, "--soft-policy-arm-assist")  # adds the trainer CLI flag when enabled
        return args  # returns assembled trainer CLI arguments
