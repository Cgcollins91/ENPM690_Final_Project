"""Reward weights for the centered red-block lift profile."""

from __future__ import annotations  # keeps annotations lazy for forward references

from dataclasses import dataclass  # imports dataclass helpers used by config groups

from .base import clean_dict  # imports shared env and CLI conversion helpers


@dataclass(frozen=True)  # makes the following config group immutable
class ReachAlignReward:  # defines the reach align reward config group
    """Dense terms before physical contact."""

    reach_alignment_error_quadratic     : float = -6  # Weights the reach alignment error quadratic term
    reach_fingertip_line_angle_quadratic: float = -0.75  # Weights the reach fingertip line angle quadratic term
    align_alignment_error_quadratic     : float = -6  # Weights the align alignment error quadratic term
    align_fingertip_line_angle_quadratic: float = -0.6  # Weights the align fingertip line angle quadratic term

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "CURRICULUM_W_REACH_ALIGNMENT_ERROR_QUADRATIC": self.reach_alignment_error_quadratic,  # Exports CURRICULUM_W_REACH_ALIGNMENT_ERROR_QUADRATIC from the reach alignment error quadratic setting
                "CURRICULUM_W_REACH_FINGERTIP_LINE_ANGLE_QUADRATIC": self.reach_fingertip_line_angle_quadratic,  # Exports CURRICULUM_W_REACH_FINGERTIP_LINE_ANGLE_QUADRATIC from the reach fingertip line angle quadratic setting
                "CURRICULUM_W_ALIGN_ALIGNMENT_ERROR_QUADRATIC": self.align_alignment_error_quadratic,  # Exports CURRICULUM_W_ALIGN_ALIGNMENT_ERROR_QUADRATIC from the align alignment error quadratic setting
                "CURRICULUM_W_ALIGN_FINGERTIP_LINE_ANGLE_QUADRATIC": self.align_fingertip_line_angle_quadratic,  # Exports CURRICULUM_W_ALIGN_FINGERTIP_LINE_ANGLE_QUADRATIC from the align fingertip line angle quadratic setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class LiftReward:  # defines the lift reward config group
    """Lift shaping and safety penalties."""

    height_progress               : float        = 2  # Weights the height progress term
    lift_with_grip                : float        = 8  # Weights the lift with grip term
    centered_lift_progress        : float        = 0  # Weights the centered lift progress term
    centered_upright_lift_bonus   : float        = 20  # Weights the centered upright lift bonus term
    lift_xy_drift_penalty         : float        = -12  # Weights the lift XY drift penalty term
    block_tilt_lift_penalty       : float        = -15  # Weights the block tilt lift penalty term
    uncentered_lift_penalty       : float        = -8  # Weights the uncentered lift penalty term
    block_xy_velocity_penalty     : float        = -2  # Weights the block XY velocity penalty term
    block_angular_velocity_penalty: float        = -1  # Weights the block angular velocity penalty term
    block_drop_penalty            : float        = -500  # Weights the block drop penalty term
    block_off_table_bonus         : float        = 15  # Weights the block off table bonus term
    lift_penalty_height_start     : float        = 0.0  # Sets the lift height where lift penalties begin
    lift_penalty_height_ramp      : float        = 0.02  # Sets the height span used to ramp lift penalties
    lift_xy_drift_penalty_free    : float | None = None  # Weights the lift XY drift penalty free term
    lift_xy_drift_penalty_zero    : float | None = None  # Weights the lift XY drift penalty zero term

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "CURRICULUM_W_LIFT_HEIGHT_PROGRESS": self.height_progress,  # Exports CURRICULUM_W_LIFT_HEIGHT_PROGRESS from the height progress setting
                "CURRICULUM_W_LIFT_WITH_GRIP": self.lift_with_grip,  # Exports CURRICULUM_W_LIFT_WITH_GRIP from the lift with grip setting
                "CURRICULUM_W_CENTERED_LIFT_PROGRESS": self.centered_lift_progress,  # Exports CURRICULUM_W_CENTERED_LIFT_PROGRESS from the centered lift progress setting
                "CURRICULUM_W_CENTERED_UPRIGHT_LIFT_BONUS": self.centered_upright_lift_bonus,  # Exports CURRICULUM_W_CENTERED_UPRIGHT_LIFT_BONUS from the centered upright lift bonus setting
                "CURRICULUM_W_LIFT_XY_DRIFT_PENALTY": self.lift_xy_drift_penalty,  # Exports CURRICULUM_W_LIFT_XY_DRIFT_PENALTY from the lift XY drift penalty setting
                "CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY": self.block_tilt_lift_penalty,  # Exports CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY from the block tilt lift penalty setting
                "CURRICULUM_W_UNCENTERED_LIFT_PENALTY": self.uncentered_lift_penalty,  # Exports CURRICULUM_W_UNCENTERED_LIFT_PENALTY from the uncentered lift penalty setting
                "CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY": self.block_xy_velocity_penalty,  # Exports CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY from the block XY velocity penalty setting
                "CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY": self.block_angular_velocity_penalty,  # Exports CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY from the block angular velocity penalty setting
                "CURRICULUM_W_BLOCK_DROP_PENALTY": self.block_drop_penalty,  # Exports CURRICULUM_W_BLOCK_DROP_PENALTY from the block drop penalty setting
                "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS": self.block_off_table_bonus,  # Exports CURRICULUM_W_BLOCK_OFF_TABLE_BONUS from the block off table bonus setting
                "CURRICULUM_LIFT_PENALTY_HEIGHT_START": self.lift_penalty_height_start,  # Exports CURRICULUM_LIFT_PENALTY_HEIGHT_START from the lift penalty height start setting
                "CURRICULUM_LIFT_PENALTY_HEIGHT_RAMP": self.lift_penalty_height_ramp,  # Exports CURRICULUM_LIFT_PENALTY_HEIGHT_RAMP from the lift penalty height ramp setting
                "CURRICULUM_LIFT_XY_DRIFT_PENALTY_FREE": self.lift_xy_drift_penalty_free,  # Exports CURRICULUM_LIFT_XY_DRIFT_PENALTY_FREE from the lift XY drift penalty free setting
                "CURRICULUM_LIFT_XY_DRIFT_PENALTY_ZERO": self.lift_xy_drift_penalty_zero,  # Exports CURRICULUM_LIFT_XY_DRIFT_PENALTY_ZERO from the lift XY drift penalty zero setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class ContactReward:  # defines the contact reward config group
    """Contact quality terms at the grasp shell."""

    opposed_contact                  : float        = 12  # Weights the opposed contact term
    one_sided                        : float        = -4  # Weights the one sided term
    bilateral_contact                : float        = 0  # Weights the bilateral contact term
    bilateral_imbalance              : float        = 0  # Weights the bilateral imbalance term
    one_sided_flip                   : float        = 0  # Weights the one sided flip term
    preunlock_pocket                 : float        = 0  # Weights the preunlock pocket term
    preunlock_no_contact             : float        = 0  # Weights the preunlock no contact term
    contact_alignment_error          : float        = -0.5  # Weights the contact alignment error term
    contact_alignment_error_quadratic: float        = 0.0  # Weights the contact alignment error quadratic term
    alignment_degradation            : float        = 0.0  # Weights the alignment degradation term
    finger_center_x_error_quadratic  : float        = -0.5  # Weights the finger center X error quadratic term
    finger_center_y_error_quadratic  : float        = -0.5  # Weights the finger center Y error quadratic term
    finger_center_err_scale          : float        = 0.025  # Scales finger-center error before reward normalization
    finger_center_err_norm_cap       : float        = 3.0  # Caps normalized finger-center error for reward shaping
    centered_contact                 : float        = 30  # Weights the centered contact term
    contact_lift_progress            : float        = 0  # Weights the contact lift progress term
    contact_success_now_continuous   : float        = 0  # Weights the contact success now continuous term
    overforce                        : float | None = None  # Weights the overforce term
    smooth_pose_no_contact           : float | None = None  # Weights the smooth pose no contact term
    smooth_success_pose              : float | None = None  # Weights the smooth success pose term
    smooth_success_with_contact      : float | None = None  # Weights the smooth success with contact term
    stage2_floor                     : float        = 0.0  # Weights the stage2 floor term

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "CURRICULUM_W_CONTACT_OPPOSED_CONTACT": self.opposed_contact,  # Exports CURRICULUM_W_CONTACT_OPPOSED_CONTACT from the opposed contact setting
                "CURRICULUM_W_CONTACT_ONE_SIDED": self.one_sided,  # Exports CURRICULUM_W_CONTACT_ONE_SIDED from the one sided setting
                "CURRICULUM_W_CONTACT_BILATERAL_CONTACT": self.bilateral_contact,  # Exports CURRICULUM_W_CONTACT_BILATERAL_CONTACT from the bilateral contact setting
                "CURRICULUM_W_CONTACT_BILATERAL_IMBALANCE": self.bilateral_imbalance,  # Exports CURRICULUM_W_CONTACT_BILATERAL_IMBALANCE from the bilateral imbalance setting
                "CURRICULUM_W_CONTACT_ONE_SIDED_FLIP": self.one_sided_flip,  # Exports CURRICULUM_W_CONTACT_ONE_SIDED_FLIP from the one sided flip setting
                "CURRICULUM_W_CONTACT_PREUNLOCK_POCKET": self.preunlock_pocket,  # Exports CURRICULUM_W_CONTACT_PREUNLOCK_POCKET from the preunlock pocket setting
                "CURRICULUM_W_CONTACT_PREUNLOCK_NO_CONTACT": self.preunlock_no_contact,  # Exports CURRICULUM_W_CONTACT_PREUNLOCK_NO_CONTACT from the preunlock no contact setting
                "CURRICULUM_W_CONTACT_ALIGNMENT_ERROR": self.contact_alignment_error,  # Exports CURRICULUM_W_CONTACT_ALIGNMENT_ERROR from the contact alignment error setting
                "CURRICULUM_W_CONTACT_ALIGNMENT_ERROR_QUADRATIC": self.contact_alignment_error_quadratic,  # Exports CURRICULUM_W_CONTACT_ALIGNMENT_ERROR_QUADRATIC from the contact alignment error quadratic setting
                "CURRICULUM_W_ALIGNMENT_DEGRADATION": self.alignment_degradation,  # Exports CURRICULUM_W_ALIGNMENT_DEGRADATION from the alignment degradation setting
                "CURRICULUM_W_CONTACT_FINGER_CENTER_X_ERROR_QUADRATIC": self.finger_center_x_error_quadratic,  # Exports CURRICULUM_W_CONTACT_FINGER_CENTER_X_ERROR_QUADRATIC from the finger center X error quadratic setting
                "CURRICULUM_W_CONTACT_FINGER_CENTER_Y_ERROR_QUADRATIC": self.finger_center_y_error_quadratic,  # Exports CURRICULUM_W_CONTACT_FINGER_CENTER_Y_ERROR_QUADRATIC from the finger center Y error quadratic setting
                "CURRICULUM_CONTACT_FINGER_CENTER_ERR_SCALE": self.finger_center_err_scale,  # Exports CURRICULUM_CONTACT_FINGER_CENTER_ERR_SCALE from the finger center error scale setting
                "CURRICULUM_CONTACT_FINGER_CENTER_ERR_NORM_CAP": self.finger_center_err_norm_cap,  # Exports CURRICULUM_CONTACT_FINGER_CENTER_ERR_NORM_CAP from the finger center error norm cap setting
                "CURRICULUM_W_CONTACT_CENTERED_CONTACT": self.centered_contact,  # Exports CURRICULUM_W_CONTACT_CENTERED_CONTACT from the centered contact setting
                "CURRICULUM_W_CONTACT_LIFT_PROGRESS": self.contact_lift_progress,  # Exports CURRICULUM_W_CONTACT_LIFT_PROGRESS from the contact lift progress setting
                "CURRICULUM_W_CONTACT_SUCCESS_NOW_CONTINUOUS": self.contact_success_now_continuous,  # Exports CURRICULUM_W_CONTACT_SUCCESS_NOW_CONTINUOUS from the contact success now continuous setting
                "CURRICULUM_W_CONTACT_OVERFORCE": self.overforce,  # Exports CURRICULUM_W_CONTACT_OVERFORCE from the overforce setting
                "CURRICULUM_W_SMOOTH_POSE_NO_CONTACT": self.smooth_pose_no_contact,  # Exports CURRICULUM_W_SMOOTH_POSE_NO_CONTACT from the smooth pose no contact setting
                "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_POSE": self.smooth_success_pose,  # Exports CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_POSE from the smooth success pose setting
                "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_WITH_CONTACT": self.smooth_success_with_contact,  # Exports CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_WITH_CONTACT from the smooth success with contact setting
                "CURRICULUM_W_STAGE2_FLOOR": self.stage2_floor,  # Exports CURRICULUM_W_STAGE2_FLOOR from the stage2 floor setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class RuntimeReward:  # defines the runtime reward config group
    """Global shaping terms that do not belong to one stage."""

    step_cost: float = -0.005  # Weights the step cost term

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict({"CURRICULUM_W_STEP_COST": self.step_cost})  # returns env vars after dropping unset values
