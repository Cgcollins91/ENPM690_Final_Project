"""Named final-project training profiles.

Profiles are the project's primary experiment ledger.  A profile should read
like a compact run card: start from a known baseline, replace only the groups
that changed, and put the reasoning next to the changed values.  The launcher
then serializes the selected profile into a manifest so GUI smoke tests and
Docker runs do not depend on today's Python defaults.

Late profiles are intentionally not "cleaned up" into a single generic config:
the comments preserve the debugging history that made v32 liftable.  When in
doubt, prefer adding a new profile over mutating an old one that has a saved
checkpoint or manifest.
"""

from __future__ import annotations  # keeps annotations lazy for forward references

from dataclasses import dataclass, replace  # imports dataclass helpers used by config groups

from .base import add_arg, assert_field_limits, bool01, clean_dict  # imports shared env and CLI conversion helpers
from .reward import ContactReward, LiftReward, ReachAlignReward, RuntimeReward  # imports config dependencies from reward
from .task import (  # imports config dependencies from task
    ArmHoldCenteringConfig,  # continues this config expression
    ContactPoseFallbackConfig,  # continues this config expression
    FingerCenteringConfig,  # continues this config expression
    LiftSuccessConfig,  # continues this config expression
    StageGateConfig,  # continues this config expression
    TaskIdentity,  # continues this config expression
)  # closes the current expression
from .teacher import (  # imports config dependencies from teacher
    ActionSurfaceConfig,  # continues this config expression
    TaskSpaceIKConfig,  # continues this config expression
    TeacherLiftConfig,  # continues this config expression
    TeacherPreholdAdvancedConfig,  # continues this config expression
    TeacherPreholdConfig,  # continues this config expression
    TeacherProfile,  # continues this config expression
)  # closes the current expression
from .training import (  # imports config dependencies from training
    CoreTrainingConfig,  # continues this config expression
    DaggerConfig,  # continues this config expression
    DeterminismConfig,  # continues this config expression
    OptimizationConfig,  # continues this config expression
    RlAssistHandoffConfig,  # continues this config expression
    RlSwitchConfig,  # continues this config expression
    RunIOConfig,  # continues this config expression
    RuntimeConfig,  # continues this config expression
)  # closes the current expression


CLONING_RUN_DIR = "runs/cloning_red_centered_lift_v1"  # Overrides the cloning run dir setting for this config preset
CLONING_CHECKPOINT = f"{CLONING_RUN_DIR}/latest.pt"  # Overrides the cloning checkpoint setting for this config preset
STRICT_CLONING_RUN_DIR = "runs/cloning_red_centered_lift_strict_v1"  # Overrides the strict cloning run dir setting for this config preset
STRICT_CLONING_CHECKPOINT = f"{STRICT_CLONING_RUN_DIR}/latest.pt"  # Overrides the strict cloning checkpoint setting for this config preset
CENTERED_V2_CLONING_RUN_DIR = "runs/cloning_red_centered_lift_v2"  # Overrides the centered v2 cloning run dir setting for this config preset
IK_V3_CLONING_RUN_DIR = "runs/cloning_red_centered_lift_ik_v3"  # Overrides the IK v3 cloning run dir setting for this config preset
IK_V4_CLONING_RUN_DIR = "runs/cloning_red_centered_lift_ik_v4"  # Overrides the IK v4 cloning run dir setting for this config preset
IK_2F_V5_CLONING_RUN_DIR = "runs/cloning_red_centered_lift_ik_2f_v5"  # Overrides the IK 2 f v5 cloning run dir setting for this config preset
V8_600K_STEP_CHECKPOINT = "runs/teacher_dagger_upstream_fasttd3_v8_600k_handoff_rl/step_600100.pt"  # Overrides the v8 600 k step checkpoint setting for this config preset


def final_stage_gate() -> StageGateConfig:  # builds the final stage gate config preset
    """Teacher-reachable gates for the final red-block topdown pipeline."""

    return StageGateConfig(  # returns a stage gate config preset
        stage1_align_err_max=0.60,  # Overrides the stage1 align error maximum setting for this config preset
        stage2_align_err_max=0.32,  # Overrides the stage2 align error maximum setting for this config preset
        stage2_line_angle_max_deg=55,  # Overrides the stage2 line angle maximum deg setting for this config preset
        stage2_palm_height_max=0.08,  # Overrides the stage2 palm height maximum setting for this config preset
    )  # closes the current expression


def final_finger_centering() -> FingerCenteringConfig:  # builds the final finger centering config preset
    """Finger unlock gate loose enough for teacher demos, still centered."""

    return FingerCenteringConfig(  # returns a finger centering config preset
        hold_steps=2,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=60,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=45,  # Overrides the align angle maximum deg setting for this config preset
        tip_xy_max=0.060,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.100,  # Overrides the max tip XY maximum setting for this config preset
    )  # closes the current expression


def strict_stage_gate() -> StageGateConfig:  # builds the strict stage gate config preset
    """Stage-two gate for redo runs where fingers must wait for a tighter pocket."""

    return StageGateConfig(  # returns a stage gate config preset
        stage1_align_err_max=0.55,  # Overrides the stage1 align error maximum setting for this config preset
        stage2_align_err_max=0.24,  # Overrides the stage2 align error maximum setting for this config preset
        stage2_line_angle_max_deg=40,  # Overrides the stage2 line angle maximum deg setting for this config preset
        stage2_palm_height_max=0.06,  # Overrides the stage2 palm height maximum setting for this config preset
    )  # closes the current expression


def strict_finger_centering() -> FingerCenteringConfig:  # builds the strict finger centering config preset
    """Continuous, tighter center/z gate before any finger closure."""

    return FingerCenteringConfig(  # returns a finger centering config preset
        latch=False,  # Overrides the latch setting for this config preset
        hold_steps=6,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=120,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=20,  # Overrides the align angle maximum deg setting for this config preset
        tip_xy_max=0.035,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.060,  # Overrides the max tip XY maximum setting for this config preset
        tip_z_max=0.045,  # Overrides the tip Z maximum setting for this config preset
    )  # closes the current expression


def strict_contact_pose_fallback() -> ContactPoseFallbackConfig:  # builds the strict contact pose fallback config preset
    """Fallback shell that delays unlock until the centered topdown path is stable."""

    return ContactPoseFallbackConfig(  # returns a contact pose fallback config preset
        fallback_steps=80,  # Overrides the fallback steps setting for this config preset
        align_err_max=0.25,  # Overrides the align error maximum setting for this config preset
        palm_dist_max=0.10,  # Overrides the palm distance maximum setting for this config preset
        palm_height_max=0.06,  # Overrides the palm height maximum setting for this config preset
        palm_orient_max_deg=45,  # Overrides the palm orient maximum deg setting for this config preset
        palm_yaw_max_deg=45,  # Overrides the palm yaw maximum deg setting for this config preset
        opposed_gate_min=0.50,  # Overrides the opposed gate minimum setting for this config preset
    )  # closes the current expression


def strict_lift_success() -> LiftSuccessConfig:  # builds the strict lift success config preset
    """Lift gate with tighter centered-contact z and angle requirements."""

    return LiftSuccessConfig(  # returns a lift success config preset
        success_height=0.035,  # Overrides the success height setting for this config preset
        success_center_tip_xy_max=0.018,  # Overrides the success center tip XY maximum setting for this config preset
        success_center_tip_z_max=0.040,  # Overrides the success center tip Z maximum setting for this config preset
        success_center_align_angle_max_deg=10,  # Overrides the success center align angle maximum deg setting for this config preset
        success_center_hold_steps=4,  # Overrides the success center hold steps setting for this config preset
    )  # closes the current expression


def centered_v2_finger_centering() -> FingerCenteringConfig:  # builds the centered v2 finger centering config preset
    """Reachable finger gate for the v2 redo.

    The first strict redo over-constrained pre-contact z and arm-freeze
    readiness, which starved replay of real closure/lift attempts. This keeps
    the old reachable path while delaying closure more gently.
    """

    return FingerCenteringConfig(  # returns a finger centering config preset
        latch=True,  # Overrides the latch setting for this config preset
        hold_steps=4,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=90,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=35,  # Overrides the align angle maximum deg setting for this config preset
        tip_xy_max=0.050,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.085,  # Overrides the max tip XY maximum setting for this config preset
        tip_z_max=0.070,  # Overrides the tip Z maximum setting for this config preset
    )  # closes the current expression


def centered_v2_lift_success() -> LiftSuccessConfig:  # builds the centered v2 lift success config preset
    """Stricter centered-lift success without blocking teacher demonstrations."""

    return LiftSuccessConfig(  # returns a lift success config preset
        success_height=0.035,  # Overrides the success height setting for this config preset
        success_center_tip_xy_max=0.020,  # Overrides the success center tip XY maximum setting for this config preset
        success_center_tip_z_max=0.050,  # Overrides the success center tip Z maximum setting for this config preset
        success_center_align_angle_max_deg=12,  # Overrides the success center align angle maximum deg setting for this config preset
        success_center_hold_steps=2,  # Overrides the success center hold steps setting for this config preset
    )  # closes the current expression


def centered_v2_contact_reward() -> ContactReward:  # defines the centered v2 contact reward helper
    """A little more soft pressure toward centered closure than v1."""

    return ContactReward(  # returns a contact reward preset
        finger_center_x_error_quadratic=-0.75,  # Overrides the finger center X error quadratic setting for this config preset
        finger_center_y_error_quadratic=-0.75,  # Overrides the finger center Y error quadratic setting for this config preset
        finger_center_err_scale=0.035,  # Overrides the finger center error scale setting for this config preset
        centered_contact=4.0,  # Overrides the centered contact setting for this config preset
    )  # closes the current expression


def ik_v3_finger_centering() -> FingerCenteringConfig:  # builds the IK v3 finger centering config preset
    """Stricter unlock gate intended for the direct fingertip IK teacher."""

    return FingerCenteringConfig(  # returns a finger centering config preset
        latch=True,  # Overrides the latch setting for this config preset
        hold_steps=5,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=120,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=30,  # Overrides the align angle maximum deg setting for this config preset
        tip_xy_max=0.040,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.075,  # Overrides the max tip XY maximum setting for this config preset
        tip_z_max=0.060,  # Overrides the tip Z maximum setting for this config preset
    )  # closes the current expression


def ik_v3_lift_success() -> LiftSuccessConfig:  # builds the IK v3 lift success config preset
    """Centered-lift success gate for the strict-contract replay run."""

    return LiftSuccessConfig(  # returns a lift success config preset
        success_height=0.035,  # Overrides the success height setting for this config preset
        success_center_tip_xy_max=0.018,  # Overrides the success center tip XY maximum setting for this config preset
        success_center_tip_z_max=0.045,  # Overrides the success center tip Z maximum setting for this config preset
        success_center_align_angle_max_deg=10,  # Overrides the success center align angle maximum deg setting for this config preset
        success_center_hold_steps=2,  # Overrides the success center hold steps setting for this config preset
    )  # closes the current expression


def lift02_phase1_success() -> LiftSuccessConfig:  # builds the lift02 phase1 success config preset
    """Phase-1 transfer success: sustained physical lift only.

    v35 can produce real stable lifts that the historical opposed/strict
    contact scalar misses.  This gate is intentionally permissive for the
    first transfer phase: count the episode once the block stays at least 2 cm
    off its spawn height for 30 consecutive env steps.
    """

    return LiftSuccessConfig(  # returns a lift success config preset
        success_mode="height_only",  # Overrides the success mode setting for this config preset
        success_height=0.020,  # Overrides the success height setting for this config preset
        success_hold_steps=30,  # Overrides the success hold steps setting for this config preset
        success_xy_drift_max=999.0,  # Overrides the success XY drift maximum setting for this config preset
        lift_requires_grip=False,  # Overrides the lift requires grip setting for this config preset
    )  # closes the current expression


def ik_v3_teacher_prehold() -> TeacherPreholdConfig:  # builds the IK v3 teacher prehold config preset
    """Direct stacked fingertip IK correction layered on the existing palm IK."""

    return TeacherPreholdConfig(  # returns a teacher prehold config preset
        tip_jacobian_ik=True,  # Overrides the tip jacobian IK setting for this config preset
        tip_jacobian_stage_min=1,  # Overrides the tip jacobian stage minimum setting for this config preset
        tip_jacobian_gain=0.9,  # Overrides the tip jacobian gain setting for this config preset
        tip_jacobian_damping=0.05,  # Overrides the tip jacobian damping setting for this config preset
        tip_jacobian_max_joint_step=0.040,  # Overrides the tip jacobian maximum joint step setting for this config preset
        tip_jacobian_xy_weight=1.0,  # Overrides the tip jacobian XY weight setting for this config preset
        tip_jacobian_z_weight=1.75,  # Overrides the tip jacobian Z weight setting for this config preset
        tip_jacobian_use_middle=True,  # Overrides the tip jacobian use middle setting for this config preset
        tip_jacobian_joints="base",  # Overrides the tip jacobian joints setting for this config preset
        ik_tip_servo_gain=0.85,  # Overrides the IK tip servo gain setting for this config preset
        ik_tip_servo_max_m=0.100,  # Overrides the IK tip servo maximum m setting for this config preset
        pocket_sweep_iters=2,  # Overrides the pocket sweep iterations setting for this config preset
    )  # closes the current expression


def ik_v3_contact_reward(stage2_floor: float = 0.0) -> ContactReward:  # defines the IK v3 contact reward helper
    """Soft rewards aligned with the stricter IK replay contract."""

    return ContactReward(  # returns a contact reward preset
        contact_alignment_error=-2.0,  # Overrides the contact alignment error setting for this config preset
        contact_alignment_error_quadratic=-10.0,  # Overrides the contact alignment error quadratic setting for this config preset
        alignment_degradation=-25.0,  # Overrides the alignment degradation setting for this config preset
        finger_center_x_error_quadratic=-1.0,  # Overrides the finger center X error quadratic setting for this config preset
        finger_center_y_error_quadratic=-1.0,  # Overrides the finger center Y error quadratic setting for this config preset
        finger_center_err_scale=0.035,  # Overrides the finger center error scale setting for this config preset
        centered_contact=6.0,  # Overrides the centered contact setting for this config preset
        stage2_floor=stage2_floor,  # Overrides the stage2 floor setting for this config preset
    )  # closes the current expression


def ik_v4_finger_centering() -> FingerCenteringConfig:  # builds the IK v4 finger centering config preset
    """Pre-close gate for the stronger v4 direct IK teacher."""

    return FingerCenteringConfig(  # returns a finger centering config preset
        latch=True,  # Overrides the latch setting for this config preset
        hold_steps=4,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=140,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=38,  # Overrides the align angle maximum deg setting for this config preset
        tip_xy_max=0.060,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.090,  # Overrides the max tip XY maximum setting for this config preset
        tip_z_max=0.095,  # Overrides the tip Z maximum setting for this config preset
    )  # closes the current expression


def ik_2f_v5_stage_gate() -> StageGateConfig:  # builds the IK 2f v5 stage gate config preset
    """Let the dual-tip teacher enter contact once the actual pocket is centered."""

    return StageGateConfig(  # returns a stage gate config preset
        stage1_align_err_max=0.55,  # Overrides the stage1 align error maximum setting for this config preset
        stage2_palm_dist_max=0.14,  # Overrides the stage2 palm distance maximum setting for this config preset
        stage2_palm_height_max=0.12,  # Overrides the stage2 palm height maximum setting for this config preset
        stage2_palm_orient_max_deg=52,  # Overrides the stage2 palm orient maximum deg setting for this config preset
        stage2_palm_yaw_max_deg=65,  # Overrides the stage2 palm yaw maximum deg setting for this config preset
        stage2_align_err_max=0.34,  # Overrides the stage2 align error maximum setting for this config preset
        stage2_line_angle_max_deg=55,  # Overrides the stage2 line angle maximum deg setting for this config preset
        stage2_opposed_gate_min=0.50,  # Overrides the stage2 opposed gate minimum setting for this config preset
    )  # closes the current expression


def ik_2f_v5_finger_centering() -> FingerCenteringConfig:  # builds the IK 2f v5 finger centering config preset
    """Dual-tip pre-close gate matching the old two-finger IK contract."""

    return FingerCenteringConfig(  # returns a finger centering config preset
        latch=True,  # Overrides the latch setting for this config preset
        hold_steps=2,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=100,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=35,  # Overrides the align angle maximum deg setting for this config preset
        three_finger_centering=False,  # Overrides the three finger centering setting for this config preset
        tip_xy_max=0.060,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.085,  # Overrides the max tip XY maximum setting for this config preset
        tip_z_max=0.060,  # Overrides the tip Z maximum setting for this config preset
    )  # closes the current expression


def ik_2f_v5_contact_pose_fallback() -> ContactPoseFallbackConfig:  # builds the IK 2f v5 contact pose fallback config preset
    """Expose finger closure before the open hand drifts out of a good pocket."""

    return ContactPoseFallbackConfig(  # returns a contact pose fallback config preset
        fallback_steps=10,  # Overrides the fallback steps setting for this config preset
        align_err_max=0.35,  # Overrides the align error maximum setting for this config preset
        palm_dist_max=0.10,  # Overrides the palm distance maximum setting for this config preset
        palm_height_max=0.05,  # Overrides the palm height maximum setting for this config preset
        palm_orient_max_deg=70,  # Overrides the palm orient maximum deg setting for this config preset
        palm_yaw_max_deg=80,  # Overrides the palm yaw maximum deg setting for this config preset
        opposed_gate_min=0.50,  # Overrides the opposed gate minimum setting for this config preset
        contact_pose_hold_steps=2,  # Overrides the contact pose hold steps setting for this config preset
        block_disp_max=0.060,  # Overrides the block disp maximum setting for this config preset
    )  # closes the current expression


def ik_2f_v5_teacher_prehold() -> TeacherPreholdConfig:  # builds the IK 2f v5 teacher prehold config preset
    """Two-tip DLS correction, close to the prior dual-tip diagnostic setup."""

    return TeacherPreholdConfig(  # returns a teacher prehold config preset
        tip_jacobian_ik=True,  # Overrides the tip jacobian IK setting for this config preset
        tip_jacobian_stage_min=0,  # Overrides the tip jacobian stage minimum setting for this config preset
        tip_jacobian_gain=1.05,  # Overrides the tip jacobian gain setting for this config preset
        tip_jacobian_damping=0.045,  # Overrides the tip jacobian damping setting for this config preset
        tip_jacobian_max_joint_step=0.060,  # Overrides the tip jacobian maximum joint step setting for this config preset
        tip_jacobian_xy_weight=1.15,  # Overrides the tip jacobian XY weight setting for this config preset
        tip_jacobian_z_weight=2.00,  # Overrides the tip jacobian Z weight setting for this config preset
        tip_jacobian_use_middle=False,  # Overrides the tip jacobian use middle setting for this config preset
        tip_jacobian_joints="base",  # Overrides the tip jacobian joints setting for this config preset
        align_angle_gain=0.90,  # Overrides the align angle gain setting for this config preset
        align_angle_max_dz=0.030,  # Overrides the align angle maximum dz setting for this config preset
        align_angle_max_joint_step=0.08,  # Overrides the align angle maximum joint step setting for this config preset
        planar_align_gain=0.90,  # Overrides the planar align gain setting for this config preset
        planar_align_max_xy=0.060,  # Overrides the planar align maximum XY setting for this config preset
        planar_align_max_joint_step=0.08,  # Overrides the planar align maximum joint step setting for this config preset
        ik_tip_servo_stage_min=0,  # Overrides the IK tip servo stage minimum setting for this config preset
        ik_tip_servo_gain=0.90,  # Overrides the IK tip servo gain setting for this config preset
        ik_tip_servo_max_m=0.100,  # Overrides the IK tip servo maximum m setting for this config preset
        pocket_sweep_stage_min=1,  # Overrides the pocket sweep stage minimum setting for this config preset
        pocket_sweep_iters=2,  # Overrides the pocket sweep iterations setting for this config preset
    )  # closes the current expression


def ik_2f_v5_teacher_prehold_advanced() -> TeacherPreholdAdvancedConfig:  # builds the IK 2f v5 teacher prehold advanced config preset
    """Keep the dual-tip correction monotonic and off the frozen-arm mask."""

    return TeacherPreholdAdvancedConfig(  # returns a teacher prehold advanced config preset
        tip_jacobian_respect_arm_hold=False,  # Overrides the tip jacobian respect arm hold setting for this config preset
        tip_jacobian_accept_worse=False,  # Overrides the tip jacobian accept worse setting for this config preset
        tip_jacobian_max_worse_m=0.001,  # Overrides the tip jacobian maximum worse m setting for this config preset
        position_only_stage_min=1,  # Overrides the position only stage minimum setting for this config preset
        align_angle_joints="all",  # Overrides the align angle joints setting for this config preset
        planar_align_joints="all",  # Overrides the planar align joints setting for this config preset
    )  # closes the current expression


def ik_2f_v6_stage_gate() -> StageGateConfig:  # builds the IK 2f v6 stage gate config preset
    """Slightly wider stage-two shell for the stronger all-joint two-tip teacher."""

    return StageGateConfig(  # returns a stage gate config preset
        reach_hold_steps=4,  # Overrides the reach hold steps setting for this config preset
        align_hold_steps=4,  # Overrides the align hold steps setting for this config preset
        stage1_align_err_max=0.60,  # Overrides the stage1 align error maximum setting for this config preset
        stage2_palm_dist_max=0.16,  # Overrides the stage2 palm distance maximum setting for this config preset
        stage2_palm_height_max=0.13,  # Overrides the stage2 palm height maximum setting for this config preset
        stage2_palm_orient_max_deg=60,  # Overrides the stage2 palm orient maximum deg setting for this config preset
        stage2_palm_yaw_max_deg=72,  # Overrides the stage2 palm yaw maximum deg setting for this config preset
        stage2_align_err_max=0.38,  # Overrides the stage2 align error maximum setting for this config preset
        stage2_line_angle_max_deg=60,  # Overrides the stage2 line angle maximum deg setting for this config preset
        stage2_opposed_gate_min=0.45,  # Overrides the stage2 opposed gate minimum setting for this config preset
    )  # closes the current expression


def ik_2f_v6_finger_centering() -> FingerCenteringConfig:  # builds the IK 2f v6 finger centering config preset
    """Keep the two-finger close gate strict enough to avoid early one-sided pushes."""

    return FingerCenteringConfig(  # returns a finger centering config preset
        latch=True,  # Overrides the latch setting for this config preset
        hold_steps=3,  # Overrides the hold steps setting for this config preset
        unlock_ramp_steps=120,  # Overrides the unlock ramp steps setting for this config preset
        align_angle_max_deg=35,  # Overrides the align angle maximum deg setting for this config preset
        three_finger_centering=False,  # Overrides the three finger centering setting for this config preset
        tip_xy_max=0.055,  # Overrides the tip XY maximum setting for this config preset
        max_tip_xy_max=0.080,  # Overrides the max tip XY maximum setting for this config preset
        tip_z_max=0.055,  # Overrides the tip Z maximum setting for this config preset
    )  # closes the current expression


def ik_2f_v6_contact_pose_fallback() -> ContactPoseFallbackConfig:  # builds the IK 2f v6 contact pose fallback config preset
    """Fallback shell for late contact, still requiring a real opposing pocket."""

    return ContactPoseFallbackConfig(  # returns a contact pose fallback config preset
        fallback_steps=12,  # Overrides the fallback steps setting for this config preset
        align_err_max=0.34,  # Overrides the align error maximum setting for this config preset
        palm_dist_max=0.105,  # Overrides the palm distance maximum setting for this config preset
        palm_height_max=0.055,  # Overrides the palm height maximum setting for this config preset
        palm_orient_max_deg=72,  # Overrides the palm orient maximum deg setting for this config preset
        palm_yaw_max_deg=82,  # Overrides the palm yaw maximum deg setting for this config preset
        opposed_gate_min=0.45,  # Overrides the opposed gate minimum setting for this config preset
        contact_pose_hold_steps=3,  # Overrides the contact pose hold steps setting for this config preset
        block_disp_max=0.055,  # Overrides the block disp maximum setting for this config preset
    )  # closes the current expression


def ik_2f_v6_teacher_prehold() -> TeacherPreholdConfig:  # builds the IK 2f v6 teacher prehold config preset
    """Stronger two-tip DLS correction for faster stage-two entry from reset."""

    return TeacherPreholdConfig(  # returns a teacher prehold config preset
        tip_jacobian_ik=True,  # Overrides the tip jacobian IK setting for this config preset
        tip_jacobian_stage_min=0,  # Overrides the tip jacobian stage minimum setting for this config preset
        tip_jacobian_gain=1.30,  # Overrides the tip jacobian gain setting for this config preset
        tip_jacobian_damping=0.035,  # Overrides the tip jacobian damping setting for this config preset
        tip_jacobian_max_joint_step=0.085,  # Overrides the tip jacobian maximum joint step setting for this config preset
        tip_jacobian_xy_weight=1.35,  # Overrides the tip jacobian XY weight setting for this config preset
        tip_jacobian_z_weight=2.60,  # Overrides the tip jacobian Z weight setting for this config preset
        tip_jacobian_use_middle=False,  # Overrides the tip jacobian use middle setting for this config preset
        tip_jacobian_joints="all",  # Overrides the tip jacobian joints setting for this config preset
        align_angle_stage_min=0,  # Overrides the align angle stage minimum setting for this config preset
        align_angle_gain=1.10,  # Overrides the align angle gain setting for this config preset
        align_angle_max_dz=0.040,  # Overrides the align angle maximum dz setting for this config preset
        align_angle_max_joint_step=0.11,  # Overrides the align angle maximum joint step setting for this config preset
        planar_align_stage_min=0,  # Overrides the planar align stage minimum setting for this config preset
        planar_align_gain=1.15,  # Overrides the planar align gain setting for this config preset
        planar_align_max_xy=0.085,  # Overrides the planar align maximum XY setting for this config preset
        planar_align_max_joint_step=0.11,  # Overrides the planar align maximum joint step setting for this config preset
        ik_tip_servo_stage_min=0,  # Overrides the IK tip servo stage minimum setting for this config preset
        ik_tip_servo_gain=1.10,  # Overrides the IK tip servo gain setting for this config preset
        ik_tip_servo_max_m=0.130,  # Overrides the IK tip servo maximum m setting for this config preset
        pocket_sweep_stage_min=0,  # Overrides the pocket sweep stage minimum setting for this config preset
        pocket_sweep_iters=4,  # Overrides the pocket sweep iterations setting for this config preset
    )  # closes the current expression


def ik_2f_v6_teacher_prehold_advanced() -> TeacherPreholdAdvancedConfig:  # builds the IK 2f v6 teacher prehold advanced config preset
    """Let the teacher use position-only IK immediately and keep all correction joints live."""

    return TeacherPreholdAdvancedConfig(  # returns a teacher prehold advanced config preset
        tip_jacobian_respect_arm_hold=False,  # Overrides the tip jacobian respect arm hold setting for this config preset
        tip_jacobian_accept_worse=False,  # Overrides the tip jacobian accept worse setting for this config preset
        tip_jacobian_max_worse_m=0.0005,  # Overrides the tip jacobian maximum worse m setting for this config preset
        position_only_stage_min=0,  # Overrides the position only stage minimum setting for this config preset
        align_angle_joints="all",  # Overrides the align angle joints setting for this config preset
        planar_align_joints="all",  # Overrides the planar align joints setting for this config preset
    )  # closes the current expression


def ik_v4_teacher_prehold() -> TeacherPreholdConfig:  # builds the IK v4 teacher prehold config preset
    """Earlier and stronger fingertip correction before the strict contact contract."""

    return TeacherPreholdConfig(  # returns a teacher prehold config preset
        tip_jacobian_ik=True,  # Overrides the tip jacobian IK setting for this config preset
        tip_jacobian_stage_min=0,  # Overrides the tip jacobian stage minimum setting for this config preset
        tip_jacobian_gain=1.25,  # Overrides the tip jacobian gain setting for this config preset
        tip_jacobian_damping=0.035,  # Overrides the tip jacobian damping setting for this config preset
        tip_jacobian_max_joint_step=0.080,  # Overrides the tip jacobian maximum joint step setting for this config preset
        tip_jacobian_xy_weight=1.25,  # Overrides the tip jacobian XY weight setting for this config preset
        tip_jacobian_z_weight=2.50,  # Overrides the tip jacobian Z weight setting for this config preset
        tip_jacobian_use_middle=True,  # Overrides the tip jacobian use middle setting for this config preset
        tip_jacobian_joints="all",  # Overrides the tip jacobian joints setting for this config preset
        align_angle_gain=1.0,  # Overrides the align angle gain setting for this config preset
        align_angle_max_dz=0.035,  # Overrides the align angle maximum dz setting for this config preset
        align_angle_max_joint_step=0.10,  # Overrides the align angle maximum joint step setting for this config preset
        planar_align_gain=1.0,  # Overrides the planar align gain setting for this config preset
        planar_align_max_xy=0.070,  # Overrides the planar align maximum XY setting for this config preset
        planar_align_max_joint_step=0.10,  # Overrides the planar align maximum joint step setting for this config preset
        ik_tip_servo_stage_min=0,  # Overrides the IK tip servo stage minimum setting for this config preset
        ik_tip_servo_gain=1.0,  # Overrides the IK tip servo gain setting for this config preset
        ik_tip_servo_max_m=0.120,  # Overrides the IK tip servo maximum m setting for this config preset
        pocket_sweep_stage_min=1,  # Overrides the pocket sweep stage minimum setting for this config preset
        pocket_sweep_iters=3,  # Overrides the pocket sweep iterations setting for this config preset
    )  # closes the current expression


def ik_v4_teacher_prehold_advanced() -> TeacherPreholdAdvancedConfig:  # builds the IK v4 teacher prehold advanced config preset
    """Experiment-only solver masks that keep v4 from freezing too early."""

    return TeacherPreholdAdvancedConfig(  # returns a teacher prehold advanced config preset
        tip_jacobian_respect_arm_hold=False,  # Overrides the tip jacobian respect arm hold setting for this config preset
        position_only_stage_min=1,  # Overrides the position only stage minimum setting for this config preset
        align_angle_joints="all",  # Overrides the align angle joints setting for this config preset
        planar_align_joints="all",  # Overrides the planar align joints setting for this config preset
    )  # closes the current expression


def ik_v4_arm_hold_centering() -> ArmHoldCenteringConfig:  # builds the IK v4 arm hold centering config preset
    """Do not freeze the in-pocket arm action until the pre-close gate is met."""

    return ArmHoldCenteringConfig(  # returns a arm hold centering config preset
        latch_hold_steps=3,  # Overrides the latch hold steps setting for this config preset
        freeze_requires_finger_center=True,  # Overrides the freeze requires finger center setting for this config preset
        center_tip_xy_max=0.060,  # Overrides the center tip XY maximum setting for this config preset
        center_tip_z_max=0.095,  # Overrides the center tip Z maximum setting for this config preset
        center_align_angle_max_deg=38.0,  # Overrides the center align angle maximum deg setting for this config preset
    )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class UpstreamFastTD3BackendConfig:  # defines the upstream fast TD3 backend config group
    """Select the upstream FastTD3 learner behind the existing teacher pipeline."""

    fasttd3_repo     : str | None = None  # Sets the FastTD3 repo filesystem path
    num_atoms        : int        = 51  # Sets the num atoms config value
    v_min            : float      = -5.0  # Sets the minimum required v
    v_max            : float      = 0.0  # Sets the maximum allowed v
    actor_hidden_dim : int        = 512  # Sets the actor hidden dim config value
    critic_hidden_dim: int        = 1024  # Sets the critic hidden dim config value
    init_scale       : float      = 0.01  # Sets scale factor for init
    weight_decay     : float      = 0.0  # Sets the weight decay config value
    std_min          : float      = 0.001  # Sets the minimum required std
    std_max          : float      = 0.4  # Sets the maximum allowed std
    use_cdq          : bool       = True  # Controls whether use CDQ is enabled

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the trainer backend."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TD3_BACKEND": "upstream_fasttd3",  # Sets TD3_BACKEND from this literal mapping entry
                "FASTTD3_REPO": self.fasttd3_repo,  # Exports FASTTD3_REPO from the FastTD3 repo setting
                "FASTTD3_NUM_ATOMS": self.num_atoms,  # Exports FASTTD3_NUM_ATOMS from the num atoms setting
                "FASTTD3_V_MIN": self.v_min,  # Exports FASTTD3_V_MIN from the v minimum setting
                "FASTTD3_V_MAX": self.v_max,  # Exports FASTTD3_V_MAX from the v maximum setting
                "FASTTD3_ACTOR_HIDDEN_DIM": self.actor_hidden_dim,  # Exports FASTTD3_ACTOR_HIDDEN_DIM from the actor hidden dim setting
                "FASTTD3_CRITIC_HIDDEN_DIM": self.critic_hidden_dim,  # Exports FASTTD3_CRITIC_HIDDEN_DIM from the critic hidden dim setting
                "FASTTD3_INIT_SCALE": self.init_scale,  # Exports FASTTD3_INIT_SCALE from the init scale setting
                "FASTTD3_WEIGHT_DECAY": self.weight_decay,  # Exports FASTTD3_WEIGHT_DECAY from the weight decay setting
                "FASTTD3_STD_MIN": self.std_min,  # Exports FASTTD3_STD_MIN from the std minimum setting
                "FASTTD3_STD_MAX": self.std_max,  # Exports FASTTD3_STD_MAX from the std maximum setting
                "FASTTD3_USE_CDQ": bool01(self.use_cdq),  # Exports FASTTD3_USE_CDQ as legacy 0 or 1 from the use CDQ setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return CLI arguments that make backend selection visible in manifests."""
        args = ["--td3-backend", "upstream_fasttd3"]  # Collects trainer CLI arguments before return
        if self.fasttd3_repo:  # Checks whether FastTD3 repo
            args.extend(["--fasttd3-repo", self.fasttd3_repo])  # appends these trainer CLI tokens
        for name, raw in (  # iterates over configured values
            ("--fasttd3-num-atoms", self.num_atoms),  # Pairs trainer option --fasttd3-num-atoms with the num atoms setting
            ("--fasttd3-v-min", self.v_min),  # Pairs trainer option --fasttd3-v-min with the v minimum setting
            ("--fasttd3-v-max", self.v_max),  # Pairs trainer option --fasttd3-v-max with the v maximum setting
            ("--fasttd3-actor-hidden-dim", self.actor_hidden_dim),  # Pairs trainer option --fasttd3-actor-hidden-dim with the actor hidden dim setting
            ("--fasttd3-critic-hidden-dim", self.critic_hidden_dim),  # Pairs trainer option --fasttd3-critic-hidden-dim with the critic hidden dim setting
            ("--fasttd3-init-scale", self.init_scale),  # Pairs trainer option --fasttd3-init-scale with the init scale setting
            ("--fasttd3-weight-decay", self.weight_decay),  # Pairs trainer option --fasttd3-weight-decay with the weight decay setting
            ("--fasttd3-std-min", self.std_min),  # Pairs trainer option --fasttd3-std-min with the std minimum setting
            ("--fasttd3-std-max", self.std_max),  # Pairs trainer option --fasttd3-std-max with the std maximum setting
            ("--fasttd3-use-cdq", int(self.use_cdq)),  # continues this config expression
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class WristYawLockConfig:  # defines the wrist yaw lock config group
    """Topdown wrist-yaw release policy for pre-descent hover."""

    release_at_stage2: bool = True  # Controls whether release at stage2 is enabled

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "WRIST_YAW_RELEASE_AT_STAGE2": bool01(self.release_at_stage2),  # Exports WRIST_YAW_RELEASE_AT_STAGE2 as legacy 0 or 1 from the release at stage2 setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class RunProfile:  # defines the run profile config group
    """One immutable training profile assembled from small config groups."""

    name       : str  # Stores the registry name for this run profile
    description: str  # Summarizes the intent of this run profile
    groups     : tuple[object, ...]  # Lists the config groups that compose this run profile
    script     : str = ""  # Selects a script launcher when a profile intentionally bypasses module launch
    module     : str = "training.native_entrypoint"  # Selects the module launcher when script is empty

    def __post_init__(self) -> None:  # defines the post init helper
        """Validate derived dataclass invariants immediately after construction."""
        assert_field_limits(self.groups)  # validates that each config group stays below the field limit

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        env: dict[str, str] = {}  # Collects env vars merged from profile groups
        for group in self.groups:  # iterates over configured values
            method = getattr(group, "env", None)  # Looks up the optional export hook on this config group
            if callable(method):  # Checks whether callable(method)
                env.update(method())  # merges override values into the current mapping
        return env  # returns the computed value

    def app_args(self) -> list[str]:  # exports this config group as Isaac app launcher arguments
        """Return Isaac application launcher arguments for this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for group in self.groups:  # iterates over configured values
            method = getattr(group, "app_args", None)  # Looks up the optional export hook on this config group
            if callable(method):  # Checks whether callable(method)
                args.extend(method())  # appends these trainer CLI tokens
        return args  # returns assembled trainer CLI arguments

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for group in self.groups:  # iterates over configured values
            method = getattr(group, "trainer_args", None)  # Looks up the optional export hook on this config group
            if callable(method):  # Checks whether callable(method)
                args.extend(method())  # appends these trainer CLI tokens
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class NativeEntrypointConfig:  # defines native entrypoint-only CLI options
    """Native launcher switches that are stripped before core trainer parsing"""

    teacher_provider      : str  = "none"  # Selects native teacher source provider
    contact_attr_parts    : bool = False  # Controls env attr contact teacher parts
    contact_middle_teacher: bool = False  # Controls middle finger contact teacher output

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return native entrypoint flags for modular training launches"""
        args: list[str] = []  # Collects trainer CLI arguments before return
        add_arg(args, "--native-teacher-provider", self.teacher_provider)  # Selects native teacher source provider
        if self.contact_attr_parts:  # Checks whether env attr contact parts are enabled
            args.append("--native-contact-attr-parts")  # Enables contact parts from env attrs
        if self.contact_middle_teacher:  # Checks whether middle contact teacher output is enabled
            args.append("--native-contact-middle-teacher")  # Enables middle finger contact teacher output
        return args  # returns assembled trainer CLI arguments


def dagger_rl_current() -> RunProfile:  # builds the dagger RL current run profile
    """Current red centered DAgger-to-RL run with shell behavior preserved."""

    run_dir = (  # Builds the run directory string for this profile
        "runs/"  # adds literal text to the surrounding expression
        "phase3_4_red_centered_dagger_nstep3_bc100k_then_rl_1m_assist025_rewardsoft_r8"  # adds literal text to the surrounding expression
    )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="dagger_rl_current",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Current red centered lift run: actor init from the validated DAgger "  # adds literal text to the surrounding expression
            "checkpoint, 100k DAgger/BC replay fill, then TD3 refinement."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=run_dir,  # Builds the run directory string for this profile
                actor_init_checkpoint="checkpoints/overnight_0430n_red_centered_dagger_400k_r1_best.pt",  # Overrides the actor init checkpoint setting for this config preset
                reset_obs_stats_on_resume=False,  # Overrides the reset obs stats on resume setting for this config preset
                allow_warmstart=True,  # Overrides the allow warmstart setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            StageGateConfig(),  # adds the stage gate config group
            FingerCenteringConfig(),  # adds the finger centering config group
            LiftSuccessConfig(),  # adds the lift success config group
            ContactPoseFallbackConfig(),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ContactReward(),  # adds the contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            CoreTrainingConfig(),  # adds the core training config group
            DeterminismConfig(),  # adds the determinism config group
            DaggerConfig(),  # adds the dagger config group
            OptimizationConfig(),  # adds the optimization config group
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(),  # adds the runtime config group
        ),  # closes the current expression
    )  # closes the current expression


def cloning() -> RunProfile:  # builds the cloning run profile
    """Pure IK-teacher behavior cloning stage."""

    return RunProfile(  # returns the assembled run profile
        name="cloning",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Pure cloning stage: execute the IK/contact teacher, train the actor "  # adds literal text to the surrounding expression
            "against teacher labels, and produce a warm-start checkpoint for RL."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(run_dir=CLONING_RUN_DIR, allow_warmstart=False, reset_obs_stats_on_resume=True),  # adds the run IO config group
            TaskIdentity(),  # adds the task identity config group
            final_stage_gate(),  # adds the final stage gate config group
            final_finger_centering(),  # adds the final finger centering config group
            LiftSuccessConfig(success_height=0.035),  # adds the lift success config group
            ContactPoseFallbackConfig(),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            TeacherPreholdConfig(),  # adds the teacher prehold config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(reward_normalization=False),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ContactReward(centered_contact=2.0),  # adds the contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=300_000,  # Overrides the total steps setting for this config preset
                start_steps=10_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=300_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                updates_per_step=16,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=False,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=-1,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=-1,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=-1,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=-1,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.03,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.01,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(eval_episodes=2, checkpoint_every=50_000, rolling_checkpoint_every=50_000, rolling_checkpoint_keep=8),  # adds the runtime config group
        ),  # closes the current expression
    )  # closes the current expression


def cloning_strict_centered() -> RunProfile:  # builds the cloning strict centered run profile
    """Redo cloning with delayed finger unlock and tighter centered/z gates."""

    return RunProfile(  # returns the assembled run profile
        name="cloning_strict_centered",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Teacher cloning redo that preserves the centered arm path but only "  # adds literal text to the surrounding expression
            "unlocks fingers after a tighter, lower, more stable pocket."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=STRICT_CLONING_RUN_DIR,  # Builds the run directory string for this profile
                allow_warmstart=False,  # Overrides the allow warmstart setting for this config preset
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            strict_stage_gate(),  # adds the strict stage gate config group
            strict_finger_centering(),  # adds the strict finger centering config group
            strict_lift_success(),  # adds the strict lift success config group
            strict_contact_pose_fallback(),  # adds the strict contact pose fallback config group
            TeacherProfile(finger_unlock_requires_arm_hold=True),  # adds the teacher profile config group
            ArmHoldCenteringConfig(),  # adds the arm hold centering config group
            TeacherPreholdConfig(),  # adds the teacher prehold config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(reward_normalization=False),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ContactReward(centered_contact=2.0),  # adds the contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=300_000,  # Overrides the total steps setting for this config preset
                start_steps=10_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=300_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                updates_per_step=16,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=False,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=-1,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=-1,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=-1,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=-1,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.03,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.01,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(  # starts the runtime config block
                eval_episodes=2,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=50_000,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=50_000,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=8,  # Overrides the rolling checkpoint keep setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def cloning_centered_v2() -> RunProfile:  # builds the cloning centered v2 run profile
    """Redo cloning with the v1 reachable path and moderately stricter closure."""

    return RunProfile(  # returns the assembled run profile
        name="cloning_centered_v2",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Teacher cloning redo after the strict run regressed: preserve the "  # adds literal text to the surrounding expression
            "working v1 topdown reach path, slow finger unlock, add moderate "  # adds literal text to the surrounding expression
            "pre-curl z/centering gates, and tighten final centered-lift success."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=CENTERED_V2_CLONING_RUN_DIR,  # Builds the run directory string for this profile
                allow_warmstart=False,  # Overrides the allow warmstart setting for this config preset
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            final_stage_gate(),  # adds the final stage gate config group
            centered_v2_finger_centering(),  # adds the centered v2 finger centering config group
            centered_v2_lift_success(),  # adds the centered v2 lift success config group
            ContactPoseFallbackConfig(fallback_steps=40),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            TeacherPreholdConfig(),  # adds the teacher prehold config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(reward_normalization=False),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            centered_v2_contact_reward(),  # adds the centered v2 contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=300_000,  # Overrides the total steps setting for this config preset
                start_steps=10_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=300_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                updates_per_step=16,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=False,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=-1,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=-1,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=-1,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=-1,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.03,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.01,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(  # starts the runtime config block
                eval_episodes=2,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=50_000,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=50_000,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=8,  # Overrides the rolling checkpoint keep setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def cloning_ik_strict_v3() -> RunProfile:  # builds the cloning IK strict v3 run profile
    """Teacher cloning with stronger direct fingertip IK and longer teacher replay."""

    return RunProfile(  # returns the assembled run profile
        name="cloning_ik_strict_v3",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Strict-contract cloning run: direct stacked fingertip IK correction "  # adds literal text to the surrounding expression
            "for thumb/index/middle, a full-episode teacher-only replay prefix, "  # adds literal text to the surrounding expression
            "then BC/policy-assist cloning on that stronger replay manifold."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=IK_V3_CLONING_RUN_DIR,  # Builds the run directory string for this profile
                allow_warmstart=False,  # Overrides the allow warmstart setting for this config preset
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            final_stage_gate(),  # adds the final stage gate config group
            ik_v3_finger_centering(),  # adds the IK v3 finger centering config group
            ik_v3_lift_success(),  # adds the IK v3 lift success config group
            ContactPoseFallbackConfig(fallback_steps=50),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            ik_v3_teacher_prehold(),  # adds the IK v3 teacher prehold config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(reward_normalization=False),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ik_v3_contact_reward(),  # adds the IK v3 contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=400_000,  # Overrides the total steps setting for this config preset
                start_steps=60_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=400_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                updates_per_step=16,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=False,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=-1,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=-1,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=-1,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=-1,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.03,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.01,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(  # starts the runtime config block
                eval_episodes=2,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=50_000,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=50_000,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=8,  # Overrides the rolling checkpoint keep setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def dagger_rl_from_cloning() -> RunProfile:  # builds the dagger RL from cloning run profile
    """DAgger/BC manifold expansion followed by TD3, initialized from cloning."""

    run_dir = "runs/dagger_bc_rl_from_cloning_v1"  # Builds the run directory string for this profile
    return RunProfile(  # returns the assembled run profile
        name="dagger_rl_from_cloning",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Clean final-project stage: initialize the actor from the local "  # adds literal text to the surrounding expression
            "teacher-cloning checkpoint, fill replay with DAgger teacher labels "  # adds literal text to the surrounding expression
            "and BC pressure, then switch in-process to TD3 refinement."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=run_dir,  # Builds the run directory string for this profile
                actor_init_checkpoint=CLONING_CHECKPOINT,  # Overrides the actor init checkpoint setting for this config preset
                reset_obs_stats_on_resume=False,  # Overrides the reset obs stats on resume setting for this config preset
                allow_warmstart=True,  # Overrides the allow warmstart setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            final_stage_gate(),  # adds the final stage gate config group
            final_finger_centering(),  # adds the final finger centering config group
            LiftSuccessConfig(),  # adds the lift success config group
            ContactPoseFallbackConfig(),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            TeacherPreholdConfig(),  # adds the teacher prehold config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ContactReward(),  # adds the contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            CoreTrainingConfig(),  # adds the core training config group
            DeterminismConfig(),  # adds the determinism config group
            DaggerConfig(),  # adds the dagger config group
            OptimizationConfig(),  # adds the optimization config group
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(),  # adds the runtime config group
        ),  # closes the current expression
    )  # closes the current expression


def pipeline_smoke() -> RunProfile:  # builds the pipeline smoke run profile
    """Short run that exercises teacher labels, DAgger/BC, and the RL switch."""

    run_dir = "runs/pipeline_smoke_teacher_dagger_rl"  # Builds the run directory string for this profile
    return RunProfile(  # returns the assembled run profile
        name="pipeline_smoke",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Tiny runtime smoke test for the supported pipeline: teacher warmup, "  # adds literal text to the surrounding expression
            "DAgger/BC relabeling, replay writes, and the in-process TD3 switch."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(run_dir=run_dir, allow_warmstart=False, reset_obs_stats_on_resume=True),  # adds the run IO config group
            TaskIdentity(),  # adds the task identity config group
            final_stage_gate(),  # adds the final stage gate config group
            final_finger_centering(),  # adds the final finger centering config group
            LiftSuccessConfig(success_height=0.035),  # adds the lift success config group
            ContactPoseFallbackConfig(),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            TeacherPreholdConfig(),  # adds the teacher prehold config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ContactReward(centered_contact=2.0),  # adds the contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            CoreTrainingConfig(  # starts the core training config block
                num_envs=4,  # Overrides the num envs setting for this config preset
                total_steps=128,  # Overrides the total steps setting for this config preset
                start_steps=16,  # Overrides the start steps setting for this config preset
                bc_only_steps=64,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=64,  # Overrides the RL phase start steps setting for this config preset
                replay_size=2048,  # Overrides the replay size setting for this config preset
                batch_size=32,  # Overrides the batch size setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                updates_per_step=1,  # Overrides the updates per step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DeterminismConfig(),  # adds the determinism config group
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=True,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=0.5,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=64,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=0.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=8.0,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=2.0,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=8.0,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=2.0,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=128,  # Overrides the teacher BC decay steps setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(  # starts the RL switch config block
                updates_per_step=1,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
                actor_freeze_steps=0,  # Overrides the actor freeze steps setting for this config preset
                teacher_bc_arm_weight=0.0,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=0.0,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
            ),  # closes the current expression
            RuntimeConfig(  # starts the runtime config block
                eval_steps=60,  # Overrides the eval steps setting for this config preset
                eval_episodes=1,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=0,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=0,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=0,  # Overrides the rolling checkpoint keep setting for this config preset
                log_every=16,  # Overrides the log every setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def cloning_ik_strict_v4() -> RunProfile:  # builds the cloning IK strict v4 run profile
    """Teacher cloning run with earlier direct fingertip IK and delayed arm freeze."""

    return RunProfile(  # returns the assembled run profile
        name="cloning_ik_strict_v4",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Strict-contract cloning run v4: run the stacked fingertip Jacobian "  # adds literal text to the surrounding expression
            "from the approach stage, allow wrist-roll correction, delay arm-hold "  # adds literal text to the surrounding expression
            "freeze until the pre-close center gate, and keep final lift success strict."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=IK_V4_CLONING_RUN_DIR,  # Builds the run directory string for this profile
                allow_warmstart=False,  # Overrides the allow warmstart setting for this config preset
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            final_stage_gate(),  # adds the final stage gate config group
            ik_v4_finger_centering(),  # adds the IK v4 finger centering config group
            ik_v3_lift_success(),  # adds the IK v3 lift success config group
            ContactPoseFallbackConfig(fallback_steps=70),  # adds the contact pose fallback config group
            TeacherProfile(),  # adds the teacher profile config group
            ik_v4_arm_hold_centering(),  # adds the IK v4 arm hold centering config group
            ik_v4_teacher_prehold(),  # adds the IK v4 teacher prehold config group
            ik_v4_teacher_prehold_advanced(),  # adds the IK v4 teacher prehold advanced config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(reward_normalization=False),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ik_v3_contact_reward(),  # adds the IK v3 contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=450_000,  # Overrides the total steps setting for this config preset
                start_steps=90_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=450_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                updates_per_step=16,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=False,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=-1,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=-1,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=-1,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=-1,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.02,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.005,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(  # starts the runtime config block
                eval_steps=80,  # Overrides the eval steps setting for this config preset
                eval_episodes=1,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=50_000,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=50_000,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=8,  # Overrides the rolling checkpoint keep setting for this config preset
                log_every=16,  # Overrides the log every setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def cloning_ik_2f_v5() -> RunProfile:  # builds the cloning IK 2f v5 run profile
    """Two-finger IK replay run based on the old dual-tip teacher contract."""

    return RunProfile(  # returns the assembled run profile
        name="cloning_ik_2f_v5",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Dual-tip cloning run: disable middle-finger centering and middle "  # adds literal text to the surrounding expression
            "contact teacher, use a thumb/index DLS post-pass, and open contact "  # adds literal text to the surrounding expression
            "readiness as soon as the two-finger pocket is stable."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=IK_2F_V5_CLONING_RUN_DIR,  # Builds the run directory string for this profile
                allow_warmstart=False,  # Overrides the allow warmstart setting for this config preset
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
            ),  # closes the current expression
            TaskIdentity(),  # adds the task identity config group
            ik_2f_v5_stage_gate(),  # adds the IK 2f v5 stage gate config group
            ik_2f_v5_finger_centering(),  # adds the IK 2f v5 finger centering config group
            ik_v3_lift_success(),  # adds the IK v3 lift success config group
            ik_2f_v5_contact_pose_fallback(),  # adds the IK 2f v5 contact pose fallback config group
            TeacherProfile(middle_scale=0.0),  # adds the teacher profile config group
            ik_v4_arm_hold_centering(),  # adds the IK v4 arm hold centering config group
            ik_2f_v5_teacher_prehold(),  # adds the IK 2f v5 teacher prehold config group
            ik_2f_v5_teacher_prehold_advanced(),  # adds the IK 2f v5 teacher prehold advanced config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(reward_normalization=False),  # adds the action surface config group
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(),  # adds the lift reward config group
            ik_v3_contact_reward(),  # adds the IK v3 contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=450_000,  # Overrides the total steps setting for this config preset
                start_steps=90_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=450_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                updates_per_step=16,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=False,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=-1,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=-1,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=-1,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=-1,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.02,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.005,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.0,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(),  # adds the RL switch config group
            RuntimeConfig(  # starts the runtime config block
                eval_steps=80,  # Overrides the eval steps setting for this config preset
                eval_episodes=1,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=50_000,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=50_000,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=8,  # Overrides the rolling checkpoint keep setting for this config preset
                log_every=16,  # Overrides the log every setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3() -> RunProfile:  # builds the teacher dagger upstream FastTD3 run profile
    """Teacher -> BC/DAgger -> upstream FastTD3 in one deterministic run."""

    run_dir = "runs/teacher_dagger_upstream_fasttd3_v6_priority_fixes"  # Builds the run directory string for this profile
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Single-process final pipeline using the existing IK/contact teacher, "  # adds literal text to the surrounding expression
            "BC and DAgger replay fill, then RL refinement with the upstream "  # adds literal text to the surrounding expression
            "FastTD3 distributional actor/critic backend."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script="",  # Selects module launch for the refactored training path
        module="training.native_entrypoint",  # Selects the native modular training entrypoint
        groups=(  # Starts the config groups included in this run profile
            RunIOConfig(  # starts the run IO config block
                run_dir=run_dir,  # Builds the run directory string for this profile
                allow_warmstart=False,  # Overrides the allow warmstart setting for this config preset
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
                handoff_checkpoint_path=f"{run_dir}/handoff_replay.pt",  # Overrides the handoff checkpoint path setting for this config preset
            ),  # closes the current expression
            UpstreamFastTD3BackendConfig(),  # adds the upstream fast TD3 backend config group
            NativeEntrypointConfig(  # starts the native entrypoint config block
                teacher_provider="env",  # Selects the env-backed teacher provider
                contact_attr_parts=True,  # Enables contact teacher parts from env attrs
            ),  # closes the current expression
            TaskIdentity(block_static_friction=1.5, block_dynamic_friction=1.5),  # adds the task identity config group
            ik_2f_v5_stage_gate(),  # adds the IK 2f v5 stage gate config group
            ik_2f_v5_finger_centering(),  # adds the IK 2f v5 finger centering config group
            ik_v3_lift_success(),  # adds the IK v3 lift success config group
            ik_2f_v5_contact_pose_fallback(),  # adds the IK 2f v5 contact pose fallback config group
            TeacherProfile(middle_scale=0.0),  # adds the teacher profile config group
            ik_v4_arm_hold_centering(),  # adds the IK v4 arm hold centering config group
            ik_2f_v5_teacher_prehold(),  # adds the IK 2f v5 teacher prehold config group
            ik_2f_v5_teacher_prehold_advanced(),  # adds the IK 2f v5 teacher prehold advanced config group
            TeacherLiftConfig(),  # adds the teacher lift config group
            ActionSurfaceConfig(  # starts the action surface config block
                reward_normalization=False,  # Overrides the reward normalization setting for this config preset
                actor_q_action_gate_mode="raw",  # Overrides the actor Q action gate mode setting for this config preset
                actor_bc_action_gate_mode="raw",  # Overrides the actor BC action gate mode setting for this config preset
            ),  # closes the current expression
            ReachAlignReward(),  # adds the reach align reward config group
            LiftReward(  # starts the lift reward config block
                block_drop_penalty=-5,  # Overrides the block drop penalty setting for this config preset
                lift_penalty_height_start=0.020,  # Overrides the lift penalty height start setting for this config preset
                lift_penalty_height_ramp=0.020,  # Overrides the lift penalty height ramp setting for this config preset
            ),  # closes the current expression
            ik_v3_contact_reward(stage2_floor=0.5),  # adds the IK v3 contact reward config group
            RuntimeReward(),  # adds the runtime reward config group
            DeterminismConfig(),  # adds the determinism config group
            CoreTrainingConfig(  # starts the core training config block
                total_steps=800_000,  # Overrides the total steps setting for this config preset
                start_steps=90_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=300_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=300_000,  # Overrides the RL phase start steps setting for this config preset
                replay_size=1_000_000,  # Overrides the replay size setting for this config preset
                batch_size=384,  # Overrides the batch size setting for this config preset
                updates_per_step=12,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=1,  # Overrides the policy delay setting for this config preset
                gamma=0.995,  # Overrides the gamma setting for this config preset
                tau=0.005,  # Overrides the tau setting for this config preset
            ),  # closes the current expression
            DaggerConfig(  # starts the dagger config block
                policy_bc_relabel=True,  # Overrides the policy BC relabel setting for this config preset
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.25,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=600_000,  # Overrides the policy assist decay steps setting for this config preset
                bc_only_weight=10.0,  # Overrides the BC only weight setting for this config preset
                bc_only_arm_weight=4.0,  # Overrides the BC only arm weight setting for this config preset
                bc_only_finger_weight=2.0,  # Overrides the BC only finger weight setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=3.0,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=1.5,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=600_000,  # Overrides the teacher BC decay steps setting for this config preset
                assist_noise_arm=0.01,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.004,  # Overrides the assist noise finger setting for this config preset
            ),  # closes the current expression
            OptimizationConfig(  # starts the optimization config block
                actor_lr=3e-4,  # Overrides the actor learning rate setting for this config preset
                critic_lr=1e-4,  # Overrides the critic learning rate setting for this config preset
                target_q_clip=50,  # Overrides the target Q clip setting for this config preset
                critic_grad_clip=5.0,  # Overrides the critic grad clip setting for this config preset
                exploration_noise=0.01,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.012,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.0,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                actor_pre_tanh_l2=0.0,  # Overrides the actor pre tanh l2 setting for this config preset
            ),  # closes the current expression
            RlSwitchConfig(  # starts the RL switch config block
                updates_per_step=8,  # Overrides the updates per step setting for this config preset
                n_step=1,  # Overrides the n step setting for this config preset
                policy_delay=4,  # Overrides the policy delay setting for this config preset
                tau=0.0005,  # Overrides the tau setting for this config preset
                actor_lr=2e-6,  # Overrides the actor learning rate setting for this config preset
                critic_lr=5e-5,  # Overrides the critic learning rate setting for this config preset
                exploration_noise=0.0,  # Overrides the exploration noise setting for this config preset
                exploration_noise_finger=0.02,  # Overrides the exploration noise finger setting for this config preset
                policy_noise=0.0,  # Overrides the policy noise setting for this config preset
                policy_noise_finger=0.02,  # Overrides the policy noise finger setting for this config preset
                noise_clip=0.05,  # Overrides the noise clip setting for this config preset
                policy_bc_relabel=True,  # Overrides the policy BC relabel setting for this config preset
                teacher_bc_weight=0.0,  # Overrides the teacher BC weight setting for this config preset
                teacher_bc_arm_weight=8.0,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=2.0,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=4_000_000,  # Overrides the teacher BC decay steps setting for this config preset
                actor_freeze_steps=2_000_000,  # Overrides the actor freeze steps setting for this config preset
            ),  # closes the current expression
            RlAssistHandoffConfig(  # starts the RL assist handoff config block
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.90,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=2_000_000,  # Overrides the policy assist decay steps setting for this config preset
            ),  # closes the current expression
            RuntimeConfig(  # starts the runtime config block
                eval_steps=600,  # Overrides the eval steps setting for this config preset
                eval_episodes=2,  # Overrides the eval episodes setting for this config preset
                checkpoint_every=50_000,  # Overrides the checkpoint every setting for this config preset
                rolling_checkpoint_every=50_000,  # Overrides the rolling checkpoint every setting for this config preset
                rolling_checkpoint_keep=16,  # Overrides the rolling checkpoint keep setting for this config preset
                log_every=2500,  # Overrides the log every setting for this config preset
            ),  # closes the current expression
        ),  # closes the current expression
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_from_handoff() -> RunProfile:  # builds the teacher dagger upstream FastTD3 from handoff run profile
    """Resume RL variants from the replay-inclusive v6 handoff checkpoint."""

    base = teacher_dagger_upstream_fasttd3()  # Loads the base run profile before replacing selected groups
    source_dir = "runs/teacher_dagger_upstream_fasttd3_v6_priority_fixes"  # Overrides the source dir setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v6_from_handoff"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    if not isinstance(groups[0], RunIOConfig):  # Checks whether not isinstance(groups[0], run IO config)
        raise TypeError("expected RunIOConfig as first profile group")  # raises an error for invalid config state
    groups[0] = replace(  # stores the resolved value in the mapping
        groups[0],  # continues this config expression
        run_dir=run_dir,  # Builds the run directory string for this profile
        resume_checkpoint=f"{source_dir}/handoff_replay.pt",  # Overrides the resume checkpoint setting for this config preset
        reset_obs_stats_on_resume=False,  # Overrides the reset obs stats on resume setting for this config preset
        resume_replay=True,  # Overrides the resume replay setting for this config preset
        resume_global_step=True,  # Overrides the resume global step setting for this config preset
        handoff_checkpoint_path=None,  # Overrides the handoff checkpoint path setting for this config preset
        stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
    )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_from_handoff",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "RL-only continuation from the replay-inclusive BC/DAgger handoff "  # adds literal text to the surrounding expression
            "checkpoint produced by teacher_dagger_upstream_fasttd3."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v7_warmstart() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v7 warmstart run profile
    """Warmstart from the replay handoff, then let FastTD3 actually move the actor."""

    base = teacher_dagger_upstream_fasttd3()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v6/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v7_warmstart_unfrozen"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                reset_obs_stats_on_resume=True,  # Overrides the reset obs stats on resume setting for this config preset
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, RlSwitchConfig):  # Checks alternate branch for isinstance(group, RL switch config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                actor_lr=8e-6,  # Overrides the actor learning rate setting for this config preset
                teacher_bc_arm_weight=4.0,  # Overrides the teacher BC arm weight setting for this config preset
                teacher_bc_finger_weight=1.0,  # Overrides the teacher BC finger weight setting for this config preset
                teacher_bc_decay_steps=1_000_000,  # Overrides the teacher BC decay steps setting for this config preset
                actor_freeze_steps=50_000,  # Overrides the actor freeze steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, RlAssistHandoffConfig):  # Checks alternate branch for isinstance(group, RL assist handoff config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.65,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=750_000,  # Overrides the policy assist decay steps setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v7_warmstart",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Warmstart-aware continuation using the replay-inclusive BC/DAgger "  # adds literal text to the surrounding expression
            "handoff when compatible, regenerating it when missing or stale. "  # adds literal text to the surrounding expression
            "Compared with v6, this keeps the priority fixes but unfreezes the "  # adds literal text to the surrounding expression
            "actor after a short critic refit and bleeds teacher assist to 0.65."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v8_600k_handoff() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v8 600k handoff run profile
    """Generate/reuse a mature 600k BC/DAgger replay handoff, then run FastTD3."""

    base = teacher_dagger_upstream_fasttd3_v7_warmstart()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v8_600k/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v8_600k_handoff_rl"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=1_100_000,  # Overrides the total steps setting for this config preset
                bc_only_steps=600_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=600_000,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v8_600k_handoff",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Warmstart-aware FastTD3 run that treats 600k as the end of the "  # adds literal text to the surrounding expression
            "BC/DAgger handoff. If the compatible replay-inclusive handoff "  # adds literal text to the surrounding expression
            "exists it resumes from that boundary; otherwise it regenerates "  # adds literal text to the surrounding expression
            "the full imitation prefix, saves the handoff, and continues RL."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v8_600k_from_handoff() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v8 600k from handoff run profile
    """Resume RL directly from the replay-inclusive v8 600k handoff."""

    base = teacher_dagger_upstream_fasttd3_v8_600k_handoff()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v8_600k/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v8_600k_from_handoff_rl"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                resume_checkpoint=handoff_path,  # Overrides the resume checkpoint setting for this config preset
                reset_obs_stats_on_resume=False,  # Overrides the reset obs stats on resume setting for this config preset
                resume_replay=True,  # Overrides the resume replay setting for this config preset
                resume_global_step=True,  # Overrides the resume global step setting for this config preset
                handoff_checkpoint_path=None,  # Overrides the handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v8_600k_from_handoff",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "RL continuation from the replay-inclusive 600k BC/DAgger handoff "  # adds literal text to the surrounding expression
            "created by teacher_dagger_upstream_fasttd3_v8_600k_handoff. This "  # adds literal text to the surrounding expression
            "skips the repeated imitation prefix and resumes with replay and "  # adds literal text to the surrounding expression
            "global_step intact."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v8 600k from step checkpoint run profile
    """Resume from the v8 600k policy checkpoint when replay is unavailable."""

    base = teacher_dagger_upstream_fasttd3_v8_600k_handoff()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint_rl"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                resume_checkpoint=V8_600K_STEP_CHECKPOINT,  # Overrides the resume checkpoint setting for this config preset
                reset_obs_stats_on_resume=False,  # Overrides the reset obs stats on resume setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=True,  # Overrides the resume global step setting for this config preset
                handoff_checkpoint_path=None,  # Overrides the handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "RL continuation from the v8 600k policy checkpoint "  # adds literal text to the surrounding expression
            f"{V8_600K_STEP_CHECKPOINT}. The checkpoint does not contain replay, "  # adds formatted data to the returned text
            "so this preserves model state and global_step but starts with a "  # adds literal text to the surrounding expression
            "fresh replay buffer."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v9_vertical_center_drop() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v9 vertical center drop run profile
    """Retrain with a centered topdown approach followed by vertical descent."""

    base = teacher_dagger_upstream_fasttd3_v8_600k_handoff()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v9_vertical_center_drop/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v9_vertical_center_drop"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                inward_m=0.0,  # Overrides the inward m setting for this config preset
                missing_contact_extra_inward=0.0,  # Overrides the missing contact extra inward setting for this config preset
                descent_requires_center=True,  # Overrides the descent requires center setting for this config preset
                inward_requires_center=True,  # Overrides the inward requires center setting for this config preset
                inward_vertical_only=True,  # Overrides the inward vertical only setting for this config preset
                vertical_drop_lock_xy=True,  # Overrides the vertical drop lock XY setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                tip_jacobian_z_requires_center=True,  # Overrides the tip jacobian Z requires center setting for this config preset
                tip_servo_z_requires_center=True,  # Overrides the tip servo Z requires center setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v9_vertical_center_drop",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v8 plus a stricter teacher approach: align the thumb/back-finger "  # adds literal text to the surrounding expression
            "plane over the block first, suppress Z tip corrections until the "  # adds literal text to the surrounding expression
            "center gate is ready, then descend vertically with XY locked."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v10 pose oriented drop run profile
    """v9 with palm orientation preserved so the finger plane stays horizontal."""

    base = teacher_dagger_upstream_fasttd3_v9_vertical_center_drop()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(group, prehold_ik_position_only=False)  # stores the resolved value in the mapping
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(group, position_only_stage_min=99)  # stores the resolved value in the mapping
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v9 vertical-center-drop plus pose-mode IK during approach/drop. "  # adds literal text to the surrounding expression
            "This preserves the palm target quaternion so the thumb/back-finger "  # adds literal text to the surrounding expression
            "plane stays parallel to the block top instead of pitching the thumb "  # adds literal text to the surrounding expression
            "down into the cube."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v11_finger_plane_basis() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v11 finger plane basis run profile
    """v10 with target palm basis prioritizing the thumb/back-finger plane."""

    base = teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v11_finger_plane_basis/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v11_finger_plane_basis"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="yaw_priority",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v11_finger_plane_basis",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v10 plus a yaw-priority palm target basis. The target quaternion now "  # adds literal text to the surrounding expression
            "keeps the thumb-to-back-finger line horizontal, so the open-hand "  # adds literal text to the surrounding expression
            "finger plane approaches parallel to the block top before vertical drop."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v12_plane_center_unlock() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v12 plane center unlock run profile
    """v11 with unlock based on centered grasp plane, not final fingertip span."""

    base = teacher_dagger_upstream_fasttd3_v11_finger_plane_basis()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v12_plane_center_unlock/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v12_plane_center_unlock"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                hold_steps=1,  # Overrides the hold steps setting for this config preset
                unlock_ramp_steps=45,  # Overrides the unlock ramp steps setting for this config preset
                align_angle_max_deg=45,  # Overrides the align angle maximum deg setting for this config preset
                tip_xy_max=0.050,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.180,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.125,  # Overrides the tip Z maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ContactPoseFallbackConfig):  # Checks alternate branch for isinstance(group, contact pose fallback config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                align_err_max=0.20,  # Overrides the align error maximum setting for this config preset
                palm_height_max=0.08,  # Overrides the palm height maximum setting for this config preset
                contact_pose_hold_steps=1,  # Overrides the contact pose hold steps setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v12_plane_center_unlock",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v11 plus a plane-center unlock gate. The teacher only requires the "  # adds literal text to the surrounding expression
            "thumb/back-finger grasp plane center to be over the block before "  # adds literal text to the surrounding expression
            "vertical descent; individual fingertip side-face errors are left "  # adds literal text to the surrounding expression
            "for closure/contact to resolve."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v13_contact_plane_basis() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v13 contact plane basis run profile
    """v12 with palm target basis forcing the finger contact plane horizontal."""

    base = teacher_dagger_upstream_fasttd3_v12_plane_center_unlock()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v13_contact_plane_basis/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v13_contact_plane_basis"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="contact_plane",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v13_contact_plane_basis",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v12 plus a contact-plane palm target basis. The target quaternion "  # adds literal text to the surrounding expression
            "maps the anatomical thumb/index/middle plane to the table plane, "  # adds literal text to the surrounding expression
            "while the active grip model remains two-finger thumb/index."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v14_segment_plane_basis() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v14 segment plane basis run profile
    """v13 with target basis built from visible thumb/index segment axes."""

    base = teacher_dagger_upstream_fasttd3_v13_contact_plane_basis()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v14_segment_plane_basis/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v14_segment_plane_basis"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="finger_segment_plane",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v14_segment_plane_basis",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v13 but builds the palm target from thumb_1->thumb_2 and "  # adds literal text to the surrounding expression
            "index_0->index_1 segment directions. This targets the visible "  # adds literal text to the surrounding expression
            "thumb-down/perpendicular approach instead of only coplanar link origins."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v15 segment plane axis X run profile
    """v14 with the thumb/index target axis rotated 90 degrees to world X."""

    base = teacher_dagger_upstream_fasttd3_v14_segment_plane_basis()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="finger_segment_plane",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_yaw_world_axis="x",  # Overrides the target palm yaw world axis setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v14 with TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS=x. This keeps the "  # adds literal text to the surrounding expression
            "visible finger segment plane horizontal but rotates the in-plane "  # adds literal text to the surrounding expression
            "thumb/index targeting axis by 90 degrees."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v16 segment plane axis neg X run profile
    """v15 mirrored around vertical so thumb/index swap sides on the block."""

    base = teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="finger_segment_plane",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_yaw_world_axis="-x",  # Overrides the target palm yaw world axis setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v15 mirrored to TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS=-x. Use this "  # adds literal text to the surrounding expression
            "when the finger plane is horizontal but the thumb/index sides are "  # adds literal text to the surrounding expression
            "swapped or the approach comes from the wrong side of the cube."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v17 segment axis X palm back run profile
    """v15 plus a small palm offset to reduce index-only block shove."""

    base = teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="finger_segment_plane",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_yaw_world_axis="x",  # Overrides the target palm yaw world axis setting for this config preset
                topdown_palm_offset_x=-0.035,  # Overrides the topdown palm offset X setting for this config preset
                topdown_palm_offset_y=0.0,  # Overrides the topdown palm offset Y setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v15 plus TEACHER_TOPDOWN_PALM_OFFSET_X=-0.035. v15 had the "  # adds literal text to the surrounding expression
            "better finger-plane orientation but hit index-first and shoved "  # adds literal text to the surrounding expression
            "the block; this shifts the palm target backward to center the "  # adds literal text to the surrounding expression
            "two-finger pocket before descent."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v18_center_span_ik() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v18 center span IK run profile
    """v15 with direct center/span Jacobian IK for the two-finger pocket."""

    base = teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v18_center_span_ik/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v18_center_span_ik"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                tip_jacobian_gain=1.35,  # Overrides the tip jacobian gain setting for this config preset
                tip_jacobian_damping=0.035,  # Overrides the tip jacobian damping setting for this config preset
                tip_jacobian_max_joint_step=0.070,  # Overrides the tip jacobian maximum joint step setting for this config preset
                tip_jacobian_stage_min=0,  # Overrides the tip jacobian stage minimum setting for this config preset
                ik_tip_servo_gain=1.35,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.150,  # Overrides the IK tip servo maximum m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="finger_segment_plane",  # Overrides the target palm basis setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_yaw_world_axis="x",  # Overrides the target palm yaw world axis setting for this config preset
                topdown_palm_offset_x=0.0,  # Overrides the topdown palm offset X setting for this config preset
                topdown_palm_offset_y=0.0,  # Overrides the topdown palm offset Y setting for this config preset
                tip_jacobian_mode="center_span",  # Overrides the tip jacobian mode setting for this config preset
                tip_jacobian_center_xy_weight=3.0,  # Overrides the tip jacobian center XY weight setting for this config preset
                tip_jacobian_center_z_weight=0.25,  # Overrides the tip jacobian center Z weight setting for this config preset
                tip_jacobian_span_xy_weight=1.0,  # Overrides the tip jacobian span XY weight setting for this config preset
                tip_jacobian_span_z_weight=4.0,  # Overrides the tip jacobian span Z weight setting for this config preset
                tip_jacobian_center_z_requires_center=True,  # Overrides the tip jacobian center Z requires center setting for this config preset
                tip_jacobian_accept_worse=False,  # Overrides the tip jacobian accept worse setting for this config preset
                tip_jacobian_max_worse_m=0.0005,  # Overrides the tip jacobian maximum worse m setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v18_center_span_ik",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v15 with the post-palm IK solving a stacked center/span system: "  # adds literal text to the surrounding expression
            "thumb-index pocket center over the cube, thumb-index span across "  # adds literal text to the surrounding expression
            "the target faces, and high span-Z weight to hold align_angle near zero."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v19 position only vertical grip run profile
    """Clean position-only teacher: grip center above cube, then vertical drop."""

    base = teacher_dagger_upstream_fasttd3_v9_vertical_center_drop()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                prehold_ik_position_only=True,  # Overrides the prehold IK position only setting for this config preset
                prehold_align_angle_servo=True,  # Overrides the prehold align angle servo setting for this config preset
                middle_scale=0.0,  # Overrides the middle scale setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                tip_jacobian_ik=False,  # Overrides the tip jacobian IK setting for this config preset
                planar_align_servo=False,  # Overrides the planar align servo setting for this config preset
                ik_tip_servo_gain=0.0,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.0,  # Overrides the IK tip servo maximum m setting for this config preset
                pocket_sweep=False,  # Overrides the pocket sweep setting for this config preset
                align_angle_stage_min=0,  # Overrides the align angle stage minimum setting for this config preset
                align_angle_gain=1.2,  # Overrides the align angle gain setting for this config preset
                align_angle_max_dz=0.045,  # Overrides the align angle maximum dz setting for this config preset
                align_angle_max_joint_step=0.10,  # Overrides the align angle maximum joint step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                position_only_stage_min=0,  # Overrides the position only stage minimum setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_basis=None,  # Overrides the target palm basis setting for this config preset
                target_palm_yaw_world_axis=None,  # Overrides the target palm yaw world axis setting for this config preset
                target_palm_position_mode="current_grip_offset",  # Overrides the target palm position mode setting for this config preset
                topdown_palm_offset_x=0.0,  # Overrides the topdown palm offset X setting for this config preset
                topdown_palm_offset_y=0.0,  # Overrides the topdown palm offset Y setting for this config preset
                tip_jacobian_mode=None,  # Overrides the tip jacobian mode setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                hold_steps=1,  # Overrides the hold steps setting for this config preset
                unlock_ramp_steps=60,  # Overrides the unlock ramp steps setting for this config preset
                align_angle_max_deg=20,  # Overrides the align angle maximum deg setting for this config preset
                tip_xy_max=0.060,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.120,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.110,  # Overrides the tip Z maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ContactPoseFallbackConfig):  # Checks alternate branch for isinstance(group, contact pose fallback config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                contact_pose_hold_steps=1,  # Overrides the contact pose hold steps setting for this config preset
                align_err_max=0.28,  # Overrides the align error maximum setting for this config preset
                palm_height_max=0.09,  # Overrides the palm height maximum setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Position-only topdown teacher. The palm target is computed from "  # adds literal text to the surrounding expression
            "the live thumb/index grip-center offset, so the first solve puts "  # adds literal text to the surrounding expression
            "the actual open two-finger grip center above the cube. Descent "  # adds literal text to the surrounding expression
            "then remains vertical with XY locked; only the align-angle servo "  # adds literal text to the surrounding expression
            "is allowed to level the thumb/index line."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v20 task space topdown IK run profile
    """Task-constrained IK: grasp center, palm-down drop vector, thumb-index axis."""

    base = teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip()  # Loads the base run profile before replacing selected groups
    handoff_path = "runs/replay_handoffs/teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik/handoff_replay.pt"  # Overrides the handoff path setting for this config preset
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                prehold_ik_position_only=True,  # Overrides the prehold IK position only setting for this config preset
                prehold_align_angle_servo=False,  # Overrides the prehold align angle servo setting for this config preset
                middle_scale=0.0,  # Overrides the middle scale setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                tip_jacobian_ik=False,  # Overrides the tip jacobian IK setting for this config preset
                planar_align_servo=False,  # Overrides the planar align servo setting for this config preset
                ik_tip_servo_gain=0.0,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.0,  # Overrides the IK tip servo maximum m setting for this config preset
                pocket_sweep=False,  # Overrides the pocket sweep setting for this config preset
                align_angle_gain=0.0,  # Overrides the align angle gain setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_position_mode="current_grip_offset",  # Overrides the target palm position mode setting for this config preset
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_basis=None,  # Overrides the target palm basis setting for this config preset
                target_palm_yaw_world_axis=None,  # Overrides the target palm yaw world axis setting for this config preset
                topdown_palm_offset_x=0.0,  # Overrides the topdown palm offset X setting for this config preset
                topdown_palm_offset_y=0.0,  # Overrides the topdown palm offset Y setting for this config preset
                tip_jacobian_mode=None,  # Overrides the tip jacobian mode setting for this config preset
            )  # closes the current expression
    groups.append(  # appends the computed value to the collection
        TaskSpaceIKConfig(  # starts the task space IK config block
            enabled=True,  # Overrides the enabled setting for this config preset
            center_xy_weight=5.0,  # Overrides the center XY weight setting for this config preset
            center_z_weight=1.0,  # Overrides the center Z weight setting for this config preset
            span_xy_weight=1.25,  # Overrides the span XY weight setting for this config preset
            span_z_weight=5.0,  # Overrides the span Z weight setting for this config preset
            drop_weight=3.0,  # Overrides the drop weight setting for this config preset
            posture_weight=0.05,  # Overrides the posture weight setting for this config preset
            damping=0.045,  # Overrides the damping setting for this config preset
            max_joint_step=0.075,  # Overrides the max joint step setting for this config preset
        )  # closes the current expression
    )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Direct constrained DLS IK on the topdown grasp frame: solve the "  # adds literal text to the surrounding expression
            "thumb/index grip center over the block, keep the palm-to-grip "  # adds literal text to the surrounding expression
            "approach vector vertical, align the thumb-index axis with the "  # adds literal text to the surrounding expression
            "block grip axis, and use null-space posture toward default joints."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v21 task space delayed unlock run profile
    """Task-space IK with the finger unlock gate restored and slowed down."""

    base = teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                bypass_unlock=False,  # Overrides the bypass unlock setting for this config preset
                close_rate=0.006,  # Overrides the close rate setting for this config preset
                finger_unlock_min=0.35,  # Overrides the finger unlock minimum setting for this config preset
                finger_arm_hold_fallback=False,  # Overrides the finger arm hold fallback setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                requires_center=True,  # Overrides the requires center setting for this config preset
                latch=True,  # Overrides the latch setting for this config preset
                hold_steps=8,  # Overrides the hold steps setting for this config preset
                unlock_ramp_steps=220,  # Overrides the unlock ramp steps setting for this config preset
                align_angle_max_deg=10,  # Overrides the align angle maximum deg setting for this config preset
                three_finger_centering=False,  # Overrides the three finger centering setting for this config preset
                tip_xy_max=0.035,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.065,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.070,  # Overrides the tip Z maximum setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Same constrained task-space topdown IK as v20, but restore the "  # adds literal text to the surrounding expression
            "finger unlock contract: no bypass, no arm-hold fallback, eight "  # adds literal text to the surrounding expression
            "centered frames required, 35 percent unlock progress before the "  # adds literal text to the surrounding expression
            "contact teacher may close, and slower finger closure."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v22 task space descend then unlock run profile
    """Allow vertical descent before the centered finger-unlock gate fires."""

    base = teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                inward_m=0.0,  # Overrides the inward m setting for this config preset
                missing_contact_extra_inward=0.0,  # Overrides the missing contact extra inward setting for this config preset
                descent_requires_center=False,  # Overrides the descent requires center setting for this config preset
                inward_requires_center=False,  # Overrides the inward requires center setting for this config preset
                inward_vertical_only=True,  # Overrides the inward vertical only setting for this config preset
                vertical_drop_lock_xy=True,  # Overrides the vertical drop lock XY setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Task-space topdown IK with separated gates: the arm may descend "  # adds literal text to the surrounding expression
            "vertically once the topdown pose is ready, while finger closure "  # adds literal text to the surrounding expression
            "still waits for the strict centered unlock gate."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v23 task space strict continuous unlock run profile
    """Lower vertical descent and require continuous two-finger centering before close."""

    base = teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                close_rate=0.004,  # Overrides the close rate setting for this config preset
                finger_unlock_min=0.55,  # Overrides the finger unlock minimum setting for this config preset
                finger_arm_hold_fallback=False,  # Overrides the finger arm hold fallback setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                descent_z=0.060,  # Overrides the descent Z setting for this config preset
                missing_contact_extra_descent=0.015,  # Overrides the missing contact extra descent setting for this config preset
                descent_tip_z_target=0.002,  # Overrides the descent tip Z target setting for this config preset
                descent_requires_center=False,  # Overrides the descent requires center setting for this config preset
                inward_requires_center=False,  # Overrides the inward requires center setting for this config preset
                inward_m=0.0,  # Overrides the inward m setting for this config preset
                missing_contact_extra_inward=0.0,  # Overrides the missing contact extra inward setting for this config preset
                inward_vertical_only=True,  # Overrides the inward vertical only setting for this config preset
                vertical_drop_lock_xy=True,  # Overrides the vertical drop lock XY setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                requires_center=True,  # Overrides the requires center setting for this config preset
                latch=False,  # Overrides the latch setting for this config preset
                hold_steps=12,  # Overrides the hold steps setting for this config preset
                unlock_ramp_steps=280,  # Overrides the unlock ramp steps setting for this config preset
                align_angle_max_deg=6,  # Overrides the align angle maximum deg setting for this config preset
                three_finger_centering=False,  # Overrides the three finger centering setting for this config preset
                tip_xy_max=0.024,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.045,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.048,  # Overrides the tip Z maximum setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v22 plus stricter continuous two-finger centering: lower vertical "  # adds literal text to the surrounding expression
            "descent, no center latch, longer hold, tighter XY/Z/angle gates, "  # adds literal text to the surrounding expression
            "and slower close so one-sided index contact cannot keep curling."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v24 task space grasp center gate run profile
    """Unlock from centered grasp-frame geometry, not final side-face residuals."""

    base = teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                close_rate=0.005,  # Overrides the close rate setting for this config preset
                finger_unlock_min=0.30,  # Overrides the finger unlock minimum setting for this config preset
                finger_arm_hold_fallback=False,  # Overrides the finger arm hold fallback setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                requires_center=True,  # Overrides the requires center setting for this config preset
                latch=False,  # Overrides the latch setting for this config preset
                hold_steps=8,  # Overrides the hold steps setting for this config preset
                unlock_ramp_steps=160,  # Overrides the unlock ramp steps setting for this config preset
                align_angle_max_deg=8,  # Overrides the align angle maximum deg setting for this config preset
                three_finger_centering=False,  # Overrides the three finger centering setting for this config preset
                tip_xy_max=0.035,  # Overrides the tip XY maximum setting for this config preset
                # Before closing, the open fingers should not need to already
                # sit on the final side-face targets The task-space IK solves
                # the thumb/index grasp center to block center; closure then
                # brings individual pads onto the opposed faces
                max_tip_xy_max=0.0,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.0,  # Overrides the tip Z maximum setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v23 with the finger-unlock contract changed to match the actual "  # adds literal text to the surrounding expression
            "pre-close task: unlock from grasp-center-over-block plus plane "  # adds literal text to the surrounding expression
            "alignment, not from each open fingertip already matching its "  # adds literal text to the surrounding expression
            "final side-face target."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v25 task space local grasp center run profile
    """Use the canonical palm-local thumb_2/index_1 midpoint as grasp center."""

    base = teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_grip_finger_model="two_finger",  # Overrides the target grip finger model setting for this config preset
                target_palm_position_mode="canonical_local",  # Overrides the target palm position mode setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v24 with the grasp center defined as the fixed palm-frame midpoint "  # adds literal text to the surrounding expression
            "0.5 * (thumb_2_local + index_1_local), and the task-space IK driving "  # adds literal text to the surrounding expression
            "that point to the block-centered target."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v26_closure_synced_drop() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v26 closure synced drop run profile
    """Hold vertical descent until the measured two-finger first-contact closure."""

    base = teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v26_closure_synced_drop/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v26_closure_synced_drop"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # The hand skeleton diagnostic shows thumb_2/index_1 first
                # span an 80 mm cube at roughly closure_frac=0 point 65:
                # (0 point 0887318 - 0 point 080) / (0 point 0887318 - 0 point 0752887)
                # Keep the
                # topdown pose centered until that measured closure point,
                # then allow the vertical drop
                descent_min_closure_fraction=0.65,  # Overrides the descent minimum closure fraction setting for this config preset
                descent_full_closure_fraction=0.65,  # Overrides the descent full closure fraction setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v26_closure_synced_drop",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v25 plus a measured closure gate on vertical descent: the arm "  # adds literal text to the surrounding expression
            "holds the centered topdown pose until thumb/index closure reaches "  # adds literal text to the surrounding expression
            "the ~0.65 first-contact fraction for the 8 cm cube."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v27 closure ramped drop run profile
    """Ramp vertical descent so touchdown coincides with measured first-contact closure."""

    base = teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Start descending while the hand is partially closed, then
                # reach full descent at the measured first-contact closure
                # This avoids v26's failure mode: curling above the cube and
                # never getting close enough for contact
                descent_min_closure_fraction=0.30,  # Overrides the descent minimum closure fraction setting for this config preset
                descent_full_closure_fraction=0.65,  # Overrides the descent full closure fraction setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v25 plus a closure-synchronized descent ramp: descent begins at "  # adds literal text to the surrounding expression
            "0.30 closure and reaches full depth at the measured ~0.65 "  # adds literal text to the surrounding expression
            "thumb/index first-contact fraction for the 8 cm cube."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v28 XYZ front close gate run profile
    """Gate finger curl from live thumb/index XYZ error and face-side validity."""

    base = teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                close_rate=0.015,  # Overrides the close rate setting for this config preset
                start_fraction=0.20,  # Overrides the start fraction setting for this config preset
                finger_unlock_min=0.30,  # Overrides the finger unlock minimum setting for this config preset
                finger_arm_hold_fallback=False,  # Overrides the finger arm hold fallback setting for this config preset
                prehold_align_angle_servo=True,  # Overrides the prehold align angle servo setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                requires_center=True,  # Overrides the requires center setting for this config preset
                latch=True,  # Overrides the latch setting for this config preset
                hold_steps=1,  # Overrides the hold steps setting for this config preset
                requires_contact_pose=False,  # Overrides the requires contact pose setting for this config preset
                tip_xy_max=0.030,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.050,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.0,  # Overrides the tip Z maximum setting for this config preset
                align_angle_max_deg=1.0,  # Overrides the align angle maximum deg setting for this config preset
                face_top_margin=0.035,  # Overrides the face top margin setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ContactPoseFallbackConfig):  # Checks alternate branch for isinstance(group, contact pose fallback config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                fallback_steps=0,  # Overrides the fallback steps setting for this config preset
                contact_pose_hold_steps=4,  # Overrides the contact pose hold steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ArmHoldCenteringConfig):  # Checks alternate branch for isinstance(group, arm hold centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                freeze_requires_finger_center=False,  # Overrides the freeze requires finger center setting for this config preset
                freeze_requires_contact_center=True,  # Overrides the freeze requires contact center setting for this config preset
                center_tip_xy_max=0.030,  # Overrides the center tip XY maximum setting for this config preset
                center_tip_z_max=0.045,  # Overrides the center tip Z maximum setting for this config preset
                center_align_angle_max_deg=1.0,  # Overrides the center align angle maximum deg setting for this config preset
            )  # closes the current expression
        elif isinstance(group, LiftSuccessConfig):  # Checks alternate branch for isinstance(group, lift success config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                latch_requires_center=True,  # Overrides the latch requires center setting for this config preset
                latch_requires_center_live=True,  # Overrides the latch requires center live setting for this config preset
                latch_requires_descent_z_min=0.050,  # Overrides the latch requires descent Z minimum setting for this config preset
                grip_settle_steps=45.0,  # Overrides the grip settle steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                descent_requires_center=True,  # Overrides the descent requires center setting for this config preset
                descent_uses_center_ready=True,  # Overrides the descent uses center ready setting for this config preset
                inward_requires_center=True,  # Overrides the inward requires center setting for this config preset
                vertical_drop_lock_xy=False,  # Overrides the vertical drop lock XY setting for this config preset
                descent_tip_servo_xy_max_m=0.030,  # Overrides the descent tip servo XY maximum m setting for this config preset
                descent_min_closure_fraction=0.0,  # Overrides the descent minimum closure fraction setting for this config preset
                descent_full_closure_fraction=0.0,  # Overrides the descent full closure fraction setting for this config preset
                descent_uses_z_need=False,  # Overrides the descent uses Z need setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ActionSurfaceConfig):  # Checks alternate branch for isinstance(group, action surface config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                contact_finger_close_cap=0.85,  # Overrides the contact finger close cap setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                align_angle_stage_min=0,  # Overrides the align angle stage minimum setting for this config preset
                align_angle_gain=1.5,  # Overrides the align angle gain setting for this config preset
                align_angle_max_dz=0.060,  # Overrides the align angle maximum dz setting for this config preset
                align_angle_max_joint_step=0.12,  # Overrides the align angle maximum joint step setting for this config preset
                ik_tip_servo_gain=1.0,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.120,  # Overrides the IK tip servo maximum m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                finger_close_gate_mode="xyz_front",  # Overrides the finger close gate mode setting for this config preset
                finger_xyz_gate_start_m=0.055,  # Overrides the finger XYZ gate start m setting for this config preset
                finger_xyz_gate_full_m=0.025,  # Overrides the finger XYZ gate full m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                direct_grip_center=True,  # Overrides the direct grip center setting for this config preset
                grip_offset_live_start_fraction=0.50,  # Overrides the grip offset live start fraction setting for this config preset
                grip_offset_live_full_fraction=0.80,  # Overrides the grip offset live full fraction setting for this config preset
                grip_offset_blend_requires_descent=True,  # Overrides the grip offset blend requires descent setting for this config preset
                span_xy_weight=3.0,  # Overrides the span XY weight setting for this config preset
                span_z_weight=8.0,  # Overrides the span Z weight setting for this config preset
                max_joint_step=0.090,  # Overrides the max joint step setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v27 with finger curl controlled by a live linear thumb/index XYZ "  # adds literal text to the surrounding expression
            "target-error gate. The xyz/front gate is independent from the "  # adds literal text to the surrounding expression
            "curriculum unlock ramp: fingers close only while they are near "  # adds literal text to the surrounding expression
            "their face targets and still outside/in front of the faces; "  # adds literal text to the surrounding expression
            "otherwise the finger action is forced open. Vertical descent is "  # adds literal text to the surrounding expression
            "independent from finger closure and only uses the centered "  # adds literal text to the surrounding expression
            "topdown pose gate."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v29_live_grip_offset() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v29 live grip offset run profile
    """Use live palm-local grip offset so targets track the curling fingers."""

    base = teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v29_live_grip_offset/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v29_live_grip_offset"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_position_mode="closure_aware_local",  # Overrides the target palm position mode setting for this config preset
                palm_local_grip_offset_mode="closure_blend_live_local",  # Overrides the palm local grip offset mode setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v29_live_grip_offset",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v28 plus closure-aware palm targeting: target construction uses "  # adds literal text to the surrounding expression
            "the live palm-frame thumb/index grasp-center offset instead of "  # adds literal text to the surrounding expression
            "one cached open-hand offset."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class PreholdClearanceConfig:  # defines the prehold clearance config group
    """Temporary vertical clearance while the hover pocket is not aligned."""

    clearance_m                       : float        = 0.0  # Sets the clearance m distance in meters
    tip_clearance_m                   : float        = 0.0  # Sets the tip clearance m distance in meters
    angle_max_deg                     : float        = 0.0  # Sets the angle maximum deg angular threshold
    align_err_max                     : float        = 0.0  # Sets the maximum allowed align err
    arm_servo_uses_contact_missing    : bool         = False  # Controls whether arm servo uses contact missing is enabled
    servo_uses_live_contact           : bool         = False  # Controls whether servo uses live contact is enabled
    extra_descent_uses_contact_missing: bool         = False  # Controls whether extra descent uses contact missing is enabled
    tip_servo_max_m                   : float | None = None  # Sets the tip servo maximum m distance in meters
    one_sided_close_boost             : float | None = None  # Sets the one sided close boost config value
    close_requires_descent_ready      : bool | None  = None  # Controls whether close requires descent ready is enabled
    pre_descent_live_debounce_steps   : int | None   = None  # Sets the number of steps for pre descent live debounce
    descent_requires_wrist_yaw_release: bool | None  = None  # Controls whether descent requires wrist yaw release is enabled
    descent_keepalive                 : bool | None  = None  # Controls whether descent keepalive is enabled
    finger_xyz_gate_start_m           : float | None = None  # Sets the finger XYZ gate start m distance in meters
    finger_xyz_gate_full_m            : float | None = None  # Sets the finger XYZ gate full m distance in meters
    explicit_prehover_waypoint        : bool | None  = None  # Controls whether explicit prehover waypoint is enabled
    explicit_prehover_height_m        : float | None = None  # Controls whether explicit prehover height m is enabled
    precenter_servo                   : bool | None  = None  # Controls whether precenter servo is enabled
    precenter_servo_gain              : float | None = None  # Controls whether precenter servo gain is enabled
    precenter_servo_max_m             : float | None = None  # Controls whether precenter servo maximum m is enabled
    precenter_stage_min               : int | None   = None  # Sets the first stage where precenter applies
    clearance_until_center            : bool | None  = None  # Controls whether clearance until center is enabled
    stage2_requires_finger_center     : bool | None  = None  # Controls whether stage2 requires finger center is enabled
    stage2_center_bypasses_opposed    : bool | None  = None  # Controls whether stage2 center bypasses opposed is enabled
    finger_requires_center            : bool | None  = None  # Requires finger-centering before scripted closure begins
    opposed_contact_uses_middle_back  : bool | None  = None  # Controls whether opposed contact uses middle back is enabled

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TOPDOWN_PREHOLD_ALIGN_CLEARANCE_M": self.clearance_m,  # Exports TOPDOWN_PREHOLD_ALIGN_CLEARANCE_M from the clearance m setting
                "TOPDOWN_PREHOLD_TIP_CLEARANCE_M": self.tip_clearance_m,  # Exports TOPDOWN_PREHOLD_TIP_CLEARANCE_M from the tip clearance m setting
                "TOPDOWN_PREHOLD_ALIGN_CLEARANCE_ANGLE_MAX_DEG": self.angle_max_deg,  # Exports TOPDOWN_PREHOLD_ALIGN_CLEARANCE_ANGLE_MAX_DEG from the angle maximum deg setting
                "TOPDOWN_PREHOLD_ALIGN_CLEARANCE_ALIGN_ERR_MAX": self.align_err_max,  # Exports TOPDOWN_PREHOLD_ALIGN_CLEARANCE_ALIGN_ERR_MAX from the align error maximum setting
                "TOPDOWN_CONTACT_TEACHER_ARM_SERVO_USES_CONTACT_MISSING": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_ARM_SERVO_USES_CONTACT_MISSING
                    self.arm_servo_uses_contact_missing  # Passes the arm servo uses contact missing setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_SERVO_USES_LIVE_CONTACT": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_SERVO_USES_LIVE_CONTACT
                    self.servo_uses_live_contact  # Passes the servo uses live contact setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_EXTRA_DESCENT_USES_CONTACT_MISSING": bool01(  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_EXTRA_DESCENT_USES_CONTACT_MISSING
                    self.extra_descent_uses_contact_missing  # Passes the extra descent uses contact missing setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_TIP_SERVO_MAX_M": self.tip_servo_max_m,  # Exports TOPDOWN_CONTACT_TEACHER_TIP_SERVO_MAX_M from the tip servo maximum m setting
                "TOPDOWN_CONTACT_TEACHER_ONE_SIDED_CLOSE_BOOST": self.one_sided_close_boost,  # Exports TOPDOWN_CONTACT_TEACHER_ONE_SIDED_CLOSE_BOOST from the one sided close boost setting
                "TOPDOWN_CONTACT_TEACHER_CLOSE_REQUIRES_DESCENT_READY": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_CLOSE_REQUIRES_DESCENT_READY
                    bool01(self.close_requires_descent_ready)  # Converts the close requires descent ready setting to legacy 0 or 1 text
                    if self.close_requires_descent_ready is not None  # Checks whether optional close requires descent ready override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_PRE_DESCENT_LIVE_DEBOUNCE_STEPS": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_PRE_DESCENT_LIVE_DEBOUNCE_STEPS
                    self.pre_descent_live_debounce_steps  # Passes the pre descent live debounce steps setting into the surrounding call
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_DESCENT_REQUIRES_WRIST_YAW_RELEASE": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_DESCENT_REQUIRES_WRIST_YAW_RELEASE
                    bool01(self.descent_requires_wrist_yaw_release)  # Converts the descent requires wrist yaw release setting to legacy 0 or 1 text
                    if self.descent_requires_wrist_yaw_release is not None  # Checks whether optional descent requires wrist yaw release override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_DESCENT_KEEPALIVE": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_DESCENT_KEEPALIVE
                    bool01(self.descent_keepalive)  # Converts the descent keepalive setting to legacy 0 or 1 text
                    if self.descent_keepalive is not None  # Checks whether optional descent keepalive override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_FINGER_XYZ_GATE_START_M": self.finger_xyz_gate_start_m,  # Exports TOPDOWN_FINGER_XYZ_GATE_START_M from the finger XYZ gate start m setting
                "TOPDOWN_FINGER_XYZ_GATE_FULL_M": self.finger_xyz_gate_full_m,  # Exports TOPDOWN_FINGER_XYZ_GATE_FULL_M from the finger XYZ gate full m setting
                "TOPDOWN_EXPLICIT_PREHOVER_WAYPOINT": (  # Starts env export expression for TOPDOWN_EXPLICIT_PREHOVER_WAYPOINT
                    bool01(self.explicit_prehover_waypoint)  # Converts the explicit prehover waypoint setting to legacy 0 or 1 text
                    if self.explicit_prehover_waypoint is not None  # Checks whether optional explicit prehover waypoint override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_EXPLICIT_PREHOVER_HEIGHT_M": self.explicit_prehover_height_m,  # Exports TOPDOWN_EXPLICIT_PREHOVER_HEIGHT_M from the explicit prehover height m setting
                "TOPDOWN_CONTACT_TEACHER_PRECENTER_SERVO": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_PRECENTER_SERVO
                    bool01(self.precenter_servo)  # Converts the precenter servo setting to legacy 0 or 1 text
                    if self.precenter_servo is not None  # Checks whether optional precenter servo override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_PRECENTER_SERVO_GAIN": self.precenter_servo_gain,  # Exports TOPDOWN_CONTACT_TEACHER_PRECENTER_SERVO_GAIN from the precenter servo gain setting
                "TOPDOWN_CONTACT_TEACHER_PRECENTER_SERVO_MAX_M": self.precenter_servo_max_m,  # Exports TOPDOWN_CONTACT_TEACHER_PRECENTER_SERVO_MAX_M from the precenter servo maximum m setting
                "TOPDOWN_CONTACT_TEACHER_PRECENTER_STAGE_MIN": self.precenter_stage_min,  # Exports TOPDOWN_CONTACT_TEACHER_PRECENTER_STAGE_MIN from the precenter stage minimum setting
                "TOPDOWN_PREHOLD_CLEARANCE_UNTIL_CENTER": (  # Starts env export expression for TOPDOWN_PREHOLD_CLEARANCE_UNTIL_CENTER
                    bool01(self.clearance_until_center)  # Converts the clearance until center setting to legacy 0 or 1 text
                    if self.clearance_until_center is not None  # Checks whether optional clearance until center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_STAGE2_REQUIRES_FINGER_CENTER": (  # Starts env export expression for TOPDOWN_STAGE2_REQUIRES_FINGER_CENTER
                    bool01(self.stage2_requires_finger_center)  # Converts the stage2 requires finger center setting to legacy 0 or 1 text
                    if self.stage2_requires_finger_center is not None  # Checks whether optional stage2 requires finger center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_STAGE2_CENTER_BYPASSES_OPPOSED": (  # Starts env export expression for TOPDOWN_STAGE2_CENTER_BYPASSES_OPPOSED
                    bool01(self.stage2_center_bypasses_opposed)  # Converts the stage2 center bypasses opposed setting to legacy 0 or 1 text
                    if self.stage2_center_bypasses_opposed is not None  # Checks whether optional stage2 center bypasses opposed override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_CONTACT_TEACHER_FINGER_REQUIRES_CENTER": (  # Starts env export expression for TOPDOWN_CONTACT_TEACHER_FINGER_REQUIRES_CENTER
                    bool01(self.finger_requires_center)  # Converts the finger requires center setting to legacy 0 or 1 text
                    if self.finger_requires_center is not None  # Checks whether optional finger requires center override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
                "TOPDOWN_OPPOSED_CONTACT_USE_MIDDLE_BACK": (  # Starts env export expression for TOPDOWN_OPPOSED_CONTACT_USE_MIDDLE_BACK
                    bool01(self.opposed_contact_uses_middle_back)  # Converts the opposed contact uses middle back setting to legacy 0 or 1 text
                    if self.opposed_contact_uses_middle_back is not None  # Checks whether optional opposed contact uses middle back override is set
                    else None  # omits the optional env var when unset
                ),  # closes the current expression
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class PhysicsOverrideConfig:  # defines the physics override config group
    """Per-profile physics overrides layered on top of a named physics profile."""

    block_static_friction     : float | None = None  # Sets static friction for the target block material
    block_dynamic_friction    : float | None = None  # Sets dynamic friction for the target block material
    fingertip_static_friction : float | None = None  # Sets static friction for the fingertip material
    fingertip_dynamic_friction: float | None = None  # Sets dynamic friction for the fingertip material

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "TOPDOWN_BLOCK_STATIC_FRICTION": self.block_static_friction,  # Exports TOPDOWN_BLOCK_STATIC_FRICTION from the block static friction setting
                "TOPDOWN_BLOCK_DYNAMIC_FRICTION": self.block_dynamic_friction,  # Exports TOPDOWN_BLOCK_DYNAMIC_FRICTION from the block dynamic friction setting
                "TOPDOWN_FINGERTIP_STATIC_FRICTION": self.fingertip_static_friction,  # Exports TOPDOWN_FINGERTIP_STATIC_FRICTION from the fingertip static friction setting
                "TOPDOWN_FINGERTIP_DYNAMIC_FRICTION": self.fingertip_dynamic_friction,  # Exports TOPDOWN_FINGERTIP_DYNAMIC_FRICTION from the fingertip dynamic friction setting
            }  # closes the current expression
        )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v30 canonical hover XYZ gate run profile
    """Use xyz-front finger gating with preload-aware grasp-center targeting."""

    base = teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, StageGateConfig):  # Checks whether isinstance(group, stage gate config)
            # The v30 task-space teacher currently plateaus around 52 point 5° of
            # drop-axis error Keep the stage gates consistent with that
            # measured floor so the teacher can leave pre-descent hover and
            # exercise the contact/descent code path This is an acceptance
            # threshold only; the 20% preload and IK targets are unchanged
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                stage1_palm_orient_max_deg=55,  # Overrides the stage1 palm orient maximum deg setting for this config preset
                stage2_palm_orient_max_deg=55,  # Overrides the stage2 palm orient maximum deg setting for this config preset
                success_palm_orient_max_deg=55,  # Overrides the success palm orient maximum deg setting for this config preset
                stage1_palm_height_max=0.030,  # Overrides the stage1 palm height maximum setting for this config preset
                stage2_palm_height_max=0.030,  # Overrides the stage2 palm height maximum setting for this config preset
            )  # closes the current expression
            continue  # skips this item and continues validation
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                close_rate=0.006,  # Overrides the close rate setting for this config preset
                start_fraction=0.20,  # Overrides the start fraction setting for this config preset
                inpocket_arm_hold=False,  # Overrides the inpocket arm hold setting for this config preset
                bc_target_includes_inpocket_arm_hold=False,  # Overrides the BC target includes inpocket arm hold setting for this config preset
                # Keep the 20% preload, but do not curl beyond preload until
                # the hover pocket is centered and the thumb/index line is
                # nearly level This prevents the one-finger graze/restart loop
                # before the vertical descent phase
                finger_requires_center=True,  # Overrides the finger requires center setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Center-ready must be a live stable hover condition, not a
                # sticky one-frame latch Run I/J reached transient strict
                # contact but lost the pocket before lift; a short live
                # debounce prevents one-frame releases without starving the
                # reachable hover pocket
                latch=False,  # Overrides the latch setting for this config preset
                hold_steps=4,  # Overrides the hold steps setting for this config preset
                # Pre-descent centering should align the thumb/index pocket
                # center over the block and level the pinch line Requiring
                # each preloaded fingertip to already sit near its final side
                # face blocks the descent phase before the fingers can curl
                # into those face targets
                max_tip_xy_max=0.0,  # Overrides the max tip XY maximum setting for this config preset
                align_angle_max_deg=8.0,  # Overrides the align angle maximum deg setting for this config preset
                align_error_max=0.28,  # Overrides the align error maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, LiftSuccessConfig):  # Checks alternate branch for isinstance(group, lift success config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                latch_requires_center=True,  # Overrides the latch requires center setting for this config preset
                latch_requires_center_live=True,  # Overrides the latch requires center live setting for this config preset
                latch_requires_descent_z_min=0.040,  # Overrides the latch requires descent Z minimum setting for this config preset
                grip_settle_steps=0.0,  # Overrides the grip settle steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # With the hover pocket reset to 5cm above the block, the old
                # 5 point 5cm descent plus missing-contact extra drove the target
                # below the block top and pushed the block after contact
                descent_z=0.040,  # Overrides the descent Z setting for this config preset
                descent_tip_z_target=0.0,  # Overrides the descent tip Z target setting for this config preset
                hold_extra_fraction=0.10,  # Overrides the hold extra fraction setting for this config preset
                hold_max_fraction=0.92,  # Overrides the hold maximum fraction setting for this config preset
                prelift_squeeze_fraction=0.68,  # Overrides the prelift squeeze fraction setting for this config preset
                lift_squeeze_fraction=0.68,  # Overrides the lift squeeze fraction setting for this config preset
                missing_contact_extra_descent=0.015,  # Overrides the missing contact extra descent setting for this config preset
                inward_m=0.003,  # Overrides the inward m setting for this config preset
                # Do not let the contact teacher drop from hover just because
                # the stage gate passed In xyz-front mode, use the debounced
                # center-ready latch so a good centered/level hover can release
                # contact even if the live angle flickers a few frames later
                # Preload remains at 20%; only the arm descent is held back
                descent_requires_center=True,  # Overrides the descent requires center setting for this config preset
                descent_uses_center_ready=True,  # Overrides the descent uses center ready setting for this config preset
                inward_requires_center=True,  # Overrides the inward requires center setting for this config preset
                pre_descent_hover_height_max=0.030,  # Overrides the pre descent hover height maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                align_angle_stage_min=0,  # Overrides the align angle stage minimum setting for this config preset
                align_angle_gain=2.0,  # Overrides the align angle gain setting for this config preset
                align_angle_max_dz=0.080,  # Overrides the align angle maximum dz setting for this config preset
                align_angle_max_joint_step=0.12,  # Overrides the align angle maximum joint step setting for this config preset
                planar_align_servo=True,  # Overrides the planar align servo setting for this config preset
                planar_align_stage_min=0,  # Overrides the planar align stage minimum setting for this config preset
                planar_align_gain=1.5,  # Overrides the planar align gain setting for this config preset
                planar_align_max_xy=0.080,  # Overrides the planar align maximum XY setting for this config preset
                planar_align_max_joint_step=0.12,  # Overrides the planar align maximum joint step setting for this config preset
                ik_tip_servo_gain=1.0,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.120,  # Overrides the IK tip servo maximum m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_position_mode="closure_aware_local",  # Overrides the target palm position mode setting for this config preset
                palm_local_grip_offset_mode="closure_blend_live_local",  # Overrides the palm local grip offset mode setting for this config preset
                tip_servo_z_requires_center=True,  # Overrides the tip servo Z requires center setting for this config preset
                # The 5 point 5cm v28 gate is too tight for the 20% preload hover:
                # good centered/level samples have max weighted xyz errors in
                # the 6-8cm band Use the audited/default 8 point 5cm start gate so
                # the contact teacher can begin descent from a valid hover
                finger_xyz_gate_start_m=0.085,  # Overrides the finger XYZ gate start m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # The hover contract is "align the preloaded pocket about
                # 10 cm above the block, then descend" The inherited task
                # space weights allowed the grip center to settle 3-4 cm below
                # the hover target while satisfying span/drop terms, causing
                # thumb-top contact at stage 0 Make hover center-Z the
                # dominant pre-descent objective; keep span/drop as secondary
                # shaping terms The temporary clearance gate above prevents
                # top-face grazing while the lower-level servos solve angle and
                # centering
                center_xy_weight=5.0,  # Overrides the center XY weight setting for this config preset
                center_z_weight=10.0,  # Overrides the center Z weight setting for this config preset
                span_xy_weight=1.25,  # Overrides the span XY weight setting for this config preset
                span_z_weight=3.0,  # Overrides the span Z weight setting for this config preset
                drop_weight=0.75,  # Overrides the drop weight setting for this config preset
                orientation_weight=2.0,  # Overrides the orientation weight setting for this config preset
                orientation_sign=1.0,  # Overrides the orientation sign setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ActionSurfaceConfig):  # Checks alternate branch for isinstance(group, action surface config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                mirror_middle_to_index=True,  # Overrides the mirror middle to index setting for this config preset
            )  # closes the current expression
    groups.append(  # appends the computed value to the collection
        WristYawLockConfig(  # starts the wrist yaw lock config block
            # Do not release wrist yaw simply because stage 2 latched The
            # smoke run reached a good pre-descent pocket, then stage 2
            # released yaw and the hand rotated out to ~25 deg yaw / ~10 deg
            # pinch-line error Keep the lock until the geometry release gate
            # itself is satisfied
            release_at_stage2=False,  # Overrides the release at stage2 setting for this config preset
        )  # closes the current expression
    )  # closes the current expression
    groups.append(  # appends the computed value to the collection
        PreholdClearanceConfig(  # starts the prehold clearance config block
            # Keep the approach above the block until the thumb/index pocket is
            # actually centered and level This prevents the diagonal approach
            # from grazing the top/front face before hover alignment is solved
            clearance_m=0.030,  # Overrides the clearance m setting for this config preset
            tip_clearance_m=0.012,  # Overrides the tip clearance m setting for this config preset
            angle_max_deg=8.0,  # Overrides the angle maximum deg setting for this config preset
            align_err_max=0.28,  # Overrides the align error maximum setting for this config preset
            arm_servo_uses_contact_missing=True,  # Overrides the arm servo uses contact missing setting for this config preset
            servo_uses_live_contact=False,  # Overrides the servo uses live contact setting for this config preset
            extra_descent_uses_contact_missing=True,  # Overrides the extra descent uses contact missing setting for this config preset
            tip_servo_max_m=0.045,  # Overrides the tip servo maximum m setting for this config preset
            one_sided_close_boost=1.75,  # Overrides the one sided close boost setting for this config preset
        )  # closes the current expression
    )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v28 with the xyz-front live finger-close gate and a "  # adds literal text to the surrounding expression
            "preload-aware thumb/index grasp-center palm target, so the "  # adds literal text to the surrounding expression
            "hover target matches the 20% preloaded hand shape."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v30 preload closure aware restore run profile
    """Restore the v30 smoke behavior before later latch/gate tightening."""

    # This intentionally rebuilds from v27 instead of inheriting current v28
    # The best GUI smoke run used the original xyz-front gate semantics:
    # no finger-center latch dependency, a wider 8 point 5 cm close-start window,
    # and preload-aware palm targeting from v30
    base = teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                close_rate=0.006,  # Overrides the close rate setting for this config preset
                start_fraction=0.20,  # Overrides the start fraction setting for this config preset
                finger_unlock_min=0.0,  # Overrides the finger unlock minimum setting for this config preset
                finger_arm_hold_fallback=False,  # Overrides the finger arm hold fallback setting for this config preset
                inpocket_arm_hold=False,  # Overrides the inpocket arm hold setting for this config preset
                bc_target_includes_inpocket_arm_hold=False,  # Overrides the BC target includes inpocket arm hold setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                requires_center=False,  # Overrides the requires center setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                descent_z=0.055,  # Overrides the descent Z setting for this config preset
                descent_tip_z_target=0.0,  # Overrides the descent tip Z target setting for this config preset
                hold_extra_fraction=0.0,  # Overrides the hold extra fraction setting for this config preset
                prelift_squeeze_fraction=0.68,  # Overrides the prelift squeeze fraction setting for this config preset
                lift_squeeze_fraction=0.68,  # Overrides the lift squeeze fraction setting for this config preset
                missing_contact_extra_descent=0.010,  # Overrides the missing contact extra descent setting for this config preset
                descent_min_closure_fraction=0.0,  # Overrides the descent minimum closure fraction setting for this config preset
                descent_full_closure_fraction=0.0,  # Overrides the descent full closure fraction setting for this config preset
                # Use the debounced/latching center-ready gate for descent and
                # xyz finger release The live angle can flicker after a good
                # centered hover; requiring it every frame prevents the contact
                # teacher from ever leaving pre-descent
                descent_uses_center_ready=True,  # Overrides the descent uses center ready setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                align_angle_stage_min=0,  # Overrides the align angle stage minimum setting for this config preset
                align_angle_gain=2.0,  # Overrides the align angle gain setting for this config preset
                align_angle_max_dz=0.080,  # Overrides the align angle maximum dz setting for this config preset
                align_angle_max_joint_step=0.12,  # Overrides the align angle maximum joint step setting for this config preset
                planar_align_servo=True,  # Overrides the planar align servo setting for this config preset
                planar_align_stage_min=0,  # Overrides the planar align stage minimum setting for this config preset
                planar_align_gain=1.5,  # Overrides the planar align gain setting for this config preset
                planar_align_max_xy=0.080,  # Overrides the planar align maximum XY setting for this config preset
                planar_align_max_joint_step=0.12,  # Overrides the planar align maximum joint step setting for this config preset
                ik_tip_servo_gain=1.0,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.120,  # Overrides the IK tip servo maximum m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                finger_close_gate_mode="xyz_front",  # Overrides the finger close gate mode setting for this config preset
                finger_xyz_gate_start_m=0.085,  # Overrides the finger XYZ gate start m setting for this config preset
                finger_xyz_gate_full_m=0.025,  # Overrides the finger XYZ gate full m setting for this config preset
                target_palm_position_mode="closure_aware_local",  # Overrides the target palm position mode setting for this config preset
                palm_local_grip_offset_mode="closure_blend_live_local",  # Overrides the palm local grip offset mode setting for this config preset
                tip_servo_z_requires_center=True,  # Overrides the tip servo Z requires center setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                direct_grip_center=True,  # Overrides the direct grip center setting for this config preset
                grip_offset_live_start_fraction=0.50,  # Overrides the grip offset live start fraction setting for this config preset
                grip_offset_live_full_fraction=0.80,  # Overrides the grip offset live full fraction setting for this config preset
                grip_offset_blend_requires_descent=True,  # Overrides the grip offset blend requires descent setting for this config preset
                span_xy_weight=3.0,  # Overrides the span XY weight setting for this config preset
                span_z_weight=8.0,  # Overrides the span Z weight setting for this config preset
                max_joint_step=0.090,  # Overrides the max joint step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ActionSurfaceConfig):  # Checks alternate branch for isinstance(group, action surface config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                mirror_middle_to_index=True,  # Overrides the mirror middle to index setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Frozen restore of the best v30 GUI-smoke lineage: v27 descent, "  # adds literal text to the surrounding expression
            "original xyz-front finger gate, 15 percent preload, and "  # adds literal text to the surrounding expression
            "closure-aware thumb/index grasp-center targeting. This excludes "  # adds literal text to the surrounding expression
            "the later strict center latch, contact-pose fallback, and tightened "  # adds literal text to the surrounding expression
            "0.055 m xyz gate edits that regressed the teacher."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v31_centered_restore() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v31 centered restore run profile
    """Restore the pre-xyz-gate centered topdown teacher baseline."""

    base = teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v31_centered_restore/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v31_centered_restore"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                start_fraction=0.15,  # Overrides the start fraction setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                descent_z=0.065,  # Overrides the descent Z setting for this config preset
                descent_tip_z_target=0.002,  # Overrides the descent tip Z target setting for this config preset
                missing_contact_extra_descent=0.015,  # Overrides the missing contact extra descent setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_position_mode="closure_aware_local",  # Overrides the target palm position mode setting for this config preset
                palm_local_grip_offset_mode="closure_blend_live_local",  # Overrides the palm local grip offset mode setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                grip_offset_live_start_fraction=0.50,  # Overrides the grip offset live start fraction setting for this config preset
                grip_offset_live_full_fraction=0.80,  # Overrides the grip offset live full fraction setting for this config preset
                grip_offset_blend_requires_descent=True,  # Overrides the grip offset blend requires descent setting for this config preset
                center_xy_weight=8.0,  # Overrides the center XY weight setting for this config preset
                center_z_weight=1.5,  # Overrides the center Z weight setting for this config preset
                max_joint_step=0.085,  # Overrides the max joint step setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v31_centered_restore",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Restore point for GUI teacher debugging: v25 canonical local "  # adds literal text to the surrounding expression
            "two-finger grasp-center IK, with only the 20 percent finger "  # adds literal text to the surrounding expression
            "preload retained. This intentionally excludes the v28/v30 "  # adds literal text to the surrounding expression
            "xyz-front close gate and later contact/latch experiments."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v32 6cm hover8 centered XYZ run profile
    """Last-try IK profile: 6cm block, centered hover, vertical descent.

    This is the profile that first produced a usable stable lift.  The key
    contract is simple even though the knobs are many:

    * use a 6cm block with no jitter,
    * keep a low, centered pre-descent hover,
    * require the open thumb/index pocket to be centered before descent,
    * descend vertically with no inward shove, and
    * scale the xyz close gate for the smaller block.

    Avoid broad face-selection/orientation rewrites here unless they are being
    evaluated as a new profile.  This function is the known-good baseline for
    MVP demos and warm starts.
    """

    base = teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate()  # Loads the base run profile before replacing selected groups
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz"  # Builds the run directory string for this profile
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskIdentity):  # Checks alternate branch for isinstance(group, task identity)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                block_size=0.06,  # Overrides the block size setting for this config preset
                # Last GUI pass showed 5cm above the top leaves the hand
                # hovering too high/side-biased Use a lower pre-descent
                # waypoint: 3cm above the top of the 6cm cube
                hover_above_block_top=0.03,  # Overrides the hover above block top setting for this config preset
                block_jitter_x=0.0,  # Overrides the block jitter X setting for this config preset
                block_jitter_y=0.0,  # Overrides the block jitter Y setting for this config preset
                episode_length_s=10.0,  # Overrides the episode length s setting for this config preset
            )  # closes the current expression
        elif isinstance(group, StageGateConfig):  # Checks alternate branch for isinstance(group, stage gate config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                stage1_palm_height_max=0.100,  # Overrides the stage1 palm height maximum setting for this config preset
                stage2_palm_height_max=0.100,  # Overrides the stage2 palm height maximum setting for this config preset
                # Stage entry should not be the final centering contract for
                # this last-try profile The strict pre-descent waypoint is
                # enforced by FingerCenteringConfig below; these looser stage
                # gates let the IK leave the long stage-0 hover once it has
                # reached the neighborhood above the 6cm block
                stage1_align_err_max=0.40,  # Overrides the stage1 align error maximum setting for this config preset
                stage2_align_err_max=0.40,  # Overrides the stage2 align error maximum setting for this config preset
                stage1_line_angle_max_deg=65,  # Overrides the stage1 line angle maximum deg setting for this config preset
                stage2_line_angle_max_deg=65,  # Overrides the stage2 line angle maximum deg setting for this config preset
                # Keep orientation as a broad safety shell r6 strict contact
                # needed a ~65deg approach moment; r7 at 65deg delayed entry
                # too long and lost strict contact The real shaping is the
                # gentle IK orientation weight below, not this hard gate
                stage1_palm_orient_max_deg=70,  # Overrides the stage1 palm orient maximum deg setting for this config preset
                stage2_palm_orient_max_deg=70,  # Overrides the stage2 palm orient maximum deg setting for this config preset
                success_palm_orient_max_deg=70,  # Overrides the success palm orient maximum deg setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ContactPoseFallbackConfig):  # Checks alternate branch for isinstance(group, contact pose fallback config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                palm_orient_max_deg=75.0,  # Overrides the palm orient maximum deg setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                latch=False,  # Overrides the latch setting for this config preset
                hold_steps=5,  # Overrides the hold steps setting for this config preset
                # This gate controls whether the 20% preloaded hand may proceed
                # into the xyz close/descent behavior It is not the final lift
                # success gate r4 showed the policy repeatedly hovering with
                # align_err under 0 point 1 but one fingertip dimension just outside
                # the previous 18mm center threshold, which pinned unlock/close
                align_angle_max_deg=8.0,  # Overrides the align angle maximum deg setting for this config preset
                align_error_max=0.10,  # Overrides the align error maximum setting for this config preset
                tip_xy_max=0.030,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.070,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.065,  # Overrides the tip Z maximum setting for this config preset
                face_half_extent=0.0225,  # Overrides the face half extent setting for this config preset
                face_top_margin=0.009,  # Overrides the face top margin setting for this config preset
            )  # closes the current expression
        elif isinstance(group, LiftSuccessConfig):  # Checks alternate branch for isinstance(group, lift success config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # r13 reached sustained strict contact, but lift never armed
                # because this profile inherited a descent-Z latch guard while
                # the successful contact formed during the hover/curl phase
                # For the MVP demo, let strict opposed contact arm lift and
                # give RL room to reduce drift instead of terminating at 12cm
                latch_requires_center=False,  # Overrides the latch requires center setting for this config preset
                latch_requires_center_live=False,  # Overrides the latch requires center live setting for this config preset
                latch_requires_descent_z_min=0.0,  # Overrides the latch requires descent Z minimum setting for this config preset
                block_drift_threshold=0.50,  # Overrides the block drift threshold setting for this config preset
                success_xy_drift_max=0.50,  # Overrides the success XY drift maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                close_rate=0.006,  # Overrides the close rate setting for this config preset
                start_fraction=0.20,  # Overrides the start fraction setting for this config preset
                finger_unlock_min=0.20,  # Overrides the finger unlock minimum setting for this config preset
                finger_requires_center=True,  # Overrides the finger requires center setting for this config preset
                inpocket_arm_hold=False,  # Overrides the inpocket arm hold setting for this config preset
                bc_target_includes_inpocket_arm_hold=False,  # Overrides the BC target includes inpocket arm hold setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Hover target is 3cm over the 6cm cube top A smaller
                # vertical descent keeps the side-face target in the face band
                # without dropping the palm through the top face
                max_fraction=0.92,  # Overrides the max fraction setting for this config preset
                descent_z=0.035,  # Overrides the descent Z setting for this config preset
                hold_max_fraction=0.92,  # Overrides the hold maximum fraction setting for this config preset
                lift_squeeze_fraction=0.92,  # Overrides the lift squeeze fraction setting for this config preset
                descent_tip_z_target=0.0,  # Overrides the descent tip Z target setting for this config preset
                missing_contact_extra_descent=0.005,  # Overrides the missing contact extra descent setting for this config preset
                inward_m=0.0,  # Overrides the inward m setting for this config preset
                descent_requires_center=True,  # Overrides the descent requires center setting for this config preset
                descent_uses_center_ready=True,  # Overrides the descent uses center ready setting for this config preset
                inward_requires_center=True,  # Overrides the inward requires center setting for this config preset
                pre_descent_hover_height_max=0.025,  # Overrides the pre descent hover height maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                align_angle_stage_min=0,  # Overrides the align angle stage minimum setting for this config preset
                align_angle_gain=2.0,  # Overrides the align angle gain setting for this config preset
                align_angle_max_dz=0.050,  # Overrides the align angle maximum dz setting for this config preset
                align_angle_max_joint_step=0.050,  # Overrides the align angle maximum joint step setting for this config preset
                planar_align_servo=True,  # Overrides the planar align servo setting for this config preset
                planar_align_stage_min=0,  # Overrides the planar align stage minimum setting for this config preset
                planar_align_gain=1.25,  # Overrides the planar align gain setting for this config preset
                planar_align_max_xy=0.040,  # Overrides the planar align maximum XY setting for this config preset
                planar_align_max_joint_step=0.050,  # Overrides the planar align maximum joint step setting for this config preset
                ik_tip_servo_gain=0.75,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.050,  # Overrides the IK tip servo maximum m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                target_palm_basis="drop_priority",  # Overrides the target palm basis setting for this config preset
                target_palm_position_mode="canonical_open",  # Overrides the target palm position mode setting for this config preset
                palm_local_grip_offset_mode="canonical_open",  # Overrides the palm local grip offset mode setting for this config preset
                align_angle_joints="all",  # Overrides the align angle joints setting for this config preset
                planar_align_joints="all",  # Overrides the planar align joints setting for this config preset
                tip_servo_z_requires_center=True,  # Overrides the tip servo Z requires center setting for this config preset
                finger_close_gate_mode="xyz_front",  # Overrides the finger close gate mode setting for this config preset
                # Let the close gate begin as soon as the fingers are in the
                # correct neighborhood; the center gate above still prevents
                # random off-block closure Full close remains tighter
                finger_xyz_gate_start_m=0.100,  # Overrides the finger XYZ gate start m setting for this config preset
                finger_xyz_gate_full_m=0.030,  # Overrides the finger XYZ gate full m setting for this config preset
                finger_xyz_gate_linear=True,  # Overrides the finger XYZ gate linear setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ActionSurfaceConfig):  # Checks alternate branch for isinstance(group, action surface config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                contact_finger_close_cap=0.92,  # Overrides the contact finger close cap setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                direct_grip_center=True,  # Overrides the direct grip center setting for this config preset
                grip_offset_blend_requires_descent=True,  # Overrides the grip offset blend requires descent setting for this config preset
                center_xy_weight=12.0,  # Overrides the center XY weight setting for this config preset
                center_z_weight=10.0,  # Overrides the center Z weight setting for this config preset
                span_xy_weight=2.0,  # Overrides the span XY weight setting for this config preset
                span_z_weight=18.0,  # Overrides the span Z weight setting for this config preset
                drop_weight=0.15,  # Overrides the drop weight setting for this config preset
                orientation_weight=0.15,  # Overrides the orientation weight setting for this config preset
                posture_weight=0.08,  # Overrides the posture weight setting for this config preset
                max_joint_step=0.050,  # Overrides the max joint step setting for this config preset
            )  # closes the current expression
    groups.append(  # appends the computed value to the collection
        PreholdClearanceConfig(  # starts the prehold clearance config block
            # Keep the 3cm grip-center hover, but prevent low fingertips from
            # sweeping through the cube while the pre-descent center/angle gate
            # is still false This is a millimeter-scale guard, not another
            # high hover waypoint
            clearance_m=0.001,  # Overrides the clearance m setting for this config preset
            tip_clearance_m=0.006,  # Overrides the tip clearance m setting for this config preset
            angle_max_deg=3.0,  # Overrides the angle maximum deg setting for this config preset
            align_err_max=0.04,  # Overrides the align error maximum setting for this config preset
            arm_servo_uses_contact_missing=True,  # Overrides the arm servo uses contact missing setting for this config preset
            servo_uses_live_contact=False,  # Overrides the servo uses live contact setting for this config preset
            extra_descent_uses_contact_missing=False,  # Overrides the extra descent uses contact missing setting for this config preset
            tip_servo_max_m=0.050,  # Overrides the tip servo maximum m setting for this config preset
            one_sided_close_boost=1.0,  # Overrides the one sided close boost setting for this config preset
            close_requires_descent_ready=False,  # Overrides the close requires descent ready setting for this config preset
            pre_descent_live_debounce_steps=10,  # Overrides the pre descent live debounce steps setting for this config preset
            precenter_servo=False,  # Overrides the precenter servo setting for this config preset
            finger_requires_center=False,  # Overrides the finger requires center setting for this config preset
            opposed_contact_uses_middle_back=True,  # Overrides the opposed contact uses middle back setting for this config preset
        )  # closes the current expression
    )  # closes the current expression
    groups.append(  # appends the computed value to the collection
        PhysicsOverrideConfig(  # starts the physics override config block
            # NVIDIA mirror defaults are 0 point 5/0 point 5 r13 achieved a real cage
            # but slid the 6cm block before lift could ramp, so use a modest
            # friction bump without returning to the old high-friction regime
            block_static_friction=1.2,  # Overrides the block static friction setting for this config preset
            block_dynamic_friction=1.0,  # Overrides the block dynamic friction setting for this config preset
            fingertip_static_friction=1.2,  # Overrides the fingertip static friction setting for this config preset
            fingertip_dynamic_friction=1.0,  # Overrides the fingertip dynamic friction setting for this config preset
        )  # closes the current expression
    )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Last-try IK solve: 6cm cube, no jitter, hover waypoint at "  # adds literal text to the surrounding expression
            "3cm above the block top, strict pre-descent centering/alignment, "  # adds literal text to the surrounding expression
            "vertical-only descent, and xyz close gate scaled for the 6cm block."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v32 6cm MVP RL 700k run profile
    """MVP Teacher -> BC/DAgger -> RL run from the v32 6cm liftable teacher.

    This profile preserves the teacher exactly enough to reproduce the liftable
    behavior, then changes only the training schedule: 50k teacher collection,
    150k BC/DAgger warmup, and 500k RL refinement.  It is the best checkpoint
    family to use when the goal is "show RL can improve something" rather than
    continuing IK research.
    """

    base = teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k"  # Builds the run directory string for this profile
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=700_000,  # Overrides the total steps setting for this config preset
                start_steps=50_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=200_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=200_000,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "MVP salvage run: v32 6cm teacher with sustained strict contact and "  # adds literal text to the surrounding expression
            "partial stable lift, 50k teacher collection, 150k BC/DAgger, then "  # adds literal text to the surrounding expression
            "500k RL refinement."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v34 6cm MVP RL 700k reward fix run profile
    """v32 MVP run with reward shaping aimed at the teacher's known shortfalls."""

    base = teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix"  # Builds the run directory string for this profile
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ContactReward):  # Checks alternate branch for isinstance(group, contact reward)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Keep the old strict-shell opposed reward, but make the
                # one-sided teacher trace net-negative unless it becomes an
                # active back-finger opposed grip
                opposed_contact=8.0,  # Overrides the opposed contact setting for this config preset
                one_sided=-2.0,  # Overrides the one sided setting for this config preset
                bilateral_contact=8.0,  # Overrides the bilateral contact setting for this config preset
                bilateral_imbalance=-8.0,  # Overrides the bilateral imbalance setting for this config preset
                one_sided_flip=-6.0,  # Overrides the one sided flip setting for this config preset
                # The smoke trajectory stalled at stage2/unlock=0 This pair
                # pays the pre-curl pocket and charges no-contact dwell outside it
                preunlock_pocket=6.0,  # Overrides the preunlock pocket setting for this config preset
                preunlock_no_contact=-4.0,  # Overrides the preunlock no contact setting for this config preset
                contact_alignment_error=-2.0,  # Overrides the contact alignment error setting for this config preset
                contact_alignment_error_quadratic=-10.0,  # Overrides the contact alignment error quadratic setting for this config preset
                alignment_degradation=-25.0,  # Overrides the alignment degradation setting for this config preset
                finger_center_x_error_quadratic=-1.5,  # Overrides the finger center X error quadratic setting for this config preset
                finger_center_y_error_quadratic=-1.5,  # Overrides the finger center Y error quadratic setting for this config preset
                finger_center_err_scale=0.035,  # Overrides the finger center error scale setting for this config preset
                centered_contact=10.0,  # Overrides the centered contact setting for this config preset
                contact_lift_progress=0.0,  # Overrides the contact lift progress setting for this config preset
                contact_success_now_continuous=4.0,  # Overrides the contact success now continuous setting for this config preset
                overforce=-4.0,  # Overrides the overforce setting for this config preset
                smooth_pose_no_contact=-8.0,  # Overrides the smooth pose no contact setting for this config preset
                smooth_success_pose=3.0,  # Overrides the smooth success pose setting for this config preset
                smooth_success_with_contact=14.0,  # Overrides the smooth success with contact setting for this config preset
                stage2_floor=0.0,  # Overrides the stage2 floor setting for this config preset
            )  # closes the current expression
        elif isinstance(group, LiftReward):  # Checks alternate branch for isinstance(group, lift reward)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Remove lift reward that only needs opposed contact Positive
                # lift shaping now requires centered bilateral contact
                height_progress=0.0,  # Overrides the height progress setting for this config preset
                lift_with_grip=0.0,  # Overrides the lift with grip setting for this config preset
                centered_lift_progress=12.0,  # Overrides the centered lift progress setting for this config preset
                centered_upright_lift_bonus=24.0,  # Overrides the centered upright lift bonus setting for this config preset
                lift_xy_drift_penalty=-24.0,  # Overrides the lift XY drift penalty setting for this config preset
                block_tilt_lift_penalty=-20.0,  # Overrides the block tilt lift penalty setting for this config preset
                uncentered_lift_penalty=-16.0,  # Overrides the uncentered lift penalty setting for this config preset
                block_xy_velocity_penalty=-4.0,  # Overrides the block XY velocity penalty setting for this config preset
                block_angular_velocity_penalty=-2.0,  # Overrides the block angular velocity penalty setting for this config preset
                block_off_table_bonus=0.0,  # Overrides the block off table bonus setting for this config preset
                lift_penalty_height_start=0.0,  # Overrides the lift penalty height start setting for this config preset
                lift_penalty_height_ramp=0.015,  # Overrides the lift penalty height ramp setting for this config preset
                lift_xy_drift_penalty_free=0.04,  # Overrides the lift XY drift penalty free setting for this config preset
                lift_xy_drift_penalty_zero=0.12,  # Overrides the lift XY drift penalty zero setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v32 MVP teacher/schedule with reward shaping for the audited smoke "  # adds literal text to the surrounding expression
            "failure: pre-unlock pocket guidance, no-contact dwell cost, active "  # adds literal text to the surrounding expression
            "bilateral contact symmetry, centered-only lift reward, and tighter "  # adds literal text to the surrounding expression
            "drift pressure."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v35 6cm XYZ align stage run profile
    """v32 MVP baseline with stage/close gates reduced to waypoint + xyz + angle.

    This profile is intentionally not another finger-latch experiment. It keeps
    the v32 6cm task, teacher, and training schedule, but removes the old
    hard yaw/orient/opposed-face requirements from curriculum staging. Stage 2
    and active close are instead authorized by:

    * palm waypoint distance/height,
    * live thumb/index xyz distance to the block center, and
    * fingertip alignment angle.

    That lets the same controller evaluate either side-face approach without
    downstream gates insisting on one historical yaw convention.
    """

    base = teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage"  # Builds the run directory string for this profile
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskIdentity):  # Checks alternate branch for isinstance(group, task identity)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # v32's 3cm hover let the side-agnostic hand sweep through the
                # cube during initial centering Make the actual curriculum
                # waypoint high, not just a late clearance correction
                hover_above_block_top=0.075,  # Overrides the hover above block top setting for this config preset
                # Put the yellow/blue source blocks back in the scene for the
                # final curriculum They remain inactive distractors with
                # contact sensors off unless source_pose_mode is changed
                keep_distractors_visible=True,  # Overrides the keep distractors visible setting for this config preset
                # Reintroduce modest spawn variability now that the v35
                # side-agnostic teacher is stable Keep it below the historical
                # 2 point 5cm default so the first training pass broadens coverage
                # without immediately invalidating the tuned hover/close gates
                block_jitter_x=0.015,  # Overrides the block jitter X setting for this config preset
                block_jitter_y=0.015,  # Overrides the block jitter Y setting for this config preset
            )  # closes the current expression
        elif isinstance(group, StageGateConfig):  # Checks alternate branch for isinstance(group, stage gate config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Yaw/orient/open-hand face residuals are no longer hard
                # curriculum contracts in this side-agnostic experiment
                # Palm distance/height remain the coarse IK waypoint gates
                stage1_palm_orient_max_deg=180.0,  # Overrides the stage1 palm orient maximum deg setting for this config preset
                stage2_palm_orient_max_deg=180.0,  # Overrides the stage2 palm orient maximum deg setting for this config preset
                success_palm_orient_max_deg=180.0,  # Overrides the success palm orient maximum deg setting for this config preset
                stage1_palm_yaw_max_deg=180.0,  # Overrides the stage1 palm yaw maximum deg setting for this config preset
                stage2_palm_yaw_max_deg=180.0,  # Overrides the stage2 palm yaw maximum deg setting for this config preset
                success_palm_yaw_max_deg=180.0,  # Overrides the success palm yaw maximum deg setting for this config preset
                stage1_align_err_max=999.0,  # Overrides the stage1 align error maximum setting for this config preset
                stage2_align_err_max=999.0,  # Overrides the stage2 align error maximum setting for this config preset
                success_align_err_max=999.0,  # Overrides the success align error maximum setting for this config preset
                stage1_opposed_gate_min=-1.0,  # Overrides the stage1 opposed gate minimum setting for this config preset
                stage2_opposed_gate_min=-1.0,  # Overrides the stage2 opposed gate minimum setting for this config preset
                success_opposed_gate_min=-1.0,  # Overrides the success opposed gate minimum setting for this config preset
                # Keep alignment angle as the stage-quality metric
                stage2_line_angle_max_deg=8.0,  # Overrides the stage2 line angle maximum deg setting for this config preset
            )  # closes the current expression
        elif isinstance(group, ContactPoseFallbackConfig):  # Checks alternate branch for isinstance(group, contact pose fallback config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Fallback is disabled in v32, but keep the manifest explicit
                # so enabling fallback later does not silently reintroduce
                # yaw/orient/opposed hard gates
                align_err_max=999.0,  # Overrides the align error maximum setting for this config preset
                palm_orient_max_deg=180.0,  # Overrides the palm orient maximum deg setting for this config preset
                palm_yaw_max_deg=180.0,  # Overrides the palm yaw maximum deg setting for this config preset
                opposed_gate_min=-1.0,  # Overrides the opposed gate minimum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                requires_center=False,  # Overrides the requires center setting for this config preset
                latch=False,  # Overrides the latch setting for this config preset
                hold_steps=1,  # Overrides the hold steps setting for this config preset
                requires_contact_pose=False,  # Overrides the requires contact pose setting for this config preset
                use_xyz_gate=True,  # Overrides the use XYZ gate setting for this config preset
                # Stage 2 is the high pre-descent hover Let it latch when the
                # open pocket is plausibly centered; active closure below keeps
                # the tighter 4 point 5cm -> 1 point 9cm gate
                stage2_xyz_gate_start_m=0.075,  # Overrides the stage2 XYZ gate start m setting for this config preset
                stage2_xyz_gate_full_m=0.035,  # Overrides the stage2 XYZ gate full m setting for this config preset
                xyz_gate_min=1.0e-6,  # Overrides the XYZ gate minimum setting for this config preset
                # Disable face-target center residuals; the center contract
                # for v35 is the shared block-center xyz gate below
                tip_xy_max=0.0,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.0,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.0,  # Overrides the tip Z maximum setting for this config preset
                align_error_max=0.0,  # Overrides the align error maximum setting for this config preset
                align_angle_max_deg=8.0,  # Overrides the align angle maximum deg setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Use block-center xyz rather than thumb/index side ordering,
                # so either side-face approach can pass if the open pocket is
                # centered and angle-aligned
                finger_close_gate_mode="xyz",  # Overrides the finger close gate mode setting for this config preset
                finger_xyz_gate_start_m=0.075,  # Overrides the finger XYZ gate start m setting for this config preset
                finger_xyz_gate_full_m=0.035,  # Overrides the finger XYZ gate full m setting for this config preset
                finger_xyz_gate_linear=True,  # Overrides the finger XYZ gate linear setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # The planar servo uses historical thumb/index face targets
                # v35 keeps only waypoint IK, xyz proximity, and vertical
                # thumb/index alignment angle as active pre-descent logic
                planar_align_servo=False,  # Overrides the planar align servo setting for this config preset
                planar_align_gain=0.0,  # Overrides the planar align gain setting for this config preset
                planar_align_max_xy=0.0,  # Overrides the planar align maximum XY setting for this config preset
                planar_align_max_joint_step=0.0,  # Overrides the planar align maximum joint step setting for this config preset
                # Pocket sweep is local and score-based Run it before Stage 2
                # so it can improve the high hover pocket enough for the
                # stage2 xyz+angle gate to latch
                pocket_sweep=True,  # Overrides the pocket sweep setting for this config preset
                pocket_sweep_stage_min=1,  # Overrides the pocket sweep stage minimum setting for this config preset
                pocket_sweep_iters=2,  # Overrides the pocket sweep iterations setting for this config preset
            )  # closes the current expression
        elif isinstance(group, PreholdClearanceConfig):  # Checks alternate branch for isinstance(group, prehold clearance config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Keep the final contact/descent target from v32, but hold a
                # higher alignment hover until the block-center xyz + angle
                # gate is live This prevents the improved side-agnostic
                # approach from doing its last centering motion at cube height
                clearance_m=0.035,  # Overrides the clearance m setting for this config preset
                tip_clearance_m=0.050,  # Overrides the tip clearance m setting for this config preset
                angle_max_deg=0.0,  # Overrides the angle maximum deg setting for this config preset
                align_err_max=0.0,  # Overrides the align error maximum setting for this config preset
                clearance_until_center=True,  # Overrides the clearance until center setting for this config preset
                stage2_requires_finger_center=True,  # Overrides the stage2 requires finger center setting for this config preset
                stage2_center_bypasses_opposed=True,  # Overrides the stage2 center bypasses opposed setting for this config preset
                finger_requires_center=False,  # Overrides the finger requires center setting for this config preset
                finger_xyz_gate_start_m=0.075,  # Overrides the finger XYZ gate start m setting for this config preset
                finger_xyz_gate_full_m=0.035,  # Overrides the finger XYZ gate full m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Preserve roughly v32's final approach depth after raising
                # the nominal hover from 3cm to 7 point 5cm above the block top
                teacher_lift_z=0.07,  # Overrides the teacher lift Z setting for this config preset
                max_fraction=0.78,  # Overrides the max fraction setting for this config preset
                descent_z=0.080,  # Overrides the descent Z setting for this config preset
                hold_max_fraction=0.78,  # Overrides the hold maximum fraction setting for this config preset
                # The inherited 0 point 68 pre-lift squeeze is enough to form the
                # pinch Do not force a 0 point 92 post-latch crush, which drives the
                # thumb through the block and then drops it
                lift_squeeze_fraction=-1.0,  # Overrides the lift squeeze fraction setting for this config preset
                # After lift latch, preserve the achieved pinch This is not a
                # pre-contact close latch; it only prevents the xyz gate from
                # reopening the fingers while the cube is already being lifted
                freeze_finger_fraction_at_latch=True,  # Overrides the freeze finger fraction at latch setting for this config preset
                lift_finger_freeze_extra_fraction=0.0,  # Overrides the lift finger freeze extra fraction setting for this config preset
                lift_finger_freeze_max_fraction=0.78,  # Overrides the lift finger freeze maximum fraction setting for this config preset
                pre_descent_hover_height_max=0.035,  # Overrides the pre descent hover height maximum setting for this config preset
                descent_requires_center=True,  # Overrides the descent requires center setting for this config preset
                descent_uses_center_ready=True,  # Overrides the descent uses center ready setting for this config preset
                inward_requires_center=True,  # Overrides the inward requires center setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Once bilateral contact has latched, the teacher should carry
                # the captured grip center upward Continuing to solve span and
                # palm-orientation rows during lift rolls the wrist around the
                # cube and turns a vertical lift into a pry motion
                lift_span_scale=0.0,  # Overrides the lift span scale setting for this config preset
                lift_orientation_scale=0.0,  # Overrides the lift orientation scale setting for this config preset
                lift_posture_weight=0.0,  # Overrides the lift posture weight setting for this config preset
            )  # closes the current expression
        elif isinstance(group, LiftSuccessConfig):  # Checks alternate branch for isinstance(group, lift success config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # v35 is side-agnostic Keep the bilateral/contact-chain lift
                # latch, but do not require the historical front/back
                # opposite-face scalar to agree with the chosen side face
                latch_opposed_face_min=-1.0,  # Overrides the latch opposed face minimum setting for this config preset
                # v35 now commands a 7cm vertical teacher lift; make the
                # success threshold match the target instead of accepting the
                # older 3 point 5cm MVP lift
                success_height=0.070,  # Overrides the success height setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherProfile):  # Checks alternate branch for isinstance(group, teacher profile)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                finger_requires_center=False,  # Overrides the finger requires center setting for this config preset
            )  # closes the current expression
    groups.append(  # adds native entrypoint switches for modular v35 launches
        NativeEntrypointConfig(  # starts the native entrypoint config block
            teacher_provider="env",  # Selects the env-backed teacher provider
            contact_attr_parts=True,  # Enables contact teacher parts from env attrs
        )  # closes the current expression
    )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v32 MVP baseline with yaw/orient/opposed removed as hard gates; "  # adds literal text to the surrounding expression
            "Stage 2 and active close use palm waypoint distance, block-center "  # adds literal text to the surrounding expression
            "finger xyz gate, and fingertip alignment angle."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script="",  # Selects module launch for the refactored v35 training path
        module="training.native_entrypoint",  # Selects the native modular training entrypoint
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v35 6cm XYZ align stage RL 1p5m run profile
    """v35 teacher with the v32 MVP Teacher -> BC -> DAgger -> RL schedule, extended to 1.5M."""

    base = teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m"  # Builds the run directory string for this profile
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=1_500_000,  # Overrides the total steps setting for this config preset
                start_steps=50_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=200_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=200_000,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, DaggerConfig):  # Checks alternate branch for isinstance(group, dagger config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                assist_noise_arm=0.0,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.0,  # Overrides the assist noise finger setting for this config preset
            )  # closes the current expression
        elif isinstance(group, RlAssistHandoffConfig):  # Checks alternate branch for isinstance(group, RL assist handoff config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.85,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=750_000,  # Overrides the policy assist decay steps setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v35 side-agnostic 6cm teacher with visible distractors and jitter; "  # adds literal text to the surrounding expression
            "v32 phase contract extended to 1.5M, clean assist targets, and RL "  # adds literal text to the surrounding expression
            "policy assist decaying only to 0.85 for the first run."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v35 6cm XYZ align stage transfer1m run profile
    """Transfer the successful v35 teacher into policy before trying RL again.

    This run is deliberately not an RL experiment.  The v35 teacher is the
    first geometry that reliably generated strict contact and lift, so keep its
    behavior fixed while collecting a long teacher-assisted replay and training
    the actor under BC/DAgger only.  A replay-inclusive final handoff checkpoint
    is supplied by the launcher via ``--final-handoff-checkpoint-path``.
    """

    base = teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m"  # Builds the run directory string for this profile
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
                save_replay_in_checkpoint=False,  # Overrides the save replay in checkpoint setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=1_000_000,  # Overrides the total steps setting for this config preset
                start_steps=200_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=1_000_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, DaggerConfig):  # Checks alternate branch for isinstance(group, dagger config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=1.0,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1,  # Overrides the policy assist decay steps setting for this config preset
                assist_noise_arm=0.0,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.0,  # Overrides the assist noise finger setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v35 successful teacher transfer: 200k teacher replay fill, then "  # adds literal text to the surrounding expression
            "BC/DAgger through 1M with teacher assist fixed at 1.0, no assist "  # adds literal text to the surrounding expression
            "noise, and no RL phase."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v35 lift02 phase1 transfer1m run profile
    """Phase-1 v35 policy transfer with success = 2cm sustained lift.

    This keeps the v35 IK teacher geometry intact and changes the task
    contract for the first learning phase only.  The goal is to transfer the
    demonstrated physical lift behavior before asking RL to optimize the
    stricter opposed/centered grasp metrics.
    """

    base = teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m"  # Builds the run directory string for this profile
    final_replay_path = (  # Overrides the final replay path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m/"  # adds literal text to the surrounding expression
        "final_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
                handoff_checkpoint_path=None,  # Overrides the handoff checkpoint path setting for this config preset
                final_handoff_checkpoint_path=final_replay_path,  # Overrides the final handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
                save_replay_in_checkpoint=False,  # Overrides the save replay in checkpoint setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=1_000_000,  # Overrides the total steps setting for this config preset
                start_steps=100_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=300_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=-1,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, DaggerConfig):  # Checks alternate branch for isinstance(group, dagger config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.85,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=900_000,  # Overrides the policy assist decay steps setting for this config preset
                policy_assist_decay_start_steps=100_000,  # Overrides the policy assist decay start steps setting for this config preset
                assist_noise_arm=0.0,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.0,  # Overrides the assist noise finger setting for this config preset
            )  # closes the current expression
        elif isinstance(group, LiftSuccessConfig):  # Checks alternate branch for isinstance(group, lift success config)
            groups[i] = lift02_phase1_success()  # stores the resolved value in the mapping
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v35 phase-1 policy transfer using the refactored training stack. "  # adds literal text to the surrounding expression
            "Terminal success is sustained physical lift only: block lift > "  # adds literal text to the surrounding expression
            "2cm for 30 consecutive env steps.  Teacher assist decays gently "  # adds literal text to the surrounding expression
            "from 1.0 to 0.85 after a 100k teacher-only warmup."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_lift02_phase1_transfer1m() -> RunProfile:
    """Public submission alias for the 2 cm lift transfer profile."""

    base = teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m()
    return replace(
        base,
        name="teacher_dagger_upstream_fasttd3_lift02_phase1_transfer1m",
        description=(
            "2 cm lift policy transfer using the modular training stack. "
            "Terminal success is sustained physical lift only: block lift > "
            "2cm for 30 consecutive environment steps."
        ),
    )


def teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v35 6cm XYZ align stage dagger1m from900k run profile
    """Continue v35 teacher transfer from the 900k policy with the full replay.

    The transfer run's best policy was around 900k, while the replay-inclusive
    handoff is written only at shutdown.  This profile expects a combined
    checkpoint whose actor/optimizer state comes from ``step_900100.pt`` and
    whose replay buffer comes from the transfer run's final replay checkpoint.
    It then stays in DAgger/BC relabel mode for one more million transitions
    and decays teacher assist gently from 1.0 to 0.9 over that continuation.
    """

    base = teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k"  # Builds the run directory string for this profile
    resume_path = (  # Overrides the resume path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m/"  # adds literal text to the surrounding expression
        "step_900100_policy_full_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    final_replay_path = (  # Overrides the final replay path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k/"  # adds literal text to the surrounding expression
        "final_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                resume_checkpoint=resume_path,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=True,  # Overrides the resume replay setting for this config preset
                resume_global_step=True,  # Overrides the resume global step setting for this config preset
                force_dagger_after_resume=True,  # Overrides the force dagger after resume setting for this config preset
                dagger_resume_policy_assist_mix=1.0,  # Overrides the dagger resume policy assist mix setting for this config preset
                dagger_resume_policy_assist_mix_floor=0.9,  # Overrides the dagger resume policy assist mix floor setting for this config preset
                dagger_resume_policy_assist_decay_steps=1_000_000,  # Overrides the dagger resume policy assist decay steps setting for this config preset
                allow_handoff_source_hash_mismatch=True,  # Overrides the allow handoff source hash mismatch setting for this config preset
                handoff_checkpoint_path=None,  # Overrides the handoff checkpoint path setting for this config preset
                final_handoff_checkpoint_path=final_replay_path,  # Overrides the final handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
                save_replay_in_checkpoint=False,  # Overrides the save replay in checkpoint setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=1_900_099,  # Overrides the total steps setting for this config preset
                start_steps=200_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=1_900_099,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=200_000,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, DaggerConfig):  # Checks alternate branch for isinstance(group, dagger config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.9,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1_000_000,  # Overrides the policy assist decay steps setting for this config preset
                policy_assist_decay_start_steps=900_099,  # Overrides the policy assist decay start steps setting for this config preset
                assist_noise_arm=0.0,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.0,  # Overrides the assist noise finger setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v35 continuation from the 900100 policy with the transfer run's "  # adds literal text to the surrounding expression
            "full replay buffer. Force DAgger after resume, keep teacher BC "  # adds literal text to the surrounding expression
            "relabels active, and decay executed teacher assist from 1.0 to "  # adds literal text to the surrounding expression
            "0.9 over one additional million transitions."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v35 6cm XYZ align stage rl1m from dagger best run profile
    """RL continuation from the best v35 DAgger policy with the final DAgger replay."""

    base = teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best"  # Builds the run directory string for this profile
    resume_path = (  # Overrides the resume path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best/"  # adds literal text to the surrounding expression
        "best_policy_full_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    final_replay_path = (  # Overrides the final replay path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best/"  # adds literal text to the surrounding expression
        "final_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                resume_checkpoint=resume_path,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=True,  # Overrides the resume replay setting for this config preset
                resume_global_step=True,  # Overrides the resume global step setting for this config preset
                force_dagger_after_resume=False,  # Overrides the force dagger after resume setting for this config preset
                allow_handoff_source_hash_mismatch=True,  # Overrides the allow handoff source hash mismatch setting for this config preset
                handoff_checkpoint_path=None,  # Overrides the handoff checkpoint path setting for this config preset
                final_handoff_checkpoint_path=final_replay_path,  # Overrides the final handoff checkpoint path setting for this config preset
                stop_after_handoff_checkpoint=False,  # Overrides the stop after handoff checkpoint setting for this config preset
                save_replay_in_checkpoint=False,  # Overrides the save replay in checkpoint setting for this config preset
            )  # closes the current expression
        elif isinstance(group, CoreTrainingConfig):  # Checks alternate branch for isinstance(group, core training config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                total_steps=2_000_098,  # Overrides the total steps setting for this config preset
                start_steps=200_000,  # Overrides the start steps setting for this config preset
                bc_only_steps=200_000,  # Overrides the BC only steps setting for this config preset
                rl_phase_start_steps=200_000,  # Overrides the RL phase start steps setting for this config preset
            )  # closes the current expression
        elif isinstance(group, DaggerConfig):  # Checks alternate branch for isinstance(group, dagger config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                assist_noise_arm=0.0,  # Overrides the assist noise arm setting for this config preset
                assist_noise_finger=0.0,  # Overrides the assist noise finger setting for this config preset
            )  # closes the current expression
        elif isinstance(group, RlAssistHandoffConfig):  # Checks alternate branch for isinstance(group, RL assist handoff config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                policy_assist_mix=1.0,  # Overrides the policy assist mix setting for this config preset
                policy_assist_mix_floor=0.9,  # Overrides the policy assist mix floor setting for this config preset
                policy_assist_decay_steps=1_000_000,  # Overrides the policy assist decay steps setting for this config preset
                policy_assist_decay_start_steps=1_000_098,  # Overrides the policy assist decay start steps setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "v35 RL continuation from the DAgger run's best policy at global "  # adds literal text to the surrounding expression
            "step 1000098, using the completed DAgger replay. Switch directly "  # adds literal text to the surrounding expression
            "to RL after resume and decay policy assist from 1.0 to 0.9 over "  # adds literal text to the surrounding expression
            "one additional million transitions."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher() -> RunProfile:  # builds the teacher dagger upstream FastTD3 v33 6cm centered descent teacher run profile
    """6cm teacher with a strict no-drift pre-hover and vertical centered descent."""

    base = teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz()  # Loads the base run profile before replacing selected groups
    run_dir = "runs/teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher"  # Builds the run directory string for this profile
    handoff_path = (  # Overrides the handoff path setting for this config preset
        "runs/replay_handoffs/"  # adds literal text to the surrounding expression
        "teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher/handoff_replay.pt"  # adds literal text to the surrounding expression
    )  # closes the current expression
    groups = list(base.groups)  # Starts the config groups included in this run profile
    for i, group in enumerate(groups):  # iterates over configured values
        if isinstance(group, RunIOConfig):  # Checks whether isinstance(group, run IO config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                run_dir=run_dir,  # Builds the run directory string for this profile
                handoff_checkpoint_path=handoff_path,  # Overrides the handoff checkpoint path setting for this config preset
                resume_checkpoint=None,  # Overrides the resume checkpoint setting for this config preset
                resume_replay=False,  # Overrides the resume replay setting for this config preset
                resume_global_step=False,  # Overrides the resume global step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskIdentity):  # Checks alternate branch for isinstance(group, task identity)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # 3cm was close enough to make good grips, but it allowed the
                # preloaded fingers to sweep the block while XY/angle were
                # still converging Use a modest 5cm staging hover, then drop
                # vertically only after the live center/angle gate is stable
                hover_above_block_top=0.05,  # Overrides the hover above block top setting for this config preset
                episode_length_s=12.0,  # Overrides the episode length s setting for this config preset
            )  # closes the current expression
        elif isinstance(group, FingerCenteringConfig):  # Checks alternate branch for isinstance(group, finger centering config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                latch=False,  # Overrides the latch setting for this config preset
                hold_steps=1,  # Overrides the hold steps setting for this config preset
                align_angle_max_deg=10.0,  # Overrides the align angle maximum deg setting for this config preset
                align_error_max=0.220,  # Overrides the align error maximum setting for this config preset
                tip_xy_max=0.085,  # Overrides the tip XY maximum setting for this config preset
                max_tip_xy_max=0.125,  # Overrides the max tip XY maximum setting for this config preset
                tip_z_max=0.080,  # Overrides the tip Z maximum setting for this config preset
                face_half_extent=0.0225,  # Overrides the face half extent setting for this config preset
                face_top_margin=0.009,  # Overrides the face top margin setting for this config preset
            )  # closes the current expression
        elif isinstance(group, StageGateConfig):  # Checks alternate branch for isinstance(group, stage gate config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # v33 accepts a side-on centered pocket instead of forcing the
                # old front/back yaw convention FingerCenteringConfig is the
                # actual pre-descent contract; yaw remains a visual/pose detail
                # and should not keep the hand low where it can knock the cube
                stage2_palm_yaw_max_deg=180.0,  # Overrides the stage2 palm yaw maximum deg setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherLiftConfig):  # Checks alternate branch for isinstance(group, teacher lift config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Same final approach depth as v32's 3cm hover + 3 point 5cm drop,
                # but reached from a safer 5cm hover after a stricter gate
                descent_z=0.055,  # Overrides the descent Z setting for this config preset
                missing_contact_extra_descent=0.005,  # Overrides the missing contact extra descent setting for this config preset
                descent_requires_center=True,  # Overrides the descent requires center setting for this config preset
                descent_uses_center_ready=True,  # Overrides the descent uses center ready setting for this config preset
                inward_requires_center=True,  # Overrides the inward requires center setting for this config preset
                inward_m=0.0,  # Overrides the inward m setting for this config preset
                missing_contact_extra_inward=0.0,  # Overrides the missing contact extra inward setting for this config preset
                vertical_drop_lock_xy=True,  # Overrides the vertical drop lock XY setting for this config preset
                descent_tip_servo_xy_max_m=0.006,  # Overrides the descent tip servo XY maximum m setting for this config preset
                descent_tip_z_target=0.0,  # Overrides the descent tip Z target setting for this config preset
                pre_descent_hover_height_max=0.040,  # Overrides the pre descent hover height maximum setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdConfig):  # Checks alternate branch for isinstance(group, teacher prehold config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                align_angle_gain=0.45,  # Overrides the align angle gain setting for this config preset
                align_angle_max_dz=0.012,  # Overrides the align angle maximum dz setting for this config preset
                align_angle_max_joint_step=0.018,  # Overrides the align angle maximum joint step setting for this config preset
                planar_align_gain=0.0,  # Overrides the planar align gain setting for this config preset
                planar_align_max_xy=0.0,  # Overrides the planar align maximum XY setting for this config preset
                planar_align_max_joint_step=0.0,  # Overrides the planar align maximum joint step setting for this config preset
                # Do not chase final side-face fingertip targets while above
                # the block; that was the path that swept into the cube The
                # task-space center/span terms align the open pocket, and the
                # contact teacher takes over after the live descent gate
                ik_tip_servo_gain=0.0,  # Overrides the IK tip servo gain setting for this config preset
                ik_tip_servo_max_m=0.0,  # Overrides the IK tip servo maximum m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TeacherPreholdAdvancedConfig):  # Checks alternate branch for isinstance(group, teacher prehold advanced config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                # Fixed-face v33 should grasp the robot front/back faces, not
                # the old block-local Y faces Keep the palm yaw target and
                # the state-machine face targets on the same world axis
                target_palm_yaw_world_axis="x",  # Overrides the target palm yaw world axis setting for this config preset
                grip_face_axis="x",  # Overrides the grip face axis setting for this config preset
                finger_front_face_tolerance_m=0.004,  # Overrides the finger front face tolerance m setting for this config preset
            )  # closes the current expression
        elif isinstance(group, TaskSpaceIKConfig):  # Checks alternate branch for isinstance(group, task space IK config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                direct_grip_center=True,  # Overrides the direct grip center setting for this config preset
                grip_offset_blend_requires_descent=True,  # Overrides the grip offset blend requires descent setting for this config preset
                center_xy_weight=18.0,  # Overrides the center XY weight setting for this config preset
                center_z_weight=14.0,  # Overrides the center Z weight setting for this config preset
                span_xy_weight=0.5,  # Overrides the span XY weight setting for this config preset
                span_z_weight=10.0,  # Overrides the span Z weight setting for this config preset
                drop_weight=0.10,  # Overrides the drop weight setting for this config preset
                orientation_weight=0.10,  # Overrides the orientation weight setting for this config preset
                posture_weight=0.18,  # Overrides the posture weight setting for this config preset
                # r26 showed that strongly suppressing span/orientation in the
                # pre-hover made the hand passive and never reached stage 2
                # Keep the new hooks available, but isolate this revision to
                # the face-axis correction
                prehover_span_scale=1.0,  # Overrides the prehover span scale setting for this config preset
                prehover_orientation_scale=0.5,  # Overrides the prehover orientation scale setting for this config preset
                prehover_posture_weight=0.25,  # Overrides the prehover posture weight setting for this config preset
                max_joint_step=0.020,  # Overrides the max joint step setting for this config preset
            )  # closes the current expression
        elif isinstance(group, PreholdClearanceConfig):  # Checks alternate branch for isinstance(group, prehold clearance config)
            groups[i] = replace(  # stores the resolved value in the mapping
                group,  # continues this config expression
                clearance_m=0.020,  # Overrides the clearance m setting for this config preset
                tip_clearance_m=0.050,  # Overrides the tip clearance m setting for this config preset
                angle_max_deg=5.0,  # Overrides the angle maximum deg setting for this config preset
                align_err_max=0.06,  # Overrides the align error maximum setting for this config preset
                arm_servo_uses_contact_missing=False,  # Overrides the arm servo uses contact missing setting for this config preset
                servo_uses_live_contact=False,  # Overrides the servo uses live contact setting for this config preset
                extra_descent_uses_contact_missing=False,  # Overrides the extra descent uses contact missing setting for this config preset
                tip_servo_max_m=0.0,  # Overrides the tip servo maximum m setting for this config preset
                one_sided_close_boost=1.0,  # Overrides the one sided close boost setting for this config preset
                close_requires_descent_ready=True,  # Overrides the close requires descent ready setting for this config preset
                pre_descent_live_debounce_steps=3,  # Overrides the pre descent live debounce steps setting for this config preset
                descent_requires_wrist_yaw_release=False,  # Overrides the descent requires wrist yaw release setting for this config preset
                descent_keepalive=True,  # Overrides the descent keepalive setting for this config preset
                # 6cm side-on pocket: allow close once the centered open
                # pocket is in the 9-10cm max fingertip-to-block band seen
                # in r9/r10, instead of waiting for the old 8 point 5cm cube gate
                finger_xyz_gate_start_m=0.125,  # Overrides the finger XYZ gate start m setting for this config preset
                finger_xyz_gate_full_m=0.055,  # Overrides the finger XYZ gate full m setting for this config preset
                explicit_prehover_waypoint=True,  # Overrides the explicit prehover waypoint setting for this config preset
                explicit_prehover_height_m=0.065,  # Overrides the explicit prehover height m setting for this config preset
                precenter_servo=True,  # Overrides the precenter servo setting for this config preset
                precenter_servo_gain=0.35,  # Overrides the precenter servo gain setting for this config preset
                precenter_servo_max_m=0.015,  # Overrides the precenter servo maximum m setting for this config preset
                precenter_stage_min=0,  # Overrides the precenter stage minimum setting for this config preset
                clearance_until_center=True,  # Overrides the clearance until center setting for this config preset
                stage2_requires_finger_center=True,  # Overrides the stage2 requires finger center setting for this config preset
                stage2_center_bypasses_opposed=False,  # Overrides the stage2 center bypasses opposed setting for this config preset
                finger_requires_center=True,  # Overrides the finger requires center setting for this config preset
                opposed_contact_uses_middle_back=True,  # Overrides the opposed contact uses middle back setting for this config preset
            )  # closes the current expression
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "6cm no-jitter teacher that stages 5cm above the block, waits for "  # adds literal text to the surrounding expression
            "a live centered pocket with align angle <=5deg, locks XY, then "  # adds literal text to the surrounding expression
            "descends vertically before closing beyond the fixed 20% preload."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


def teacher_dagger_upstream_fasttd3_handoff_only() -> RunProfile:  # builds the teacher dagger upstream FastTD3 handoff only run profile
    """Run only the shared BC/DAgger prefix and stop after saving replay."""

    base = teacher_dagger_upstream_fasttd3()  # Loads the base run profile before replacing selected groups
    groups = list(base.groups)  # Starts the config groups included in this run profile
    if not isinstance(groups[0], RunIOConfig):  # Checks whether not isinstance(groups[0], run IO config)
        raise TypeError("expected RunIOConfig as first profile group")  # raises an error for invalid config state
    groups[0] = replace(groups[0], stop_after_handoff_checkpoint=True)  # stores the resolved value in the mapping
    return RunProfile(  # returns the assembled run profile
        name="teacher_dagger_upstream_fasttd3_handoff_only",  # Overrides the name setting for this config preset
        description=(  # Describes the run profile purpose
            "Generate the replay-inclusive BC/DAgger handoff checkpoint, then "  # adds literal text to the surrounding expression
            "exit before collecting RL transitions."  # adds literal text to the surrounding expression
        ),  # closes the current expression
        script=base.script,  # Selects the launcher script for this run profile
        groups=tuple(groups),  # Starts the config groups included in this run profile
    )  # closes the current expression


PROFILES = {  # registers all named run profile factories
    "cloning": cloning,  # Registers cloning as a run profile factory
    "cloning_strict_centered": cloning_strict_centered,  # Registers cloning_strict_centered as a run profile factory
    "cloning_centered_v2": cloning_centered_v2,  # Registers cloning_centered_v2 as a run profile factory
    "cloning_ik_strict_v3": cloning_ik_strict_v3,  # Registers cloning_ik_strict_v3 as a run profile factory
    "cloning_ik_strict_v4": cloning_ik_strict_v4,  # Registers cloning_ik_strict_v4 as a run profile factory
    "cloning_ik_2f_v5": cloning_ik_2f_v5,  # Registers cloning_ik_2f_v5 as a run profile factory
    "teacher_dagger_upstream_fasttd3": teacher_dagger_upstream_fasttd3,  # Registers teacher_dagger_upstream_fasttd3 as a run profile factory
    "teacher_dagger_upstream_fasttd3_handoff_only": teacher_dagger_upstream_fasttd3_handoff_only,  # Registers teacher_dagger_upstream_fasttd3_handoff_only as a run profile factory
    "teacher_dagger_upstream_fasttd3_from_handoff": teacher_dagger_upstream_fasttd3_from_handoff,  # Registers teacher_dagger_upstream_fasttd3_from_handoff as a run profile factory
    "teacher_dagger_upstream_fasttd3_v7_warmstart": teacher_dagger_upstream_fasttd3_v7_warmstart,  # Registers teacher_dagger_upstream_fasttd3_v7_warmstart as a run profile factory
    "teacher_dagger_upstream_fasttd3_v8_600k_handoff": teacher_dagger_upstream_fasttd3_v8_600k_handoff,  # Registers teacher_dagger_upstream_fasttd3_v8_600k_handoff as a run profile factory
    "teacher_dagger_upstream_fasttd3_v8_600k_from_handoff": teacher_dagger_upstream_fasttd3_v8_600k_from_handoff,  # Registers teacher_dagger_upstream_fasttd3_v8_600k_from_handoff as a run profile factory
    "teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint
        teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v9_vertical_center_drop": teacher_dagger_upstream_fasttd3_v9_vertical_center_drop,  # Registers teacher_dagger_upstream_fasttd3_v9_vertical_center_drop as a run profile factory
    "teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop": teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop,  # Registers teacher_dagger_upstream_fasttd3_v10_pose_oriented_drop as a run profile factory
    "teacher_dagger_upstream_fasttd3_v11_finger_plane_basis": teacher_dagger_upstream_fasttd3_v11_finger_plane_basis,  # Registers teacher_dagger_upstream_fasttd3_v11_finger_plane_basis as a run profile factory
    "teacher_dagger_upstream_fasttd3_v12_plane_center_unlock": teacher_dagger_upstream_fasttd3_v12_plane_center_unlock,  # Registers teacher_dagger_upstream_fasttd3_v12_plane_center_unlock as a run profile factory
    "teacher_dagger_upstream_fasttd3_v13_contact_plane_basis": teacher_dagger_upstream_fasttd3_v13_contact_plane_basis,  # Registers teacher_dagger_upstream_fasttd3_v13_contact_plane_basis as a run profile factory
    "teacher_dagger_upstream_fasttd3_v14_segment_plane_basis": teacher_dagger_upstream_fasttd3_v14_segment_plane_basis,  # Registers teacher_dagger_upstream_fasttd3_v14_segment_plane_basis as a run profile factory
    "teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x": teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x,  # Registers teacher_dagger_upstream_fasttd3_v15_segment_plane_axis_x as a run profile factory
    "teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x": teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x,  # Registers teacher_dagger_upstream_fasttd3_v16_segment_plane_axis_neg_x as a run profile factory
    "teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back": teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back,  # Registers teacher_dagger_upstream_fasttd3_v17_segment_axis_x_palm_back as a run profile factory
    "teacher_dagger_upstream_fasttd3_v18_center_span_ik": teacher_dagger_upstream_fasttd3_v18_center_span_ik,  # Registers teacher_dagger_upstream_fasttd3_v18_center_span_ik as a run profile factory
    "teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip": teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip,  # Registers teacher_dagger_upstream_fasttd3_v19_position_only_vertical_grip as a run profile factory
    "teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik": teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik,  # Registers teacher_dagger_upstream_fasttd3_v20_task_space_topdown_ik as a run profile factory
    "teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock": teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock,  # Registers teacher_dagger_upstream_fasttd3_v21_task_space_delayed_unlock as a run profile factory
    "teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock": teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock,  # Registers teacher_dagger_upstream_fasttd3_v22_task_space_descend_then_unlock as a run profile factory
    "teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock": teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock,  # Registers teacher_dagger_upstream_fasttd3_v23_task_space_strict_continuous_unlock as a run profile factory
    "teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate": teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate,  # Registers teacher_dagger_upstream_fasttd3_v24_task_space_grasp_center_gate as a run profile factory
    "teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center": teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center,  # Registers teacher_dagger_upstream_fasttd3_v25_task_space_local_grasp_center as a run profile factory
    "teacher_dagger_upstream_fasttd3_v26_closure_synced_drop": teacher_dagger_upstream_fasttd3_v26_closure_synced_drop,  # Registers teacher_dagger_upstream_fasttd3_v26_closure_synced_drop as a run profile factory
    "teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop": teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop,  # Registers teacher_dagger_upstream_fasttd3_v27_closure_ramped_drop as a run profile factory
    "teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate": teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate,  # Registers teacher_dagger_upstream_fasttd3_v28_xyz_front_close_gate as a run profile factory
    "teacher_dagger_upstream_fasttd3_v29_live_grip_offset": teacher_dagger_upstream_fasttd3_v29_live_grip_offset,  # Registers teacher_dagger_upstream_fasttd3_v29_live_grip_offset as a run profile factory
    "teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate": teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate,  # Registers teacher_dagger_upstream_fasttd3_v30_canonical_hover_xyz_gate as a run profile factory
    "teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore
        teacher_dagger_upstream_fasttd3_v30_preload_closure_aware_restore  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v31_centered_restore": teacher_dagger_upstream_fasttd3_v31_centered_restore,  # Registers teacher_dagger_upstream_fasttd3_v31_centered_restore as a run profile factory
    "teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz
        teacher_dagger_upstream_fasttd3_v32_6cm_hover8_centered_xyz  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k
        teacher_dagger_upstream_fasttd3_v32_6cm_mvp_rl_700k  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix
        teacher_dagger_upstream_fasttd3_v34_6cm_mvp_rl_700k_reward_fix  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage
        teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m
        teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl_1p5m  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m
        teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_transfer1m  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m
        teacher_dagger_upstream_fasttd3_v35_lift02_phase1_transfer1m  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_lift02_phase1_transfer1m": (
        teacher_dagger_upstream_fasttd3_lift02_phase1_transfer1m
    ),
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k
        teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best
        teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best  # continues this config expression
    ),  # closes the current expression
    "teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher": (  # Starts env export expression for teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher
        teacher_dagger_upstream_fasttd3_v33_6cm_centered_descent_teacher  # continues this config expression
    ),  # closes the current expression
    "dagger_rl_from_cloning": dagger_rl_from_cloning,  # Registers dagger_rl_from_cloning as a run profile factory
    "dagger_rl_current": dagger_rl_current,  # Registers dagger_rl_current as a run profile factory
    "pipeline_smoke": pipeline_smoke,  # Registers pipeline_smoke as a run profile factory
}  # closes the current expression


CHECKPOINT_DEPENDENT_PROFILES = {  # historical profiles that require omitted run/checkpoint artifacts
    "dagger_rl_current",
    "dagger_rl_from_cloning",
    "teacher_dagger_upstream_fasttd3_from_handoff",
    "teacher_dagger_upstream_fasttd3_v8_600k_from_handoff",
    "teacher_dagger_upstream_fasttd3_v8_600k_from_step_checkpoint",
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_dagger1m_from900k",
    "teacher_dagger_upstream_fasttd3_v35_6cm_xyz_align_stage_rl1m_from_dagger_best",
}

SUBMISSION_PROFILES = {
    name: factory
    for name, factory in PROFILES.items()
    if name not in CHECKPOINT_DEPENDENT_PROFILES
}
