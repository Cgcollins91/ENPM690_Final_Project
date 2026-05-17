"""
Minimal robot configuration helpers


This config file provides helper functions to generate articulation configurations for the Unitree G1 29DOF Dex3 robot used by this project.
The `RobotJointTemplates` class defines methods to return dictionaries of joint names grouped by leg, waist, arm, and hand joints. 
The `RobotBaseCfg` class provides a factory method to create a base articulation configuration for the robot, allowing customization of 
the initial position, rotation, included joints, and more. Finally, the `G1RobotPresets` class retains named presets for compatibility
with the topdown task modules.

"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX


@configclass
class RobotJointTemplates:
    """Joint templates for the G1 29DOF Dex3 robot used by this project."""

    @classmethod
    def get_leg_joints(cls) -> Dict[str, float]:
        """Return the Unitree G1 leg joint names used by the articulation config."""
        return {
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
        }

    @classmethod
    def get_waist_joints(cls, include_waist: bool = True) -> Dict[str, float]:
        """Return the Unitree G1 waist joint names used by the articulation config."""
        if not include_waist:
            return {}
        return {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        }

    @classmethod
    def get_arm_joints(cls) -> Dict[str, float]:
        """Return the Unitree G1 arm joint names used by the articulation config."""
        return {
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

    @classmethod
    def get_hand_joints(cls, hand_type: Literal["dex3"] = "dex3") -> Dict[str, float]:
        """Return Dex3 hand joint names grouped by left and right hand."""
        if hand_type != "dex3":
            raise ValueError("Standalone ENPM690 project supports only the Dex3 hand.")
        return {
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        }


@configclass
class RobotBaseCfg:
    """Factory for the single robot variant used by the topdown curriculum."""

    @classmethod
    def get_base_config(
        cls,
        prim_path               : str                               = "/World/envs/env_.*/Robot",
        init_pos                : Tuple[float, float, float]        = (-0.15, 0.0, 0.744),
        init_rot                : Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071),
        include_waist           : bool                              = True,
        hand_type               : Literal["dex3"]                   = "dex3",
        base_config             : ArticulationCfg | None            = None,
        custom_joint_pos        : Optional[Dict[str, float]]        = None,
        is_have_hand            : bool                              = True,
        update_default_joint_pos: bool                              = True,
        robot_type              : Literal["g129dof"]                = "g129dof",
    ) -> ArticulationCfg:
        """Return the base Unitree G1 articulation configuration."""
        if robot_type != "g129dof":
            raise ValueError("Standalone ENPM690 project supports only G1 29DOF Dex3.")
        if base_config is None:
            base_config = G129_CFG_WITH_DEX3_BASE_FIX

        if update_default_joint_pos:
            joint_pos: Dict[str, float] = {}
            joint_pos.update(RobotJointTemplates.get_leg_joints())
            joint_pos.update(RobotJointTemplates.get_waist_joints(include_waist))
            joint_pos.update(RobotJointTemplates.get_arm_joints())
            if is_have_hand:
                joint_pos.update(RobotJointTemplates.get_hand_joints(hand_type))
        else:
            joint_pos = base_config.init_state.joint_pos.copy()

        if custom_joint_pos:
            joint_pos = {**joint_pos, **custom_joint_pos}

        return base_config.replace(
            prim_path=prim_path,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot,
                joint_pos=joint_pos,
                joint_vel={".*": 0.0},
            ),
        )


@configclass
class G1RobotPresets:
    """Named preset retained for compatibility with the topdown task modules."""

    @classmethod
    def g1_29dof_dex3_base_fix(
        cls,
        init_pos: Tuple[float, float, float]        = (-0.15, 0.0, 0.76),
        init_rot: Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071),
    ) -> ArticulationCfg:
        """Return the fixed-base G1/Dex3 robot config used by the task."""
        return RobotBaseCfg.get_base_config(
            init_pos=init_pos,
            init_rot=init_rot,
            include_waist=False,
            hand_type="dex3",
            base_config=G129_CFG_WITH_DEX3_BASE_FIX,
        )
