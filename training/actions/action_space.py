"""

Reduced action layout helpers for the topdown trainer


DEFAULT_POLICY_OBS_KEYS:                         Define default policy obs keys constant
OBS_SCHEMA_VERSION:                              Define obs schema version constant
REMOVED_TEACHER_STATE_OBS_KEYS:                  Define removed teacher state obs keys constant
TOPDOWN_POLICY_OBS_DIM:                          Define topdown policy obs dim constant
TOPDOWN_PRIVILEGED_OBS_DIM:                      Define topdown privileged obs dim constant
TOPDOWN_FINGER_UNLOCK_PROGRESS_COL:              Define topdown finger unlock progress col constant
TOPDOWN_STAGE_ONE_HOT_OBS_COL:                   Define topdown stage one hot obs col constant
TOPDOWN_POLICY_OBS_KEYS:                         Define topdown policy obs keys constant
BASE_ARM_JOINTS:                                 Define base arm joints constant
WRIST_ROLL_ARM_JOINTS:                           Define wrist roll arm joints constant
FINGER_JOINTS:                                   Define finger joints constant
FINGER_ACTION_SCALES:                            Define finger action scales constant
RIGHT_PALM_LINK:                                 Define right palm link constant
WORKSPACE_CAMERA_TARGET:                         Define workspace camera target constant
WORKSPACE_CAMERA_EYES:                           Define workspace camera eyes constant
ReducedActionSpec:                               Description of the reduced policy action surface
ActionLayoutOptions:                             CLI-derived options that define the action layout
ActionLayout:                                    Resolved reduced action layout for policy and environment actions
_right_arm_action_scales:                        Handle right arm action scales logic
resolve_action_layout:                           Resolve action joints and scales from trainer options
get_joint_indices:                               Return joint indices by robot joint name
get_action_mapping:                              Return mapped joint indices and action scales for a reduced action spec
expand_reduced_action:                           Expand reduced policy actions into full robot joint target tensors
get_finger_state:                                Return current/default positions, scales, and ids for finger joints
convert_finger_delta_to_reduced:                 Convert per-finger delta commands into absolute reduced-action frame
compute_teacher_finger_reduced_in_current_mode:  Return teacher finger action in the configured action frame
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_POLICY_OBS_KEYS = (
    "robot_joint_state",
    "robot_dex3_state",
    "red_block_perception",
    "palm_pos_world",
    "block_pos_world",
    "block_minus_palm",
    "topdown_center_pos_world",
    "block_minus_topdown_center",
    "topdown_axis_world",
    "topdown_drop_axis_world",
)

OBS_SCHEMA_VERSION = 2
REMOVED_TEACHER_STATE_OBS_KEYS = (
    "controller_state_scalars",
    "teacher_contact_state_scalars",
    "teacher_ik_state_scalars",
    "arm_hold_action_scalars",
)
TOPDOWN_POLICY_OBS_DIM = 171
TOPDOWN_PRIVILEGED_OBS_DIM = 78
TOPDOWN_FINGER_UNLOCK_PROGRESS_COL = 170
TOPDOWN_STAGE_ONE_HOT_OBS_COL = 166
TOPDOWN_POLICY_OBS_KEYS = (
    "robot_joint_state",
    "robot_dex3_state",
    "palm_pos_world",
    "block_pos_world",
    "block_minus_palm",
    "palm_quat_world",
    "block_quat_world",
    "grip_target_pos_world",
    "thumb_tip_pos_world",
    "index_tip_pos_world",
    "middle_tip_pos_world",
    "thumb_minus_block",
    "index_minus_block",
    "middle_minus_block",
    "palm_pose_scalars",
    "axis_orientation_scalars",
    "alignment_scalars",
    "contact_strengths",
    "lift_scalars",
    "tip_target_error_scalars",
    "physical_block_scalars",
    "source_block_one_hot",
    "stage_one_hot",
    "episode_step",
    "finger_unlock_progress",
)

BASE_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
WRIST_ROLL_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
FINGER_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
FINGER_ACTION_SCALES = (+0.942, -0.964, -1.658, +1.492, +1.658, +1.492, +1.658)
RIGHT_PALM_LINK = "right_hand_palm_link"
WORKSPACE_CAMERA_TARGET = (-4.20, -3.74, 0.94)
WORKSPACE_CAMERA_EYES = {
    "world"         : (-2.98, -2.46, 1.98),
    "overview"      : (-2.98, -2.46, 1.98),
    "table_overhead": (-3.34, -2.82, 2.18),
    "top"           : (-4.20, -3.74, 3.20),
}


@dataclass(frozen=True)
class ReducedActionSpec:
    """Description of the reduced policy action surface"""

    joint_names: tuple[str, ...]  # ordered names used to resolve joint attributes
    scales     : tuple[float, ...]  # floating-point scales value used by reduced action spec


@dataclass(frozen=True)
class ActionLayoutOptions:
    """CLI-derived options that define the action layout"""

    include_wrist_roll      : bool  = False  # boolean value indicating the include wrist roll state for action layout options
    include_waist_yaw       : bool  = False  # boolean value indicating the include waist yaw state for action layout options
    waist_yaw_action_scale  : float = 1.0  # multiplier applied to waist yaw action terms
    arm_action_scale_profile: str   = "side"  # string arm action scale profile value used by action layout options
    arm_controller          : str   = "policy"  # string arm controller value used by action layout options


@dataclass(frozen=True)
class ActionLayout:
    """Resolved reduced action layout for policy and environment actions"""

    arm_joints        : tuple[str, ...]  # string arm joints value used by action layout
    finger_joints     : tuple[str, ...]  # string finger joints value used by action layout
    env_action_spec   : ReducedActionSpec  # action layout spec expected by the environment
    policy_action_spec: ReducedActionSpec  # action layout spec expected by the policy output

    @property
    def num_arm(self) -> int:
        """Return the number of arm dimensions"""
        return len(self.arm_joints)

    @property
    def num_fingers(self) -> int:
        """Return the number of finger dimensions"""
        return len(self.finger_joints)

    @property
    def reduced_action_joints(self) -> tuple[str, ...]:
        """Return the full environment reduced-action joint tuple"""
        return self.env_action_spec.joint_names


def _right_arm_action_scales(options: ActionLayoutOptions) -> tuple[float, ...]:
    if options.arm_action_scale_profile == "topdown":
        if options.include_wrist_roll:
            return (4.00, 2.80, 4.00, 1.90, 3.00, 1.80, 1.80)
        return (4.00, 2.80, 4.00, 1.90, 1.80, 1.80)
    if options.include_wrist_roll:
        return (0.50, 0.25, 0.60, 0.80, 0.50, 0.50, 0.50)
    return (0.50, 0.25, 0.60, 0.80, 0.50, 0.50)


def resolve_action_layout(options: ActionLayoutOptions | None = None) -> ActionLayout:
    """Resolve action joints and scales from trainer options

    Steps:
    - Resolve inputs for `resolve_action_layout` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    opts = ActionLayoutOptions() if options is None else options
    selected_arm = WRIST_ROLL_ARM_JOINTS if opts.include_wrist_roll else BASE_ARM_JOINTS
    arm_joints = ("waist_yaw_joint", *selected_arm) if opts.include_waist_yaw else selected_arm
    right_scales = _right_arm_action_scales(opts)
    arm_scales = (
        (float(opts.waist_yaw_action_scale), *right_scales)
        if opts.include_waist_yaw
        else right_scales
    )
    env_spec = ReducedActionSpec(
        joint_names=arm_joints + FINGER_JOINTS,
        scales=tuple(float(x) for x in (*arm_scales, *FINGER_ACTION_SCALES)),
    )
    if opts.arm_controller == "ik":
        policy_spec = ReducedActionSpec(
            joint_names=FINGER_JOINTS,
            scales=tuple(float(x) for x in FINGER_ACTION_SCALES),
        )
    else:
        policy_spec = env_spec
    return ActionLayout(
        arm_joints=arm_joints,
        finger_joints=FINGER_JOINTS,
        env_action_spec=env_spec,
        policy_action_spec=policy_spec,
    )


def get_joint_indices(robot) -> dict[str, int]:
    """Return joint indices by robot joint name"""
    return {name: idx for idx, name in enumerate(robot.data.joint_names)}


def get_action_mapping(
    robot,                           # Param: input value used as robot
    device     : torch.device | str,  # Param: torch device where tensors are read or allocated
    action_spec: ReducedActionSpec,  # Param: input value used as action spec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mapped joint indices and action scales for a reduced action spec"""
    joint_indices = get_joint_indices(robot)
    full_indices = [joint_indices[name] for name in action_spec.joint_names]
    return (
        torch.tensor(full_indices, dtype=torch.long, device=device),
        torch.tensor(action_spec.scales, dtype=torch.float32, device=device),
    )


def expand_reduced_action(
    reduced_action: torch.Tensor,  # Param: tensor input carrying reduced action values
    action_dim    : int,  # Param: integer input for action dim
    mapped_indices: torch.Tensor,  # Param: tensor input carrying mapped indices values
    mapped_scales : torch.Tensor,  # Param: tensor input carrying mapped scales values
) -> torch.Tensor:
    """Expand reduced policy actions into full robot joint target tensors"""
    full_action = torch.zeros((reduced_action.shape[0], action_dim), device=reduced_action.device)
    full_action[:, mapped_indices] = reduced_action * mapped_scales
    return full_action


def get_finger_state(
    env,                           # Param: environment or backend object used for runtime calls
    mapped_indices: torch.Tensor,  # Param: tensor input carrying mapped indices values
    mapped_scales : torch.Tensor,  # Param: tensor input carrying mapped scales values
    *,
    num_arm: int,                  # Param: number of arm action dimensions in the active layout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current/default positions, scales, and ids for finger joints

    Steps:
    - Resolve inputs for `get_finger_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    robot = env.scene["robot"]
    finger_joint_ids = mapped_indices[num_arm:]
    finger_scales = mapped_scales[num_arm:]
    current_pos = robot.data.joint_pos[:, finger_joint_ids]
    default_pos = robot.data.default_joint_pos[:, finger_joint_ids]
    return current_pos, default_pos, finger_scales, finger_joint_ids


def convert_finger_delta_to_reduced(
    env,                                # Param: environment or backend object used for runtime calls
    finger_delta_action: torch.Tensor,  # Param: tensor input carrying finger delta action values
    mapped_indices     : torch.Tensor,  # Param: tensor input carrying mapped indices values
    mapped_scales      : torch.Tensor,  # Param: tensor input carrying mapped scales values
    *,
    num_arm           : int,  # Param: number of arm action dimensions in the active layout
    finger_delta_scale: float,  # Param: multiplier applied to finger delta
) -> torch.Tensor:
    """Convert per-finger delta commands into absolute reduced-action frame

    Steps:
    - Resolve inputs for `convert_finger_delta_to_reduced` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    current_pos, default_pos, finger_scales, _ = get_finger_state(
        env,
        mapped_indices,
        mapped_scales,
        num_arm=num_arm,
    )
    scale_sign = torch.sign(finger_scales)
    raw_delta = finger_delta_action * scale_sign.unsqueeze(0) * float(finger_delta_scale)
    desired_target = current_pos + raw_delta
    safe_scales = torch.where(
        finger_scales.abs() > 1e-6,
        finger_scales,
        torch.ones_like(finger_scales),
    )
    reduced_out = (desired_target - default_pos) / safe_scales.unsqueeze(0)
    return reduced_out.clamp(-1.0, 1.0)


def compute_teacher_finger_reduced_in_current_mode(
    env,                           # Param: environment or backend object used for runtime calls
    mapped_indices: torch.Tensor,  # Param: tensor input carrying mapped indices values
    mapped_scales : torch.Tensor,  # Param: tensor input carrying mapped scales values
    fraction      : torch.Tensor,  # Param: tensor input carrying fraction values
    *,
    num_arm           : int,  # Param: number of arm action dimensions in the active layout
    num_fingers       : int,  # Param: number of finger action dimensions in the active layout
    finger_action_mode: str,  # Param: mode string selecting the finger action behavior
    finger_delta_scale: float,  # Param: multiplier applied to finger delta
) -> torch.Tensor:
    """Return teacher finger action in the configured action frame

    Steps:
    - Resolve inputs for `compute_teacher_finger_reduced_in_current_mode` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if fraction.dim() == 1:
        reduced = fraction.unsqueeze(-1).expand(env.num_envs, num_fingers)
    else:
        reduced = fraction.reshape(env.num_envs, num_fingers)
    reduced = reduced.clamp(-1.0, 1.0)
    if finger_action_mode == "absolute":
        return reduced
    current_pos, default_pos, finger_scales, _ = get_finger_state(
        env,
        mapped_indices,
        mapped_scales,
        num_arm=num_arm,
    )
    abs_target = default_pos + reduced * finger_scales.unsqueeze(0)
    raw_delta = abs_target - current_pos
    scale_sign = torch.sign(finger_scales).unsqueeze(0)
    reduced_delta = (raw_delta * scale_sign) / max(float(finger_delta_scale), 1e-6)
    return reduced_delta.clamp(-1.0, 1.0)
