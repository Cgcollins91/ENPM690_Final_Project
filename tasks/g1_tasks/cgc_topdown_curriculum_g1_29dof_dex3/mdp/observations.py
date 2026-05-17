""" Defines observation composition functions for the topdown reach-align-contact curriculum, including geometric, contact, 
and teacher state features.


Critical contract: the policy obs MUST include ``finger_unlock_progress`` as
a single scalar so the trainer's TD3 actor/target update can mask finger
columns symmetrically with the rollout finger gate. See
``configure_finger_unlock_progress_obs_col`` in the trainer.

Keep observation widths stable once a checkpoint exists.  The trainer performs
schema checks on checkpoint load because changing actor-visible terms changes
the network input width and invalidates old policies.  Add new diagnostics as
critic-only terms or gated profile variants unless retraining from scratch.




"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tasks.common_observations.dex3_state import get_robot_dex3_joint_states
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states

from .state_machine import (
    ensure_curriculum_stage_updated,    # Refresh per-env curriculum state before stage-dependent obs
    palm_distance,                      # Grip-center 3D distance to reach/align hover target
    palm_height_error,                  # Grip-center vertical error from reach/align hover target
    palm_drop_axis_error_rad,           # Palm-to-grip drop-axis angle from world -Z
    palm_yaw_axis_error_rad,            # Thumb/back-finger yaw error against grip face axis
    palm_spread_axis_error_rad,         # Index/middle spread error against block spread axis
    open_hand_alignment_error,          # Active fingertip distance sum to face targets
    fingertip_line_angle_rad,           # Thumb-to-back-finger pinch-line elevation angle
    opposite_face_gate,                 # Smooth [0, 1] score for opposed-face fingertip placement
    any_hand_contact_strength,          # Max normalized contact across fingertips and palm
    thumb_contact_strength,             # Normalized contact for thumb chain
    index_contact_strength,             # Normalized contact for index-finger chain
    palm_contact_strength,              # Normalized contact for palm sensor
    block_displacement,                 # Full 3D block displacement from episode spawn position
    block_xy_displacement,              # Horizontal block drift from episode spawn position
    block_lift_height,                  # Positive block height gain from episode spawn height
    opposed_contact_strength,           # Thumb-plus-opposed-back-finger pinch contact
    stage_one_hot as _stage_one_hot,    # Three-column curriculum stage one-hot
    finger_unlock_progress as _finger_unlock_progress,  # Single-column finger-close gate progress
    _active_source_pose_idx,            # Per-env active visible/source block pose index
    _block_pose,                        # Active block world-position and quaternion lookup
    _palm_pose,                         # Palm world-position and quaternion lookup
    _link_pos,                          # Named robot link world-position lookup
    _grip_target_position,              # World-space reach/align hover target for grip center
    _face_targets,                      # Thumb/index opposed face targets near block top
    _THUMB_LINK,                        # Rigid-body name for thumb link in observations
    _INDEX_LINK,                        # Rigid-body name for index link in observations
    _MIDDLE_LINK,                       # Rigid-body name for middle link in observations
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _to_env_local_world_frame(env: "ManagerBasedRLEnv", pos_w: torch.Tensor) -> torch.Tensor:
    """
    Return world-axis-aligned positions with env-grid offsets removed.\
        The "world" frame for the block and palm may have a constant offset from the "world" 
        frame of the robot root and teacher state, which are used for curriculum logic and may be more stable for learning. 
        Removing the offset keeps the block and palm positions consistent across envs with different grid placements, and 
        consistent with the robot-relative features.
    """
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        return pos_w
    return pos_w - env_origins.to(device=pos_w.device, dtype=pos_w.dtype)


def get_palm_pos_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Return palm position in world coordinates
    
    """
    pos, _ = _palm_pose(env)
    return _to_env_local_world_frame(env, pos)


def get_palm_quat_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Return palm orientation quaternion in world coordinates
    """
    _, quat = _palm_pose(env)
    return quat


def get_block_pos_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return active block position in world coordinates."""
    pos, _ = _block_pose(env)
    return _to_env_local_world_frame(env, pos)


def get_block_quat_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return active block orientation quaternion in world coordinates."""
    _, quat = _block_pose(env)
    return quat


def get_grip_target_pos_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the target point the palm should approach for topdown grip."""
    return _to_env_local_world_frame(env, _grip_target_position(env))


def get_block_minus_palm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the active block position relative to the palm."""
    palm, _ = _palm_pose(env)
    block, _ = _block_pose(env)
    return block - palm


def get_thumb_tip_pos_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return thumb-tip position in world coordinates."""
    return _to_env_local_world_frame(env, _link_pos(env, _THUMB_LINK))


def get_index_tip_pos_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return index-tip position in world coordinates."""
    return _to_env_local_world_frame(env, _link_pos(env, _INDEX_LINK))


def get_middle_tip_pos_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return middle-tip position in world coordinates."""
    return _to_env_local_world_frame(env, _link_pos(env, _MIDDLE_LINK))


def get_thumb_minus_block(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return thumb-tip position relative to the active block."""
    block, _ = _block_pose(env)
    return _link_pos(env, _THUMB_LINK) - block


def get_index_minus_block(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return index-tip position relative to the active block."""
    block, _ = _block_pose(env)
    return _link_pos(env, _INDEX_LINK) - block


def get_middle_minus_block(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return middle-tip position relative to the active block."""
    block, _ = _block_pose(env)
    return _link_pos(env, _MIDDLE_LINK) - block


def get_palm_pose_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """4-dim: palm_distance, palm_height_err, palm_drop_axis_rad, alignment_err.

    Width preserved (4) to keep the trainer's ``finger_unlock_progress``
    obs-column offset stable. New yaw / spread axis errors are exposed via
    ``get_axis_orientation_scalars`` instead.
    """
    return torch.stack(
        (
            palm_distance(env),
            palm_height_error(env),
            palm_drop_axis_error_rad(env),
            open_hand_alignment_error(env),
        ),
        dim=-1,
    )


def get_axis_orientation_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """2-dim: palm_yaw_axis_rad, palm_spread_axis_rad. Block-relative."""
    return torch.stack(
        (
            palm_yaw_axis_error_rad(env),
            palm_spread_axis_error_rad(env),
        ),
        dim=-1,
    )


def get_alignment_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """3-dim: fingertip_line_angle_rad, opposite_face_gate, block_displacement."""
    return torch.stack(
        (
            fingertip_line_angle_rad(env),
            opposite_face_gate(env),
            block_displacement(env),
        ),
        dim=-1,
    )


def get_contact_strengths(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """4-dim: thumb, index, palm, any."""
    return torch.stack(
        (
            thumb_contact_strength(env),
            index_contact_strength(env),
            palm_contact_strength(env),
            any_hand_contact_strength(env),
        ),
        dim=-1,
    )


def get_lift_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1-dim: block lift height."""
    return block_lift_height(env).unsqueeze(-1)


def get_tip_target_error_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """6-dim: thumb/index face-target error vectors in world frame."""
    thumb_target, index_target = _face_targets(env)
    thumb_error = thumb_target - _link_pos(env, _THUMB_LINK)
    index_error = index_target - _link_pos(env, _INDEX_LINK)
    return torch.cat((thumb_error, index_error), dim=-1)


def _active_block_ang_vel_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return active block angular velocity in world coordinates."""
    names = ("object", "object_yellow", "object_blue")
    try:
        vel_stack = torch.stack(
            [env.scene[name].data.root_ang_vel_w[:, :3] for name in names],
            dim=0,
        )
        source_idx = _active_source_pose_idx(env)
        env_idx = torch.arange(env.num_envs, device=env.device)
        return vel_stack[source_idx, env_idx]
    except (AttributeError, KeyError):
        pass

    active_name = str(getattr(env, "_topdown_active_object_name", "object"))
    try:
        return env.scene[active_name].data.root_ang_vel_w[:, :3]
    except (AttributeError, KeyError):
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)


def get_physical_block_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """4-dim: xy drift, lift height, opposed contact strength, block angular speed."""
    ang_vel = _active_block_ang_vel_world(env)
    return torch.stack(
        (
            block_xy_displacement(env),
            block_lift_height(env),
            opposed_contact_strength(env),
            torch.linalg.norm(ang_vel, dim=-1),
        ),
        dim=-1,
    )


def get_source_block_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """3-dim one-hot source block id: red/object, yellow, blue."""
    idx = _active_source_pose_idx(env).reshape(env.num_envs, 1)
    out = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    return out.scatter_(1, idx.clamp(0, 2), 1.0)


def episode_step_fraction(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return normalized episode progress as a single deployable scalar."""
    ep_len = getattr(env, "episode_length_buf", None)
    if not torch.is_tensor(ep_len) or ep_len.shape[:1] != (env.num_envs,):
        ep_len = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    max_len = float(getattr(env, "max_episode_length", 1.0) or 1.0)
    progress = ep_len.to(device=env.device, dtype=torch.float32) / max(max_len, 1.0)
    return progress.clamp(0.0, 1.0).unsqueeze(-1)


def _state_scalar(env: "ManagerBasedRLEnv", name: str, *, default: float = 0.0) -> torch.Tensor:
    """Return one scalar curriculum-state tensor as an observation term."""
    value = getattr(env, name, None)
    if torch.is_tensor(value) and value.shape[:1] == (env.num_envs,):
        return value.to(device=env.device, dtype=torch.float32).reshape(env.num_envs)
    return torch.full((env.num_envs,), float(default), device=env.device, dtype=torch.float32)


def _counter_scalar(env: "ManagerBasedRLEnv", name: str, *, scale: float = 100.0) -> torch.Tensor:
    """Return one scalar curriculum counter normalized for observation use."""
    return _state_scalar(env, name) / float(scale)


def get_controller_state_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """22-dim controller/latch state needed to imitate the IK/contact teacher.

    The teacher is stateful: contact-pose readiness, finger-center readiness,
    and lift latches change its action even when geometry is nearly identical.
    Exposing these scalars makes the BC target Markovian for the student.
    """
    ensure_curriculum_stage_updated(env)
    return torch.stack(
        (
            _state_scalar(env, "_topdown_contact_pose_ready"),
            _state_scalar(env, "_topdown_finger_center_ready"),
            _state_scalar(env, "_topdown_finger_center_live"),
            _state_scalar(env, "reach_align_finger_unlocked"),
            _state_scalar(env, "_arm_lift_latched"),
            _state_scalar(env, "_teacher_ik_topdown_lift_progress"),
            _state_scalar(env, "_inpocket_arm_hold_lift_release"),
            _state_scalar(env, "_inpocket_arm_hold_live_gate"),
            _state_scalar(env, "_inpocket_arm_hold_active"),
            _state_scalar(env, "_inpocket_arm_hold_freeze_ready"),
            _state_scalar(env, "_inpocket_arm_hold_valid"),
            _state_scalar(env, "_inpocket_arm_hold_frozen"),
            _state_scalar(env, "_contact_preroll_touch_phase_latched"),
            _state_scalar(env, "_topdown_contact_teacher_arm_hold_unlock_fallback"),
            _state_scalar(env, "_teacher_ik_topdown_lift_freeze_active"),
            _counter_scalar(env, "_topdown_reach_hold"),
            _counter_scalar(env, "_topdown_align_hold"),
            _counter_scalar(env, "_topdown_stage2_age"),
            _counter_scalar(env, "_topdown_stage2_fallout_hold"),
            _counter_scalar(env, "_topdown_contact_pose_hold"),
            _counter_scalar(env, "_topdown_contact_pose_age"),
            _counter_scalar(env, "_topdown_finger_center_hold"),
        ),
        dim=-1,
    )


def get_teacher_contact_state_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """34-dim contact-teacher finite-state and servo state.

    These fields are written by ``compute_topdown_contact_teacher_parts`` in
    the trainer. Defaults keep the observation well-defined before the teacher
    has run on the first step of an episode.
    """
    return torch.stack(
        (
            _state_scalar(env, "_topdown_contact_teacher_ready"),
            _state_scalar(env, "_topdown_contact_teacher_finger_ready"),
            _state_scalar(env, "_topdown_contact_teacher_center_gate"),
            _state_scalar(env, "_topdown_contact_teacher_wrist_yaw_release_gate"),
            _state_scalar(env, "_topdown_contact_teacher_descent_ready"),
            _state_scalar(env, "_topdown_contact_teacher_thumb_fraction"),
            _state_scalar(env, "_topdown_contact_teacher_index_fraction"),
            _state_scalar(env, "_topdown_contact_teacher_thumb_latched"),
            _state_scalar(env, "_topdown_contact_teacher_index_latched"),
            _state_scalar(env, "_topdown_contact_teacher_thumb_hold_fraction", default=-1.0),
            _state_scalar(env, "_topdown_contact_teacher_index_hold_fraction", default=-1.0),
            _state_scalar(env, "_topdown_contact_teacher_thumb_lift_freeze_fraction", default=-1.0),
            _state_scalar(env, "_topdown_contact_teacher_index_lift_freeze_fraction", default=-1.0),
            _state_scalar(env, "_topdown_contact_teacher_closure_fraction"),
            _state_scalar(env, "_topdown_contact_teacher_thumb_missing"),
            _state_scalar(env, "_topdown_contact_teacher_index_missing"),
            _state_scalar(env, "_topdown_contact_teacher_live_thumb_missing"),
            _state_scalar(env, "_topdown_contact_teacher_live_index_missing"),
            _state_scalar(env, "_topdown_contact_teacher_servo_thumb_missing"),
            _state_scalar(env, "_topdown_contact_teacher_servo_index_missing"),
            _state_scalar(env, "_topdown_contact_teacher_thumb_z_gap"),
            _state_scalar(env, "_topdown_contact_teacher_index_z_gap"),
            _state_scalar(env, "_topdown_contact_teacher_thumb_geom_done"),
            _state_scalar(env, "_topdown_contact_teacher_index_geom_done"),
            _state_scalar(env, "_topdown_contact_teacher_descent_z"),
            _state_scalar(env, "_topdown_contact_teacher_descent_z_need"),
            _state_scalar(env, "_topdown_contact_teacher_inward_m"),
            _state_scalar(env, "_topdown_contact_teacher_tip_servo_m"),
            _state_scalar(env, "_topdown_contact_teacher_precenter_servo_m"),
            _state_scalar(env, "_topdown_contact_teacher_precenter_active"),
            _state_scalar(env, "_topdown_contact_teacher_center_servo_m"),
            _state_scalar(env, "_topdown_contact_teacher_center_servo_active"),
            _state_scalar(env, "_topdown_contact_teacher_center_err_xy"),
            torch.linalg.norm(_state_vector(env, "_topdown_contact_teacher_xy_offset", 2), dim=-1),
        ),
        dim=-1,
    )


def get_teacher_ik_state_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """14-dim arm-IK teacher servo state."""
    return torch.stack(
        (
            _state_scalar(env, "_teacher_ik_topdown_inward_m"),
            _state_scalar(env, "_teacher_ik_topdown_tip_servo_m"),
            _state_scalar(env, "_teacher_ik_topdown_prehold_tip_servo_m"),
            _state_scalar(env, "_teacher_ik_topdown_align_line_z"),
            _state_scalar(env, "_teacher_ik_topdown_align_servo_q"),
            _state_scalar(env, "_teacher_ik_topdown_align_servo_dz"),
            _state_scalar(env, "_teacher_ik_topdown_align_servo_active"),
            _state_scalar(env, "_teacher_ik_topdown_planar_align_err_xy"),
            _state_scalar(env, "_teacher_ik_topdown_planar_align_servo_q"),
            _state_scalar(env, "_teacher_ik_topdown_planar_align_servo_m"),
            _state_scalar(env, "_teacher_ik_topdown_planar_align_servo_active"),
            _state_scalar(env, "_teacher_ik_topdown_pocket_sweep_q"),
            _state_scalar(env, "_teacher_ik_topdown_pocket_sweep_active"),
            _state_scalar(env, "_teacher_ik_topdown_lift_freeze_active"),
        ),
        dim=-1,
    )


def _state_vector(env: "ManagerBasedRLEnv", name: str, width: int) -> torch.Tensor:
    """Return a vector observation term copied from curriculum state buffers."""
    value = getattr(env, name, None)
    if torch.is_tensor(value) and value.shape == (env.num_envs, width):
        return value.to(device=env.device, dtype=torch.float32)
    return torch.zeros((env.num_envs, width), device=env.device, dtype=torch.float32)


def get_arm_hold_action_scalars(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """8-dim canonical frozen arm-hold target: waist + right arm joint actions."""
    held = getattr(env, "_inpocket_arm_hold_action", None)
    joint_names = tuple(getattr(env, "_topdown_arm_hold_joint_names", ()))
    out = torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)
    if not torch.is_tensor(held) or held.shape[:1] != (env.num_envs,):
        return out
    held = held.to(device=env.device, dtype=torch.float32)
    canonical = (
        "waist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    for src_idx, joint_name in enumerate(joint_names[: held.shape[-1]]):
        try:
            dst_idx = canonical.index(str(joint_name))
        except ValueError:
            continue
        out[:, dst_idx] = held[:, src_idx]
    return out


def stage_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """3-dim one-hot of curriculum stage (0/1/2)."""
    return _stage_one_hot(env)


def finger_unlock_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1-dim scalar in [0, 1] — load-bearing for TD3 update finger mask."""
    return _finger_unlock_progress(env)
