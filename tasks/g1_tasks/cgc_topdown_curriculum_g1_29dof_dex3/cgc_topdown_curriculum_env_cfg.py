"""Env config for the topdown reach-align-contact curriculum task.

Self-contained: scene, observations, actions, events, rewards, terminations
all defined in this package. No subclassing of legacy task env cfgs.
The trainer detects this task via
``env.cfg.phase1_target_mode == "topdown_grip"``.
"""

# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

# Isaac Sim's bundled Python launches us in a subprocess that drops PYTHONPATH
# (see training_engine.build_plan); only PROJECT_ROOT is added to sys.path
# (see topdown_dagger_td3.py). Add src/ so enpm690_final_project is importable.
_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX
from tasks.common_config.robot_configs import RobotBaseCfg
from enpm690_final_project.config.physics_profile import (
    ResolvedPhysicsProfile,
    resolve_physics_profile,
)

from . import mdp

# Resolve TOPDOWN_PHYSICS_PROFILE once at module load so every consumer
# (block factory, fingertip material binder, __post_init__) sees a
# consistent set of values.
_PHYSICS_PROFILE: ResolvedPhysicsProfile = resolve_physics_profile(os.environ)


# --- Geometry constants (own copies, no imports from sibling packages) -------


def _env_float(name: str, default: float) -> float:
    """Read a float task configuration override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean task configuration override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_TABLE_POS = (-4.3, -4.2, 0.77)
_TABLE_SIZE = (0.9, 0.9, 0.08)
_TABLE_TOP_Z = _TABLE_POS[2] + 0.5 * _TABLE_SIZE[2]
_WRAP_TABLE_ENABLED = _env_bool("TOPDOWN_WRAP_TABLE", False)
_WRAP_TABLE_POS = (
    _env_float("TOPDOWN_WRAP_TABLE_X", -4.57),
    _env_float("TOPDOWN_WRAP_TABLE_Y", -3.615),
    _TABLE_POS[2],
)
_WRAP_TABLE_SIZE = (
    _env_float("TOPDOWN_WRAP_TABLE_SIZE_X", 0.38),
    _env_float("TOPDOWN_WRAP_TABLE_SIZE_Y", 0.27),
    _TABLE_SIZE[2],
)
_WRAP_TABLE_SPAWN_POS = _WRAP_TABLE_POS if _WRAP_TABLE_ENABLED else (0.0, 0.0, -10.0)
_WRAP_TABLE_SPAWN_SIZE = _WRAP_TABLE_SIZE if _WRAP_TABLE_ENABLED else (0.01, 0.01, 0.01)
_TOPDOWN_BLOCK_SIZE = float(os.environ.get("TOPDOWN_BLOCK_SIZE", "0.08"))
_TOPDOWN_BLOCK_HALF_EXTENT = 0.5 * _TOPDOWN_BLOCK_SIZE
_BLOCK_INIT_Z = _TABLE_TOP_Z + _TOPDOWN_BLOCK_HALF_EXTENT
_BLOCK_INIT_POS = (
    _env_float("TOPDOWN_RED_SOURCE_X", -4.52),
    _env_float("TOPDOWN_RED_SOURCE_Y", -3.83),
    _BLOCK_INIT_Z,
)
_BLOCK_JITTER_X = _env_float("TOPDOWN_BLOCK_JITTER_X", 0.025)
_BLOCK_JITTER_Y = _env_float("TOPDOWN_BLOCK_JITTER_Y", 0.025)
_BLOCK_POSE_RANGE = {"x": (-_BLOCK_JITTER_X, _BLOCK_JITTER_X), "y": (-_BLOCK_JITTER_Y, _BLOCK_JITTER_Y)}
_SOURCE_POSE_MODE = os.environ.get("TOPDOWN_SOURCE_POSE_MODE", "red").lower()
_PYRAMID_SOURCE_POSE_MODES = {"pyramid", "pyramid3", "three", "all3", "all"}
_FIXED_SOURCE_POSE_INDEX = {
    "red": 0,
    "yellow": 1,
    "blue": 2,
}
_KEEP_DISTRACTORS_VISIBLE = os.environ.get("TOPDOWN_KEEP_DISTRACTORS_VISIBLE", "0") == "1"
_ACTIVE_BLOCK_COLOR = {
    "red": (1.0, 0.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}.get(_SOURCE_POSE_MODE, (1.0, 0.0, 0.0)) if not _KEEP_DISTRACTORS_VISIBLE else (1.0, 0.0, 0.0)
_ROBOT_INIT_POS = (-4.2, -3.60, 0.76)
_ROBOT_INIT_ROT = (0.7071, 0, 0, -0.7071)
_INCLUDE_WAIST_YAW = os.environ.get(
    "TOPDOWN_INCLUDE_WAIST",
    os.environ.get("INCLUDE_WAIST_YAW", "0"),
) == "1"
_WAIST_YAW_INIT_RAD = math.radians(
    _env_float("TOPDOWN_WAIST_YAW_INIT_DEG", _env_float("WAIST_YAW_INIT_DEG", 0.0))
)

_OBJECT_FILTER_EXPR = ["/World/envs/env_.*/Object"]
if _KEEP_DISTRACTORS_VISIBLE:
    _OBJECT_FILTER_EXPR = [
        "/World/envs/env_.*/Object",
        "/World/envs/env_.*/ObjectYellow",
        "/World/envs/env_.*/ObjectBlue",
    ]

# Topdown ready pose for the right arm — natural L-shape parallel to table.
# Upper arm forward (horizontal), elbow bent 90°, forearm horizontal so the
# entire arm lies in a plane parallel to the table with the hand at shoulder
# height. Wrist neutral so hand/index continue inline with the forearm.
_CURRICULUM_CUSTOM_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 1.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.843999981880188,
    "right_shoulder_roll_joint": -1.1070587139917407,
    "right_shoulder_yaw_joint": -1.218999981880188,
    "right_elbow_joint": 0.2460000067949295,
    "right_wrist_roll_joint": 0.9409999847412109,
    "right_wrist_pitch_joint": 0.07500000298023224,
    "right_wrist_yaw_joint": 0.12399999797344208,
}
if _INCLUDE_WAIST_YAW:
    _CURRICULUM_CUSTOM_JOINT_POS.update(
        {
            "waist_yaw_joint": _WAIST_YAW_INIT_RAD,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        }
    )

_CURRICULUM_ARM_RESET_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


# --- Reset helpers (self-contained) ------------------------------------------


def _resolve_env_ids(env, env_ids):
    """Normalize optional environment ids to a tensor on the simulation device."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if torch.is_tensor(env_ids):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.tensor(env_ids, device=env.device, dtype=torch.long)


def _reset_robot_to_default(env, env_ids=None):
    """Reset robot joint state to the configured default pose."""
    env_ids = _resolve_env_ids(env, env_ids)
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def _reset_scene_object_pose(env, object_name: str, pos: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write root pose and velocity state for one scene object."""
    try:
        obj = env.scene[object_name]
    except KeyError:
        return
    n = env_ids.shape[0]
    quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).expand(n, 4).clone()
    root_pose = torch.cat((pos, quat), dim=-1)
    root_vel = torch.zeros((n, 6), device=env.device)
    obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def _sample_block_jitter(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-environment xy jitter for block reset positions."""
    dx_lo, dx_hi = _BLOCK_POSE_RANGE["x"]
    dy_lo, dy_hi = _BLOCK_POSE_RANGE["y"]
    dx = torch.empty(n, device=device).uniform_(dx_lo, dx_hi)
    dy = torch.empty(n, device=device).uniform_(dy_lo, dy_hi)
    return dx, dy


def _reset_inactive_distractors_for_source_pose_mode(env, env_ids: torch.Tensor) -> None:
    """Move inactive distractor blocks out of the active scene area."""
    if _SOURCE_POSE_MODE not in _PYRAMID_SOURCE_POSE_MODES and _SOURCE_POSE_MODE not in _FIXED_SOURCE_POSE_INDEX:
        return
    if _KEEP_DISTRACTORS_VISIBLE:
        return
    n = env_ids.shape[0]
    env_origins = env.scene.env_origins[env_ids]
    disabled = torch.tensor((0.0, 0.0, -10.0), device=env.device).expand(n, 3).clone()
    disabled = disabled + env_origins
    _reset_scene_object_pose(env, "object_blue", disabled, env_ids)
    _reset_scene_object_pose(env, "object_yellow", disabled, env_ids)


def _reset_visible_source_blocks(env, env_ids: torch.Tensor) -> None:
    """Spawn all color-coded source blocks at their fixed source positions."""
    if not _KEEP_DISTRACTORS_VISIBLE:
        return
    if _SOURCE_POSE_MODE not in _PYRAMID_SOURCE_POSE_MODES and _SOURCE_POSE_MODE not in _FIXED_SOURCE_POSE_INDEX:
        return
    env_origins = env.scene.env_origins[env_ids]
    sources = torch.tensor(_PYRAMID_SOURCE_POSITIONS, device=env.device)
    positions = sources.unsqueeze(1) + env_origins.unsqueeze(0)
    dx, dy = _sample_block_jitter(env_ids.shape[0], env.device)
    positions[:, :, 0] = positions[:, :, 0] + dx.unsqueeze(0)
    positions[:, :, 1] = positions[:, :, 1] + dy.unsqueeze(0)
    # Source index convention: 0=red active Object, 1=yellow ObjectYellow,
    # 2=blue ObjectBlue. The state machine gathers from these by source idx.
    _reset_scene_object_pose(env, "object", positions[0], env_ids)
    _reset_scene_object_pose(env, "object_yellow", positions[1], env_ids)
    _reset_scene_object_pose(env, "object_blue", positions[2], env_ids)
    env._topdown_use_visible_source_objects = True


def _reset_block_randomized(env, env_ids=None):
    """Reset the active block and distractors using the configured source-pose mode."""
    env_ids = _resolve_env_ids(env, env_ids)
    obj = env.scene["object"]
    n = env_ids.shape[0]
    sources = torch.tensor(_PYRAMID_SOURCE_POSITIONS, device=env.device)
    if _SOURCE_POSE_MODE in _PYRAMID_SOURCE_POSE_MODES:
        source_idx = torch.randint(0, sources.shape[0], (n,), device=env.device)
        base_pos = sources[source_idx].clone()
        full_idx = getattr(env, "_topdown_source_pose_idx", None)
        if full_idx is None or full_idx.shape[0] != env.num_envs:
            full_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        full_idx[env_ids] = source_idx
        env._topdown_source_pose_idx = full_idx
        if _KEEP_DISTRACTORS_VISIBLE:
            _reset_visible_source_blocks(env, env_ids)
            return
    elif _SOURCE_POSE_MODE in _FIXED_SOURCE_POSE_INDEX:
        source_idx_value = int(_FIXED_SOURCE_POSE_INDEX[_SOURCE_POSE_MODE])
        source_idx = torch.full((n,), source_idx_value, device=env.device, dtype=torch.long)
        base_pos = sources[source_idx].clone()
        full_idx = getattr(env, "_topdown_source_pose_idx", None)
        if full_idx is None or full_idx.shape[0] != env.num_envs:
            full_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        full_idx[env_ids] = source_idx
        env._topdown_source_pose_idx = full_idx
        if _KEEP_DISTRACTORS_VISIBLE:
            _reset_visible_source_blocks(env, env_ids)
            return
    else:
        base_pos = torch.tensor(_BLOCK_INIT_POS, device=env.device).expand(n, 3).clone()
    env_origins = env.scene.env_origins[env_ids]
    base_pos = base_pos + env_origins
    dx, dy = _sample_block_jitter(n, env.device)
    pos = base_pos.clone()
    pos[:, 0] = pos[:, 0] + dx
    pos[:, 1] = pos[:, 1] + dy
    quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).expand(n, 4).clone()
    root_pose = torch.cat((pos, quat), dim=-1)
    root_vel = torch.zeros((n, 6), device=env.device)
    obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    _reset_inactive_distractors_for_source_pose_mode(env, env_ids)


def _reset_all(env, env_ids=None):
    """Reset robot and object state for the requested environments."""
    env_ids = _resolve_env_ids(env, env_ids)
    base_mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)
    _reset_robot_to_default(env, env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)
    _reset_block_randomized(env, env_ids)


# --- Object factory (own copy, kinematic + gravity-disabled for stability) ---


def _topdown_robot_base_cfg() -> ArticulationCfg:
    """Build the fixed-base G1/Dex3 articulation config for the task scene."""
    cfg = copy.deepcopy(G129_CFG_WITH_DEX3_BASE_FIX)
    hands = cfg.actuators.get("hands")
    if hands is not None:
        hands.effort_limit_sim = _env_float(
            "TOPDOWN_FINGER_EFFORT_LIMIT_SIM",
            _env_float("DEX3_FINGER_EFFORT_LIMIT_SIM", 40.0),
        )
    if not _INCLUDE_WAIST_YAW:
        return cfg
    waist = cfg.actuators.get("waist")
    if waist is not None:
        waist.velocity_limit_sim = {
            "waist_yaw_joint": float(os.environ.get("WAIST_YAW_VELOCITY_LIMIT", "8.0")),
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        }
        waist.stiffness = {
            "waist_yaw_joint": float(os.environ.get("WAIST_YAW_STIFFNESS", "400.0")),
            "waist_roll_joint": 10000.0,
            "waist_pitch_joint": 10000.0,
        }
        waist.damping = {
            "waist_yaw_joint": float(os.environ.get("WAIST_YAW_DAMPING", "40.0")),
            "waist_roll_joint": 10000.0,
            "waist_pitch_joint": 10000.0,
        }
    return cfg


def _make_block_cfg(
    *,
    prim_path               : str                        = "/World/envs/env_.*/Object",
    init_pos                : tuple[float, float, float] = _BLOCK_INIT_POS,
    diffuse_color           : tuple[float, float, float] = (1.0, 0.0, 0.0),
    activate_contact_sensors: bool                       = True,
    dynamic_env_var         : str                        = "TOPDOWN_DYNAMIC_BLOCK",
) -> RigidObjectCfg:
    """Build a cuboid block rigid-object config for one source color."""
    # The active red block uses the profile-resolved value for its
    # kinematic/dynamic state; distractors (which pass a different env
    # var name) still read their own var. Distractor mass, friction,
    # and contact offsets still come from the profile, but distractors
    # are kinematic by default so this is benign.
    if dynamic_env_var == "TOPDOWN_DYNAMIC_BLOCK":
        dynamic_enabled = _PHYSICS_PROFILE.block_dynamic
    else:
        dynamic_enabled = os.environ.get(dynamic_env_var, "0") == "1"
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(init_pos),
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TOPDOWN_BLOCK_SIZE, _TOPDOWN_BLOCK_SIZE, _TOPDOWN_BLOCK_SIZE),
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Overnight-0426 experiment: TOPDOWN_DYNAMIC_BLOCK=1 enables a
                # dynamic block (gravity on, low mass) so contacts produce
                # block displacement signal that the policy can learn from.
                # Default kinematic (no movement under contact).
                # Note: TOPDOWN_PHYSICS_PROFILE=nvidia_mirror also activates
                # dynamic; TOPDOWN_DYNAMIC_BLOCK=0 forces kinematic even then.
                kinematic_enabled=not dynamic_enabled,
                disable_gravity=not dynamic_enabled,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=_PHYSICS_PROFILE.block_mass
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=_PHYSICS_PROFILE.contact_offset,
                rest_offset=_PHYSICS_PROFILE.rest_offset,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse_color,
                metallic=0.0,
                roughness=0.9,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=_PHYSICS_PROFILE.block_friction_combine_mode,
                restitution_combine_mode=_PHYSICS_PROFILE.block_restitution_combine_mode,
                static_friction=_PHYSICS_PROFILE.block_static_friction,
                dynamic_friction=_PHYSICS_PROFILE.block_dynamic_friction,
                restitution=0.01,
            ),
        ),
    )


def _apply_topdown_fingertip_physics_material(env, env_ids=None) -> None:
    """Bind high-friction physics material to distal Dex3 contact links."""
    del env_ids
    if not _env_bool("TOPDOWN_FINGERTIP_MATERIAL", True):
        return

    from isaaclab.sim.utils import bind_physics_material, get_current_stage

    stage = get_current_stage()
    material_path = "/World/topdown_fingertip_physics_material"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=_PHYSICS_PROFILE.fingertip_friction_combine_mode,
        restitution_combine_mode=_PHYSICS_PROFILE.fingertip_restitution_combine_mode,
        static_friction=_PHYSICS_PROFILE.fingertip_static_friction,
        dynamic_friction=_PHYSICS_PROFILE.fingertip_dynamic_friction,
        restitution=0.0,
    )
    if not stage.GetPrimAtPath(material_path).IsValid():
        material_cfg.func(material_path, material_cfg)

    link_suffixes = (
        "/Robot/right_hand_thumb_1_link",
        "/Robot/right_hand_thumb_2_link",
        "/Robot/right_hand_index_0_link",
        "/Robot/right_hand_index_1_link",
        "/Robot/right_hand_middle_0_link",
        "/Robot/right_hand_middle_1_link",
    )
    bind_count = 0
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not prim_path.endswith(link_suffixes):
            continue
        try:
            bind_physics_material(
                prim_path,
                material_path,
                stage=stage,
                stronger_than_descendants=True,
            )
        except ValueError:
            continue
        bind_count += 1
    env._topdown_fingertip_material_bind_count = bind_count


# Decorative distractor block positions (kept off the active grasp footprint
# so they neither occlude the IK target nor produce stray contact signals).
_BLUE_BLOCK_INIT_POS = (
    _env_float("TOPDOWN_BLUE_SOURCE_X", -4.42 if _WRAP_TABLE_ENABLED else -4.35),
    _env_float("TOPDOWN_BLUE_SOURCE_Y", -3.66 if _WRAP_TABLE_ENABLED else -4.0632),
    _BLOCK_INIT_Z,
)
_YELLOW_BLOCK_INIT_POS = (
    _env_float("TOPDOWN_YELLOW_SOURCE_X", -4.70 if _WRAP_TABLE_ENABLED else -4.55),
    _env_float("TOPDOWN_YELLOW_SOURCE_Y", -3.66 if _WRAP_TABLE_ENABLED else -4.0632),
    _BLOCK_INIT_Z,
)
_PYRAMID_SOURCE_POSITIONS = (
    _BLOCK_INIT_POS,
    _YELLOW_BLOCK_INIT_POS,
    _BLUE_BLOCK_INIT_POS,
)


def _validate_visible_source_clearance() -> None:
    """Validate that visible source blocks do not overlap at reset."""
    if not _KEEP_DISTRACTORS_VISIBLE:
        return
    if _SOURCE_POSE_MODE not in _PYRAMID_SOURCE_POSE_MODES and _SOURCE_POSE_MODE not in _FIXED_SOURCE_POSE_INDEX:
        return
    min_center_distance = _env_float(
        "TOPDOWN_SOURCE_MIN_CENTER_DISTANCE",
        _TOPDOWN_BLOCK_SIZE + 0.04,
    )
    labels = ("red", "yellow", "blue")
    positions = _PYRAMID_SOURCE_POSITIONS
    failures: list[str] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = float(positions[i][0]) - float(positions[j][0])
            dy = float(positions[i][1]) - float(positions[j][1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_center_distance:
                failures.append(f"{labels[i]}-{labels[j]}={dist:.3f}m")
    if failures:
        raise ValueError(
            "TOPDOWN_KEEP_DISTRACTORS_VISIBLE=1 source layout lacks pickup clearance: "
            + ", ".join(failures)
            + f" < TOPDOWN_SOURCE_MIN_CENTER_DISTANCE={min_center_distance:.3f}m"
        )


_validate_visible_source_clearance()


# --- Scene cfg ---------------------------------------------------------------


@configclass
class CurriculumSceneCfg(InteractiveSceneCfg):
    """Robot + table + kinematic block + 4 contact sensors. Minimal."""

    ground = AssetBaseCfg(
        prim_path="/World/envs/env_.*/LocalGround",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.005]),
        spawn=sim_utils.CuboidCfg(
            size=(12.0, 12.0, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.19, 0.20),
                metallic=0.0,
                roughness=1.0,
            ),
        ),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(_TABLE_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=_TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.72, 0.70),
                metallic=0.0,
                roughness=0.95,
            ),
        ),
    )

    right_arm_wrap_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/RightArmWrapTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(_WRAP_TABLE_SPAWN_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=_WRAP_TABLE_SPAWN_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.68),
                metallic=0.0,
                roughness=0.95,
            ),
        ),
    )

    object = _make_block_cfg(diffuse_color=_ACTIVE_BLOCK_COLOR)
    object_blue = _make_block_cfg(
        prim_path="/World/envs/env_.*/ObjectBlue",
        init_pos=_BLUE_BLOCK_INIT_POS,
        diffuse_color=(0.0, 0.0, 1.0),
        activate_contact_sensors=False,
        dynamic_env_var="TOPDOWN_DYNAMIC_DISTRACTORS",
    )
    object_yellow = _make_block_cfg(
        prim_path="/World/envs/env_.*/ObjectYellow",
        init_pos=_YELLOW_BLOCK_INIT_POS,
        diffuse_color=(1.0, 1.0, 0.0),
        activate_contact_sensors=False,
        dynamic_env_var="TOPDOWN_DYNAMIC_DISTRACTORS",
    )

    robot: ArticulationCfg = RobotBaseCfg.get_base_config(
        init_pos=_ROBOT_INIT_POS,
        init_rot=_ROBOT_INIT_ROT,
        include_waist=_INCLUDE_WAIST_YAW,
        hand_type="dex3",
        base_config=_topdown_robot_base_cfg(),
        custom_joint_pos=_CURRICULUM_CUSTOM_JOINT_POS,
    )

    thumb_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_thumb_2_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )
    thumb_mid_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_thumb_1_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )
    index_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_index_1_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )
    index_mid_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_index_0_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )
    middle_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_middle_1_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )
    middle_mid_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_middle_0_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )
    palm_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_palm_link",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=_OBJECT_FILTER_EXPR,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=2500.0),
    )


# --- Actions cfg --------------------------------------------------------------


@configclass
class ActionsCfg:
    """Full joint-position action interface; trainer masks down to 14 reduced DoFs."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


# --- Observations cfg --------------------------------------------------------


@configclass
class CurriculumObservationsCfg:
    """Actor obs plus critic-only privileged teacher state.

    The order of terms is load-bearing: ``finger_unlock_progress`` MUST be
    the last term so the trainer can slice its column at a fixed offset
    without depending on every upstream term's width.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor-visible observation group."""

        # Robot state (87 + 14 = 101 dims: body pos/vel/torque plus Dex3 positions)
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_dex3_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        # Geometry
        palm_pos_world = ObsTerm(func=mdp.get_palm_pos_world)
        palm_quat_world = ObsTerm(func=mdp.get_palm_quat_world)
        block_pos_world = ObsTerm(func=mdp.get_block_pos_world)
        block_quat_world = ObsTerm(func=mdp.get_block_quat_world)
        grip_target_pos_world = ObsTerm(func=mdp.get_grip_target_pos_world)
        block_minus_palm = ObsTerm(func=mdp.get_block_minus_palm)
        thumb_tip_pos_world = ObsTerm(func=mdp.get_thumb_tip_pos_world)
        index_tip_pos_world = ObsTerm(func=mdp.get_index_tip_pos_world)
        middle_tip_pos_world = ObsTerm(func=mdp.get_middle_tip_pos_world)
        thumb_minus_block = ObsTerm(func=mdp.get_thumb_minus_block)
        index_minus_block = ObsTerm(func=mdp.get_index_minus_block)
        middle_minus_block = ObsTerm(func=mdp.get_middle_minus_block)
        # Reward-shaping scalars
        palm_pose_scalars = ObsTerm(func=mdp.get_palm_pose_scalars)              # 4
        axis_orientation_scalars = ObsTerm(func=mdp.get_axis_orientation_scalars)  # 2
        alignment_scalars = ObsTerm(func=mdp.get_alignment_scalars)              # 3
        contact_strengths = ObsTerm(func=mdp.get_contact_strengths)              # 4
        lift_scalars = ObsTerm(func=mdp.get_lift_scalars)                        # 1
        tip_target_error_scalars = ObsTerm(func=mdp.get_tip_target_error_scalars)  # 6
        physical_block_scalars = ObsTerm(func=mdp.get_physical_block_scalars)      # 4
        source_block_one_hot = ObsTerm(func=mdp.get_source_block_one_hot)          # 3
        # Curriculum signals
        stage_one_hot = ObsTerm(func=mdp.stage_one_hot)                          # 3
        episode_step = ObsTerm(func=mdp.episode_step_fraction)                   # 1
        # MUST BE LAST: trainer reads this column offset for finger close/unlock mask
        finger_unlock_progress = ObsTerm(func=mdp.finger_unlock_progress)        # 1

        def __post_init__(self):
            """Validate derived dataclass invariants immediately after construction."""
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        # Teacher/controller state is critic-only. The actor must stay deployable.
        controller_state_scalars = ObsTerm(func=mdp.get_controller_state_scalars)  # 22
        teacher_contact_state_scalars = ObsTerm(
            func=mdp.get_teacher_contact_state_scalars
        )                                                                        # 34
        teacher_ik_state_scalars = ObsTerm(func=mdp.get_teacher_ik_state_scalars)  # 14
        arm_hold_action_scalars = ObsTerm(func=mdp.get_arm_hold_action_scalars)    # 8

        def __post_init__(self):
            """Validate derived dataclass invariants immediately after construction."""
            self.enable_corruption = False
            self.concatenate_terms = False

    policy    : PolicyCfg     = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


# --- Rewards cfg --------------------------------------------------------------


def _w(name: str, default: float) -> float:
    """Resolve reward weight from CURRICULUM_W_<NAME> env var, fallback to default.

    Lets overnight refinement runs sweep weights without code edits.
    Empty string treated as "use default".
    """
    raw = os.environ.get(f"CURRICULUM_W_{name.upper()}", "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@configclass
class CurriculumRewardsCfg:
    """Stage-routed reward terms.

    Per-stage primary weights mid (-1.0 to +50). Maintenance weights are
    50% (Stage 1) and 30% (Stage 2) of original primary weights so the
    actor pays a small cost for breaking earlier shells without overwhelming
    the current stage's gradient.
    """

    # Stage 0 (reach) primary
    reach_palm_distance = RewTerm(func=mdp.reach_palm_distance, weight=_w("reach_palm_distance", -1.0))
    reach_palm_height = RewTerm(func=mdp.reach_palm_height, weight=_w("reach_palm_height", -1.0))
    reach_palm_orientation = RewTerm(func=mdp.reach_palm_orientation, weight=_w("reach_palm_orientation", -1.0))
    reach_palm_yaw_axis = RewTerm(func=mdp.reach_palm_yaw_axis, weight=_w("reach_palm_yaw_axis", -1.0))
    reach_palm_spread_axis = RewTerm(func=mdp.reach_palm_spread_axis, weight=_w("reach_palm_spread_axis", -0.5))
    reach_alignment_error_quadratic = RewTerm(
        func=mdp.reach_alignment_error_quadratic,
        weight=_w("reach_alignment_error_quadratic", 0.0),
    )
    reach_fingertip_line_angle_quadratic = RewTerm(
        func=mdp.reach_fingertip_line_angle_quadratic,
        weight=_w("reach_fingertip_line_angle_quadratic", 0.0),
    )
    reach_shell_bonus = RewTerm(func=mdp.reach_shell_bonus, weight=_w("reach_shell_bonus", +5.0))

    # Stage 1 (alignment) maintenance for prior stage + primary
    align_palm_distance = RewTerm(func=mdp.align_palm_distance_maintenance, weight=_w("align_palm_distance", -0.5))
    align_palm_height = RewTerm(func=mdp.align_palm_height_maintenance, weight=_w("align_palm_height", -0.5))
    align_palm_orientation = RewTerm(func=mdp.align_palm_orientation_maintenance, weight=_w("align_palm_orientation", -1.5))
    align_palm_yaw_axis = RewTerm(func=mdp.align_palm_yaw_axis_maintenance, weight=_w("align_palm_yaw_axis", -0.5))
    align_alignment_error = RewTerm(func=mdp.align_alignment_error, weight=_w("align_alignment_error", -1.5))
    align_fingertip_line_angle = RewTerm(func=mdp.align_fingertip_line_angle, weight=_w("align_fingertip_line_angle", -1.0))
    align_alignment_error_quadratic = RewTerm(
        func=mdp.align_alignment_error_quadratic,
        weight=_w("align_alignment_error_quadratic", 0.0),
    )
    align_fingertip_line_angle_quadratic = RewTerm(
        func=mdp.align_fingertip_line_angle_quadratic,
        weight=_w("align_fingertip_line_angle_quadratic", 0.0),
    )
    align_opposite_face = RewTerm(func=mdp.align_opposite_face, weight=_w("align_opposite_face", +1.0))
    align_shell_bonus = RewTerm(func=mdp.align_shell_bonus, weight=_w("align_shell_bonus", +10.0))

    # Stage 2 (contact) maintenance for prior stages + primary
    contact_palm_distance = RewTerm(func=mdp.contact_palm_distance_maintenance, weight=_w("contact_palm_distance", -0.3))
    contact_palm_height = RewTerm(func=mdp.contact_palm_height_maintenance, weight=_w("contact_palm_height", 0.0))
    # Stage-2 IK is position-dominant; palm orientation/yaw penalties fight the
    # teacher redundancy and are better left to the stage gate/success shell.
    contact_palm_orientation = RewTerm(func=mdp.contact_palm_orientation_maintenance, weight=_w("contact_palm_orientation", 0.0))
    contact_palm_yaw_axis = RewTerm(func=mdp.contact_palm_yaw_axis_maintenance, weight=_w("contact_palm_yaw_axis", 0.0))
    contact_alignment_error = RewTerm(func=mdp.contact_alignment_error_maintenance, weight=_w("contact_alignment_error", -0.5))
    contact_alignment_error_quadratic = RewTerm(
        func=mdp.contact_alignment_error_quadratic,
        weight=_w("contact_alignment_error_quadratic", 0.0),
    )
    alignment_degradation = RewTerm(
        func=mdp.alignment_degradation_penalty,
        weight=_w("alignment_degradation", 0.0),
    )
    contact_fingertip_line_angle = RewTerm(func=mdp.contact_fingertip_line_angle_maintenance, weight=_w("contact_fingertip_line_angle", -0.3))
    contact_opposite_face = RewTerm(func=mdp.contact_opposite_face_maintenance, weight=_w("contact_opposite_face", +1.0))
    contact_target_distance = RewTerm(func=mdp.contact_target_distance, weight=_w("contact_target_distance", -4.0))
    contact_vertical_gap = RewTerm(func=mdp.contact_vertical_gap, weight=_w("contact_vertical_gap", -4.0))
    contact_thumb_contact = RewTerm(func=mdp.contact_thumb_contact_bonus, weight=_w("contact_thumb_contact", +0.75))
    contact_index_contact = RewTerm(func=mdp.contact_index_contact_bonus, weight=_w("contact_index_contact", +0.5))
    contact_opposed_contact = RewTerm(func=mdp.contact_opposed_bonus, weight=_w("contact_opposed_contact", +6.0))
    contact_preunlock_pocket = RewTerm(
        func=mdp.contact_preunlock_pocket_quality,
        weight=_w("contact_preunlock_pocket", 0.0),
    )
    contact_preunlock_no_contact = RewTerm(
        func=mdp.contact_preunlock_no_contact_penalty,
        weight=_w("contact_preunlock_no_contact", 0.0),
    )
    contact_bilateral_contact = RewTerm(
        func=mdp.contact_bilateral_contact_bonus,
        weight=_w("contact_bilateral_contact", 0.0),
    )
    contact_bilateral_imbalance = RewTerm(
        func=mdp.contact_bilateral_imbalance_penalty,
        weight=_w("contact_bilateral_imbalance", 0.0),
    )
    contact_lift_progress = RewTerm(func=mdp.contact_lift_progress, weight=_w("contact_lift_progress", +15.0))
    lift_height_progress = RewTerm(func=mdp.lift_height_progress, weight=_w("lift_height_progress", 0.0))
    lift_with_grip = RewTerm(func=mdp.lift_with_grip, weight=_w("lift_with_grip", 0.0))
    centered_lift_progress = RewTerm(
        func=mdp.centered_lift_progress,
        weight=_w("centered_lift_progress", 0.0),
    )
    vertical_lift_velocity_bonus = RewTerm(
        func=mdp.vertical_lift_velocity_bonus,
        weight=_w("vertical_lift_velocity_bonus", 0.0),
    )
    block_xy_velocity_penalty = RewTerm(
        func=mdp.block_xy_velocity_penalty,
        weight=_w("block_xy_velocity_penalty", 0.0),
    )
    block_angular_velocity_penalty = RewTerm(
        func=mdp.block_angular_velocity_penalty,
        weight=_w("block_angular_velocity_penalty", 0.0),
    )
    block_upright_lift_bonus = RewTerm(
        func=mdp.block_upright_lift_bonus,
        weight=_w("block_upright_lift_bonus", 0.0),
    )
    centered_upright_lift_bonus = RewTerm(
        func=mdp.centered_upright_lift_bonus,
        weight=_w("centered_upright_lift_bonus", 0.0),
    )
    lift_xy_drift_penalty = RewTerm(
        func=mdp.lift_xy_drift_penalty,
        weight=_w("lift_xy_drift_penalty", 0.0),
    )
    block_tilt_lift_penalty = RewTerm(
        func=mdp.block_tilt_lift_penalty,
        weight=_w("block_tilt_lift_penalty", 0.0),
    )
    uncentered_lift_penalty = RewTerm(
        func=mdp.uncentered_lift_penalty,
        weight=_w("uncentered_lift_penalty", 0.0),
    )
    block_off_table_bonus = RewTerm(func=mdp.block_off_table_bonus, weight=_w("block_off_table_bonus", 0.0))
    sustained_lift_grip_bonus = RewTerm(
        func=mdp.sustained_lift_grip_bonus,
        weight=_w("sustained_lift_grip_bonus", 0.0),
    )
    block_drop_penalty = RewTerm(func=mdp.block_drop_penalty, weight=_w("block_drop_penalty", 0.0))
    contact_deep_shell = RewTerm(func=mdp.contact_deep_shell_bonus, weight=_w("contact_deep_shell", +5.0))
    stage2_floor = RewTerm(func=mdp.stage2_floor_reward, weight=_w("stage2_floor", 0.0))
    contact_one_sided = RewTerm(func=mdp.contact_one_sided_penalty, weight=_w("contact_one_sided", -3.0))
    contact_one_sided_flip = RewTerm(
        func=mdp.contact_one_sided_flip_penalty,
        weight=_w("contact_one_sided_flip", 0.0),
    )
    contact_overforce = RewTerm(func=mdp.contact_overforce_penalty, weight=_w("contact_overforce", -2.0))
    contact_pose_ready_no_contact = RewTerm(
        func=mdp.contact_pose_ready_no_contact_penalty,
        weight=_w("pose_ready_no_contact", -3.0),
    )
    # Overnight-0426 E1: penalty in the SOFT pose neighborhood when no contact.
    # Mirror band of contact_smooth_success_pose so it fires at palm_c=0.085
    # (the boundary equilibrium that ``pose_ready_no_contact`` cannot reach
    # because that one is gated on the strict latch). Disabled by default;
    # set CURRICULUM_W_SMOOTH_POSE_NO_CONTACT=-12.0 to activate.
    contact_smooth_pose_no_contact = RewTerm(
        func=mdp.contact_smooth_pose_no_contact_penalty,
        weight=_w("smooth_pose_no_contact", 0.0),
    )
    contact_success_now_continuous = RewTerm(func=mdp.contact_success_now_continuous, weight=_w("contact_success_now_continuous", +10.0))
    # Legacy joint smooth-success: kept at 0.0 by default. The split pose/contact
    # variants below carry the gradient now. Set CURRICULUM_W_CONTACT_SMOOTH_SUCCESS
    # to bring it back if needed for ablation.
    contact_smooth_success = RewTerm(func=mdp.contact_smooth_success_continuous, weight=_w("contact_smooth_success", 0.0))
    # Split smooth-success: pose track has no contact requirement before the
    # contact-pose latch, so the gradient pulls the policy into the shell
    # without keeping a positive hover reward after latch.
    contact_smooth_success_pose = RewTerm(
        func=mdp.contact_smooth_success_pose_continuous,
        weight=_w("contact_smooth_success_pose", +8.0),
    )
    # Split smooth-success: contact track requires contact strength so contact
    # itself still pays. Total magnitude (8 + 12) preserves the original 20.
    contact_smooth_success_with_contact = RewTerm(
        func=mdp.contact_smooth_success_with_contact_continuous,
        weight=_w("contact_smooth_success_with_contact", +12.0),
    )
    contact_finger_center_x_error_quadratic = RewTerm(
        func=mdp.contact_finger_center_x_error_quadratic,
        weight=_w("contact_finger_center_x_error_quadratic", 0.0),
    )
    contact_finger_center_y_error_quadratic = RewTerm(
        func=mdp.contact_finger_center_y_error_quadratic,
        weight=_w("contact_finger_center_y_error_quadratic", 0.0),
    )
    contact_centered_contact = RewTerm(
        func=mdp.contact_centered_contact_continuous,
        weight=_w("contact_centered_contact", +2.0),
    )
    light_contact_success_bonus = RewTerm(func=mdp.light_contact_success_bonus, weight=_w("light_contact_success_bonus", +50.0))

    # Always-on smoothness
    step_cost = RewTerm(func=mdp.step_cost, weight=_w("step_cost", -0.05))
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=_w("action_rate_l2", -0.01))


# --- Terminations cfg --------------------------------------------------------


@configclass
class CurriculumTerminationsCfg:
    """Termination terms used by the topdown curriculum task."""
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.light_contact_success)
    block_drift = DoneTerm(func=mdp.block_drifted)
    alignment_timeout = DoneTerm(func=mdp.alignment_timeout_bad)


# --- Events cfg --------------------------------------------------------------


@configclass
class CurriculumEventCfg:
    """Reset-time event terms used by the topdown curriculum task."""
    apply_fingertip_physics_material = EventTermCfg(
        func=_apply_topdown_fingertip_physics_material,
        mode="startup",
    )

    reset_all = EventTermCfg(
        func=_reset_all,
        mode="reset",
    )


# --- Top-level env cfg -------------------------------------------------------


@configclass
class TopdownCurriculumEnvCfg(ManagerBasedRLEnvCfg):
    """Topdown reach-align-contact curriculum env cfg.

    Sets ``phase1_target_mode = "topdown_grip"`` so the trainer's IK teacher
    routing picks the topdown palm target without relying on hardcoded task
    string membership.
    """

    # Trainer-side detection key (used by _uses_topdown_grip_targets refactor)
    phase1_target_mode: str = "topdown_grip"

    scene: CurriculumSceneCfg = CurriculumSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: CurriculumObservationsCfg = CurriculumObservationsCfg()
    actions     : ActionsCfg                = ActionsCfg()
    terminations: CurriculumTerminationsCfg = CurriculumTerminationsCfg()
    events = CurriculumEventCfg()
    commands = None
    rewards: CurriculumRewardsCfg = CurriculumRewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Validate derived dataclass invariants immediately after construction."""
        self.decimation = 2
        self.episode_length_s = float(os.environ.get("TOPDOWN_EPISODE_LENGTH_S", "10.0"))
        self.sim.dt = _PHYSICS_PROFILE.physics_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = (
            _PHYSICS_PROFILE.friction_correlation_distance
        )
        self.sim.physx.enable_ccd = False
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = _PHYSICS_PROFILE.num_substeps
        self.sim.physx.contact_offset = _PHYSICS_PROFILE.contact_offset
        self.sim.physx.rest_offset = _PHYSICS_PROFILE.rest_offset
        self.sim.physx.num_position_iterations = _PHYSICS_PROFILE.num_position_iterations
        self.sim.physx.num_velocity_iterations = _PHYSICS_PROFILE.num_velocity_iterations
        # Single-line summary of resolved physics config — grep this
        # in run logs to confirm TOPDOWN_PHYSICS_PROFILE actually applied.
        print(_PHYSICS_PROFILE.as_log_line(), flush=True)
