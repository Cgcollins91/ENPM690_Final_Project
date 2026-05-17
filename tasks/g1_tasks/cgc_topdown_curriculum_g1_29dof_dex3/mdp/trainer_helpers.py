"""
Trainer-side topdown diagnostics and compatibility metrics

These functions are used by the topdown curriculum trainer to compute diagnostics and metrics for training and evaluation

"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from . import state_machine as _sm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_float(name: str, default: float) -> float:
    """Read a float override used by trainer diagnostics"""
    raw = os.environ.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


CONTACT_FORCE_THRESHOLD = 0.5
CONTACT_FORCE_SATURATION = 4.0
CONTACT_SUCCESS_STRENGTH = 0.35
HAND_CONTACT_THRESHOLD = CONTACT_SUCCESS_STRENGTH
LIGHT_CONTACT_SUCCESS_CONTACT_THRESHOLD = _env_float("LIGHT_CONTACT_SUCCESS_CONTACT_THRESHOLD", 0.08)
GRASP_SUCCESS_BLOCK_DISP_MAX = 0.03
GRASP_SUCCESS_OPPOSED_FACE_THRESHOLD = 0.70
GRASP_SUCCESS_PALM_READY_TOLERANCE = _env_float("GRASP_SUCCESS_PALM_READY_TOLERANCE", 0.06)
GRASP_SUCCESS_PALM_READY_OUTER_DISTANCE = 0.12
GRASP_SUCCESS_PALM_HEIGHT_TOLERANCE = 0.05
GRASP_SUCCESS_PALM_HEIGHT_INNER_DISTANCE = 0.01
GRASP_SUCCESS_PALM_HEIGHT_OUTER_DISTANCE = 0.08
GRASP_SUCCESS_PALM_ORIENT_ERR_RAD = math.radians(_env_float("GRASP_SUCCESS_PALM_ORIENT_DEG", 22.0))
OPEN_HAND_ALIGN_FACE_DISTANCE_TOLERANCE = _env_float("OPEN_HAND_ALIGN_FACE_DISTANCE_TOLERANCE", 0.125)
PREGRASP_TIGHT_TOLERANCE = 0.07
CURL_SUCCESS_THRESHOLD = 2.20
FINGER_CURL_CLOSURE_SCALE = 6.00

_FINGER_CLOSE_DIRECTION = {
    "right_hand_thumb_0_joint": +1.0,
    "right_hand_thumb_1_joint": -1.0,
    "right_hand_thumb_2_joint": -1.0,
    "right_hand_index_0_joint": +1.0,
    "right_hand_index_1_joint": +1.0,
    "right_hand_middle_0_joint": +1.0,
    "right_hand_middle_1_joint": +1.0,
}
_PINCH_CLOSE_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
_HAND_CONTACT_SENSOR_NAMES = (
    "thumb_contact",
    "thumb_mid_contact",
    "index_contact",
    "index_mid_contact",
    "middle_contact",
    "middle_mid_contact",
    "palm_contact",
)


def _scene_entity(env: "ManagerBasedRLEnv", name: str):
    """Return a scene entity by name from either scene mapping surface"""
    try:
        return env.scene[name]
    except KeyError:
        return env.scene.sensors[name]


def _contact_force_magnitude(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    """Return raw contact-force magnitude for one topdown contact sensor"""
    sensor = _scene_entity(env, sensor_name)
    force_matrix = getattr(sensor.data, "force_matrix_w", None)
    if force_matrix is not None:
        forces = _sm._active_filter_forces(env, force_matrix)
    else:
        net_forces = getattr(sensor.data, "net_forces_w", None)
        if net_forces is None:
            return torch.zeros(env.num_envs, device=env.device)
        forces = net_forces[:, 0, :] if net_forces.dim() == 3 else net_forces
    return torch.linalg.norm(forces, dim=-1)


def _effective_contact_force_magnitude(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    """Return strongest raw contact magnitude for the active topdown sensor"""
    candidates = [sensor_name]
    if sensor_name.endswith("_contact"):
        try:
            candidates.insert(0, _sm._active_contact_sensor_name(env, sensor_name))
        except Exception:
            pass

    force_mag = torch.zeros(env.num_envs, device=env.device)
    for candidate in dict.fromkeys(candidates):
        try:
            force_mag = torch.maximum(force_mag, _contact_force_magnitude(env, candidate))
        except KeyError:
            continue
    return force_mag


def total_hand_contact_force(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return summed raw contact force across topdown hand sensors"""
    total = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in _HAND_CONTACT_SENSOR_NAMES:
        total = total + _effective_contact_force_magnitude(env, sensor_name)
    return total


def _joint_cache_name(joint_names_to_sum: tuple[str, ...]) -> str:
    """Return env attribute name for cached curl joint tensors"""
    return "_topdown_trainer_curl_joint_cache::" + "|".join(joint_names_to_sum)


def _signed_finger_curl(env: "ManagerBasedRLEnv", joint_names_to_sum: tuple[str, ...]) -> torch.Tensor:
    """Return signed Dex3 curl over selected joints"""
    robot = env.scene["robot"]
    cache_name = _joint_cache_name(joint_names_to_sum)
    cached = getattr(env, cache_name, None)
    if cached is None:
        joint_names = robot.data.joint_names
        device = robot.data.joint_pos.device
        indices = torch.tensor(
            [joint_names.index(name) for name in joint_names_to_sum],
            device=device,
            dtype=torch.long,
        )
        directions = torch.tensor(
            [_FINGER_CLOSE_DIRECTION[name] for name in joint_names_to_sum],
            device=device,
            dtype=robot.data.joint_pos.dtype,
        )
        cached = (indices, directions)
        setattr(env, cache_name, cached)
    indices, directions = cached
    indices = indices.to(device=robot.data.joint_pos.device)
    directions = directions.to(device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype)
    return (robot.data.joint_pos[:, indices] * directions).sum(dim=1)


def finger_curl(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return all-finger signed curl clipped to trainer scale"""
    curl = _signed_finger_curl(env, tuple(_FINGER_CLOSE_DIRECTION.keys()))
    return torch.clamp(curl, min=0.0, max=FINGER_CURL_CLOSURE_SCALE)


def pinch_curl(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return thumb-index signed curl clipped to contact-success scale"""
    curl = _signed_finger_curl(env, _PINCH_CLOSE_JOINTS)
    return torch.clamp(curl, min=0.0, max=CURL_SUCCESS_THRESHOLD)


def block_displacement(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return active block 3D displacement from episode spawn"""
    return _sm.block_displacement(env)


def block_lift_height(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return active block lift height from episode spawn"""
    return _sm.block_lift_height(env)


def phase1_palm_distance(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown grip-center distance to the reach target"""
    return _sm.palm_distance(env)


def phase1_palm_height_error(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown grip-center height error to the reach target"""
    return _sm.palm_height_error(env)


def phase1_palm_orientation_error_rad(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown drop-axis orientation error"""
    return _sm.palm_drop_axis_error_rad(env)


def pregrasp_target_positions(env: "ManagerBasedRLEnv", *_, **__) -> tuple[torch.Tensor, torch.Tensor]:
    """Return topdown thumb and index face targets"""
    return _sm._face_targets(env)


def open_hand_face_distances(env: "ManagerBasedRLEnv", *_, **__) -> tuple[torch.Tensor, torch.Tensor]:
    """Return thumb and index distances to topdown face targets"""
    thumb_target, index_target = _sm._face_targets(env)
    thumb = _sm._link_pos(env, _sm._THUMB_LINK)
    index = _sm._link_pos(env, _sm._INDEX_LINK)
    return torch.linalg.norm(thumb - thumb_target, dim=-1), torch.linalg.norm(index - index_target, dim=-1)


def thumb_distance(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown thumb distance to its face target"""
    thumb_dist, _ = open_hand_face_distances(env)
    return thumb_dist


def index_distance(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown index distance to its face target"""
    _, index_dist = open_hand_face_distances(env)
    return index_dist


def max_tip_distance(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return max thumb/index distance to topdown face targets"""
    thumb_dist, index_dist = open_hand_face_distances(env)
    return torch.maximum(thumb_dist, index_dist)


def open_hand_alignment_error(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown open-hand alignment error"""
    return _sm.open_hand_alignment_error(env)


def open_hand_fingertip_line_elevation_angle(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return topdown fingertip line elevation angle"""
    return _sm.fingertip_line_angle_rad(env)


def contact_phase1_palm_unlock_gate(env: "ManagerBasedRLEnv", *_, **__) -> torch.Tensor:
    """Return smooth palm shell gate used by trainer finger masks"""
    palm_dist = phase1_palm_distance(env)
    palm_span = max(GRASP_SUCCESS_PALM_READY_OUTER_DISTANCE - GRASP_SUCCESS_PALM_READY_TOLERANCE, 1.0e-6)
    palm_progress = (GRASP_SUCCESS_PALM_READY_OUTER_DISTANCE - palm_dist) / palm_span

    height_err = phase1_palm_height_error(env)
    height_span = max(GRASP_SUCCESS_PALM_HEIGHT_OUTER_DISTANCE - GRASP_SUCCESS_PALM_HEIGHT_INNER_DISTANCE, 1.0e-6)
    height_progress = (GRASP_SUCCESS_PALM_HEIGHT_OUTER_DISTANCE - height_err) / height_span

    orient_rad = phase1_palm_orientation_error_rad(env)
    orient_outer = math.radians(35.0)
    orient_span = max(orient_outer - GRASP_SUCCESS_PALM_ORIENT_ERR_RAD, 1.0e-6)
    orient_progress = (orient_outer - orient_rad) / orient_span

    return (
        torch.clamp(palm_progress, 0.0, 1.0)
        * torch.clamp(height_progress, 0.0, 1.0)
        * torch.clamp(orient_progress, 0.0, 1.0)
    )
