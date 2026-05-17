"""Topdown hand geometry shared by the trainer and reward helpers.

This module is the geometry contract between three systems:

* the scripted IK teacher,
* the curriculum state/reward terms in this task package, and
* the profile/env-var layer that selects two-finger vs three-finger variants.

All functions work on batched IsaacLab tensors and should respect per-env
block/source selection.  A recurring failure mode during teacher debugging was
accidentally caching env-0 geometry and reusing it for every environment; new
helpers should either cache only canonical palm-local constants or recompute
world-frame targets from the active block every call.
"""

from __future__ import annotations

import os

import torch


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    """Return a unit-length vector with a small norm floor for stability."""
    return vector / torch.linalg.norm(vector, dim=-1, keepdim=True).clamp_min(1.0e-9)


def _grip_finger_model() -> str:
    """Return the active finger model used for palm/grip geometry."""
    model = os.environ.get("TOPDOWN_GRIP_FINGER_MODEL", "auto").strip().lower()
    if model in {"two", "2", "two_finger", "two-finger", "thumb_index", "thumb-index"}:
        return "two_finger"
    if model in {"three", "3", "three_finger", "three-finger", "thumb_index_middle"}:
        return "three_finger"
    if model not in {"", "auto"}:
        raise RuntimeError(f"unsupported TOPDOWN_GRIP_FINGER_MODEL={model!r}")

    middle_scale = float(os.environ.get("TOPDOWN_CONTACT_TEACHER_MIDDLE_SCALE", "0.0"))
    three_finger_centering = os.environ.get("CURRICULUM_THREE_FINGER_CENTERING", "0") == "1"
    if middle_scale <= 1.0e-6 or not three_finger_centering:
        return "two_finger"
    return "three_finger"


def _back_center_position(index_pos: torch.Tensor, middle_pos: torch.Tensor, model: str) -> torch.Tensor:
    """Return the active back-finger reference point for the configured grip model."""
    if model == "two_finger":
        return index_pos
    if model == "three_finger":
        return 0.5 * (index_pos + middle_pos)
    raise RuntimeError(f"unsupported topdown grip finger model={model!r}")


def _target_world_basis_for_yaw_axis(
    env,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, torch.Tensor]:
    """Return target basis columns for local x/y/z, with y as thumb-index axis.

    The world basis is deliberately per-env.  It may follow block-local +Y
    (the default front/back grip axis) or one of the fixed world axes for
    diagnostic runs.  Only the *local* hand basis is cacheable.
    """
    axis_mode = os.environ.get("TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS", "block_y").strip().lower()
    if axis_mode in {"", "block", "block_y", "block-y", "grip", "grip_axis", "grip-axis"}:
        _, block_quat = _active_block_pose(env)
        y_world = _block_axis_xy_from_quat(
            block_quat.to(device=device, dtype=dtype),
            axis_index=1,
            fallback_xy=(0.0, 1.0),
        )
        normalized_mode = "block_y"
    elif axis_mode in {"block_x", "block-x"}:
        _, block_quat = _active_block_pose(env)
        y_world = _block_axis_xy_from_quat(
            block_quat.to(device=device, dtype=dtype),
            axis_index=0,
            fallback_xy=(1.0, 0.0),
        )
        normalized_mode = "block_x"
    elif axis_mode in {"+y", "y", "world_y", "world-y"}:
        y_world = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype).view(1, 3)
        y_world = y_world.expand(env.num_envs, -1)
        normalized_mode = "world_y"
    elif axis_mode in {"-y", "neg_y", "negative_y"}:
        y_world = torch.tensor((0.0, -1.0, 0.0), device=device, dtype=dtype).view(1, 3)
        y_world = y_world.expand(env.num_envs, -1)
        normalized_mode = "world_-y"
    elif axis_mode in {"+x", "x", "world_x", "world-x"}:
        y_world = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype).view(1, 3)
        y_world = y_world.expand(env.num_envs, -1)
        normalized_mode = "world_x"
    elif axis_mode in {"-x", "neg_x", "negative_x"}:
        y_world = torch.tensor((-1.0, 0.0, 0.0), device=device, dtype=dtype).view(1, 3)
        y_world = y_world.expand(env.num_envs, -1)
        normalized_mode = "world_-x"
    else:
        raise RuntimeError(f"unsupported TOPDOWN_TARGET_PALM_YAW_WORLD_AXIS={axis_mode!r}")

    z_world = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).view(1, 3)
    z_world = z_world.expand_as(y_world)
    x_world = _normalize(torch.cross(y_world, z_world, dim=-1))
    z_world = _normalize(torch.cross(x_world, y_world, dim=-1))
    return normalized_mode, torch.stack((x_world, y_world, z_world), dim=-1)


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion tensors to rotation matrices."""

    squeeze = False
    if quat.ndim == 1:
        quat = quat.unsqueeze(0)
        squeeze = True
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-9)
    w, x, y, z = quat.unbind(dim=-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1)
    row1 = torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1)
    row2 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    matrix = torch.stack((row0, row1, row2), dim=-2)
    return matrix[0] if squeeze else matrix


def matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix tensors to normalized wxyz quaternions."""
    squeeze = False
    if matrix.shape == (3, 3):
        matrix = matrix.unsqueeze(0)
        squeeze = True
    if matrix.ndim < 3 or matrix.shape[-2:] != (3, 3):
        raise RuntimeError(f"matrix_to_quat_wxyz expects (..., 3, 3), got {tuple(matrix.shape)}")

    m = matrix
    m00 = m[..., 0, 0]
    m01 = m[..., 0, 1]
    m02 = m[..., 0, 2]
    m10 = m[..., 1, 0]
    m11 = m[..., 1, 1]
    m12 = m[..., 1, 2]
    m20 = m[..., 2, 0]
    m21 = m[..., 2, 1]
    m22 = m[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    qx = torch.where((m21 - m12) < 0.0, -qx, qx)
    qy = torch.where((m02 - m20) < 0.0, -qy, qy)
    qz = torch.where((m10 - m01) < 0.0, -qz, qz)
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-9)
    return quat[0] if squeeze else quat


def _active_source_pose_idx(env) -> torch.Tensor:
    """Return the per-env active source index used by multi-block topdown modes."""
    idx = getattr(env, "_topdown_source_pose_idx", None)
    if torch.is_tensor(idx) and idx.shape == (env.num_envs,):
        return idx.to(device=env.device, dtype=torch.long).clamp(0, 2)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)


def _active_block_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active block pose, respecting colored source-object selection."""
    if bool(getattr(env, "_topdown_use_visible_source_objects", False)):
        names = ("object", "object_yellow", "object_blue")
        pos = torch.stack([env.scene[name].data.root_pos_w[:, :3] for name in names], dim=1)
        quat = torch.stack([env.scene[name].data.root_quat_w for name in names], dim=1)
        source_idx = _active_source_pose_idx(env).view(env.num_envs, 1, 1)
        pos_idx = source_idx.expand(-1, 1, 3)
        quat_idx = source_idx.expand(-1, 1, 4)
        return pos.gather(1, pos_idx).squeeze(1), quat.gather(1, quat_idx).squeeze(1)
    active = str(getattr(env, "_topdown_active_object_name", "object"))
    obj = env.scene[active]
    return obj.data.root_pos_w[:, :3], obj.data.root_quat_w


def _block_axis_xy_from_quat(
    block_quat: torch.Tensor,
    *,
    axis_index: int,
    fallback_xy: tuple[float, float],
) -> torch.Tensor:
    """Project one block-local axis into world XY and normalize it per env."""
    rot = quat_wxyz_to_matrix(block_quat)
    axis = rot[..., :, axis_index].clone()
    axis[..., 2] = 0.0
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    fallback = torch.tensor(
        (float(fallback_xy[0]), float(fallback_xy[1]), 0.0),
        device=axis.device,
        dtype=axis.dtype,
    ).view(1, 3).expand_as(axis)
    normalized = axis / norm.clamp_min(1.0e-6)
    return torch.where((norm >= 1.0e-3).expand_as(axis), normalized, fallback)


def get_topdown_target_palm_quat(env) -> torch.Tensor:
    """Per-env palm-down target orientation for the topdown teacher.

    The canonical local hand basis is cached from the reset/open hand, but the
    world yaw basis is recomputed from the active block every call. This keeps
    multi-env and multi-source runs from reusing an env-0/global target
    quaternion when block yaw/source selection differs per environment.

    Basis mode matters:

    * ``drop_priority`` aligns the palm-to-grip-center axis with world down.
      It matches ``palm_drop_axis_error_rad`` in ``state_machine.py`` and is
      the safest default for the final centered-descent teacher.
    * ``finger_segment_plane`` and ``contact_plane`` are diagnostic modes for
      making visible finger planes horizontal.  They can be useful visually,
      but they do not necessarily match the success metric.
    * ``yaw_priority`` preserves thumb/index line yaw first, then chooses the
      nearest compatible drop axis.
    """

    finger_model = _grip_finger_model()
    basis_mode = os.environ.get("TOPDOWN_TARGET_PALM_BASIS", "drop_priority").strip().lower()
    cached_basis_local = getattr(env, "_topdown_cached_target_palm_basis_local", None)
    cached_model = getattr(env, "_topdown_cached_target_palm_basis_local_finger_model", None)
    cached_basis = getattr(env, "_topdown_cached_target_palm_basis_local_basis", None)
    yaw_axis_mode, target_basis_world = _target_world_basis_for_yaw_axis(
        env,
        device=env.device,
        dtype=torch.float32,
    )
    if (
        cached_basis_local is not None
        and cached_model == finger_model
        and cached_basis == basis_mode
        and cached_basis_local.shape == (3, 3)
    ):
        basis_local = cached_basis_local.to(device=env.device, dtype=target_basis_world.dtype)
    else:
        robot = env.scene["robot"]
        body_names = list(robot.data.body_names)
        palm_idx = body_names.index("right_hand_palm_link")
        thumb_base_idx = body_names.index("right_hand_thumb_1_link")
        thumb_idx = body_names.index("right_hand_thumb_2_link")
        index_base_idx = body_names.index("right_hand_index_0_link")
        index_idx = body_names.index("right_hand_index_1_link")
        middle_base_idx = body_names.index("right_hand_middle_0_link")
        middle_idx = body_names.index("right_hand_middle_1_link")

        palm_pose = robot.data.body_link_pose_w[0, palm_idx].detach()
        palm_pos = palm_pose[:3]
        palm_rot = quat_wxyz_to_matrix(palm_pose[3:7])
        thumb_base_pos = robot.data.body_link_pose_w[0, thumb_base_idx, :3].detach()
        thumb_pos = robot.data.body_link_pose_w[0, thumb_idx, :3].detach()
        index_base_pos = robot.data.body_link_pose_w[0, index_base_idx, :3].detach()
        index_pos = robot.data.body_link_pose_w[0, index_idx, :3].detach()
        middle_base_pos = robot.data.body_link_pose_w[0, middle_base_idx, :3].detach()
        middle_pos = robot.data.body_link_pose_w[0, middle_idx, :3].detach()

        thumb_base_local = palm_rot.transpose(0, 1) @ (thumb_base_pos - palm_pos)
        thumb_local = palm_rot.transpose(0, 1) @ (thumb_pos - palm_pos)
        index_base_local = palm_rot.transpose(0, 1) @ (index_base_pos - palm_pos)
        index_local = palm_rot.transpose(0, 1) @ (index_pos - palm_pos)
        middle_base_local = palm_rot.transpose(0, 1) @ (middle_base_pos - palm_pos)
        middle_local = palm_rot.transpose(0, 1) @ (middle_pos - palm_pos)
        back_center_local = _back_center_position(index_local, middle_local, finger_model)
        grasp_center_local = 0.5 * (thumb_local + back_center_local)

        yaw_local = thumb_local - back_center_local
        if basis_mode in {
            "finger_segment_plane",
            "finger-segment-plane",
            "segment_plane",
            "segment-plane",
        }:
            # Make the visible finger long-axis plane horizontal. Tip-origin
            # triangles can look numerically flat while the thumb pad still points
            # down; the distal segment vectors match the visual failure mode.
            thumb_axis = thumb_local - thumb_base_local
            index_axis = index_local - index_base_local
            middle_axis = middle_local - middle_base_local
            z_local = torch.cross(thumb_axis, index_axis, dim=0)
            if float(torch.linalg.norm(z_local).item()) < 1.0e-6:
                z_local = torch.cross(thumb_axis, middle_axis, dim=0)
            if float(torch.linalg.norm(z_local).item()) < 1.0e-6:
                z_local = torch.cross(yaw_local, -grasp_center_local, dim=0)
            if float(torch.linalg.norm(z_local).item()) < 1.0e-6:
                z_local = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=palm_pos.dtype)
            z_local = _normalize(z_local)
            if float(torch.dot(grasp_center_local, z_local).item()) > 0.0:
                z_local = -z_local
            y_local = yaw_local - torch.dot(yaw_local, z_local) * z_local
            if float(torch.linalg.norm(y_local).item()) < 1.0e-6:
                y_local = index_axis - torch.dot(index_axis, z_local) * z_local
            if float(torch.linalg.norm(y_local).item()) < 1.0e-6:
                y_local = torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=palm_pos.dtype)
                y_local = y_local - torch.dot(y_local, z_local) * z_local
            y_local = _normalize(y_local)
            x_local = _normalize(torch.cross(y_local, z_local, dim=0))
            z_local = _normalize(torch.cross(x_local, y_local, dim=0))
        elif basis_mode in {"contact_plane", "contact-plane", "thumb_plane", "thumb-plane"}:
            # Make the anatomical thumb/index/middle contact plane parallel to the
            # table. This is different from the active grip model: even in a
            # two-finger task, the middle tip at reset defines the hand's local
            # finger plane so the thumb is not driven straight down into the block.
            z_local = torch.cross(index_local - thumb_local, middle_local - thumb_local, dim=0)
            if float(torch.linalg.norm(z_local).item()) < 1.0e-6:
                z_local = torch.cross(yaw_local, -grasp_center_local, dim=0)
            if float(torch.linalg.norm(z_local).item()) < 1.0e-6:
                z_local = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=palm_pos.dtype)
            z_local = _normalize(z_local)
            if float(torch.dot(grasp_center_local, z_local).item()) > 0.0:
                z_local = -z_local
            y_local = yaw_local - torch.dot(yaw_local, z_local) * z_local
            if float(torch.linalg.norm(y_local).item()) < 1.0e-6:
                y_local = torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=palm_pos.dtype)
                y_local = y_local - torch.dot(y_local, z_local) * z_local
            y_local = _normalize(y_local)
            x_local = _normalize(torch.cross(y_local, z_local, dim=0))
            z_local = _normalize(torch.cross(x_local, y_local, dim=0))
        elif basis_mode in {"yaw_priority", "finger_plane", "finger-plane"}:
            # Prioritize the thumb-to-back-finger line being horizontal. The
            # previous basis made the palm-to-grip vector exactly vertical first,
            # but any non-orthogonality in the real hand geometry left a vertical
            # component in the thumb/back-finger line, producing thumb-down entry.
            y_local = _normalize(yaw_local)
            z_hint = -grasp_center_local
            z_local = z_hint - torch.dot(z_hint, y_local) * y_local
            if float(torch.linalg.norm(z_local).item()) < 1.0e-6:
                z_local = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=palm_pos.dtype)
                z_local = z_local - torch.dot(z_local, y_local) * y_local
            z_local = _normalize(z_local)
            x_local = _normalize(torch.cross(y_local, z_local, dim=0))
            z_local = _normalize(torch.cross(x_local, y_local, dim=0))
        else:
            z_local = _normalize(-grasp_center_local)
            y_local = yaw_local - torch.dot(yaw_local, z_local) * z_local
            if float(torch.linalg.norm(y_local).item()) < 1.0e-6:
                y_local = torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=palm_pos.dtype)
                y_local = y_local - torch.dot(y_local, z_local) * z_local
            y_local = _normalize(y_local)
            x_local = _normalize(torch.cross(y_local, z_local, dim=0))
            z_local = _normalize(torch.cross(x_local, y_local, dim=0))
        basis_local = torch.stack((x_local, y_local, z_local), dim=-1).detach()
        setattr(env, "_topdown_cached_target_palm_basis_local", basis_local)
        setattr(env, "_topdown_cached_target_palm_basis_local_finger_model", finger_model)
        setattr(env, "_topdown_cached_target_palm_basis_local_basis", basis_mode)

    target_basis_world = target_basis_world.to(device=env.device, dtype=basis_local.dtype)
    target_rot = torch.matmul(
        target_basis_world,
        basis_local.transpose(0, 1).view(1, 3, 3).expand(env.num_envs, -1, -1),
    )
    target_quat = matrix_to_quat_wxyz(target_rot).detach()
    setattr(env, "_topdown_cached_target_palm_quat", target_quat)
    setattr(env, "_topdown_cached_target_palm_quat_finger_model", finger_model)
    setattr(env, "_topdown_cached_target_palm_quat_basis", basis_mode)
    setattr(env, "_topdown_cached_target_palm_quat_yaw_axis", yaw_axis_mode)
    return target_quat


def _palm_local_grip_offset(env) -> torch.Tensor:
    """Return cached canonical palm-frame offset from palm link to open-hand grip center."""
    finger_model = _grip_finger_model()
    cached = getattr(env, "_topdown_cached_palm_local_grip_offset", None)
    cached_model = getattr(env, "_topdown_cached_palm_local_grip_offset_finger_model", None)
    if cached is not None and cached_model == finger_model:
        return cached

    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)
    palm_idx = body_names.index("right_hand_palm_link")
    thumb_idx = body_names.index("right_hand_thumb_2_link")
    index_idx = body_names.index("right_hand_index_1_link")
    middle_idx = body_names.index("right_hand_middle_1_link")

    palm_pose = robot.data.body_link_pose_w[0, palm_idx]
    palm_pos = palm_pose[:3]
    palm_rot = quat_wxyz_to_matrix(palm_pose[3:7])
    thumb_pos = robot.data.body_link_pose_w[0, thumb_idx, :3]
    index_pos = robot.data.body_link_pose_w[0, index_idx, :3]
    middle_pos = robot.data.body_link_pose_w[0, middle_idx, :3]
    thumb_local = palm_rot.transpose(0, 1) @ (thumb_pos - palm_pos)
    index_local = palm_rot.transpose(0, 1) @ (index_pos - palm_pos)
    middle_local = palm_rot.transpose(0, 1) @ (middle_pos - palm_pos)
    back_center_local = _back_center_position(index_local, middle_local, finger_model)
    grip_local = (0.5 * (thumb_local + back_center_local)).detach().clone()
    setattr(env, "_topdown_cached_palm_local_grip_offset", grip_local)
    setattr(env, "_topdown_cached_palm_local_grip_offset_finger_model", finger_model)
    return grip_local


def _live_palm_local_grip_offset(env) -> torch.Tensor:
    """Return per-env palm-frame grip offsets from the current finger geometry."""
    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)
    palm_idx = body_names.index("right_hand_palm_link")
    thumb_idx = body_names.index("right_hand_thumb_2_link")
    index_idx = body_names.index("right_hand_index_1_link")
    middle_idx = body_names.index("right_hand_middle_1_link")

    palm_pose = robot.data.body_link_pose_w[:, palm_idx]
    palm_pos = palm_pose[:, :3]
    palm_rot = quat_wxyz_to_matrix(palm_pose[:, 3:7])
    thumb_pos = robot.data.body_link_pose_w[:, thumb_idx, :3]
    index_pos = robot.data.body_link_pose_w[:, index_idx, :3]
    middle_pos = robot.data.body_link_pose_w[:, middle_idx, :3]
    back_center_pos = _back_center_position(index_pos, middle_pos, _grip_finger_model())
    grip_pos = 0.5 * (thumb_pos + back_center_pos)
    grip_local = torch.matmul(
        palm_rot.transpose(1, 2),
        (grip_pos - palm_pos).unsqueeze(-1),
    ).squeeze(-1)
    grip_local = grip_local.detach().clone()
    setattr(env, "_topdown_live_palm_local_grip_offset", grip_local)
    return grip_local


def _palm_local_grip_offset_mode() -> str:
    """Return how target palm/grip offsets should be selected."""
    mode = os.environ.get("TOPDOWN_PALM_LOCAL_GRIP_OFFSET_MODE", "").strip().lower()
    if mode:
        return mode
    position_mode = os.environ.get("TOPDOWN_TARGET_PALM_POSITION_MODE", "").strip().lower()
    if position_mode in {
        "live_local",
        "live-local",
        "current_local",
        "current-local",
        "closure_aware_local",
        "closure-aware-local",
    }:
        return "closure_blend_live_local"
    return "canonical_open"


def palm_local_grip_offset_for_target(env, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return an (N, 3) palm-local grip offset for target construction.

    ``canonical_open`` keeps the historical cached open-hand offset.  This is
    the mode used by the v32 6cm liftable teacher because it creates a stable
    pre-descent waypoint independent of transient finger curl. ``live_local``
    recomputes the thumb/back-finger grasp center from the current hand state,
    which can keep contact/lift targets aligned as the fingers curl, but can
    also move the palm target while descent is active.  ``closure_blend`` is
    therefore gated by descent/fraction knobs so it does not pull the hand down
    early.
    """
    mode = _palm_local_grip_offset_mode()
    canonical = _palm_local_grip_offset(env).view(1, 3).expand(env.num_envs, -1).detach().clone()
    if mode in {"live", "live_local", "current", "current_local"}:
        grip_local = _live_palm_local_grip_offset(env)
        normalized_mode = "live_local"
    elif mode in {"closure_blend", "closure_blend_live_local", "closure_aware", "closure_aware_local"}:
        live = _live_palm_local_grip_offset(env)
        closure = getattr(env, "_topdown_contact_teacher_closure_fraction", None)
        if not torch.is_tensor(closure) or closure.shape != (env.num_envs,):
            closure = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        descent = getattr(env, "_topdown_contact_teacher_descent_z", None)
        if os.environ.get("TOPDOWN_PALM_LOCAL_GRIP_OFFSET_BLEND_REQUIRES_DESCENT", "1") == "1":
            if torch.is_tensor(descent) and descent.shape == (env.num_envs,):
                contact_gate = (descent.to(device=env.device) > 1.0e-6).to(dtype=torch.float32)
            else:
                contact_gate = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        else:
            contact_gate = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
        start = max(
            float(os.environ.get("TOPDOWN_PALM_LOCAL_GRIP_OFFSET_LIVE_START_FRACTION", "0.50")),
            0.0,
        )
        full = max(
            float(os.environ.get("TOPDOWN_PALM_LOCAL_GRIP_OFFSET_LIVE_FULL_FRACTION", "0.80")),
            start,
        )
        if full > start + 1.0e-6:
            blend = torch.clamp((closure.to(device=env.device) - start) / (full - start), 0.0, 1.0)
        else:
            blend = (closure.to(device=env.device) >= start).to(dtype=torch.float32)
        blend = blend * contact_gate
        grip_local = canonical + (live - canonical) * blend.unsqueeze(-1)
        normalized_mode = "closure_blend_live_local"
        setattr(env, "_topdown_palm_local_grip_offset_live_blend", blend.detach().clone())
    elif mode in {"canonical", "canonical_open", "open", "cached", "cached_open"}:
        grip_local = canonical
        normalized_mode = "canonical_open"
    else:
        raise RuntimeError(f"unsupported TOPDOWN_PALM_LOCAL_GRIP_OFFSET_MODE={mode!r}")

    if dtype is not None:
        grip_local = grip_local.to(device=env.device, dtype=dtype)
    setattr(env, "_topdown_selected_palm_local_grip_offset", grip_local.detach().clone())
    setattr(env, "_topdown_palm_local_grip_offset_mode", normalized_mode)
    return grip_local


def topdown_palm_position_from_grip_target(env, target_grip_pos: torch.Tensor) -> torch.Tensor:
    """Palm position whose open-hand grip center reaches ``target_grip_pos``.

    This is the inverse of "where is the grasp center relative to the palm?".
    Most teacher profiles drive a block-centered grip target, then convert it
    into a palm-link target using the selected palm orientation and local grip
    offset.  Keeping that conversion here avoids duplicating hand-geometry
    assumptions in reward code and trainer code.
    """

    position_mode = os.environ.get("TOPDOWN_TARGET_PALM_POSITION_MODE", "").strip().lower()
    if position_mode in {"current", "current_grip_offset", "current-grip-offset", "live_grip_offset"}:
        robot = env.scene["robot"]
        body_names = list(robot.data.body_names)
        palm_idx = body_names.index("right_hand_palm_link")
        thumb_idx = body_names.index("right_hand_thumb_2_link")
        index_idx = body_names.index("right_hand_index_1_link")
        middle_idx = body_names.index("right_hand_middle_1_link")
        palm_pos = robot.data.body_link_pose_w[:, palm_idx, :3].to(
            device=env.device,
            dtype=target_grip_pos.dtype,
        )
        thumb_pos = robot.data.body_link_pose_w[:, thumb_idx, :3].to(
            device=env.device,
            dtype=target_grip_pos.dtype,
        )
        index_pos = robot.data.body_link_pose_w[:, index_idx, :3].to(
            device=env.device,
            dtype=target_grip_pos.dtype,
        )
        middle_pos = robot.data.body_link_pose_w[:, middle_idx, :3].to(
            device=env.device,
            dtype=target_grip_pos.dtype,
        )
        finger_model = _grip_finger_model()
        back_center_pos = _back_center_position(index_pos, middle_pos, finger_model)
        grip_pos = 0.5 * (thumb_pos + back_center_pos)
        return target_grip_pos - (grip_pos - palm_pos)

    target_quat = get_topdown_target_palm_quat(env).to(
        device=env.device,
        dtype=target_grip_pos.dtype,
    )
    target_rot = quat_wxyz_to_matrix(target_quat)
    grip_local = palm_local_grip_offset_for_target(env, dtype=target_grip_pos.dtype)
    grip_offset_w = torch.matmul(
        target_rot,
        grip_local.unsqueeze(-1),
    ).squeeze(-1)
    return target_grip_pos - grip_offset_w
