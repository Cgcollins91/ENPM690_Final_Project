"""Curriculum stage state machine for the topdown reach-align-contact task.

This module owns the per-env stage state and the geometric primitives the
reward / observation / termination terms need. It is intentionally
self-contained — no imports from other task packages — so the new task is
isolated from the legacy reward stacks.

Stages:
- 0 = reach (palm approach)
- 1 = open-hand alignment (thumb/index opposed-face geometry)
- 2 = contact-pose shell then light contact (descend, curl fingers, first contact)

Advancement is gated on shell predicates held for HOLD steps. Stage 1 also
requires no premature contact. Stage 2 can fall back to Stage 1 after a
sustained loss of the loose contact-pose region; this prevents "zombie"
Stage-2/pose-ready samples from training contact rewards far from the block.

State exposed on env (after ``ensure_curriculum_stage_updated``):
- ``_topdown_stage``            : int tensor (num_envs,)
- ``_topdown_reach_hold``       : int tensor (num_envs,)
- ``_topdown_align_hold``       : int tensor (num_envs,)
- ``_topdown_stage2_age``       : int tensor (num_envs,)
- ``_topdown_stage2_fallout_hold``: int tensor (num_envs,)
- ``_topdown_contact_pose_hold``  : int tensor (num_envs,)
- ``_topdown_contact_pose_ready`` : bool tensor (num_envs,)
- ``_topdown_contact_pose_age``   : int tensor (num_envs,)
- ``_topdown_finger_unlock_progress``: float tensor in [0, 1]. In
  ``TOPDOWN_FINGER_CLOSE_GATE_MODE=xyz`` / ``xyz_front`` modes, this is geometry-only progress toward the fingertip-xy block-center gate
- ``reach_align_finger_unlocked``: bool tensor (num_envs,) — convenience flag


Tunables (env-var overridable for easy tuning):
- ``CURRICULUM_REACH_HOLD_STEPS``: int steps required to hold reach shell for Stage 1 advancement
- ``CURRICULUM_ALIGN_HOLD_STEPS``: int steps required to hold align shell for Stage 2 advancement

Stage 1 Tunables (Reach) Shell:
- ``CURRICULUM_REACH_PALM_DIST_MAX``: float max palm-block distance for reach shell
- ``CURRICULUM_REACH_PALM_HEIGHT_MAX``: float max palm height above block for reach shell
- ``CURRICULUM_REACH_PALM_ORIENT_MAX_DEG``: float max palm-block orientation angle for reach shell
- ``CURRICULUM_REACH_PALM_YAW_MAX_DEG``: float max palm yaw angle for reach shell
- ``CURRICULUM_STAGE1_PALM_DIST_MAX``: float max palm-block distance for Stage 1 shell (can be different from reach shell)
- ``CURRICULUM_STAGE1_PALM_HEIGHT_MAX``: float max palm height above block for Stage 1 shell
- ``CURRICULUM_STAGE1_PALM_ORIENT_MAX_DEG``: float max palm-block orientation angle for Stage 1 shell
- ``CURRICULUM_STAGE1_PALM_YAW_MAX_DEG``: float max palm yaw angle for Stage 1 shell
- ``CURRICULUM_STAGE1_ALIGN_ERR_MAX``: float max alignment error for Stage 1 shell
- ``CURRICULUM_STAGE1_LINE_ANGLE_MAX_DEG``: float max angle between fingertip-block vector and block top normal for Stage 1 shell
- ``CURRICULUM_STAGE1_OPPOSED_GATE_MIN``: float minimum opposedness for Stage 1 shell
- ``CURRICULUM_STAGE1_NO_CONTACT_MAX``: float max non-opposed fingertip contact for Stage 1 shell

Stage 2 Tunables (Align + Pre-Contact):
- ``CURRICULUM_STAGE2_PALM_DIST_MAX``: float max palm-block distance for Stage 2 shell (can be different from reach shell)
- ``CURRICULUM_STAGE2_PALM_HEIGHT_MAX``: float max palm height above block for Stage 2 shell
- ``CURRICULUM_STAGE2_PALM_ORIENT_MAX_DEG``: float max palm-block orientation angle for Stage 2 shell
- ``CURRICULUM_STAGE2_PALM_YAW_MAX_DEG``: float max palm yaw angle for Stage 2 shell
- ``CURRICULUM_STAGE2_ALIGN_ERR_MAX``: float max alignment error for Stage 2 shell
- ``CURRICULUM_STAGE2_LINE_ANGLE_MAX_DEG``: float max angle between fingertip-block vector and block top normal for Stage 2 shell
- ``CURRICULUM_STAGE2_OPPOSED_GATE_MIN``: float minimum opposedness for Stage 2 shell

- ``CURRICULUM_ALIGN_ERR_MAX``: float max alignment error for align shell
- ``CURRICULUM_ALIGN_LINE_ANGLE_MAX_DEG``: float max angle between fingertip-block vector and block top normal for align shell
- ``CURRICULUM_ALIGN_OPPOSED_GATE_MIN``: float minimum opposedness for align shell (0-1, where 1 means perfectly opposed)
- ``CURRICULUM_ALIGN_NO_CONTACT_MAX``: float max non-opposed fingertip contact for align shell
- ``CURRICULUM_STAGE2_NO_CONTACT_MAX``: float max non-opposed fingertip contact for Stage 2 shell

- ``CURRICULUM_CONTACT_POSE_HOLD_STEPS``: int steps required to hold the contact pose for it to count toward contact success
- ``CURRICULUM_CONTACT_POSE_READY_FALLBACK_STEPS``: int steps of being outside the contact pose after achieving it before falling back to Stage 1
- ``CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_DIST_MAX``: float max palm-block distance for fallback to Stage 1 (only relevant if fallback steps > 0)
- ``CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_HEIGHT_MAX``: float max palm height above block for fallback to Stage 1 (only relevant if fallback steps > 0)
- ``CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_ORIENT_MAX_DEG``: float max palm-block orientation angle for fallback to Stage 1 (only relevant if fallback steps > 0)

Contact Stage Tunables:
- ``CURRICULUM_FINGER_UNLOCK_RAMP_STEPS``: int ramp steps for finger unlock progress (after which it saturates at max)
- ``CURRICULUM_FINGER_UNLOCK_MAX_PROGRESS``: float in (0, 1] max geometry-only finger unlock progress
- ``CURRICULUM_FINGER_UNLOCK_REQUIRES_CENTER``: bool whether finger unlock progress requires the fingertips to be near the block center (not just the face targets)
- ``CURRICULUM_FINGER_CENTER_LATCH``: bool whether to latch the "centered contact" state once achieved, or allow it to turn on/off based on current geometry
- ``CURRICULUM_FINGER_CENTER_HOLD_STEPS``: int steps required to hold centered contact for it to count toward finger unlock progress
- ``CURRICULUM_FINGER_CENTER_TIP_XY_MAX``: float max fingertip-xy distance from block center for "centered contact" state
- ``CURRICULUM_FINGER_CENTER_MAX_TIP_XY_MAX``: float max fingertip-xy distance from block center for any finger unlock progress (including non-centered)
- ``CURRICULUM_FINGER_CENTER_TIP_Z_MAX``: float max fingertip-z distance from block top for "centered contact" state
- ``CURRICULUM_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG``: float max angle between fingertip-block vector and block top normal for "centered contact" state
- ``CURRICULUM_FINGER_CENTER_ALIGN_ERR_MAX``: float max fingertip-block alignment error for "centered contact" state
- ``CURRICULUM_FINGER_CENTER_USE_XYZ_GATE``: bool whether to use an XYZ gate for finger unlock progress instead of just a radius-based gate
- ``CURRICULUM_FINGER_CENTER_XYZ_GATE_MIN``: float minimum fingertip-xyz distance from block center for zero finger unlock progress (only relevant if using XYZ gate)
- ``CURRICULUM_STAGE2_FINGER_XYZ_GATE_START_M``: float fingertip-xyz distance from block center at which finger unlock progress starts ramping up (only relevant if using XYZ gate)
- ``CURRICULUM_STAGE2_FINGER_XYZ_GATE_FULL_M``: float fingertip-xyz distance from block center at which finger unlock progress reaches max (only relevant if using XYZ gate)
- ``TOPDOWN_FINGER_XYZ_GATE_Z_WEIGHT``: float how much to weight the Z dimension of the fingertip-block distance for the finger unlock gate, between 0 (ignore Z) and 1 (standard XYZ distance). Only relevant if using an XYZ gate for finger unlock progress.
- ``TOPDOWN_FINGER_CLOSE_GATE_MODE``: str in {"radius", "xyz", "xyz_front"} whether to gate finger unlock progress based on a simple radius around the block center, an XYZ box gate, or an XYZ box gate with an additional front-face requirement.


- ``CURRICULUM_CONTACT_STRENGTH_MIN``: float minimum contact strength for light contact success
- ``CURRICULUM_CONTACT_BLOCK_DISP_MAX``: float maximum block displacement for light contact success
- ``CURRICULUM_CONTACT_LIFT_MAX``: float maximum block lift for light contact success

Unused Tunables (potential future use):
- ``CURRICULUM_THREE_FINGER_CENTERING``: bool whether to require all three fingers (thumb, index, middle) to be near the block center for "centered contact" state, or just the thumb and index
- ``CURRICULUM_BACK_FINGER_SPREAD_OFFSET``: float how much to offset the back (middle) finger's face target away from the block center, as a fraction of the block half-extent. This encourages a more natural spread grasp and prevents collisions between the middle finger and the block when the thumb/index are centered on the front face.


"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_float(name: str, default: float) -> float:
    """Read a float state-machine override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an integer state-machine override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean state-machine override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# --- Tunables (env-var overridable for easy tuning) ----------------------------
_REACH_HOLD_STEPS                  = int(os.environ.get("CURRICULUM_REACH_HOLD_STEPS", "5"))
_ALIGN_HOLD_STEPS                  = int(os.environ.get("CURRICULUM_ALIGN_HOLD_STEPS", "5"))
_FINGER_UNLOCK_RAMP_STEPS          = int(os.environ.get("CURRICULUM_FINGER_UNLOCK_RAMP_STEPS", "50"))
_FINGER_UNLOCK_MAX_PROGRESS        = float(os.environ.get("CURRICULUM_FINGER_UNLOCK_MAX_PROGRESS", "1.0"))
_FINGER_UNLOCK_REQUIRES_CENTER     = os.environ.get("CURRICULUM_FINGER_UNLOCK_REQUIRES_CENTER", "0") == "1"
_FINGER_CENTER_LATCH               = os.environ.get("CURRICULUM_FINGER_CENTER_LATCH", "1") == "1"
_FINGER_CENTER_HOLD_STEPS          = int(os.environ.get("CURRICULUM_FINGER_CENTER_HOLD_STEPS", "2"))
_FINGER_CENTER_TIP_XY_MAX          = float(os.environ.get("CURRICULUM_FINGER_CENTER_TIP_XY_MAX", "0.025"))
_FINGER_CENTER_MAX_TIP_XY_MAX      = float(os.environ.get("CURRICULUM_FINGER_CENTER_MAX_TIP_XY_MAX", "0.055"))
_FINGER_CENTER_TIP_Z_MAX           = float(os.environ.get("CURRICULUM_FINGER_CENTER_TIP_Z_MAX", "0.075"))
_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG = float(os.environ.get("CURRICULUM_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG", "15.0"))
_FINGER_CENTER_ALIGN_ERR_MAX       = float(os.environ.get("CURRICULUM_FINGER_CENTER_ALIGN_ERR_MAX", "0.0"))
_FINGER_CENTER_USE_XYZ_GATE        = _env_bool("CURRICULUM_FINGER_CENTER_USE_XYZ_GATE", False)
_FINGER_CENTER_XYZ_GATE_MIN        = _env_float("CURRICULUM_FINGER_CENTER_XYZ_GATE_MIN", 1.0e-6)
_STAGE2_FINGER_XYZ_GATE_START_M    = _env_float("CURRICULUM_STAGE2_FINGER_XYZ_GATE_START_M", -1.0)
_STAGE2_FINGER_XYZ_GATE_FULL_M     = _env_float("CURRICULUM_STAGE2_FINGER_XYZ_GATE_FULL_M", -1.0)
_THREE_FINGER_CENTERING            = _env_bool("CURRICULUM_THREE_FINGER_CENTERING", False)
_BACK_FINGER_SPREAD_OFFSET         = _env_float("CURRICULUM_BACK_FINGER_SPREAD_OFFSET", 0.020)

# Reach shell thresholds. These are latch gates, not final-quality targets:
# the IK teacher must be able to reach them reliably so Stage 1 rewards can
# turn on. Dense rewards continue pulling tighter after the latch.
_REACH_PALM_DIST_MAX       = float(os.environ.get("CURRICULUM_REACH_PALM_DIST_MAX", "0.14"))
_REACH_PALM_HEIGHT_MAX     = float(os.environ.get("CURRICULUM_REACH_PALM_HEIGHT_MAX", "0.08"))
_REACH_PALM_ORIENT_MAX_DEG = float(os.environ.get("CURRICULUM_REACH_PALM_ORIENT_MAX_DEG", "45.0"))
_REACH_PALM_YAW_MAX_DEG    = float(os.environ.get("CURRICULUM_REACH_PALM_YAW_MAX_DEG", "45.0"))

# Align shell thresholds (in addition to reach shell). Also teacher-reachable:
# Stage 2 contact rewards handle the final tightening to the opposed faces.
_ALIGN_ERR_MAX              = float(os.environ.get("CURRICULUM_ALIGN_ERR_MAX", "0.40"))
_ALIGN_LINE_ANGLE_MAX_DEG   = float(os.environ.get("CURRICULUM_ALIGN_LINE_ANGLE_MAX_DEG", "45.0"))
_ALIGN_OPPOSED_GATE_MIN     = float(os.environ.get("CURRICULUM_ALIGN_OPPOSED_GATE_MIN", "0.5"))
_ALIGN_NO_CONTACT_MAX       = float(os.environ.get("CURRICULUM_ALIGN_NO_CONTACT_MAX", "0.04"))

# Stage-entry shells use the same predicate shape. Stage 1 is the loose shell;
# Stage 2 is the tighter shell that unlocks contact/lift rewards.
_STAGE1_PALM_DIST_MAX       = float(os.environ.get("CURRICULUM_STAGE1_PALM_DIST_MAX", str(_REACH_PALM_DIST_MAX)))
_STAGE1_PALM_HEIGHT_MAX     = float(os.environ.get("CURRICULUM_STAGE1_PALM_HEIGHT_MAX", str(_REACH_PALM_HEIGHT_MAX)))
_STAGE1_PALM_ORIENT_MAX_DEG = float(
    os.environ.get("CURRICULUM_STAGE1_PALM_ORIENT_MAX_DEG", str(_REACH_PALM_ORIENT_MAX_DEG))
)
_STAGE1_PALM_YAW_MAX_DEG    = float(os.environ.get("CURRICULUM_STAGE1_PALM_YAW_MAX_DEG", str(_REACH_PALM_YAW_MAX_DEG)))
_STAGE1_ALIGN_ERR_MAX       = float(os.environ.get("CURRICULUM_STAGE1_ALIGN_ERR_MAX", str(_ALIGN_ERR_MAX)))
_STAGE1_LINE_ANGLE_MAX_DEG  = float(
    os.environ.get("CURRICULUM_STAGE1_LINE_ANGLE_MAX_DEG", str(_ALIGN_LINE_ANGLE_MAX_DEG))
)
_STAGE1_OPPOSED_GATE_MIN = float(
    os.environ.get("CURRICULUM_STAGE1_OPPOSED_GATE_MIN", str(_ALIGN_OPPOSED_GATE_MIN))
)
_STAGE1_NO_CONTACT_MAX      = float(os.environ.get("CURRICULUM_STAGE1_NO_CONTACT_MAX", str(_ALIGN_NO_CONTACT_MAX)))

_STAGE2_PALM_DIST_MAX       = float(os.environ.get("CURRICULUM_STAGE2_PALM_DIST_MAX", str(_REACH_PALM_DIST_MAX)))
_STAGE2_PALM_HEIGHT_MAX     = float(os.environ.get("CURRICULUM_STAGE2_PALM_HEIGHT_MAX", str(_REACH_PALM_HEIGHT_MAX)))
_STAGE2_PALM_ORIENT_MAX_DEG = float(
    os.environ.get("CURRICULUM_STAGE2_PALM_ORIENT_MAX_DEG", str(_REACH_PALM_ORIENT_MAX_DEG))
)
_STAGE2_PALM_YAW_MAX_DEG    = float(os.environ.get("CURRICULUM_STAGE2_PALM_YAW_MAX_DEG", str(_REACH_PALM_YAW_MAX_DEG)))
_STAGE2_ALIGN_ERR_MAX       = float(os.environ.get("CURRICULUM_STAGE2_ALIGN_ERR_MAX", str(_ALIGN_ERR_MAX)))
_STAGE2_LINE_ANGLE_MAX_DEG  = float(
    os.environ.get("CURRICULUM_STAGE2_LINE_ANGLE_MAX_DEG", str(_ALIGN_LINE_ANGLE_MAX_DEG))
)
_STAGE2_OPPOSED_GATE_MIN = float(
    os.environ.get("CURRICULUM_STAGE2_OPPOSED_GATE_MIN", str(_ALIGN_OPPOSED_GATE_MIN))
)
_STAGE2_NO_CONTACT_MAX = float(os.environ.get("CURRICULUM_STAGE2_NO_CONTACT_MAX", str(_ALIGN_NO_CONTACT_MAX)))

# Light-contact success thresholds (Stage 2)
_CONTACT_OPPOSED_STRENGTH_MIN   = float(os.environ.get("CURRICULUM_CONTACT_STRENGTH_MIN", "0.08"))
_CONTACT_BLOCK_DISP_MAX         = float(os.environ.get("CURRICULUM_CONTACT_BLOCK_DISP_MAX", "0.02"))
_CONTACT_LIFT_MAX               = float(os.environ.get("CURRICULUM_CONTACT_LIFT_MAX", "0.04"))
_CONTACT_POSE_HOLD_STEPS        = int(os.environ.get("CURRICULUM_CONTACT_POSE_HOLD_STEPS", "5"))
_CONTACT_POSE_READY_FALLBACK_STEPS = int(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_STEPS", "0")
)
_CONTACT_POSE_READY_FALLBACK_PALM_DIST_MAX = float(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_DIST_MAX", "0.12")
)
_CONTACT_POSE_READY_FALLBACK_PALM_HEIGHT_MAX = float(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_HEIGHT_MAX", "0.05")
)
_CONTACT_POSE_READY_FALLBACK_PALM_ORIENT_MAX_DEG = float(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_ORIENT_MAX_DEG", "45.0")
)
_CONTACT_POSE_READY_FALLBACK_PALM_YAW_MAX_DEG = float(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_PALM_YAW_MAX_DEG", "65.0")
)
_CONTACT_POSE_READY_FALLBACK_ALIGN_ERR_MAX = float(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_ALIGN_ERR_MAX", "0.25")
)
_CONTACT_POSE_READY_FALLBACK_OPPOSED_GATE_MIN = float(
    os.environ.get("CURRICULUM_CONTACT_POSE_READY_FALLBACK_OPPOSED_GATE_MIN", "0.5")
)
_CONTACT_HOLD_STEPS = int(os.environ.get("CURRICULUM_CONTACT_HOLD_STEPS", "2"))

# Stage 2 success-shell upstream thresholds (env-overridable).
_SUCCESS_PALM_DIST_MAX            = float(os.environ.get("CURRICULUM_SUCCESS_PALM_DIST_MAX", "0.08"))
_SUCCESS_PALM_HEIGHT_MAX          = float(os.environ.get("CURRICULUM_SUCCESS_PALM_HEIGHT_MAX", "0.04"))
_SUCCESS_PALM_ORIENT_MAX_DEG      = float(os.environ.get("CURRICULUM_SUCCESS_PALM_ORIENT_MAX_DEG", "35.0"))
_SUCCESS_PALM_YAW_MAX_DEG         = float(os.environ.get("CURRICULUM_SUCCESS_PALM_YAW_MAX_DEG", "35.0"))
_SUCCESS_ALIGN_ERR_MAX            = float(os.environ.get("CURRICULUM_SUCCESS_ALIGN_ERR_MAX", "0.20"))
_SUCCESS_OPPOSED_GATE_MIN         = float(os.environ.get("CURRICULUM_SUCCESS_OPPOSED_GATE_MIN", "0.5"))
_SUCCESS_REQUIRE_CENTERED_CONTACT = os.environ.get("CURRICULUM_SUCCESS_REQUIRE_CENTERED_CONTACT", "0") == "1"
_SUCCESS_CENTER_TIP_XY_MAX        = float(os.environ.get("CURRICULUM_SUCCESS_CENTER_TIP_XY_MAX", "0.015"))
_SUCCESS_CENTER_TIP_Z_MAX         = float(os.environ.get("CURRICULUM_SUCCESS_CENTER_TIP_Z_MAX", "0.060"))
_SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG = float(
    os.environ.get("CURRICULUM_SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG", "8.0")
)
_SUCCESS_CENTER_HOLD_STEPS = int(os.environ.get("CURRICULUM_SUCCESS_CENTER_HOLD_STEPS", "0"))
_LIFT_SUCCESS_HEIGHT = float(os.environ.get("TOPDOWN_LIFT_SUCCESS_HEIGHT", "0.035"))
_LIFT_SUCCESS_HOLD_STEPS = int(os.environ.get("TOPDOWN_LIFT_SUCCESS_HOLD_STEPS", "5"))
_LIFT_SUCCESS_MODE = os.environ.get("TOPDOWN_LIFT_SUCCESS_MODE", "gated").strip().lower()
_LIFT_SUCCESS_XY_DRIFT_MAX = float(os.environ.get("TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX", "0.04"))
_LIFT_SUCCESS_REQUIRES_CONTACT = os.environ.get("TOPDOWN_LIFT_SUCCESS_REQUIRES_CONTACT", "1") == "1"
_LIFT_SUCCESS_CONTACT_MODE = os.environ.get("TOPDOWN_LIFT_SUCCESS_CONTACT_MODE", "opposed").strip().lower()
_LIFT_SUCCESS_CONTACT_MIN = float(os.environ.get("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", "0.30"))

# Loose Stage-2 retention shell. Unlike the strict contact-pose latch, this is
# only a safety rail: after Stage 2 has had a short grace period, sustained
# fall-out demotes back to Stage 1 and clears pose-ready/unlock state.
_STAGE2_FALLOUT_GRACE_STEPS = int(os.environ.get("CURRICULUM_STAGE2_FALLOUT_GRACE_STEPS", "50"))
_STAGE2_FALLOUT_HOLD_STEPS = int(os.environ.get("CURRICULUM_STAGE2_FALLOUT_HOLD_STEPS", "40"))
_STAGE2_FALLOUT_PALM_DIST_MAX = float(os.environ.get("CURRICULUM_STAGE2_FALLOUT_PALM_DIST_MAX", "0.16"))
_STAGE2_FALLOUT_PALM_HEIGHT_MAX = float(os.environ.get("CURRICULUM_STAGE2_FALLOUT_PALM_HEIGHT_MAX", "0.10"))
_STAGE2_FALLOUT_PALM_ORIENT_MAX_DEG = float(
    os.environ.get("CURRICULUM_STAGE2_FALLOUT_PALM_ORIENT_MAX_DEG", "75.0")
)
_STAGE2_FALLOUT_PALM_YAW_MAX_DEG = float(os.environ.get("CURRICULUM_STAGE2_FALLOUT_PALM_YAW_MAX_DEG", "80.0"))
_STAGE2_FALLOUT_ALIGN_ERR_MAX = float(os.environ.get("CURRICULUM_STAGE2_FALLOUT_ALIGN_ERR_MAX", "0.45"))

# Topdown grip target geometry (own copy — not imported)
_TOPDOWN_BLOCK_SIZE = float(os.environ.get("TOPDOWN_BLOCK_SIZE", "0.08"))
_BLOCK_HALF_HEIGHT = 0.5 * _TOPDOWN_BLOCK_SIZE
_HOVER_ABOVE_BLOCK_TOP = _env_float("CURRICULUM_TOPDOWN_HOVER_ABOVE_BLOCK_TOP", 0.05)
_GRIP_TARGET_Z_OFFSET = _BLOCK_HALF_HEIGHT + _HOVER_ABOVE_BLOCK_TOP

# Open-hand fingertip face target geometry (own copy).
# Default 0.025 (5cm span) is inside the 8cm block faces but matches the
# Dex3 open-hand thumb-index reach, so face targets are achievable without
# requiring the hand to curl mid-descent.
_FACE_HALF_EXTENT = _env_float(
    "CURRICULUM_FACE_HALF_EXTENT",
    0.025,
)
_FACE_TOP_MARGIN = _env_float("CURRICULUM_FINGER_FACE_TOP_MARGIN",
0.012,
)  # fingertip target sits this far below block top corner

# Stage 2 contact-pose target geometry (distinct from the hover target used
# for reach/align). Defines a palm grip-center hover offset just above the
# block top so success and any future contact-relative shaping reward descent.
_CONTACT_PALM_HEIGHT_TARGET_ABOVE_BLOCK_TOP = float(
    os.environ.get("CURRICULUM_CONTACT_PALM_HEIGHT_ABOVE_BLOCK_TOP", "0.02")
)
_ONE_SIDED_RAMP_STEPS = int(os.environ.get("CURRICULUM_ONE_SIDED_RAMP_STEPS", "50"))

# Body link names
_PALM_LINK   = "right_hand_palm_link"
_THUMB_LINK  = "right_hand_thumb_2_link"
_INDEX_LINK  = "right_hand_index_1_link"
_MIDDLE_LINK = "right_hand_middle_1_link"

# Contact sensor names registered in the new SceneCfg
_CONTACT_SENSOR_NAMES = (
    "thumb_contact",
    "thumb_mid_contact",
    "index_contact",
    "index_mid_contact",
    "middle_contact",
    "middle_mid_contact",
    "palm_contact",
)

# Force shaping
_CONTACT_FORCE_THRESHOLD = _env_float("TOPDOWN_CONTACT_FORCE_THRESHOLD", 0.5)  # N — below this counts as no contact
_CONTACT_FORCE_SATURATION = _env_float("TOPDOWN_CONTACT_FORCE_SATURATION", 4.0)  # N — strength saturates here

# --- Body / sensor index caches ----------------------------------------------


def _cached_body_index(env: "ManagerBasedRLEnv", body_name: str) -> int:
    """Return and cache the rigid-body index for a named articulation body."""
    cache_key = f"_topdown_curr_body_idx::{body_name}"
    cached = getattr(env, cache_key, None)
    if cached is not None:
        return cached
    idx = env.scene["robot"].data.body_names.index(body_name)
    setattr(env, cache_key, idx)
    return idx


def _has_sensor(env: "ManagerBasedRLEnv", sensor_name: str) -> bool:
    """Return whether a named contact sensor exists in the scene."""
    try:
        env.scene[sensor_name]
        return True
    except KeyError:
        return False


def _active_object_name(env: "ManagerBasedRLEnv") -> str:
    """Scene object currently used for topdown block-relative geometry."""
    return str(getattr(env, "_topdown_active_object_name", "object"))


def _use_visible_source_objects(env: "ManagerBasedRLEnv") -> bool:
    """Return whether source-colored block objects are active in the scene."""
    return bool(getattr(env, "_topdown_use_visible_source_objects", False))


def _active_source_pose_idx(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the active source-pose index for each environment."""
    idx = getattr(env, "_topdown_source_pose_idx", None)
    if torch.is_tensor(idx) and idx.shape[0] == env.num_envs:
        return idx.to(device=env.device, dtype=torch.long).clamp(0, 2)
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _active_contact_sensor_name(env: "ManagerBasedRLEnv", sensor_name: str) -> str:
    """Return the active-object filtered sensor when a multi-block env has one."""
    active = _active_object_name(env)
    if active == "object" or not sensor_name.endswith("_contact"):
        return sensor_name
    color = active.removesuffix("_block")
    base = sensor_name.removesuffix("_contact")
    candidate = f"{base}_{color}_contact"
    if _has_sensor(env, candidate):
        return candidate
    return sensor_name


def _block_pose(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active block world position and orientation."""
    if _use_visible_source_objects(env):
        names = ("object", "object_yellow", "object_blue")
        try:
            pos_stack = torch.stack(
                [env.scene[name].data.root_pos_w[:, :3] for name in names],
                dim=0,
            )
            quat_stack = torch.stack(
                [env.scene[name].data.root_quat_w for name in names],
                dim=0,
            )
        except KeyError:
            pass
        else:
            source_idx = _active_source_pose_idx(env)
            env_idx = torch.arange(env.num_envs, device=env.device)
            return pos_stack[source_idx, env_idx], quat_stack[source_idx, env_idx]
    obj = env.scene[_active_object_name(env)]
    return obj.data.root_pos_w[:, :3], obj.data.root_quat_w


def _palm_pose(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the palm world position and orientation."""
    robot = env.scene["robot"]
    idx = _cached_body_index(env, _PALM_LINK)
    pose = robot.data.body_link_pose_w[:, idx]
    return pose[:, :3], pose[:, 3:7]


def _link_pos(env: "ManagerBasedRLEnv", body_name: str) -> torch.Tensor:
    """Return a named robot link position in world coordinates."""
    robot = env.scene["robot"]
    idx = _cached_body_index(env, body_name)
    return robot.data.body_link_pose_w[:, idx, :3]


def _active_finger_points(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the distal link origins used for centered grasp geometry."""
    return _link_pos(env, _THUMB_LINK), _link_pos(env, _INDEX_LINK), _link_pos(env, _MIDDLE_LINK)


def _finger_close_gate_mode() -> str:
    """Return the configured trainer/env finger close gate mode."""
    return os.environ.get("TOPDOWN_FINGER_CLOSE_GATE_MODE", "center").strip().lower()


def finger_xyz_block_center_gate(
    env: "ManagerBasedRLEnv",
    *,
    write_diagnostics: bool         = False,
    start_m_override : float | None = None,
    full_m_override  : float | None = None,
) -> torch.Tensor:
    """Return the geometry-only xyz close gate.

    The rule is intentionally independent of curriculum stage, contact latches,
    face targets, and center-hold state: when both active pinch fingertips are
    close to the active block center, closure is allowed; otherwise the gate is
    closed.
    
        The default gate geometry is a simple XYZ ball around the block center, but
        the Z dimension can be downweighted to allow easier vertical descent while
        still gating out horizontally distant grasps. In "xyz_front" mode, an additional 
        front-face requirement is added: the fingertips must be between the block center and a plane parallel to
        the grip approach direction, which prevents "backwards" grasps that approach the block from behind and 
        can be more challenging to recover from.
    """
    block_pos, _ = _block_pose(env)
    thumb_pos    = _link_pos(env, _THUMB_LINK)
    index_pos   = _link_pos(env, _INDEX_LINK)

    z_weight    = max(_env_float("TOPDOWN_FINGER_XYZ_GATE_Z_WEIGHT", 0.25), 0.0)
    thumb_delta = block_pos - thumb_pos
    index_delta = block_pos - index_pos
    thumb_weighted_delta = thumb_delta.clone()
    index_weighted_delta = index_delta.clone()
    thumb_weighted_delta[:, 2] = thumb_weighted_delta[:, 2] * z_weight
    index_weighted_delta[:, 2] = index_weighted_delta[:, 2] * z_weight

    thumb_err = torch.linalg.norm(thumb_weighted_delta, dim=-1)
    index_err = torch.linalg.norm(index_weighted_delta, dim=-1)
    max_err = torch.maximum(thumb_err, index_err)

    start_m = max(
        float(start_m_override)
        if start_m_override is not None and start_m_override >= 0.0
        else _env_float("TOPDOWN_FINGER_XYZ_GATE_START_M", 0.085),
        0.0,
    )
    full_m = max(
        float(full_m_override)
        if full_m_override is not None and full_m_override >= 0.0
        else _env_float("TOPDOWN_FINGER_XYZ_GATE_FULL_M", 0.025),
        0.0,
    )
    if start_m > full_m + 1.0e-6:
        linear_gate = torch.clamp((start_m - max_err) / (start_m - full_m), 0.0, 1.0)
    else:
        linear_gate = (max_err <= full_m).to(dtype=torch.float32)
    if _env_bool("TOPDOWN_FINGER_XYZ_GATE_LINEAR", False):
        gate = linear_gate
    else:
        gate = (max_err <= start_m).to(dtype=torch.float32)

    close_gate_mode = _finger_close_gate_mode()
    front_gate = torch.ones_like(max_err)
    thumb_front_margin = torch.zeros_like(max_err)
    index_front_margin = torch.zeros_like(max_err)
    if close_gate_mode == "xyz_front":
        grip_axis = _grip_axis_xy(env)
        thumb_side = torch.sum((thumb_pos - block_pos) * grip_axis, dim=-1)
        index_side = torch.sum((index_pos - block_pos) * grip_axis, dim=-1)
        tol = max(_env_float("TOPDOWN_FINGER_FRONT_FACE_TOLERANCE_M", 0.0), 0.0)
        min_sep = max(_env_float("TOPDOWN_FINGER_FRONT_FACE_MIN_SEPARATION_M", 0.0), 0.0)
        thumb_front_margin = thumb_side + tol
        index_front_margin = -index_side + tol
        sep_margin = (thumb_side - index_side) - min_sep
        front_gate_bool = (
            (thumb_front_margin >= 0.0)
            & (index_front_margin >= 0.0)
            & (sep_margin >= 0.0)
        )
        front_gate = front_gate_bool.to(dtype=torch.float32)
        gate = gate * front_gate

    if write_diagnostics:
        env._topdown_finger_thumb_xyz_error = thumb_err.detach().clone()
        env._topdown_finger_index_xyz_error = index_err.detach().clone()
        env._topdown_finger_xyz_error = max_err.detach().clone()
        env._topdown_finger_xyz_z_weight = torch.full_like(max_err, z_weight).detach().clone()
        env._topdown_finger_xyz_linear_gate = linear_gate.detach().clone()
        env._topdown_finger_thumb_front_margin = thumb_front_margin.detach().clone()
        env._topdown_finger_index_front_margin = index_front_margin.detach().clone()
        env._topdown_finger_front_gate = front_gate.detach().clone()
        env._topdown_finger_xyz_close_gate = gate.detach().clone()

    return gate


# --- Quat / vector primitives -------------------------------------------------


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Batched (N, 4) wxyz -> (N, 3, 3) rotation matrix."""
    w, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack(
        (
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ),
        dim=-1,
    )
    return m.reshape(*quat.shape[:-1], 3, 3)


# --- Topdown grip target (palm and fingertip face) ----------------------------


def _grip_target_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the world-space target used for topdown grip alignment."""
    block_pos, _ = _block_pose(env)
    target = block_pos.clone()
    target[:, 2] = block_pos[:, 2] + _GRIP_TARGET_Z_OFFSET
    return target


def _block_axis_xy(
    env        : "ManagerBasedRLEnv",
    axis_index : int,
    fallback_xy: tuple[float, float],
) -> torch.Tensor:
    """
    Block-local axis projected into world XY and normalized.
    """
    
    _, block_quat = _block_pose(env)
    R = _quat_wxyz_to_matrix(block_quat)
    axis = R[..., :, axis_index].clone()
    axis[..., 2] = 0.0
    n = torch.linalg.norm(axis, dim=-1, keepdim=True)  # (N, 1)
    fallback = torch.tensor(
        (float(fallback_xy[0]), float(fallback_xy[1]), 0.0),
        device=axis.device,
        dtype=axis.dtype,
    ).expand_as(axis)
    normalized = axis / n.clamp_min(1.0e-6)
    valid = (n >= 1.0e-3).expand_as(axis)  # broadcast (N,1) -> (N,3)
    return torch.where(valid, normalized, fallback)


def _grip_axis_xy(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Thumb-side direction for topdown hand, projected to XY.

    Defaulting to block body-Y preserved the original cube-local convention, but
    for the fixed tabletop block we often want robot/world front-back faces
    instead.  ``TOPDOWN_GRIP_FACE_AXIS`` therefore mirrors the palm-yaw axis
    selector while keeping the legacy default.
    """
    mode = os.environ.get("TOPDOWN_GRIP_FACE_AXIS", "block_y").strip().lower()
    if mode in {"", "block", "block_y", "block-y", "y", "+y", "world_y", "world-y"}:
        if mode in {"y", "+y", "world_y", "world-y"}:
            axis = torch.tensor((0.0, 1.0, 0.0), device=env.device).view(1, 3)
            return axis.expand(env.num_envs, -1)
        return _block_axis_xy(env, axis_index=1, fallback_xy=(0.0, 1.0))
    if mode in {"-y", "neg_y", "negative_y", "world_-y", "world-neg-y"}:
        axis = torch.tensor((0.0, -1.0, 0.0), device=env.device).view(1, 3)
        return axis.expand(env.num_envs, -1)
    if mode in {"block_x", "block-x"}:
        return _block_axis_xy(env, axis_index=0, fallback_xy=(1.0, 0.0))
    if mode in {"x", "+x", "world_x", "world-x"}:
        axis = torch.tensor((1.0, 0.0, 0.0), device=env.device).view(1, 3)
        return axis.expand(env.num_envs, -1)
    if mode in {"-x", "neg_x", "negative_x", "world_-x", "world-neg-x"}:
        axis = torch.tensor((-1.0, 0.0, 0.0), device=env.device).view(1, 3)
        return axis.expand(env.num_envs, -1)
    raise RuntimeError(f"unsupported TOPDOWN_GRIP_FACE_AXIS={mode!r}")


def _spread_axis_xy(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Back-finger spread direction: horizontal perpendicular to grip axis."""
    grip_axis = _grip_axis_xy(env)
    spread = torch.stack((-grip_axis[:, 1], grip_axis[:, 0], torch.zeros_like(grip_axis[:, 0])), dim=-1)
    n = torch.linalg.norm(spread, dim=-1, keepdim=True)
    fallback = torch.tensor((1.0, 0.0, 0.0), device=spread.device, dtype=spread.dtype).expand_as(spread)
    return torch.where(n >= 1.0e-3, spread / n.clamp_min(1.0e-6), fallback)


def topdown_grip_center_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Open-hand grip center used as the topdown reach end-effector."""
    thumb_pos, index_pos, middle_pos = _active_finger_points(env)
    back_center = 0.5 * (index_pos + middle_pos) if _THREE_FINGER_CENTERING else index_pos
    return 0.5 * (thumb_pos + back_center)


def _face_targets(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Return opposed thumb/index face-centerline targets near the block top."""
    block_pos, _ = _block_pose(env)
    grip_axis = _grip_axis_xy(env)
    top_z = block_pos[:, 2] + _BLOCK_HALF_HEIGHT - _FACE_TOP_MARGIN

    thumb_face = block_pos + grip_axis * _FACE_HALF_EXTENT
    thumb_face = thumb_face.clone()
    thumb_face[:, 2] = top_z

    index_face = block_pos - grip_axis * _FACE_HALF_EXTENT
    index_face = index_face.clone()
    index_face[:, 2] = top_z
    return thumb_face, index_face


def _three_finger_face_targets(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return thumb/index/middle targets for a balanced three-finger pinch.

    Thumb targets one face. Index and middle target the opposite face with
    symmetric offsets along the block spread axis, so a centered grasp requires
    the back two fingers to straddle the block instead of curling around one
    side.
    """
    thumb_face, back_center = _face_targets(env)
    spread_axis = _spread_axis_xy(env)
    offset = max(_BACK_FINGER_SPREAD_OFFSET, 0.0)

    index_face = back_center + spread_axis * offset
    middle_face = back_center - spread_axis * offset
    index_face = index_face.clone()
    middle_face = middle_face.clone()
    return thumb_face, index_face, middle_face


# --- Geometric primitives (callable from anywhere; do not require stage update)


def palm_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return distance from the palm target point to the active block target."""
    grip_center = topdown_grip_center_position(env)
    target = _grip_target_position(env)
    return torch.linalg.norm(grip_center - target, dim=-1)


def palm_height_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return vertical palm error relative to the target topdown height."""
    grip_center = topdown_grip_center_position(env)
    target = _grip_target_position(env)
    return torch.abs(grip_center[:, 2] - target[:, 2])


def _grip_target_position_contact(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Contact-pose grip-center target: just above the block top.

    Distinct from ``_grip_target_position`` (which sits at the hover height
    used by reach/align). Stage 2 success and contact-relative maintenance
    measure against this target so descent toward contact is rewarded
    rather than penalized.
    """
    block_pos, _ = _block_pose(env)
    target = block_pos.clone()
    target[:, 2] = (
        block_pos[:, 2] + _BLOCK_HALF_HEIGHT + _CONTACT_PALM_HEIGHT_TARGET_ABOVE_BLOCK_TOP
    )
    return target


def palm_distance_contact(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """3D distance from the open-hand grip center to the contact-pose target."""
    grip_center = topdown_grip_center_position(env)
    target = _grip_target_position_contact(env)
    return torch.linalg.norm(grip_center - target, dim=-1)


def palm_height_error_contact(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vertical residual against the contact-pose Z target."""
    grip_center = topdown_grip_center_position(env)
    target = _grip_target_position_contact(env)
    return torch.abs(grip_center[:, 2] - target[:, 2])


def palm_drop_axis_error_rad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Angle between the palm-to-grip-center direction and world -Z.

    Mirrors ``topdown_drop_axis_error_rad`` from the existing reach package
    (canonical reward drop-axis metric — duplicated, not imported). Note this
    is *only* the drop-axis: it says nothing about wrist yaw or spread; see
    ``palm_yaw_axis_error_rad`` and ``palm_spread_axis_error_rad`` for those.
    """
    palm_pos, _ = _palm_pose(env)
    grip_center = topdown_grip_center_position(env)
    drop = grip_center - palm_pos
    drop = drop / torch.linalg.norm(drop, dim=-1, keepdim=True).clamp_min(1.0e-6)
    target = torch.tensor((0.0, 0.0, -1.0), device=drop.device, dtype=drop.dtype)
    cos = torch.sum(drop * target.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


# Backward-compat alias. Prefer ``palm_drop_axis_error_rad`` in new code.
def palm_orient_error_rad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return palm orientation error in radians for the topdown approach."""
    return palm_drop_axis_error_rad(env)


def palm_yaw_axis_error_rad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Yaw error: angle between (thumb − back-finger-center) projected to XY
    and the block grip axis (block body-Y in world XY).

    Mirrors ``topdown_yaw_axis_error_rad`` from the Phase 1 reach package but
    targets the block-local axis so the metric stays correct under any block
    yaw at spawn. Built from finger geometry, not palm-link body frame.
    """
    thumb_pos = _link_pos(env, _THUMB_LINK)
    index_pos = _link_pos(env, _INDEX_LINK)
    middle_pos = _link_pos(env, _MIDDLE_LINK)
    back_center = 0.5 * (index_pos + middle_pos) if _THREE_FINGER_CENTERING else index_pos
    yaw_axis = thumb_pos - back_center
    yaw_xy = yaw_axis[:, :2]
    norm = torch.linalg.norm(yaw_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    yaw_xy = yaw_xy / norm
    target = _grip_axis_xy(env)
    target = target[:, :2]
    cos = torch.sum(yaw_xy * target, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


def palm_spread_axis_error_rad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Spread error: angle between (index − middle) projected to XY and the
    block spread axis (block body-X in world XY), mod 180°.

    Mirrors ``topdown_spread_axis_error_rad`` from Phase 1 but targets the
    block-local spread axis. Used for shaping/diagnostics only — not gated.
    """
    index_pos = _link_pos(env, _INDEX_LINK)
    middle_pos = _link_pos(env, _MIDDLE_LINK)
    spread = index_pos - middle_pos
    spread_xy = spread[:, :2]
    norm = torch.linalg.norm(spread_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    spread_xy = spread_xy / norm
    target = _spread_axis_xy(env)
    target = target[:, :2]
    cos = torch.sum(spread_xy * target, dim=-1).abs().clamp(0.0, 1.0)
    return torch.acos(cos)


def open_hand_face_distances(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distances from thumb and fingers to opposing block faces."""
    thumb_pos, index_pos, _ = _active_finger_points(env)
    thumb_face, index_face = _face_targets(env)
    thumb_dist = torch.linalg.norm(thumb_pos - thumb_face, dim=-1)
    index_dist = torch.linalg.norm(index_pos - index_face, dim=-1)
    return thumb_dist, index_dist


def open_hand_alignment_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the open-hand alignment error used by reach and align gates."""
    if _THREE_FINGER_CENTERING:
        thumb_pos = _link_pos(env, _THUMB_LINK)
        index_pos = _link_pos(env, _INDEX_LINK)
        middle_pos = _link_pos(env, _MIDDLE_LINK)
        thumb_target, index_target, middle_target = _three_finger_face_targets(env)
        thumb_dist = torch.linalg.norm(thumb_pos - thumb_target, dim=-1)
        index_dist = torch.linalg.norm(index_pos - index_target, dim=-1)
        middle_dist = torch.linalg.norm(middle_pos - middle_target, dim=-1)
        return thumb_dist + index_dist + middle_dist
    thumb_dist, index_dist = open_hand_face_distances(env)
    return thumb_dist + index_dist


def centered_contact_errors(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Max centered-grasp residual to face targets in XY and Z.

    In two-finger mode this is thumb/index. In three-finger mode this is
    thumb/index/middle, with index and middle split around the opposite face.
    """
    thumb_pos, index_pos, middle_contact_pos = _active_finger_points(env)
    if _THREE_FINGER_CENTERING:
        middle_pos = middle_contact_pos
        thumb_target, index_target, middle_target = _three_finger_face_targets(env)
    else:
        middle_pos = None
        thumb_target, index_target = _face_targets(env)
        middle_target = None
    thumb_delta = thumb_target - thumb_pos
    index_delta = index_target - index_pos
    thumb_xy = torch.linalg.norm(thumb_delta[:, :2], dim=-1)
    index_xy = torch.linalg.norm(index_delta[:, :2], dim=-1)
    max_xy = torch.maximum(thumb_xy, index_xy)
    max_z = torch.maximum(torch.abs(thumb_delta[:, 2]), torch.abs(index_delta[:, 2]))
    if _THREE_FINGER_CENTERING and middle_pos is not None and middle_target is not None:
        middle_delta = middle_target - middle_pos
        middle_xy = torch.linalg.norm(middle_delta[:, :2], dim=-1)
        max_xy = torch.maximum(max_xy, middle_xy)
        max_z = torch.maximum(max_z, torch.abs(middle_delta[:, 2]))
    return max_xy, max_z


def finger_unlock_center_errors(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-curl pocket gate errors.

    Before finger curl, the open-hand contact links do not need to already sit on
    their final side-face targets. The useful latch is the mean target residual:
    once the finger group center is over the pocket and no individual finger is
    far out of bounds, finger curl can close the remaining span.
    """
    thumb_pos, index_pos, middle_contact_pos = _active_finger_points(env)
    if _THREE_FINGER_CENTERING:
        middle_pos = middle_contact_pos
        thumb_target, index_target, middle_target = _three_finger_face_targets(env)
    else:
        middle_pos = None
        thumb_target, index_target = _face_targets(env)
        middle_target = None
    thumb_delta = thumb_target - thumb_pos
    index_delta = index_target - index_pos
    if _THREE_FINGER_CENTERING and middle_pos is not None and middle_target is not None:
        middle_delta = middle_target - middle_pos
        center_delta_xy = (thumb_delta[:, :2] + index_delta[:, :2] + middle_delta[:, :2]) / 3.0
    else:
        middle_delta = None
        center_delta_xy = 0.5 * (thumb_delta[:, :2] + index_delta[:, :2])
    pair_center_xy = torch.linalg.norm(center_delta_xy, dim=-1)
    thumb_xy = torch.linalg.norm(thumb_delta[:, :2], dim=-1)
    index_xy = torch.linalg.norm(index_delta[:, :2], dim=-1)
    max_xy = torch.maximum(thumb_xy, index_xy)
    max_z = torch.maximum(torch.abs(thumb_delta[:, 2]), torch.abs(index_delta[:, 2]))
    if middle_delta is not None:
        middle_xy = torch.linalg.norm(middle_delta[:, :2], dim=-1)
        max_xy = torch.maximum(max_xy, middle_xy)
        max_z = torch.maximum(max_z, torch.abs(middle_delta[:, 2]))
    return pair_center_xy, max_xy, max_z


def centered_contact_shell_now(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Final centered-contact gate used to prevent early reset on edge touches."""
    xy_err, z_err = centered_contact_errors(env)
    align_angle_deg = torch.rad2deg(fingertip_line_angle_rad(env))
    ready = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if _SUCCESS_CENTER_TIP_XY_MAX > 0.0:
        ready = ready & (xy_err <= _SUCCESS_CENTER_TIP_XY_MAX)
    if _SUCCESS_CENTER_TIP_Z_MAX > 0.0:
        ready = ready & (z_err <= _SUCCESS_CENTER_TIP_Z_MAX)
    if _SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG > 0.0:
        ready = ready & (align_angle_deg <= _SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG)
    env._topdown_success_center_xy_err = xy_err.detach().clone()
    env._topdown_success_center_z_err = z_err.detach().clone()
    env._topdown_success_center_align_angle_deg = align_angle_deg.detach().clone()
    env._topdown_success_center_ready = ready.detach().clone()
    return ready


def fingertip_line_angle_rad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Elevation angle of the opposed pinch line above the table plane."""
    thumb_pos = _link_pos(env, _THUMB_LINK)
    index_pos = _link_pos(env, _INDEX_LINK)
    if _THREE_FINGER_CENTERING:
        middle_pos = _link_pos(env, _MIDDLE_LINK)
        back_pos = 0.5 * (index_pos + middle_pos)
    else:
        back_pos = index_pos
    delta = thumb_pos - back_pos
    horizontal = torch.linalg.norm(delta[:, :2], dim=-1).clamp_min(1.0e-6)
    return torch.atan2(torch.abs(delta[:, 2]), horizontal)


def _opposite_face_gate_for_back_pos(env: "ManagerBasedRLEnv", back_pos: torch.Tensor) -> torch.Tensor:
    """Smooth gate (0..1) for thumb and one back-finger point on opposed block sides."""
    block_pos, _ = _block_pose(env)
    axis = _grip_axis_xy(env)
    thumb_pos = _link_pos(env, _THUMB_LINK)
    thumb_side = torch.sum((thumb_pos - block_pos) * axis, dim=-1)
    back_side = torch.sum((back_pos - block_pos) * axis, dim=-1)
    # Smooth opposed signal: 0 on same-side contacts, ramps to 1 as the tips
    # separate onto opposite faces. The scale is the block half-extent squared
    # (the physical distance the tips traverse) — decoupled from
    # ``_FACE_HALF_EXTENT`` so that retuning the fingertip face-target placement
    # for hand reach does not change the gate magnitude downstream rewards
    # consume.
    scale = max(_BLOCK_HALF_HEIGHT * _BLOCK_HALF_HEIGHT, 1.0e-6)
    return torch.clamp(-(thumb_side * back_side) / scale, 0.0, 1.0)


def opposite_face_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Smooth gate (0..1) for thumb/back-finger center on opposed block sides."""
    index_pos = _link_pos(env, _INDEX_LINK)
    if _THREE_FINGER_CENTERING:
        middle_pos = _link_pos(env, _MIDDLE_LINK)
        back_pos = 0.5 * (index_pos + middle_pos)
    else:
        back_pos = index_pos
    return _opposite_face_gate_for_back_pos(env, back_pos)


# --- Contact primitives -------------------------------------------------------


def _per_link_contact_strength(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    """Smoothstepped contact strength in [0, 1] for one filtered ContactSensor."""
    sensor_name = _active_contact_sensor_name(env, sensor_name)
    if not _has_sensor(env, sensor_name):
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene[sensor_name]
    force_matrix = getattr(sensor.data, "force_matrix_w", None)
    if force_matrix is not None:
        # Single-link filtered ContactSensor: (N, 1, num_filters, 3).
        forces = force_matrix[:, 0].sum(dim=1)
    else:
        net_forces = getattr(sensor.data, "net_forces_w", None)
        if net_forces is None:
            return torch.zeros(env.num_envs, device=env.device)
        forces = net_forces[:, 0, :] if net_forces.dim() == 3 else net_forces
    mag = torch.linalg.norm(forces, dim=-1)
    span = max(_CONTACT_FORCE_SATURATION - _CONTACT_FORCE_THRESHOLD, 1.0e-6)
    progress = torch.clamp((mag - _CONTACT_FORCE_THRESHOLD) / span, 0.0, 1.0)
    # Smoothstep
    return progress * progress * (3.0 - 2.0 * progress)


def _max_contact_strength(env: "ManagerBasedRLEnv", *sensor_names: str) -> torch.Tensor:
    """Return max contact strength across a small chain of same-finger sensors."""
    if not sensor_names:
        return torch.zeros(env.num_envs, device=env.device)
    strength = _per_link_contact_strength(env, sensor_names[0])
    for sensor_name in sensor_names[1:]:
        strength = torch.maximum(strength, _per_link_contact_strength(env, sensor_name))
    return strength


def thumb_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return normalized contact strength for the thumb contact chain."""
    return _max_contact_strength(env, "thumb_contact", "thumb_mid_contact")


def index_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return normalized contact strength for the index-finger contact chain."""
    return _max_contact_strength(env, "index_contact", "index_mid_contact")


def middle_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return normalized contact strength for the middle-finger contact chain."""
    return _max_contact_strength(env, "middle_contact", "middle_mid_contact")


def palm_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return normalized contact strength for the palm sensor."""
    return _per_link_contact_strength(env, "palm_contact")


def any_hand_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return max normalized contact strength across palm and fingertips."""
    return torch.maximum(
        torch.maximum(thumb_contact_strength(env), index_contact_strength(env)),
        torch.maximum(middle_contact_strength(env), palm_contact_strength(env)),
    )


def any_fingertip_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return max normalized contact strength across fingertip sensors."""
    return torch.maximum(
        torch.maximum(thumb_contact_strength(env), index_contact_strength(env)),
        middle_contact_strength(env),
    )


def opposed_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Geometric AND-of-strengths: requires thumb plus an opposed back finger."""
    t = thumb_contact_strength(env)
    i = index_contact_strength(env)
    index_score = torch.minimum(t, i) * _opposite_face_gate_for_back_pos(env, _link_pos(env, _INDEX_LINK))
    if os.environ.get("TOPDOWN_OPPOSED_CONTACT_USE_MIDDLE_BACK", "0") == "1":
        m = middle_contact_strength(env)
        middle_score = torch.minimum(t, m) * _opposite_face_gate_for_back_pos(env, _link_pos(env, _MIDDLE_LINK))
        return torch.maximum(index_score, middle_score)
    return index_score


def block_displacement(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Distance the block has moved from its episode-start position."""
    pos, _ = _block_pose(env)
    cached = getattr(env, "_topdown_block_spawn_pos", None)
    just_reset = env.episode_length_buf <= 1
    if cached is None or cached.shape != pos.shape:
        cached = pos.detach().clone()
    if just_reset.any():
        cached = torch.where(just_reset.unsqueeze(-1), pos.detach(), cached)
    setattr(env, "_topdown_block_spawn_pos", cached)
    return torch.linalg.norm(pos - cached, dim=-1)


def block_xy_displacement(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Horizontal block drift from episode-start position."""
    pos, _ = _block_pose(env)
    pos_xy = pos[:, :2]
    cached = getattr(env, "_topdown_block_spawn_xy", None)
    just_reset = env.episode_length_buf <= 1
    if cached is None or cached.shape != pos_xy.shape:
        cached = pos_xy.detach().clone()
    if just_reset.any():
        cached = torch.where(just_reset.unsqueeze(-1), pos_xy.detach(), cached)
    setattr(env, "_topdown_block_spawn_xy", cached)
    return torch.linalg.norm(pos_xy - cached, dim=-1)


def block_lift_height(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the block height gained relative to its initial tabletop height."""
    pos, _ = _block_pose(env)
    pos_z = pos[:, 2]
    cached = getattr(env, "_topdown_block_spawn_z", None)
    just_reset = env.episode_length_buf <= 1
    if cached is None or cached.shape != pos_z.shape:
        cached = pos_z.detach().clone()
    if just_reset.any():
        cached = torch.where(just_reset, pos_z.detach(), cached)
    setattr(env, "_topdown_block_spawn_z", cached)
    return torch.clamp(pos_z - cached, min=0.0)


def block_tilt_angle_rad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Tilt of the active block's local Z axis away from world up."""
    _, quat = _block_pose(env)
    R = _quat_wxyz_to_matrix(quat)
    up_dot = R[:, 2, 2].clamp(-1.0, 1.0)
    tilt = torch.acos(up_dot)
    env._topdown_block_tilt_angle_rad = tilt.detach().clone()
    return tilt


def block_linear_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Linear velocity of the active topdown source block in world frame."""
    if _use_visible_source_objects(env):
        names = ("object", "object_yellow", "object_blue")
        try:
            vel_stack = torch.stack(
                [env.scene[name].data.root_lin_vel_w[:, :3] for name in names],
                dim=0,
            )
        except KeyError:
            pass
        else:
            source_idx = _active_source_pose_idx(env)
            env_idx = torch.arange(env.num_envs, device=env.device)
            return vel_stack[source_idx, env_idx]
    obj = env.scene[_active_object_name(env)]
    return obj.data.root_lin_vel_w[:, :3]


def block_z_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the active block vertical velocity."""
    return block_linear_velocity(env)[:, 2]


def block_xy_velocity_norm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the planar speed of the active block."""
    return torch.linalg.norm(block_linear_velocity(env)[:, :2], dim=-1)


def block_angular_velocity_norm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Angular velocity magnitude of the active topdown source block."""
    if _use_visible_source_objects(env):
        names = ("object", "object_yellow", "object_blue")
        try:
            vel_stack = torch.stack(
                [env.scene[name].data.root_ang_vel_w[:, :3] for name in names],
                dim=0,
            )
        except KeyError:
            pass
        else:
            source_idx = _active_source_pose_idx(env)
            env_idx = torch.arange(env.num_envs, device=env.device)
            return torch.linalg.norm(vel_stack[source_idx, env_idx], dim=-1)
    obj = env.scene[_active_object_name(env)]
    return torch.linalg.norm(obj.data.root_ang_vel_w[:, :3], dim=-1)


# --- Stage state machine (the singleton update path) --------------------------


def _ensure_state_buffers(env: "ManagerBasedRLEnv") -> None:
    """Allocate and reset curriculum state buffers stored on the Isaac environment.

    The state machine persists stage, hold counters, lift latches, finger
    unlock progress, and failure flags across reward, observation, and
    termination terms.
    """
    n = env.num_envs
    # Reward, observation, and termination managers can ask for curriculum
    # state in different orders, so buffers are created lazily on the shared env.
    if not hasattr(env, "_topdown_stage"):
        env._topdown_stage = torch.zeros(n, dtype=torch.long, device=env.device)
        env._topdown_reach_hold = torch.zeros(n, dtype=torch.long, device=env.device)
        env._topdown_align_hold = torch.zeros(n, dtype=torch.long, device=env.device)
        env._topdown_stage2_age = torch.zeros(n, dtype=torch.long, device=env.device)
        env._topdown_finger_unlock_progress = torch.zeros(n, device=env.device)
        env._topdown_raw_finger_unlock_progress = torch.zeros(n, device=env.device)
        env.reach_align_finger_unlocked = torch.zeros(n, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_topdown_stage2_fallout_hold"):
        env._topdown_stage2_fallout_hold = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_topdown_contact_pose_hold"):
        env._topdown_contact_pose_hold = torch.zeros(n, dtype=torch.long, device=env.device)
        env._topdown_contact_pose_ready = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._topdown_contact_pose_age = torch.zeros(n, dtype=torch.long, device=env.device)
    if not hasattr(env, "_topdown_finger_center_hold"):
        env._topdown_finger_center_hold = torch.zeros(n, dtype=torch.long, device=env.device)
        env._topdown_finger_center_ready = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._topdown_finger_center_live = torch.zeros(n, dtype=torch.bool, device=env.device)


def _contact_pose_shell_raw(
    env       : "ManagerBasedRLEnv",
    in_stage_2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage-2 posture gate before finger unlock.

    This is the success shell with all contact-force requirements removed. It
    confirms the hand has descended to the contact pose, preserved the
    topdown orientation, kept opposed-face alignment, and has not displaced or
    lifted the block. The state machine latches this shell before fingers can
    unlock.
    """
    if in_stage_2 is None:
        stage = getattr(env, "_topdown_stage", None)
        if stage is None:
            in_stage_2 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        else:
            in_stage_2 = stage == 2
    palm_d = palm_distance_contact(env)
    palm_h = palm_height_error_contact(env)
    palm_drop_deg = torch.rad2deg(palm_drop_axis_error_rad(env))
    palm_yaw_deg = torch.rad2deg(palm_yaw_axis_error_rad(env))
    align_e = open_hand_alignment_error(env)
    opposed = opposite_face_gate(env)
    blk_disp = block_displacement(env)
    lift = block_lift_height(env)
    return (
        in_stage_2
        & (palm_d <= _SUCCESS_PALM_DIST_MAX)
        & (palm_h <= _SUCCESS_PALM_HEIGHT_MAX)
        & (palm_drop_deg <= _SUCCESS_PALM_ORIENT_MAX_DEG)
        & (palm_yaw_deg <= _SUCCESS_PALM_YAW_MAX_DEG)
        & (align_e <= _SUCCESS_ALIGN_ERR_MAX)
        & (opposed >= _SUCCESS_OPPOSED_GATE_MIN)
        & (blk_disp <= _CONTACT_BLOCK_DISP_MAX)
        & (lift <= _CONTACT_LIFT_MAX)
    )


def _contact_pose_ready_fallback_shell_raw(
    env       : "ManagerBasedRLEnv",
    in_stage_2: torch.Tensor,
) -> torch.Tensor:
    """Looser fallback gate for exposing partial finger unlock.

    This is intentionally tighter than Stage-2 retention and still requires
    opposed-face geometry. It avoids latching finger unlock from generic
    Stage-2 drift poses where opening/closing fingers destabilizes the already
    learned reach/align behavior.
    """
    palm_d = palm_distance_contact(env)
    palm_h = palm_height_error_contact(env)
    palm_drop_deg = torch.rad2deg(palm_drop_axis_error_rad(env))
    palm_yaw_deg = torch.rad2deg(palm_yaw_axis_error_rad(env))
    align_e = open_hand_alignment_error(env)
    opposed = opposite_face_gate(env)
    return (
        in_stage_2
        & (palm_d <= _CONTACT_POSE_READY_FALLBACK_PALM_DIST_MAX)
        & (palm_h <= _CONTACT_POSE_READY_FALLBACK_PALM_HEIGHT_MAX)
        & (palm_drop_deg <= _CONTACT_POSE_READY_FALLBACK_PALM_ORIENT_MAX_DEG)
        & (palm_yaw_deg <= _CONTACT_POSE_READY_FALLBACK_PALM_YAW_MAX_DEG)
        & (align_e <= _CONTACT_POSE_READY_FALLBACK_ALIGN_ERR_MAX)
        & (opposed >= _CONTACT_POSE_READY_FALLBACK_OPPOSED_GATE_MIN)
        & (block_displacement(env) <= _CONTACT_BLOCK_DISP_MAX)
        & (block_lift_height(env) <= _CONTACT_LIFT_MAX)
    )


def _stage2_retention_shell_raw(
    env       : "ManagerBasedRLEnv",
    in_stage_2: torch.Tensor,
) -> torch.Tensor:
    """Loose shell that keeps Stage 2 latched while the policy recovers.

    The strict contact-pose shell can flicker during descent and finger
    closing. This wider gate only decides whether a long-running Stage-2 env
    has drifted so far away that contact rewards should stop applying. Any
    actual hand contact keeps Stage 2 active so contact transients are not
    cleared mid-grasp.
    """
    palm_d = palm_distance_contact(env)
    palm_h = palm_height_error_contact(env)
    palm_drop_deg = torch.rad2deg(palm_drop_axis_error_rad(env))
    palm_yaw_deg = torch.rad2deg(palm_yaw_axis_error_rad(env))
    align_e = open_hand_alignment_error(env)
    contact = any_hand_contact_strength(env)
    loose_shell = (
        (palm_d <= _STAGE2_FALLOUT_PALM_DIST_MAX)
        & (palm_h <= _STAGE2_FALLOUT_PALM_HEIGHT_MAX)
        & (palm_drop_deg <= _STAGE2_FALLOUT_PALM_ORIENT_MAX_DEG)
        & (palm_yaw_deg <= _STAGE2_FALLOUT_PALM_YAW_MAX_DEG)
        & (align_e <= _STAGE2_FALLOUT_ALIGN_ERR_MAX)
    )
    has_contact = contact > _ALIGN_NO_CONTACT_MAX
    return in_stage_2 & (loose_shell | has_contact)


def _step_token(env: "ManagerBasedRLEnv") -> tuple[int, torch.Tensor]:
    """Cache token = (common_step, episode_length_buf snapshot)."""
    common = int(getattr(env, "common_step_counter", 0))
    return common, env.episode_length_buf.detach().clone()


def lift_drop_from_max_bad(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """True when lift falls too far from this episode's best lift.

    This catches the failure mode where the policy earns lift reward for a
    twisted/off-center grasp, then drops the block and keeps collecting replay
    from a bad terminal tail. Disabled unless
    TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX is positive.
    """
    n = env.num_envs
    common, ep_buf = _step_token(env)
    cached_common = getattr(env, "_topdown_lift_drop_last_common_step", None)
    cached_ep = getattr(env, "_topdown_lift_drop_last_ep_buf", None)
    cached_bad = getattr(env, "_topdown_lift_drop_from_max_bad", None)
    if (
        cached_common == common
        and cached_ep is not None
        and cached_ep.shape == ep_buf.shape
        and cached_bad is not None
        and cached_bad.shape == (n,)
        and bool(torch.equal(cached_ep, ep_buf))
    ):
        return cached_bad

    lift = block_lift_height(env)
    max_lift = getattr(env, "_topdown_episode_max_lift_height", None)
    hold = getattr(env, "_topdown_lift_drop_from_max_hold", None)
    if max_lift is None or max_lift.shape != lift.shape:
        max_lift = lift.detach().clone()
    if hold is None or hold.shape != lift.shape:
        hold = torch.zeros(n, dtype=torch.long, device=env.device)

    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        max_lift = torch.where(just_reset, lift.detach(), max_lift)
        hold = torch.where(just_reset, torch.zeros_like(hold), hold)

    max_lift = torch.maximum(max_lift, lift.detach())
    drop = torch.clamp(max_lift - lift, min=0.0)

    threshold = _env_float("TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX", 0.0)
    min_peak = _env_float("TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK", 0.035)
    hold_steps = max(_env_int("TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS", 2), 1)
    if threshold > 0.0:
        raw_bad = (max_lift >= min_peak) & (drop >= threshold)
    else:
        raw_bad = torch.zeros(n, dtype=torch.bool, device=env.device)
    hold = torch.where(raw_bad, hold + 1, torch.zeros_like(hold))
    if just_reset.any():
        hold = torch.where(just_reset, torch.zeros_like(hold), hold)
    bad = hold >= hold_steps

    env._topdown_episode_max_lift_height = max_lift.detach()
    env._topdown_lift_drop_from_max = drop.detach()
    env._topdown_lift_drop_from_max_hold = hold
    env._topdown_lift_drop_from_max_bad = bad
    env._topdown_lift_drop_last_common_step = common
    env._topdown_lift_drop_last_ep_buf = ep_buf
    return bad


def ensure_curriculum_stage_updated(env: "ManagerBasedRLEnv") -> None:
    """Idempotent per-step stage update. Safe to call from any term.

    Reward, observation, termination, and teacher code all call this helper.
    The cached common-step token prevents multiple updates in one simulation
    frame from advancing hold counters more than once.  Any new term that needs
    stage-dependent state should call this function rather than reading or
    mutating the buffers directly.
    
    Steps:
    1. Check if the current step token matches the cached one. If it does, return immediately to avoid redundant updates.
    2. If the episode has just reset, initialize all stage-related buffers to their default values.
    3. Compute the current palm distance, height error, orientation errors, alignment error, line angle, opposed-face gate, and contact strength.
    4. Determine if the conditions for transitioning from stage 0 to stage 1 (reach_now) are met based on the computed values and predefined thresholds.
    5. Determine if the conditions for transitioning from stage 1 to stage 2 (align_now) are met, which may include additional checks for opposed-face alignment and contact strength.
    6. If the configuration requires it, check for the finger-center gate as well and incorporate it into the align_now condition.
    7. Update the stage buffer based on the reach_now and align_now conditions, and manage hold counters and latches accordingly.
    
    
    
    """
    _ensure_state_buffers(env)
    common, ep_buf = _step_token(env)
    cached_common = getattr(env, "_topdown_stage_last_common_step", None)
    cached_ep = getattr(env, "_topdown_stage_last_ep_buf", None)
    if (
        cached_common == common
        and cached_ep is not None
        and cached_ep.shape == ep_buf.shape
        and bool(torch.equal(cached_ep, ep_buf))
    ):
        return

    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_stage[just_reset] = 0
        env._topdown_reach_hold[just_reset] = 0
        env._topdown_align_hold[just_reset] = 0
        env._topdown_stage2_age[just_reset] = 0
        env._topdown_stage2_fallout_hold[just_reset] = 0
        env._topdown_contact_pose_hold[just_reset] = 0
        env._topdown_contact_pose_ready[just_reset] = False
        env._topdown_contact_pose_age[just_reset] = 0
        env._topdown_finger_unlock_progress[just_reset] = 0.0
        env._topdown_raw_finger_unlock_progress[just_reset] = 0.0
        env._topdown_finger_center_hold[just_reset] = 0
        env._topdown_finger_center_ready[just_reset] = False
        env._topdown_finger_center_live[just_reset] = False
        env.reach_align_finger_unlocked[just_reset] = False

    palm_d = palm_distance(env)
    palm_h = palm_height_error(env)
    palm_drop_deg = torch.rad2deg(palm_drop_axis_error_rad(env))
    palm_yaw_deg = torch.rad2deg(palm_yaw_axis_error_rad(env))
    align_e = open_hand_alignment_error(env)
    line_deg = torch.rad2deg(fingertip_line_angle_rad(env))
    opposed = opposite_face_gate(env)
    contact = any_hand_contact_strength(env)

    # Stage 0 -> 1 is a broad "the hand is in the neighborhood" latch.  It
    # should be reachable early enough that the prehold teacher can keep
    # improving pose without being trapped in pure reach shaping.
    reach_now = (
        (palm_d <= _STAGE1_PALM_DIST_MAX)
        & (palm_h <= _STAGE1_PALM_HEIGHT_MAX)
        & (palm_drop_deg <= _STAGE1_PALM_ORIENT_MAX_DEG)
        & (palm_yaw_deg <= _STAGE1_PALM_YAW_MAX_DEG)
        & (align_e <= _STAGE1_ALIGN_ERR_MAX)
        & (line_deg <= _STAGE1_LINE_ANGLE_MAX_DEG)
        & (opposed >= _STAGE1_OPPOSED_GATE_MIN)
        & (contact <= _STAGE1_NO_CONTACT_MAX)
    )
    stage2_opposed_ok = opposed >= _STAGE2_OPPOSED_GATE_MIN
    if os.environ.get("TOPDOWN_STAGE2_CENTER_BYPASSES_OPPOSED", "0") == "1":
        # New centered-descent teachers use the live finger-pocket gate as the
        # pre-hover contract. Requiring the historical opposed-face scalar here
        # keeps the hand low while it twists into a front/back orientation and
        # causes early block knocks before descent has actually begun.
        stage2_opposed_ok = torch.ones_like(stage2_opposed_ok, dtype=torch.bool)

    # Stage 1 -> 2 is the contact-attempt latch.  Profiles that use a centered
    # descent can require the live finger-center gate below; profiles that use
    # the historical opposed-face scalar can keep the stricter orientation/yaw
    # shell.  This separation is why stage gates and finger-centering gates are
    # configured independently.
    align_now = (
        (palm_d <= _STAGE2_PALM_DIST_MAX)
        & (palm_h <= _STAGE2_PALM_HEIGHT_MAX)
        & (palm_drop_deg <= _STAGE2_PALM_ORIENT_MAX_DEG)
        & (palm_yaw_deg <= _STAGE2_PALM_YAW_MAX_DEG)
        & (align_e <= _STAGE2_ALIGN_ERR_MAX)
        & (line_deg <= _STAGE2_LINE_ANGLE_MAX_DEG)
        & stage2_opposed_ok
        & (contact <= _STAGE2_NO_CONTACT_MAX)
    )
    if os.environ.get("TOPDOWN_STAGE2_REQUIRES_FINGER_CENTER", "0") == "1":
        if _FINGER_CENTER_USE_XYZ_GATE:
            xyz_gate = (
                finger_xyz_block_center_gate(
                    env,
                    write_diagnostics=True,
                    start_m_override=(
                        _STAGE2_FINGER_XYZ_GATE_START_M
                        if _STAGE2_FINGER_XYZ_GATE_START_M >= 0.0
                        else None
                    ),
                    full_m_override=(
                        _STAGE2_FINGER_XYZ_GATE_FULL_M
                        if _STAGE2_FINGER_XYZ_GATE_FULL_M >= 0.0
                        else None
                    ),
                )
                > _FINGER_CENTER_XYZ_GATE_MIN
            )
            align_now = align_now & xyz_gate
            if _FINGER_CENTER_ALIGN_ANGLE_MAX_DEG > 0.0:
                align_now = align_now & (line_deg <= _FINGER_CENTER_ALIGN_ANGLE_MAX_DEG)
        else:
            pre_stage2_center_xy, pre_stage2_center_max_xy, pre_stage2_center_z = (
                finger_unlock_center_errors(env)
            )
            pre_stage2_center_angle = line_deg
            if _FINGER_CENTER_TIP_XY_MAX > 0.0:
                align_now = align_now & (pre_stage2_center_xy <= _FINGER_CENTER_TIP_XY_MAX)
            if _FINGER_CENTER_MAX_TIP_XY_MAX > 0.0:
                align_now = align_now & (pre_stage2_center_max_xy <= _FINGER_CENTER_MAX_TIP_XY_MAX)
            if _FINGER_CENTER_TIP_Z_MAX > 0.0:
                align_now = align_now & (pre_stage2_center_z <= _FINGER_CENTER_TIP_Z_MAX)
            if _FINGER_CENTER_ALIGN_ANGLE_MAX_DEG > 0.0:
                align_now = align_now & (
                    pre_stage2_center_angle <= _FINGER_CENTER_ALIGN_ANGLE_MAX_DEG
                )
            if _FINGER_CENTER_ALIGN_ERR_MAX > 0.0:
                align_now = align_now & (align_e <= _FINGER_CENTER_ALIGN_ERR_MAX)

    env._topdown_reach_hold = torch.where(
        reach_now,
        env._topdown_reach_hold + 1,
        torch.zeros_like(env._topdown_reach_hold),
    )
    env._topdown_align_hold = torch.where(
        align_now,
        env._topdown_align_hold + 1,
        torch.zeros_like(env._topdown_align_hold),
    )

    advance_to_1 = (env._topdown_stage == 0) & (env._topdown_reach_hold >= _REACH_HOLD_STEPS)
    advance_to_2 = (env._topdown_stage == 1) & (env._topdown_align_hold >= _ALIGN_HOLD_STEPS)
    new_stage = env._topdown_stage.clone()
    new_stage = torch.where(advance_to_1, torch.full_like(new_stage, 1), new_stage)
    new_stage = torch.where(advance_to_2, torch.full_like(new_stage, 2), new_stage)
    env._topdown_stage = torch.maximum(env._topdown_stage, new_stage)

    in_stage_2 = env._topdown_stage == 2
    can_fallout = in_stage_2 & (env._topdown_stage2_age >= _STAGE2_FALLOUT_GRACE_STEPS)
    retained_stage_2 = _stage2_retention_shell_raw(env, in_stage_2)
    env._topdown_stage2_fallout_hold = torch.where(
        can_fallout & (~retained_stage_2),
        env._topdown_stage2_fallout_hold + 1,
        torch.zeros_like(env._topdown_stage2_fallout_hold),
    )
    stage2_fallout = can_fallout & (
        env._topdown_stage2_fallout_hold >= _STAGE2_FALLOUT_HOLD_STEPS
    )
    if bool(stage2_fallout.any().item()):
        env._topdown_stage = torch.where(
            stage2_fallout,
            torch.ones_like(env._topdown_stage),
            env._topdown_stage,
        )
        env._topdown_stage2_fallout_hold = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_stage2_fallout_hold),
            env._topdown_stage2_fallout_hold,
        )
        env._topdown_contact_pose_hold = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_contact_pose_hold),
            env._topdown_contact_pose_hold,
        )
        env._topdown_contact_pose_ready = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_contact_pose_ready),
            env._topdown_contact_pose_ready,
        )
        env._topdown_contact_pose_age = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_contact_pose_age),
            env._topdown_contact_pose_age,
        )
        env._topdown_finger_unlock_progress = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_finger_unlock_progress),
            env._topdown_finger_unlock_progress,
        )
        env._topdown_raw_finger_unlock_progress = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_raw_finger_unlock_progress),
            env._topdown_raw_finger_unlock_progress,
        )
        env._topdown_finger_center_hold = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_finger_center_hold),
            env._topdown_finger_center_hold,
        )
        env._topdown_finger_center_ready = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_finger_center_ready),
            env._topdown_finger_center_ready,
        )
        env._topdown_finger_center_live = torch.where(
            stage2_fallout,
            torch.zeros_like(env._topdown_finger_center_live),
            env._topdown_finger_center_live,
        )
        env.reach_align_finger_unlocked = torch.where(
            stage2_fallout,
            torch.zeros_like(env.reach_align_finger_unlocked),
            env.reach_align_finger_unlocked,
        )
        # Contact-teacher latch/fraction reset on stage demotion. Without this,
        # ``contact_protect`` in compute_topdown_contact_teacher_parts (which
        # preserves close progress through transient gate flicker) would leak
        # stale latch state from the prior stage-2 attempt back into a fresh
        # stage-2 entry, freezing closure on geometry the env no longer touches.
        # State is owned by the trainer; getattr guards make this no-op when
        # the contact-teacher buffers have not yet been initialized.
        for _attr in (
            "_topdown_contact_teacher_thumb_latched",
            "_topdown_contact_teacher_index_latched",
            "_topdown_contact_teacher_middle_latched",
            "_topdown_contact_teacher_descent_started",
            "_topdown_contact_teacher_thumb_fraction",
            "_topdown_contact_teacher_index_fraction",
            "_topdown_contact_teacher_middle_fraction",
        ):
            _tensor = getattr(env, _attr, None)
            if torch.is_tensor(_tensor) and _tensor.shape == stage2_fallout.shape:
                setattr(env, _attr, torch.where(stage2_fallout, torch.zeros_like(_tensor), _tensor))
        for _attr in (
            "_topdown_contact_teacher_thumb_hold_fraction",
            "_topdown_contact_teacher_index_hold_fraction",
            "_topdown_contact_teacher_middle_hold_fraction",
        ):
            _tensor = getattr(env, _attr, None)
            if torch.is_tensor(_tensor) and _tensor.shape == stage2_fallout.shape:
                setattr(
                    env,
                    _attr,
                    torch.where(stage2_fallout, torch.full_like(_tensor, -1.0), _tensor),
                )

    in_stage_2 = env._topdown_stage == 2
    env._topdown_stage2_age = torch.where(
        in_stage_2,
        env._topdown_stage2_age + 1,
        torch.zeros_like(env._topdown_stage2_age),
    )
    contact_pose_now = _contact_pose_shell_raw(env, in_stage_2)
    env._topdown_contact_pose_hold = torch.where(
        contact_pose_now,
        env._topdown_contact_pose_hold + 1,
        torch.zeros_like(env._topdown_contact_pose_hold),
    )
    contact_pose_latched = env._topdown_contact_pose_hold >= _CONTACT_POSE_HOLD_STEPS
    if _CONTACT_POSE_READY_FALLBACK_STEPS > 0:
        # Exposure mode: if the policy can remain in Stage 2 inside the loose
        # fallback shell but never hits the strict contact-pose latch, unlock
        # fingers after a dwell so replay contains actual contact attempts.
        fallback_pose_latched = (
            (env._topdown_stage2_age >= _CONTACT_POSE_READY_FALLBACK_STEPS)
            & _contact_pose_ready_fallback_shell_raw(env, in_stage_2)
        )
        contact_pose_latched = contact_pose_latched | fallback_pose_latched
    contact_pose_latched = in_stage_2 & contact_pose_latched
    env._topdown_contact_pose_ready = torch.where(
        in_stage_2,
        env._topdown_contact_pose_ready | contact_pose_latched,
        torch.zeros_like(env._topdown_contact_pose_ready),
    )
    env._topdown_contact_pose_age = torch.where(
        env._topdown_contact_pose_ready,
        env._topdown_contact_pose_age + 1,
        torch.zeros_like(env._topdown_contact_pose_age),
    )
    raw_progress = (
        env._topdown_contact_pose_age.float()
        / max(float(_FINGER_UNLOCK_RAMP_STEPS), 1.0)
    ).clamp(0.0, min(max(_FINGER_UNLOCK_MAX_PROGRESS, 0.0), 1.0))
    raw_progress = torch.where(
        env._topdown_contact_pose_ready, raw_progress, torch.zeros_like(raw_progress)
    )
    env._topdown_raw_finger_unlock_progress = raw_progress

    center_xy, center_max_xy, center_z = finger_unlock_center_errors(env)
    center_angle_deg = torch.rad2deg(fingertip_line_angle_rad(env))
    center_live = in_stage_2
    if os.environ.get("CURRICULUM_FINGER_CENTER_REQUIRES_CONTACT_POSE", "1") == "1":
        center_live = center_live & env._topdown_contact_pose_ready
    if _FINGER_CENTER_USE_XYZ_GATE:
        center_live = center_live & (
            finger_xyz_block_center_gate(env, write_diagnostics=True)
            > _FINGER_CENTER_XYZ_GATE_MIN
        )
    else:
        if _FINGER_CENTER_TIP_XY_MAX > 0.0:
            center_live = center_live & (center_xy <= _FINGER_CENTER_TIP_XY_MAX)
        if _FINGER_CENTER_MAX_TIP_XY_MAX > 0.0:
            center_live = center_live & (center_max_xy <= _FINGER_CENTER_MAX_TIP_XY_MAX)
        if _FINGER_CENTER_TIP_Z_MAX > 0.0:
            center_live = center_live & (center_z <= _FINGER_CENTER_TIP_Z_MAX)
    if _FINGER_CENTER_ALIGN_ANGLE_MAX_DEG > 0.0:
        center_live = center_live & (center_angle_deg <= _FINGER_CENTER_ALIGN_ANGLE_MAX_DEG)
    if _FINGER_CENTER_ALIGN_ERR_MAX > 0.0:
        center_live = center_live & (open_hand_alignment_error(env) <= _FINGER_CENTER_ALIGN_ERR_MAX)
    env._topdown_finger_center_live = center_live.detach().clone()
    env._topdown_finger_center_xy_err = center_xy.detach().clone()
    env._topdown_finger_center_max_xy_err = center_max_xy.detach().clone()
    env._topdown_finger_center_z_err = center_z.detach().clone()
    env._topdown_finger_center_align_angle_deg = center_angle_deg.detach().clone()
    env._topdown_finger_center_hold = torch.where(
        center_live,
        env._topdown_finger_center_hold + 1,
        torch.zeros_like(env._topdown_finger_center_hold),
    )
    center_debounced = env._topdown_finger_center_hold >= max(_FINGER_CENTER_HOLD_STEPS, 1)
    if _FINGER_CENTER_LATCH:
        env._topdown_finger_center_ready = torch.where(
            in_stage_2,
            env._topdown_finger_center_ready | center_debounced,
            torch.zeros_like(env._topdown_finger_center_ready),
        )
    else:
        env._topdown_finger_center_ready = center_debounced
    center_gate = env._topdown_finger_center_ready

    close_gate_mode = _finger_close_gate_mode()
    if close_gate_mode in {"xyz", "xyz_front"}:
        # xyz is a geometry gate, but curriculum finger unlock must still be
        # stage-scoped. Without this, a centered-ish stage-0 approach unlocks
        # the 20% preloaded fingers before the pre-descent hover has latched,
        # producing the thumb-top graze/restart loop.
        progress = finger_xyz_block_center_gate(env, write_diagnostics=True) * in_stage_2.to(
            dtype=raw_progress.dtype
        )
        if _FINGER_UNLOCK_REQUIRES_CENTER:
            progress = progress * center_gate.to(dtype=raw_progress.dtype)
    elif _FINGER_UNLOCK_REQUIRES_CENTER:
        progress = raw_progress * center_gate.to(dtype=raw_progress.dtype)
    else:
        progress = raw_progress
    env._topdown_finger_unlock_progress = progress
    if close_gate_mode in {"xyz", "xyz_front"}:
        env.reach_align_finger_unlocked = progress > 1.0e-6
    else:
        env.reach_align_finger_unlocked = env._topdown_contact_pose_ready & (
            center_gate if _FINGER_UNLOCK_REQUIRES_CENTER else torch.ones_like(center_gate)
        )

    env._topdown_stage_last_common_step = common
    env._topdown_stage_last_ep_buf = ep_buf


# --- Stage indicators (cheap helpers for reward routing) ---------------------


def stage_is(env: "ManagerBasedRLEnv", stage: int) -> torch.Tensor:
    """Return a mask for environments currently in the requested curriculum stage."""
    ensure_curriculum_stage_updated(env)
    return (env._topdown_stage == stage).float()


def stage_at_least(env: "ManagerBasedRLEnv", stage: int) -> torch.Tensor:
    """Return a mask for environments that have reached at least the requested stage."""
    ensure_curriculum_stage_updated(env)
    return (env._topdown_stage >= stage).float()


def stage_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return a one-hot encoding of the curriculum stage."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    out = torch.zeros((n, 3), device=env.device)
    s = env._topdown_stage
    out[torch.arange(n, device=env.device), s.clamp(0, 2)] = 1.0
    return out


def finger_unlock_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the curriculum ramp that allows fingers to close."""
    ensure_curriculum_stage_updated(env)
    return env._topdown_finger_unlock_progress.unsqueeze(-1)


def contact_pose_shell_now(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Instantaneous Stage-2 contact-pose shell predicate, excluding forces."""
    ensure_curriculum_stage_updated(env)
    return _contact_pose_shell_raw(env, env._topdown_stage == 2)


def stage2_warmup_factor(
    env: "ManagerBasedRLEnv", ramp_steps: int | None = None
) -> torch.Tensor:
    """Linear [0, 1] ramp after the contact-pose shell latches; 0 before it.

    Used to ramp Stage 2 penalties in over the early window so the policy
    isn't punished for its first contact attempt before it has a chance to
    learn bilateral closure.
    """
    ensure_curriculum_stage_updated(env)
    steps = float(ramp_steps if ramp_steps is not None else _ONE_SIDED_RAMP_STEPS)
    progress = (env._topdown_contact_pose_age.float() / max(steps, 1.0)).clamp(0.0, 1.0)
    return torch.where(
        env._topdown_contact_pose_ready, progress, torch.zeros_like(progress)
    )


# --- Stage 2 success predicate -----------------------------------------------


def light_contact_success_now(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return whether the light-contact success gate is currently satisfied."""
    ensure_curriculum_stage_updated(env)
    in_stage_2 = env._topdown_stage == 2
    opp_strength = opposed_contact_strength(env)
    pose_ready = env._topdown_contact_pose_ready & in_stage_2
    base_success = (
        pose_ready
        & (opp_strength >= _CONTACT_OPPOSED_STRENGTH_MIN)
    )
    env._topdown_light_contact_success_base = base_success.detach().clone()
    if _SUCCESS_REQUIRE_CENTERED_CONTACT:
        return base_success & centered_contact_shell_now(env)
    centered_contact_shell_now(env)
    return base_success


def light_contact_success_held(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Debounced: returns True only after success_now held for HOLD steps."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_success_hold"):
        env._topdown_success_hold = torch.zeros(n, dtype=torch.long, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_success_hold[just_reset] = 0
    now = light_contact_success_now(env)
    env._topdown_success_hold = torch.where(
        now,
        env._topdown_success_hold + 1,
        torch.zeros_like(env._topdown_success_hold),
    )
    hold_steps = _CONTACT_HOLD_STEPS
    if _SUCCESS_REQUIRE_CENTERED_CONTACT:
        hold_steps = max(hold_steps, _SUCCESS_CENTER_HOLD_STEPS)
    return env._topdown_success_hold >= hold_steps


def lift_success_now(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Physical lift success: live contact remains while the block rises."""
    lift_ok = block_lift_height(env) >= _LIFT_SUCCESS_HEIGHT
    if _LIFT_SUCCESS_MODE in {"height", "height_only", "lift", "lift_only", "off_table"}:
        ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._topdown_lift_success_center_ok = ok.detach().clone()
        env._topdown_lift_success_tilt_ok = ok.detach().clone()
        return lift_ok

    if _LIFT_SUCCESS_REQUIRES_CONTACT:
        if _LIFT_SUCCESS_CONTACT_MODE in {"finger", "fingertip", "fingertips", "any_finger"}:
            ensure_curriculum_stage_updated(env)
            contact_ok = any_fingertip_contact_strength(env) >= _LIFT_SUCCESS_CONTACT_MIN
        elif _LIFT_SUCCESS_CONTACT_MODE in {"hand", "any", "any_hand"}:
            ensure_curriculum_stage_updated(env)
            contact_ok = any_hand_contact_strength(env) >= _LIFT_SUCCESS_CONTACT_MIN
        elif _LIFT_SUCCESS_CONTACT_MODE in {"opposed", "thumb_index", "thumb-index", "pinch"}:
            ensure_curriculum_stage_updated(env)
            contact_ok = opposed_contact_strength(env) >= _LIFT_SUCCESS_CONTACT_MIN
        else:
            contact_ok = light_contact_success_now(env)
    else:
        contact_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    drift_ok = block_xy_displacement(env) <= _LIFT_SUCCESS_XY_DRIFT_MAX
    if _SUCCESS_REQUIRE_CENTERED_CONTACT:
        center_ok = centered_contact_shell_now(env)
    else:
        center_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    tilt_max_deg = _env_float("TOPDOWN_LIFT_SUCCESS_BLOCK_TILT_MAX_DEG", 0.0)
    if tilt_max_deg > 0.0:
        tilt_ok = block_tilt_angle_rad(env) <= math.radians(tilt_max_deg)
    else:
        tilt_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._topdown_lift_success_center_ok = center_ok.detach().clone()
    env._topdown_lift_success_tilt_ok = tilt_ok.detach().clone()
    return contact_ok & lift_ok & drift_ok & center_ok & tilt_ok


def lift_success_held(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Debounced lift success for dynamic-block lift training."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_lift_success_hold"):
        env._topdown_lift_success_hold = torch.zeros(n, dtype=torch.long, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_lift_success_hold[just_reset] = 0
    now = lift_success_now(env)
    env._topdown_lift_success_hold = torch.where(
        now,
        env._topdown_lift_success_hold + 1,
        torch.zeros_like(env._topdown_lift_success_hold),
    )
    return env._topdown_lift_success_hold >= _LIFT_SUCCESS_HOLD_STEPS
