"""

Rollout drift and contact diagnostic helpers

File map:

CONTACT_SENSOR_KEYS:     Define contact sensor keys constant
block_xy_drift:          Return block XY offset from spawn when available
palm_to_block_xy:        Return palm XY offset from block XY
signed_finger_curl_sum:  Return signed curl sum for one finger joint prefix
contact_force_slots:     Return per-sensor force diagnostics with zero fallbacks
finger_curl_slots:       Return thumb index and middle curl diagnostic slots
drift_instrumentation:   Return JSON-safe rollout drift and force diagnostics
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


CONTACT_SENSOR_KEYS = (
    ("thumb_contact", "thumb_tip_N"),
    ("index_contact", "index_tip_N"),
    ("middle_contact", "middle_tip_N"),
    ("palm_contact", "palm_N"),
    ("thumb_prox_contact", "thumb_prox_N"),
    ("index_prox_contact", "index_prox_N"),
    ("middle_prox_contact", "middle_prox_N"),
)


def block_xy_drift(env, env_id: int = 0, *, spawn_attr: str = "_grasp_block_spawn_pos") -> list[float]:
    """Return block XY offset from spawn when available"""
    try:
        obj = env.scene["object"]
        block_pos = obj.data.root_pos_w[int(env_id), :3].detach().cpu()
        if hasattr(env, spawn_attr):
            spawn = getattr(env, spawn_attr)[int(env_id), :3].detach().cpu()
            return [float(block_pos[0] - spawn[0]), float(block_pos[1] - spawn[1])]
    except (KeyError, AttributeError, IndexError):
        pass
    return [0.0, 0.0]


def palm_to_block_xy(env, env_id: int = 0, *, palm_link_name: str = "right_hand_palm_link") -> list[float]:
    """Return palm XY offset from block XY"""
    try:
        robot = env.scene["robot"]
        palm_idx = robot.data.body_names.index(palm_link_name)
        palm_xy = robot.data.body_link_pose_w[int(env_id), palm_idx, :2].detach().cpu()
        block_xy = env.scene["object"].data.root_pos_w[int(env_id), :2].detach().cpu()
        return [float(palm_xy[0] - block_xy[0]), float(palm_xy[1] - block_xy[1])]
    except (KeyError, ValueError, AttributeError, IndexError):
        return [0.0, 0.0]


def signed_finger_curl_sum(
    joint_names  : Sequence[str],  # Param: ordered candidate names used to resolve joint
    joint_pos_row: torch.Tensor,  # Param: tensor input carrying joint pos row values
    prefix       : str,  # Param: string input for prefix
) -> float:
    """Return signed curl sum for one finger joint prefix"""
    total = 0.0
    for joint_idx, joint_name in enumerate(joint_names):
        if not str(joint_name).startswith(prefix):
            continue
        direction = 1.0 if "thumb" not in joint_name or str(joint_name).endswith("thumb_0_joint") else -1.0
        total += direction * float(joint_pos_row[joint_idx].item())
    return total


def contact_force_slots(
    env,                                                                    # Param: environment or backend object used for runtime calls
    env_id                 : int,  # Param: integer input for env id
    contact_force_magnitude: Callable[[object, str], torch.Tensor] | None,  # Param: callback used to compute or fetch contact force magnitude
    *,
    sensor_keys: tuple[tuple[str, str], ...] = CONTACT_SENSOR_KEYS,         # Param: ordered mapping keys used to resolve sensor
) -> dict[str, float]:
    """Return per-sensor force diagnostics with zero fallbacks"""
    out: dict[str, float] = {}
    for sensor_name, key in sensor_keys:
        try:
            if contact_force_magnitude is None:
                raise KeyError(sensor_name)
            out[key] = float(contact_force_magnitude(env, sensor_name)[int(env_id)].item())
        except (KeyError, IndexError, AttributeError):
            out[key] = 0.0
    return out


def finger_curl_slots(env, env_id: int = 0) -> dict[str, float]:
    """Return thumb index and middle curl diagnostic slots"""
    try:
        robot = env.scene["robot"]
        joint_names = tuple(robot.data.joint_names)
        joint_pos_row = robot.data.joint_pos[int(env_id)]
        return {
            "thumb_curl" : signed_finger_curl_sum(joint_names, joint_pos_row, "right_hand_thumb_"),
            "index_curl" : signed_finger_curl_sum(joint_names, joint_pos_row, "right_hand_index_"),
            "middle_curl": signed_finger_curl_sum(joint_names, joint_pos_row, "right_hand_middle_"),
        }
    except (KeyError, AttributeError, ValueError, IndexError):
        return {"thumb_curl": 0.0, "index_curl": 0.0, "middle_curl": 0.0}


def drift_instrumentation(
    env,                                                                           # Param: environment or backend object used for runtime calls
    env_id: int = 0,                                                               # Param: integer input for env id
    *,
    contact_force_magnitude: Callable[[object, str], torch.Tensor] | None = None,  # Param: callback used to compute or fetch contact force magnitude
) -> dict[str, object]:
    """Return JSON-safe rollout drift and force diagnostics"""
    out: dict[str, object] = {
        "block_xy_drift"  : block_xy_drift(env, env_id),
        "palm_to_block_xy": palm_to_block_xy(env, env_id),
    }
    out.update(contact_force_slots(env, env_id, contact_force_magnitude))
    out.update(finger_curl_slots(env, env_id))
    return out
