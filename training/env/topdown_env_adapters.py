"""

Import-safe adapters for topdown env state-machine reads

File map:

_STATE_INT_ATTRS:                      Define state int attrs constant
_STATE_FLOAT_ATTRS:                    Define state float attrs constant
_STATE_VEC3_ATTRS:                     Define state vec3 attrs constant
_scalar_attr:                          Handle scalar attr logic
_vec3_attr:                            Handle vec3 attr logic
_attr_tensor:                          Handle attr tensor logic
_call_state_machine:                   Handle call state machine logic
thumb_contact_strength:                Read topdown thumb contact strength
index_contact_strength:                Read topdown index contact strength
fingertip_contact_strength:            Read any topdown fingertip contact strength
hand_block_contact_strength:           Read any topdown hand contact strength
opposite_face_gate:                    Read topdown opposed-face geometry gate
topdown_light_opposed_contact_gate:    Read topdown strict light-contact opposed gate
uses_topdown_grip_targets:             Return whether env config selects topdown grip targets
topdown_teacher_palm_target_position:  Return palm target from the configured topdown grip target
topdown_curriculum_state:              Return one-env topdown curriculum diagnostic snapshot
topdown_curriculum_batch_metrics:      Aggregate topdown curriculum env attrs over active rows
phase1_palm_height_error_for_task:     Select the Phase 1 palm-height error for the active task
thumb_index_diagnostic_errors:         Return thumb and index diagnostic errors for current target mode
open_hand_alignment_error_for_task:    Select the open-hand alignment metric for the active task
opposite_face_gate_for_task:           Select the opposed-face gate for the active task
topdown_curriculum_axis_tensors:       Read topdown shell axis tensors from state-machine functions
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..geometry.contact_metrics import target_link_distances
from ..geometry.topdown_metrics import topdown_stage_metrics


_STATE_INT_ATTRS: tuple[tuple[str, str, int], ...] = (
    ("topdown_stage", "_topdown_stage", -1),
    ("topdown_source_pose_idx", "_topdown_source_pose_idx", 0),
    ("topdown_reach_hold", "_topdown_reach_hold", 0),
    ("topdown_align_hold", "_topdown_align_hold", 0),
    ("topdown_stage2_age", "_topdown_stage2_age", 0),
    ("topdown_stage2_fallout_hold", "_topdown_stage2_fallout_hold", 0),
    ("topdown_contact_pose_hold", "_topdown_contact_pose_hold", 0),
    ("topdown_contact_pose_ready", "_topdown_contact_pose_ready", 0),
    ("topdown_contact_pose_age", "_topdown_contact_pose_age", 0),
    ("topdown_finger_center_live", "_topdown_finger_center_live", 0),
    ("topdown_finger_center_ready", "_topdown_finger_center_ready", 0),
    ("topdown_finger_center_hold", "_topdown_finger_center_hold", 0),
    ("topdown_contact_teacher_thumb_latched", "_topdown_contact_teacher_thumb_latched", 0),
    ("topdown_contact_teacher_index_latched", "_topdown_contact_teacher_index_latched", 0),
    ("topdown_contact_teacher_descent_ready", "_topdown_contact_teacher_descent_ready", 0),
    ("topdown_contact_teacher_prelift_contact_loss", "_topdown_contact_teacher_prelift_contact_loss", 0),
    ("topdown_contact_teacher_hover_height_gate", "_topdown_contact_teacher_hover_height_gate", 0),
    ("topdown_contact_teacher_wrist_yaw_release_gate", "_topdown_contact_teacher_wrist_yaw_release_gate", 0),
    ("topdown_contact_teacher_ready", "_topdown_contact_teacher_ready", 0),
    ("topdown_contact_teacher_finger_ready", "_topdown_contact_teacher_finger_ready", 0),
    ("topdown_contact_teacher_center_gate", "_topdown_contact_teacher_center_gate", 0),
    ("topdown_contact_teacher_post_latch_servo_active", "_topdown_contact_teacher_post_latch_servo_active", 0),
    ("topdown_contact_teacher_one_sided_z_active", "_topdown_contact_teacher_one_sided_z_active", 0),
    ("topdown_contact_teacher_live_thumb_missing", "_topdown_contact_teacher_live_thumb_missing", 0),
    ("topdown_contact_teacher_live_index_missing", "_topdown_contact_teacher_live_index_missing", 0),
    ("topdown_contact_teacher_servo_thumb_missing", "_topdown_contact_teacher_servo_thumb_missing", 0),
    ("topdown_contact_teacher_servo_index_missing", "_topdown_contact_teacher_servo_index_missing", 0),
    ("topdown_contact_teacher_preload_one_sided_reject", "_topdown_contact_teacher_preload_one_sided_reject", 0),
    ("topdown_contact_teacher_preload_recovery_active", "_topdown_contact_teacher_preload_recovery_active", 0),
    ("topdown_contact_teacher_preload_recovery_unload_active", "_topdown_contact_teacher_preload_recovery_unload_active", 0),
    ("teacher_ik_topdown_preload_recovery_hover_active", "_teacher_ik_topdown_preload_recovery_hover_active", 0),
    ("topdown_contact_teacher_recovery_reacquire_active", "_topdown_contact_teacher_recovery_reacquire_active", 0),
    ("topdown_arm_lift_latched", "_arm_lift_latched", 0),
    ("topdown_arm_lift_contact_counter", "_arm_lift_contact_counter", 0),
)

_STATE_FLOAT_ATTRS: tuple[tuple[str, str, float], ...] = (
    ("topdown_finger_unlock_progress", "_topdown_finger_unlock_progress", 0.0),
    ("topdown_raw_finger_unlock_progress", "_topdown_raw_finger_unlock_progress", 0.0),
    ("topdown_finger_center_xy_err", "_topdown_finger_center_xy_err", 0.0),
    ("topdown_finger_center_max_xy_err", "_topdown_finger_center_max_xy_err", 0.0),
    ("topdown_finger_center_z_err", "_topdown_finger_center_z_err", 0.0),
    ("topdown_finger_center_align_angle_deg", "_topdown_finger_center_align_angle_deg", 0.0),
    ("topdown_finger_xyz_close_gate", "_topdown_finger_xyz_close_gate", 0.0),
    ("topdown_finger_xyz_error", "_topdown_finger_xyz_error", 0.0),
    ("topdown_finger_front_gate", "_topdown_finger_front_gate", 0.0),
    ("topdown_finger_thumb_front_margin", "_topdown_finger_thumb_front_margin", 0.0),
    ("topdown_finger_index_front_margin", "_topdown_finger_index_front_margin", 0.0),
    ("topdown_contact_teacher_thumb_fraction", "_topdown_contact_teacher_thumb_fraction", 0.0),
    ("topdown_contact_teacher_index_fraction", "_topdown_contact_teacher_index_fraction", 0.0),
    ("topdown_contact_teacher_thumb_hold_fraction", "_topdown_contact_teacher_thumb_hold_fraction", -1.0),
    ("topdown_contact_teacher_index_hold_fraction", "_topdown_contact_teacher_index_hold_fraction", -1.0),
    ("topdown_contact_teacher_thumb_missing", "_topdown_contact_teacher_thumb_missing", 0.0),
    ("topdown_contact_teacher_index_missing", "_topdown_contact_teacher_index_missing", 0.0),
    ("topdown_contact_teacher_descent_ready_age", "_topdown_contact_teacher_descent_ready_age", 0.0),
    ("topdown_contact_teacher_descent_z", "_topdown_contact_teacher_descent_z", 0.0),
    ("topdown_contact_teacher_descent_z_need", "_topdown_contact_teacher_descent_z_need", 0.0),
    ("topdown_contact_teacher_descent_closure_gate", "_topdown_contact_teacher_descent_closure_gate", 0.0),
    ("topdown_contact_teacher_finger_close_gate", "_topdown_contact_teacher_finger_close_gate", 0.0),
    ("topdown_contact_teacher_tip_servo_m", "_topdown_contact_teacher_tip_servo_m", 0.0),
    ("topdown_contact_teacher_post_latch_servo_m", "_topdown_contact_teacher_post_latch_servo_m", 0.0),
    ("teacher_ik_topdown_tip_servo_m", "_teacher_ik_topdown_tip_servo_m", 0.0),
    ("topdown_contact_teacher_preload_recovery_clear_age", "_topdown_contact_teacher_preload_recovery_clear_age", 0.0),
    ("topdown_contact_teacher_preload_recovery_no_contact_age", "_topdown_contact_teacher_preload_recovery_no_contact_age", 0.0),
    ("topdown_contact_teacher_recovery_reacquire_servo_m", "_topdown_contact_teacher_recovery_reacquire_servo_m", 0.0),
    ("topdown_contact_teacher_preload_reject_servo_m", "_topdown_contact_teacher_preload_reject_servo_m", 0.0),
    ("topdown_arm_lift_latch_step", "_arm_lift_latch_step", -1.0),
    ("topdown_arm_lift_latch_signal", "_arm_lift_latch_signal", 0.0),
)

_STATE_VEC3_ATTRS: tuple[tuple[str, str], ...] = (
    ("topdown_contact_teacher_tip_servo_xyz", "_topdown_contact_teacher_tip_servo"),
    ("topdown_contact_teacher_post_latch_servo_xyz", "_topdown_contact_teacher_post_latch_servo"),
    ("teacher_ik_topdown_tip_servo_xyz", "_teacher_ik_topdown_tip_servo"),
    ("topdown_contact_teacher_recovery_reacquire_servo_xyz", "_topdown_contact_teacher_recovery_reacquire_servo"),
    ("topdown_contact_teacher_preload_reject_servo_xyz", "_topdown_contact_teacher_preload_reject_servo"),
)


def _scalar_attr(env: Any, attr_name: str, env_id: int, default: float) -> float:
    tensor = getattr(env, attr_name, None)
    if not torch.is_tensor(tensor):
        return float(default)
    try:
        return float(tensor[int(env_id)].item())
    except (IndexError, RuntimeError, TypeError, ValueError):
        return float(default)


def _vec3_attr(env: Any, attr_name: str, env_id: int) -> list[float]:
    tensor = getattr(env, attr_name, None)
    if not torch.is_tensor(tensor):
        return [0.0, 0.0, 0.0]
    try:
        values = tensor[int(env_id)].detach().reshape(-1)[:3].tolist()
    except (IndexError, RuntimeError, TypeError, ValueError):
        return [0.0, 0.0, 0.0]
    return [float(value) for value in values]


def _attr_tensor(env: Any, attr_name: str, reference: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    tensor = getattr(env, attr_name, None)
    if not torch.is_tensor(tensor) or tensor.shape != reference.shape:
        selected = torch.zeros_like(reference, dtype=torch.float32)
    else:
        selected = tensor.to(device=reference.device, dtype=torch.float32)
    if mask is not None and mask.shape == reference.shape and bool(mask.any().item()):
        selected = selected[mask.to(device=reference.device, dtype=torch.bool)]
    return selected


def _call_state_machine(env: Any, state_machine_module: Any, name: str) -> torch.Tensor:
    fn = getattr(state_machine_module, name)
    return fn(env)


def thumb_contact_strength(env: Any, state_machine_module: Any) -> torch.Tensor:
    """Read topdown thumb contact strength"""
    return _call_state_machine(env, state_machine_module, "thumb_contact_strength")


def index_contact_strength(env: Any, state_machine_module: Any) -> torch.Tensor:
    """Read topdown index contact strength"""
    return _call_state_machine(env, state_machine_module, "index_contact_strength")


def fingertip_contact_strength(env: Any, state_machine_module: Any) -> torch.Tensor:
    """Read any topdown fingertip contact strength"""
    return _call_state_machine(env, state_machine_module, "any_fingertip_contact_strength")


def hand_block_contact_strength(env: Any, state_machine_module: Any) -> torch.Tensor:
    """Read any topdown hand contact strength"""
    return _call_state_machine(env, state_machine_module, "any_hand_contact_strength")


def opposite_face_gate(env: Any, state_machine_module: Any) -> torch.Tensor:
    """Read topdown opposed-face geometry gate"""
    return _call_state_machine(env, state_machine_module, "opposite_face_gate")


def topdown_light_opposed_contact_gate(env: Any, state_machine_module: Any) -> torch.Tensor:
    """Read topdown strict light-contact opposed gate"""
    return _call_state_machine(env, state_machine_module, "opposed_contact_strength")


def uses_topdown_grip_targets(env: Any, *, topdown_curriculum_task: bool) -> bool:
    """Return whether env config selects topdown grip targets"""
    cfg = getattr(env, "cfg", None)
    return bool(topdown_curriculum_task) and getattr(cfg, "phase1_target_mode", None) == "topdown_grip"


def topdown_teacher_palm_target_position(
    env: Any,                                                                     # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task       : bool,  # Param: boolean input controlling topdown curriculum task
    grip_target_position          : Callable[[Any], torch.Tensor],  # Param: callback used to compute or fetch grip target position
    palm_position_from_grip_target: Callable[[Any, torch.Tensor], torch.Tensor],  # Param: callback used to compute or fetch palm position from grip target
) -> torch.Tensor:
    """Return palm target from the configured topdown grip target"""
    if not bool(topdown_curriculum_task):
        raise RuntimeError("ENPM690 trainer supports only Isaac-Topdown-Curriculum-G129-Dex3-Joint")
    return palm_position_from_grip_target(env, grip_target_position(env))


def topdown_curriculum_state(
    env: Any,                       # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    env_id                 : int = 0,  # Param: integer input for env id
) -> dict[str, int | float | list[float]]:
    """Return one-env topdown curriculum diagnostic snapshot

    Steps:
    - Resolve inputs for `topdown_curriculum_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not bool(topdown_curriculum_task):
        return {}
    if not torch.is_tensor(getattr(env, "_topdown_stage", None)):
        return {}
    state: dict[str, int | float | list[float]] = {}
    for key, attr_name, default in _STATE_INT_ATTRS:
        state[key] = int(_scalar_attr(env, attr_name, env_id, float(default)))
    for key, attr_name, default in _STATE_FLOAT_ATTRS:
        state[key] = float(_scalar_attr(env, attr_name, env_id, float(default)))
    for key, attr_name in _STATE_VEC3_ATTRS:
        state[key] = _vec3_attr(env, attr_name, env_id)
    return state


def topdown_curriculum_batch_metrics(
    env: Any,                          # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    mask                   : torch.Tensor | None = None,  # Param: boolean mask selecting mask rows
) -> dict[str, float]:
    """Aggregate topdown curriculum env attrs over active rows"""
    if not bool(topdown_curriculum_task):
        return {}
    stage = getattr(env, "_topdown_stage", None)
    if not torch.is_tensor(stage):
        return {}
    return topdown_stage_metrics(
        stage=stage,
        mask=mask,
        source_idx=getattr(env, "_topdown_source_pose_idx", None),
        finger_unlock_progress=getattr(env, "_topdown_finger_unlock_progress", None),
        reach_hold=getattr(env, "_topdown_reach_hold", None),
        align_hold=getattr(env, "_topdown_align_hold", None),
        stage2_age=getattr(env, "_topdown_stage2_age", None),
        stage2_fallout_hold=getattr(env, "_topdown_stage2_fallout_hold", None),
        contact_pose_hold=getattr(env, "_topdown_contact_pose_hold", None),
        contact_pose_ready=getattr(env, "_topdown_contact_pose_ready", None),
        contact_pose_age=getattr(env, "_topdown_contact_pose_age", None),
    )


def phase1_palm_height_error_for_task(
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    topdown_value          : torch.Tensor,  # Param: tensor input carrying topdown value values
    fallback_value         : torch.Tensor,  # Param: tensor input carrying fallback value values
) -> torch.Tensor:
    """Select the Phase 1 palm-height error for the active task"""
    return topdown_value if bool(topdown_curriculum_task) else fallback_value


def thumb_index_diagnostic_errors(
    env: Any,                                                                              # Param: environment or backend object used for runtime calls
    *,
    uses_topdown_targets: bool,  # Param: target values for uses topdown
    thumb_pos           : torch.Tensor | None                                       = None,  # Param: tensor input carrying thumb pos values
    index_pos           : torch.Tensor | None                                       = None,  # Param: tensor input carrying index pos values
    thumb_target        : torch.Tensor | None                                       = None,  # Param: target value for thumb
    index_target        : torch.Tensor | None                                       = None,  # Param: target value for index
    fallback_distances  : Callable[[Any], tuple[torch.Tensor, torch.Tensor]] | None = None,  # Param: callback used to compute or fetch fallback distances
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return thumb and index diagnostic errors for current target mode"""
    if bool(uses_topdown_targets):
        if thumb_pos is None or index_pos is None or thumb_target is None or index_target is None:
            raise ValueError("topdown target errors require thumb/index positions and targets")
        return target_link_distances(
            thumb_pos=thumb_pos,
            index_pos=index_pos,
            thumb_target=thumb_target,
            index_target=index_target,
        )
    if fallback_distances is None:
        raise ValueError("fallback_distances is required when topdown targets are disabled")
    return fallback_distances(env)


def open_hand_alignment_error_for_task(
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    topdown_value          : torch.Tensor,  # Param: tensor input carrying topdown value values
    fallback_value         : torch.Tensor,  # Param: tensor input carrying fallback value values
) -> torch.Tensor:
    """Select the open-hand alignment metric for the active task"""
    return topdown_value if bool(topdown_curriculum_task) else fallback_value


def opposite_face_gate_for_task(
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    topdown_value          : torch.Tensor,  # Param: tensor input carrying topdown value values
    fallback_value         : torch.Tensor,  # Param: tensor input carrying fallback value values
) -> torch.Tensor:
    """Select the opposed-face gate for the active task"""
    return topdown_value if bool(topdown_curriculum_task) else fallback_value


def topdown_curriculum_axis_tensors(
    env                 : Any,  # Param: environment or backend object used for runtime calls
    state_machine_module: Any,  # Param: input value used as state machine module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read topdown shell axis tensors from state-machine functions"""
    with torch.no_grad():
        palm_d = state_machine_module.palm_distance(env)
        palm_h = state_machine_module.palm_height_error(env)
        drop_deg = torch.rad2deg(state_machine_module.palm_drop_axis_error_rad(env))
        yaw_deg = torch.rad2deg(state_machine_module.palm_yaw_axis_error_rad(env))
        spread_deg = torch.rad2deg(state_machine_module.palm_spread_axis_error_rad(env))
    return palm_d, palm_h, drop_deg, yaw_deg, spread_deg
