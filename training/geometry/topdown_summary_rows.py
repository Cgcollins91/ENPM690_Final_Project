"""

Topdown curriculum summary row builders

File map:

TopdownSummaryContext:               Loop-level fields for a topdown curriculum summary row
TopdownEnv0Summary:                  Env0 scalar fields for a topdown curriculum summary row
INPOCKET_ARM_HOLD_ATTRS:             Define inpocket arm hold attrs constant
TOPDOWN_GATE_ATTRS:                  Define topdown gate attrs constant
VERBOSE_TOPDOWN_SCALAR_ATTRS:        Define verbose topdown scalar attrs constant
VERBOSE_TOPDOWN_VEC3_ATTRS:          Define verbose topdown vec3 attrs constant
CONTACT_TEACHER_ATTRS:               Define contact teacher attrs constant
CONTACT_TEACHER_VERBOSE_ATTRS:       Define contact teacher verbose attrs constant
source_pose_index_from_tensor:       Read one source-pose index from a tensor
build_topdown_summary_row:           Build the base topdown_curriculum_summary JSON row
add_existing_tensor_scalar_attrs:    Add scalar tensor attrs only when present
add_existing_vec3_tensor_attrs:      Add vec3 tensor attrs only when present
add_topdown_summary_optional_attrs:  Add optional env tensor diagnostics to a topdown summary row
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from .topdown_metrics import topdown_source_block_name


@dataclass(frozen=True)
class TopdownSummaryContext:
    """Loop-level fields for a topdown curriculum summary row"""

    global_step            : int  # training step associated with this record or action
    episode_idx            : int  # training episode index associated with this record
    episode_step           : int  # per-env step count inside the current episode
    reward                 : float  # reward tensor or scalar produced by the environment step
    done_envs              : int  # integer done envs value tracked by topdown summary context
    replay_size            : int  # configured or observed replay-buffer size
    training_phase         : str  # string training phase value used by topdown summary context
    training_phase_id      : int  # integer training phase id value tracked by topdown summary context
    active_n_step          : int  # step count used for active n step scheduling or reporting
    active_updates_per_step: int  # step count used for active updates per step scheduling or reporting
    active_policy_delay    : int  # integer active policy delay value tracked by topdown summary context
    assist_mix             : float  # floating-point assist mix value used by topdown summary context
    assist_arm_mix         : float  # floating-point assist arm mix value used by topdown summary context
    assist_finger_mix      : float  # floating-point assist finger mix value used by topdown summary context
    action_source          : str  # string action source value used by topdown summary context


@dataclass(frozen=True)
class TopdownEnv0Summary:
    """Env0 scalar fields for a topdown curriculum summary row"""

    topdown_stage                  : int  # current topdown curriculum stage per environment
    topdown_source_pose_idx        : int  # index identifying the topdown source pose entry
    topdown_reach_hold             : int  # integer topdown reach hold value tracked by topdown env0 summary
    topdown_align_hold             : int  # integer topdown align hold value tracked by topdown env0 summary
    topdown_stage2_age             : int  # integer topdown stage2 age value tracked by topdown env0 summary
    topdown_contact_pose_hold      : int  # integer topdown contact pose hold value tracked by topdown env0 summary
    topdown_contact_pose_ready     : int  # boolean/tensor readiness state for topdown contact pose
    topdown_contact_pose_age       : int  # integer topdown contact pose age value tracked by topdown env0 summary
    topdown_unlock                 : float  # floating-point topdown unlock value used by topdown env0 summary
    topdown_effective_unlock       : float  # floating-point topdown effective unlock value used by topdown env0 summary
    finger_unlock_arm_hold_gate    : float  # floating-point finger unlock arm hold gate value used by topdown env0 summary
    topdown_contact_pose_shell     : int  # integer topdown contact pose shell value tracked by topdown env0 summary
    topdown_contact_palm_dist      : float  # floating-point topdown contact palm dist value used by topdown env0 summary
    topdown_contact_palm_height    : float  # floating-point topdown contact palm height value used by topdown env0 summary
    best_topdown_stage_this_episode: int  # integer best topdown stage this episode value tracked by topdown env0 summary
    max_topdown_unlock_this_episode: float  # floating-point max topdown unlock this episode value used by topdown env0 summary
    palm_dist                      : float  # floating-point palm dist value used by topdown env0 summary
    orient_deg                     : float  # floating-point orient deg value used by topdown env0 summary
    align_face_dist                : float  # floating-point align face dist value used by topdown env0 summary
    align_angle                    : float  # alignment angle value used by topdown/contact metrics
    any_contact_strength           : float  # floating-point any contact strength value used by topdown env0 summary
    strict_light_contact           : float  # floating-point strict light contact value used by topdown env0 summary


INPOCKET_ARM_HOLD_ATTRS = (
    ("env0_inpocket_arm_hold_active", "_inpocket_arm_hold_active"),
    ("env0_inpocket_arm_hold_valid", "_inpocket_arm_hold_valid"),
    ("env0_inpocket_arm_hold_live_gate", "_inpocket_arm_hold_live_gate"),
    ("env0_inpocket_arm_hold_freeze_ready", "_inpocket_arm_hold_freeze_ready"),
    ("env0_inpocket_arm_hold_frozen", "_inpocket_arm_hold_frozen"),
)

TOPDOWN_GATE_ATTRS = (
    ("env0_topdown_raw_finger_unlock_progress", "_topdown_raw_finger_unlock_progress"),
    ("env0_finger_unlock_center_gate", "_finger_unlock_center_gate"),
    ("env0_topdown_finger_close_gate", "_topdown_finger_close_gate"),
    ("env0_topdown_finger_xyz_close_gate", "_topdown_finger_xyz_close_gate"),
    ("env0_topdown_finger_xyz_error", "_topdown_finger_xyz_error"),
    ("env0_topdown_finger_front_gate", "_topdown_finger_front_gate"),
    ("env0_topdown_finger_thumb_front_margin", "_topdown_finger_thumb_front_margin"),
    ("env0_topdown_finger_index_front_margin", "_topdown_finger_index_front_margin"),
    ("env0_topdown_finger_center_live", "_topdown_finger_center_live"),
    ("env0_topdown_finger_center_ready", "_topdown_finger_center_ready"),
    ("env0_topdown_finger_center_hold", "_topdown_finger_center_hold"),
    ("env0_topdown_finger_center_xy_err", "_topdown_finger_center_xy_err"),
    ("env0_topdown_finger_center_max_xy_err", "_topdown_finger_center_max_xy_err"),
    ("env0_topdown_finger_center_z_err", "_topdown_finger_center_z_err"),
    ("env0_topdown_finger_center_align_angle_deg", "_topdown_finger_center_align_angle_deg"),
    ("env0_topdown_light_contact_success_base", "_topdown_light_contact_success_base"),
    ("env0_topdown_success_center_ready", "_topdown_success_center_ready"),
    ("env0_topdown_success_center_xy_err", "_topdown_success_center_xy_err"),
    ("env0_topdown_success_center_z_err", "_topdown_success_center_z_err"),
    ("env0_topdown_success_center_align_angle_deg", "_topdown_success_center_align_angle_deg"),
)

VERBOSE_TOPDOWN_SCALAR_ATTRS = (
    ("env0_topdown_success_hold", "_topdown_success_hold"),
    ("env0_teacher_ik_task_space_q", "_teacher_ik_topdown_task_space_q"),
    ("env0_teacher_ik_task_space_center_err", "_teacher_ik_topdown_task_space_center_err"),
    ("env0_teacher_ik_task_space_center_err_after", "_teacher_ik_topdown_task_space_center_err_after"),
    ("env0_teacher_ik_task_space_span_z", "_teacher_ik_topdown_task_space_span_z"),
    ("env0_teacher_ik_task_space_span_z_after", "_teacher_ik_topdown_task_space_span_z_after"),
    ("env0_teacher_ik_task_space_drop_err", "_teacher_ik_topdown_task_space_drop_err"),
    ("env0_teacher_ik_task_space_drop_err_after", "_teacher_ik_topdown_task_space_drop_err_after"),
    ("env0_topdown_palm_local_grip_offset_live_blend", "_topdown_palm_local_grip_offset_live_blend"),
)

VERBOSE_TOPDOWN_VEC3_ATTRS = (
    ("env0_teacher_ik_task_space_center_pos", "_teacher_ik_topdown_task_space_center_pos"),
    ("env0_teacher_ik_task_space_center_target", "_teacher_ik_topdown_task_space_center_target"),
    ("env0_teacher_ik_task_space_center_err_vec", "_teacher_ik_topdown_task_space_center_err_vec"),
    ("env0_topdown_selected_palm_local_grip_offset", "_topdown_selected_palm_local_grip_offset"),
)

CONTACT_TEACHER_ATTRS = (
    ("env0_contact_teacher_thumb_fraction", "_topdown_contact_teacher_thumb_fraction"),
    ("env0_contact_teacher_index_fraction", "_topdown_contact_teacher_index_fraction"),
    ("env0_contact_teacher_thumb_latched", "_topdown_contact_teacher_thumb_latched"),
    ("env0_contact_teacher_index_latched", "_topdown_contact_teacher_index_latched"),
    ("env0_contact_teacher_descent_z", "_topdown_contact_teacher_descent_z"),
    ("env0_contact_teacher_descent_z_need", "_topdown_contact_teacher_descent_z_need"),
    ("env0_contact_teacher_tip_servo_m", "_topdown_contact_teacher_tip_servo_m"),
    ("env0_contact_teacher_ready", "_topdown_contact_teacher_ready"),
    ("env0_contact_teacher_finger_ready", "_topdown_contact_teacher_finger_ready"),
    ("env0_contact_teacher_center_gate", "_topdown_contact_teacher_center_gate"),
    ("env0_contact_teacher_finger_close_gate", "_topdown_contact_teacher_finger_close_gate"),
    ("env0_contact_teacher_center_err_xy", "_topdown_contact_teacher_center_err_xy"),
    ("env0_contact_teacher_thumb_z_gap", "_topdown_contact_teacher_thumb_z_gap"),
    ("env0_contact_teacher_index_z_gap", "_topdown_contact_teacher_index_z_gap"),
    ("env0_contact_teacher_thumb_geom_done", "_topdown_contact_teacher_thumb_geom_done"),
    ("env0_contact_teacher_index_geom_done", "_topdown_contact_teacher_index_geom_done"),
    ("env0_contact_teacher_thumb_missing", "_topdown_contact_teacher_thumb_missing"),
    ("env0_contact_teacher_index_missing", "_topdown_contact_teacher_index_missing"),
)

CONTACT_TEACHER_VERBOSE_ATTRS = (
    ("env0_contact_teacher_thumb_hold_fraction", "_topdown_contact_teacher_thumb_hold_fraction"),
    ("env0_contact_teacher_index_hold_fraction", "_topdown_contact_teacher_index_hold_fraction"),
    ("env0_contact_teacher_inward_m", "_topdown_contact_teacher_inward_m"),
    ("env0_contact_teacher_precenter_servo_m", "_topdown_contact_teacher_precenter_servo_m"),
    ("env0_contact_teacher_precenter_active", "_topdown_contact_teacher_precenter_active"),
    ("env0_contact_teacher_arm_hold_unlock_fallback", "_topdown_contact_teacher_arm_hold_unlock_fallback"),
    ("env0_contact_teacher_center_servo_m", "_topdown_contact_teacher_center_servo_m"),
    ("env0_contact_teacher_center_servo_active", "_topdown_contact_teacher_center_servo_active"),
)


def source_pose_index_from_tensor(
    source_idx: torch.Tensor | None,  # Param: index selecting the source entry
    *,
    env_id : int = 0,  # Param: integer input for env id
    default: int = 0,  # Param: fallback value used when the input omits or rejects a setting
) -> int:
    """Read one source-pose index from a tensor"""
    if not torch.is_tensor(source_idx) or source_idx.numel() <= int(env_id):
        return int(default)
    return int(source_idx.reshape(-1)[int(env_id)].item())


def build_topdown_summary_row(
    *,
    context        : TopdownSummaryContext,  # Param: runtime context carrying validated trainer settings
    env0           : TopdownEnv0Summary,  # Param: input value used as env0
    topdown_metrics: Mapping[str, float] | None = None,  # Param: string input for topdown metrics
) -> dict[str, object]:
    """Build the base topdown_curriculum_summary JSON row"""
    row: dict[str, object] = {
        "mode"                                         : "topdown_curriculum_summary",
        "global_step"                                  : int(context.global_step),
        "episode_idx"                                  : int(context.episode_idx),
        "episode_step"                                 : int(context.episode_step),
        "reward"                                       : float(context.reward),
        "done_envs"                                    : int(context.done_envs),
        "replay_size"                                  : int(context.replay_size),
        "training_phase"                               : str(context.training_phase),
        "training_phase_id"                            : int(context.training_phase_id),
        "active_n_step"                                : int(context.active_n_step),
        "active_updates_per_step"                      : int(context.active_updates_per_step),
        "active_policy_delay"                          : int(context.active_policy_delay),
        "assist_mix"                                   : float(context.assist_mix),
        "assist_arm_mix"                               : float(context.assist_arm_mix),
        "assist_finger_mix"                            : float(context.assist_finger_mix),
        "action_source"                                : str(context.action_source),
        "env0_topdown_stage"                           : int(env0.topdown_stage),
        "env0_topdown_source_pose_idx"                 : int(env0.topdown_source_pose_idx),
        "env0_topdown_source_block"                    : topdown_source_block_name(env0.topdown_source_pose_idx),
        "env0_topdown_reach_hold"                      : int(env0.topdown_reach_hold),
        "env0_topdown_align_hold"                      : int(env0.topdown_align_hold),
        "env0_topdown_stage2_age"                      : int(env0.topdown_stage2_age),
        "env0_topdown_contact_pose_hold"               : int(env0.topdown_contact_pose_hold),
        "env0_topdown_contact_pose_ready"              : int(env0.topdown_contact_pose_ready),
        "env0_topdown_contact_pose_age"                : int(env0.topdown_contact_pose_age),
        "env0_topdown_finger_unlock_progress"          : float(env0.topdown_unlock),
        "env0_topdown_effective_finger_unlock_progress": float(env0.topdown_effective_unlock),
        "env0_finger_unlock_arm_hold_gate"             : float(env0.finger_unlock_arm_hold_gate),
        "env0_topdown_contact_pose_shell"              : int(env0.topdown_contact_pose_shell),
        "env0_topdown_contact_palm_dist"               : float(env0.topdown_contact_palm_dist),
        "env0_topdown_contact_palm_height"             : float(env0.topdown_contact_palm_height),
        "env0_best_topdown_stage_this_episode"         : int(env0.best_topdown_stage_this_episode),
        "env0_max_topdown_unlock_this_episode"         : float(env0.max_topdown_unlock_this_episode),
        "env0_palm_dist"                               : float(env0.palm_dist),
        "env0_orient_deg"                              : float(env0.orient_deg),
        "env0_align_face_dist"                         : float(env0.align_face_dist),
        "env0_align_angle"                             : float(env0.align_angle),
        "env0_any_contact_strength"                    : float(env0.any_contact_strength),
        "env0_strict_light_contact"                    : float(env0.strict_light_contact),
    }
    if topdown_metrics:
        row.update({str(key): float(value) for key, value in topdown_metrics.items()})
    return row


def add_existing_tensor_scalar_attrs(
    row   : dict[str, object],  # Param: string input for row
    owner : object,  # Param: input value used as owner
    env_id: int,  # Param: integer input for env id
    specs : Iterable[tuple[str, str]],  # Param: string input for specs
) -> dict[str, object]:
    """Add scalar tensor attrs only when present"""
    for key, attr_name in specs:
        attr = getattr(owner, attr_name, None)
        if not torch.is_tensor(attr) or attr.dim() == 0 or attr.shape[0] <= int(env_id):
            continue
        row[key] = float(attr[int(env_id)].detach().reshape(-1)[0].item())
    return row


def add_existing_vec3_tensor_attrs(
    row   : dict[str, object],  # Param: string input for row
    owner : object,  # Param: input value used as owner
    env_id: int,  # Param: integer input for env id
    specs : Iterable[tuple[str, str]],  # Param: string input for specs
) -> dict[str, object]:
    """Add vec3 tensor attrs only when present"""
    for prefix, attr_name in specs:
        attr = getattr(owner, attr_name, None)
        if not torch.is_tensor(attr) or attr.dim() < 2 or attr.shape[0] <= int(env_id):
            continue
        vec = attr[int(env_id)].detach().reshape(-1)
        if vec.numel() < 3:
            continue
        row[f"{prefix}_x"] = float(vec[0].item())
        row[f"{prefix}_y"] = float(vec[1].item())
        row[f"{prefix}_z"] = float(vec[2].item())
    return row


def add_topdown_summary_optional_attrs(
    row: dict[str, object],  # Param: string input for row
    env: object,  # Param: environment or backend object used for runtime calls
    *,
    log_env_id             : int  = 0,  # Param: integer input for log env id
    verbose                : bool = False,  # Param: boolean input controlling verbose
    contact_teacher_enabled: bool = False,  # Param: boolean input enabling contact teacher
) -> dict[str, object]:
    """Add optional env tensor diagnostics to a topdown summary row

    Steps:
    - Resolve inputs for `add_topdown_summary_optional_attrs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    add_existing_tensor_scalar_attrs(row, env, log_env_id, INPOCKET_ARM_HOLD_ATTRS)
    add_existing_tensor_scalar_attrs(row, env, log_env_id, TOPDOWN_GATE_ATTRS)
    if verbose:
        add_existing_tensor_scalar_attrs(row, env, log_env_id, VERBOSE_TOPDOWN_SCALAR_ATTRS)
        add_existing_vec3_tensor_attrs(row, env, log_env_id, VERBOSE_TOPDOWN_VEC3_ATTRS)
    if contact_teacher_enabled:
        add_existing_tensor_scalar_attrs(row, env, log_env_id, CONTACT_TEACHER_ATTRS)
        if verbose:
            add_existing_tensor_scalar_attrs(row, env, log_env_id, CONTACT_TEACHER_VERBOSE_ATTRS)
    return row
