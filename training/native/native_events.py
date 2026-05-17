"""

Native loop event handlers for live backend assembly

File map:

NativeLogEventCallbacks:            Injected side effects for native progress logging
NativeCheckpointEventCallbacks:     Injected side effects for native scheduled checkpoints
NativeEvalEventCallbacks:           Injected side effects for native scheduled eval
_native_components:                 Handle native components logic
_compact_native_step_line:          Handle compact native step line logic
_state_machine_module:              Handle state machine module logic
_live_rollout:                      Handle live rollout logic
_last_step_result:                  Handle last step result logic
_last_action_selection:             Handle last action selection logic
_env_tensor:                        Handle env tensor logic
_default_env_mask:                  Handle default env mask logic
_step_done_flags:                   Handle step done flags logic
_step_active_mask:                  Handle step active mask logic
_safe_call_tensor:                  Handle safe call tensor logic
_safe_call_pair:                    Handle safe call pair logic
_env_float:                         Handle env float logic
_metric_mask:                       Handle metric mask logic
_masked_tensor_values:              Handle masked tensor values logic
_safe_scalar:                       Handle safe scalar logic
_safe_int:                          Handle safe int logic
_episode_step:                      Handle episode step logic
_episode_idx:                       Handle episode idx logic
_phase_name:                        Handle phase name logic
_phase_name_from_context:           Handle phase name from context logic
_action_source:                     Handle action source logic
_assist_values:                     Handle assist values logic
_topdown_tensors:                   Handle topdown tensors logic
_topdown_metrics:                   Handle topdown metrics logic
_lift_stage_metrics:                Handle lift stage metrics logic
_target_error_tensors:              Handle target error tensors logic
_topdown_stage_bits:                Handle topdown stage bits logic
_write_jsonl:                       Handle write jsonl logic
_masked_mean:                       Handle masked mean logic
_train_env_score:                   Handle train env score logic
_save_train_best_checkpoint:        Handle save train best checkpoint logic
_persist_native_training_log:       Handle persist native training log logic
native_log_event_line:              Format a topdown native loop progress event
trace_native_log_event:             Trace one native progress event and return the line
native_log_event:                   Build a PlanFn for native progress logging
_save_scheduled_native_checkpoint:  Handle save scheduled native checkpoint logic
run_native_checkpoint_event:        Run scheduled native checkpoint jobs for one loop plan
native_checkpoint_event:            Build a PlanFn for native scheduled checkpoint jobs
run_native_eval_event:              Run configured native eval episodes for one loop plan
native_eval_event:                  Build a PlanFn for native scheduled eval episodes
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import os
from typing import Any

import torch

from ..io.checkpoint_io import capture_rng_state, save_training_checkpoint
from ..io.checkpoint_schedule import RemovePathFn, ScheduledCheckpointJob, run_scheduled_checkpoint_plan
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..geometry.topdown_metrics import topdown_progress_metrics
from ..env.topdown_env_adapters import topdown_curriculum_state
from .native_backend import NativeTrainerState
from .native_components import NativeTrainingComponents
from .native_eval import NativeEvalCallbacks, NativeEvalConfig, NativeEvalResult, run_native_eval_episode
from .native_finalization import NativeCheckpointSaveFn, build_native_checkpoint_metadata
from .native_loop import PlanFn
from ..state.run_state import TrainingLoopStartupState
from ..logging.diagnostics import per_term_reward_means, termination_term_flags
from ..logging.jsonl import write_jsonl_row
from ..logging.progress import format_reward_term_bits, format_update_bits
from ..logging.progress_lines import ProgressLineSummary, format_progress_line
from ..logging.stage_bits import (
    TopdownStageBitInputs,
    format_topdown_done_bits,
    format_topdown_stage_bits,
    tensor_env_bit,
    tensor_env_float,
    tensor_env_int,
)
from ..logging.tensorboard_logging import (
    ScalarEvent,
    finite_scalar_events,
    reward_term_events,
    topdown_metric_events,
    train_env_metric_events,
    update_info_events,
    write_scalar_events,
)


TraceFn = Callable[[str], None]
NativeEvalCallbacksFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState, TrainingLoopStartupState, int],
    NativeEvalCallbacks,
]
NativeEvalConfigFn = Callable[
    [
        TrainerRuntimeContext,
        RuntimeConfigBundle,
        NativeTrainerState,
        TrainingLoopStartupState,
        object,
        int,
        NativeEvalConfig,
    ],
    NativeEvalConfig,
]
NativeEvalResultFn = Callable[[NativeEvalResult], None]
NativeEvalPostFn = Callable[
    [
        TrainerRuntimeContext,
        RuntimeConfigBundle,
        NativeTrainerState,
        TrainingLoopStartupState,
        object,
        tuple[NativeEvalResult, ...],
    ],
    None,
]


@dataclass(frozen=True)
class NativeLogEventCallbacks:
    """Injected side effects for native progress logging"""

    trace_fn : TraceFn | None = None  # Field: callback used for the trace fn operation


@dataclass(frozen=True)
class NativeCheckpointEventCallbacks:
    """Injected side effects for native scheduled checkpoints"""

    save_checkpoint_fn: NativeCheckpointSaveFn = save_training_checkpoint  # Field: callback used for the save checkpoint fn operation
    remove_path_fn    : RemovePathFn           = os.remove  # Field: callback used for the remove path fn operation
    trace_fn          : TraceFn | None         = None  # Field: callback used for the trace fn operation


@dataclass(frozen=True)
class NativeEvalEventCallbacks:
    """Injected side effects for native scheduled eval"""

    eval_callbacks_fn: NativeEvalCallbacksFn  # Field: callback used for the eval callbacks fn operation
    eval_config_fn   : NativeEvalConfigFn | None = None  # Field: callback used for the eval config fn operation
    result_fn        : NativeEvalResultFn | None = None  # Field: callback used for the result fn operation
    post_eval_fn     : NativeEvalPostFn | None   = None  # Field: callback used for the post eval fn operation


def _native_components(state: NativeTrainerState) -> NativeTrainingComponents:
    components = state.get("components")
    if not isinstance(components, NativeTrainingComponents):
        raise TypeError(f"native state components is {type(components)!r}")
    return components


def _compact_native_step_line(
    context   : TrainerRuntimeContext,
    loop_state: TrainingLoopStartupState,
    plan,
    *,
    log_error: str | None = None,
) -> str:
    suffix = "" if log_error is None else f" log_error={str(log_error).splitlines()[0][:120]}"
    return (
        f"native_step step={int(plan.global_step):05d} "
        f"episode={int(loop_state.episode.episode_idx.max()):03d} "
        f"transitions={int(loop_state.transitions_collected)} "
        f"added={int(plan.num_added)} "
        f"replay={int(plan.replay_size)} "
        f"task={context.task}"
        f"{suffix}"
    )


def _state_machine_module() -> Any | None:
    try:
        from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp import state_machine
    except Exception:
        return None
    return state_machine


def _live_rollout(state: NativeTrainerState) -> Any | None:
    return state.get("live_rollout")


def _last_step_result(state: NativeTrainerState) -> Any | None:
    live = _live_rollout(state)
    return getattr(live, "last_step_result", None)


def _last_action_selection(state: NativeTrainerState) -> Any | None:
    live = _live_rollout(state)
    return getattr(live, "last_action_selection", None)


def _env_tensor(env: Any, attr_name: str, *, default: float = 0.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Process for `_env_tensor`

    Steps:
    - Resolve inputs for `_env_tensor` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    tensor = getattr(env, attr_name, None)
    if torch.is_tensor(tensor):
        return tensor.detach().to(dtype=dtype).reshape(-1)
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    device = getattr(env, "device", "cpu")
    return torch.full((num_envs,), float(default), device=device, dtype=dtype)


def _default_env_mask(env: Any, *, value: bool) -> torch.Tensor:
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    device = getattr(env, "device", "cpu")
    return torch.full((num_envs,), bool(value), device=device, dtype=torch.bool)


def _step_done_flags(step_result: Any, env: Any) -> torch.Tensor:
    batch = getattr(step_result, "batch", None)
    done_flags = getattr(batch, "done_flags", None)
    if torch.is_tensor(done_flags):
        return done_flags.to(dtype=torch.bool).reshape(-1)
    return _default_env_mask(env, value=False)


def _step_active_mask(step_result: Any, env: Any) -> torch.Tensor:
    batch = getattr(step_result, "batch", None)
    active_mask = getattr(batch, "active_env_mask", None)
    if torch.is_tensor(active_mask):
        return active_mask.to(dtype=torch.bool).reshape(-1)
    return _default_env_mask(env, value=True)


def _safe_call_tensor(env: Any, fn: Any, *, default: float = 0.0) -> torch.Tensor:
    try:
        tensor = fn(env)
    except Exception:
        return _env_tensor(env, "_topdown_stage", default=default)
    if not torch.is_tensor(tensor):
        return _env_tensor(env, "_topdown_stage", default=default)
    return tensor.detach().to(dtype=torch.float32).reshape(-1)


def _safe_call_pair(env: Any, fn: Any) -> tuple[torch.Tensor, torch.Tensor]:
    zero = _env_tensor(env, "_topdown_stage", default=0.0)
    try:
        a, b = fn(env)
    except Exception:
        return zero, zero
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return zero, zero
    return a.detach().to(dtype=torch.float32).reshape(-1), b.detach().to(dtype=torch.float32).reshape(-1)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _metric_mask(active_env_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = active_env_mask.to(device=reference.device, dtype=torch.bool).reshape(-1)
    if mask.shape == reference.shape and bool(mask.any().item()):
        return mask
    return torch.ones_like(reference, dtype=torch.bool)


def _masked_tensor_values(tensor: torch.Tensor | None, reference: torch.Tensor, mask: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if not torch.is_tensor(tensor) or tensor.reshape(-1).shape != reference.shape:
        values = torch.full_like(reference, float(default), dtype=torch.float32)
    else:
        values = tensor.detach().to(device=reference.device, dtype=torch.float32).reshape(-1)
    return values[mask]


def _safe_scalar(tensor: torch.Tensor | None, env_id: int = 0, default: float = 0.0) -> float:
    if not torch.is_tensor(tensor) or tensor.numel() <= int(env_id):
        return float(default)
    try:
        value = float(tensor.reshape(-1)[int(env_id)].detach().item())
    except (RuntimeError, ValueError, TypeError, IndexError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _safe_int(tensor: torch.Tensor | None, env_id: int = 0, default: int = 0) -> int:
    return int(round(_safe_scalar(tensor, env_id=env_id, default=float(default))))


def _episode_step(env: Any, loop_state: TrainingLoopStartupState, env_id: int = 0) -> int:
    buf = getattr(env, "episode_length_buf", None)
    if torch.is_tensor(buf) and buf.numel() > int(env_id):
        return int(buf.reshape(-1)[int(env_id)].item())
    if torch.is_tensor(loop_state.episode.step) and loop_state.episode.step.numel() > int(env_id):
        return int(loop_state.episode.step.reshape(-1)[int(env_id)].item())
    return 0


def _episode_idx(loop_state: TrainingLoopStartupState, env_id: int = 0) -> int:
    idx = loop_state.episode.episode_idx
    if torch.is_tensor(idx) and idx.numel() > int(env_id):
        return int(idx.reshape(-1)[int(env_id)].item())
    return 0


def _phase_name(configs: RuntimeConfigBundle, global_step: int) -> str:
    """Process for `_phase_name`

    Steps:
    - Resolve inputs for `_phase_name` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if int(global_step) < int(configs.counts.start_steps):
        return "teacher"
    if int(global_step) < int(configs.assist.bc_only_steps):
        return "bc"
    rl_start = int(configs.counts.rl_phase_start_steps)
    if rl_start >= 0 and int(global_step) >= rl_start:
        return "rl"
    if int(configs.assist.bc_only_steps) > 0:
        return "dagger"
    return "rl"


def _phase_name_from_context(context: TrainerRuntimeContext, configs: RuntimeConfigBundle, global_step: int) -> str:
    current = str(context.args.get("_training_phase_name", "") or "")
    if current and current != "single":
        return current
    return _phase_name(configs, global_step)


def _action_source(state: NativeTrainerState) -> str:
    selection = _last_action_selection(state)
    if selection is None:
        return "unknown"
    if getattr(selection, "teacher_action", None) is not None:
        diagnostics = getattr(selection, "diagnostics", None)
        if bool(getattr(diagnostics, "teacher_warmup", False)):
            return "teacher_ik"
        return "policy_assist" if bool(getattr(diagnostics, "teacher_available", False)) else "teacher_ik"
    return "policy"


def _assist_values(state: NativeTrainerState, configs: RuntimeConfigBundle) -> tuple[float, float, float]:
    selection = _last_action_selection(state)
    diagnostics = getattr(selection, "diagnostics", None)
    if diagnostics is not None:
        return (
            float(getattr(diagnostics, "assist_mix", configs.assist.policy_assist_mix)),
            float(getattr(diagnostics, "assist_arm_mix", configs.assist.policy_assist_mix)),
            float(getattr(diagnostics, "assist_finger_mix", configs.assist.policy_assist_mix)),
        )
    return (
        float(configs.assist.policy_assist_mix),
        float(configs.assist.policy_assist_mix),
        float(configs.assist.policy_assist_mix),
    )


def _topdown_tensors(env: Any, state_machine: Any | None) -> dict[str, torch.Tensor]:
    """Process for `_topdown_tensors`

    Steps:
    - Resolve inputs for `_topdown_tensors` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    stage = _env_tensor(env, "_topdown_stage", default=0.0)
    zero = torch.zeros_like(stage, dtype=torch.float32)
    if state_machine is None:
        return {"finger_unlock": _env_tensor(env, "_topdown_finger_unlock_progress"), "stage": stage}

    thumb = _safe_call_tensor(env, getattr(state_machine, "thumb_contact_strength"), default=0.0)
    index = _safe_call_tensor(env, getattr(state_machine, "index_contact_strength"), default=0.0)
    middle_fn = getattr(state_machine, "middle_contact_strength", None)
    middle = _safe_call_tensor(env, middle_fn, default=0.0) if callable(middle_fn) else zero
    fingertip = torch.maximum(torch.maximum(thumb, index), middle)
    hand = _safe_call_tensor(env, getattr(state_machine, "any_hand_contact_strength"), default=0.0)
    strict = _safe_call_tensor(env, getattr(state_machine, "opposed_contact_strength"), default=0.0)
    opposed = _safe_call_tensor(env, getattr(state_machine, "opposite_face_gate"), default=0.0)
    align = _safe_call_tensor(env, getattr(state_machine, "open_hand_alignment_error"), default=0.0)
    align_angle = torch.rad2deg(_safe_call_tensor(env, getattr(state_machine, "fingertip_line_angle_rad"), default=0.0))
    lift = _safe_call_tensor(env, getattr(state_machine, "block_lift_height"), default=0.0)
    disp = _safe_call_tensor(env, getattr(state_machine, "block_displacement"), default=0.0)
    return {
        "stage"                : stage,
        "contact"              : fingertip,
        "both_contact"         : torch.minimum(thumb, index),
        "fingertip_contact"    : fingertip,
        "hand_contact"         : hand,
        "thumb_contact"        : thumb,
        "index_contact"        : index,
        "thumb_contact_force_N": thumb,
        "index_contact_force_N": index,
        "strict_light_contact" : strict,
        "lift"                 : lift,
        "finger_unlock"        : _env_tensor(env, "_topdown_finger_unlock_progress"),
        "curl"                 : zero,
        "opposed_face"         : opposed,
        "align_face_dist"      : align,
        "align_angle"          : align_angle,
        "phase1_ready"         : (stage >= 1).to(dtype=torch.float32),
        "success": _safe_call_tensor(env, getattr(state_machine, "lift_success_held"), default=0.0)
        if callable(getattr(state_machine, "lift_success_held", None))
        else zero,
        "block_disp": disp,
    }


def _topdown_metrics(env: Any, tensors: Mapping[str, torch.Tensor], active_env_mask: torch.Tensor) -> dict[str, float]:
    stage = tensors.get("stage")
    if not torch.is_tensor(stage):
        return {}
    return topdown_progress_metrics(
        active_env_mask=active_env_mask.to(device=stage.device, dtype=torch.bool),
        stage=stage.to(dtype=torch.long),
        finger_unlock_progress=_env_tensor(env, "_topdown_finger_unlock_progress"),
        reach_hold=_env_tensor(env, "_topdown_reach_hold"),
        align_hold=_env_tensor(env, "_topdown_align_hold"),
        stage2_age=_env_tensor(env, "_topdown_stage2_age"),
        contact_pose_hold=_env_tensor(env, "_topdown_contact_pose_hold"),
        contact_pose_ready=_env_tensor(env, "_topdown_contact_pose_ready"),
        contact_pose_age=_env_tensor(env, "_topdown_contact_pose_age"),
        contact_pose_shell=_env_tensor(env, "_topdown_contact_pose_shell"),
        contact_palm_dist=_env_tensor(env, "_topdown_contact_palm_distance"),
        contact_palm_height=_env_tensor(env, "_topdown_contact_palm_height_error"),
        source_idx=_env_tensor(env, "_topdown_source_pose_idx", dtype=torch.long),
        success_flags=tensors.get("success"),
        strict_contact=tensors.get("strict_light_contact"),
        contact=tensors.get("contact"),
        lift=tensors.get("lift"),
    )


def _lift_stage_metrics(
    env: Any,
    tensors: Mapping[str, torch.Tensor],
    active_env_mask: torch.Tensor,
    state_machine: Any | None,
) -> dict[str, float]:
    stage = tensors.get("stage")
    if state_machine is None or not torch.is_tensor(stage) or stage.numel() == 0:
        return {}
    stage = stage.detach().to(dtype=torch.float32).reshape(-1)
    mask = _metric_mask(active_env_mask, stage)
    if not bool(mask.any().item()):
        return {}

    def tensor_from_call(name: str, default: float = 0.0) -> torch.Tensor:
        return _safe_call_tensor(env, getattr(state_machine, name, None), default=default)

    stage_t = _masked_tensor_values(stage, stage, mask)
    strict_t = _masked_tensor_values(tensors.get("strict_light_contact"), stage, mask)
    lift_t = _masked_tensor_values(tensors.get("lift"), stage, mask)
    xy_drift = _masked_tensor_values(tensor_from_call("block_xy_displacement"), stage, mask)
    z_vel = _masked_tensor_values(tensor_from_call("block_z_velocity"), stage, mask)
    xy_vel = _masked_tensor_values(tensor_from_call("block_xy_velocity_norm"), stage, mask)
    ang_vel = _masked_tensor_values(tensor_from_call("block_angular_velocity_norm"), stage, mask)
    tilt = _masked_tensor_values(tensor_from_call("block_tilt_angle_rad"), stage, mask)
    block_tilt_deg = torch.rad2deg(tilt)

    lift_latched = _masked_tensor_values(_env_tensor(env, "_arm_lift_latched"), stage, mask).to(dtype=torch.bool)
    lift_freeze = _masked_tensor_values(_env_tensor(env, "_teacher_ik_topdown_lift_freeze_active"), stage, mask).to(dtype=torch.bool)
    lift_release = _masked_tensor_values(_env_tensor(env, "_inpocket_arm_hold_lift_release"), stage, mask).to(dtype=torch.bool)
    center_ok = _masked_tensor_values(_env_tensor(env, "_topdown_lift_success_center_ok"), stage, mask).to(dtype=torch.bool)
    lift_contact_counter = _masked_tensor_values(_env_tensor(env, "_arm_lift_contact_counter"), stage, mask)
    lift_latch_signal = _masked_tensor_values(_env_tensor(env, "_arm_lift_latch_signal"), stage, mask)
    teacher_lift_progress = _masked_tensor_values(_env_tensor(env, "_teacher_ik_topdown_lift_progress"), stage, mask)
    lift_success_hold = _masked_tensor_values(_env_tensor(env, "_topdown_lift_success_hold"), stage, mask)
    sustained_hold = _masked_tensor_values(_env_tensor(env, "_topdown_sustained_lift_grip_hold"), stage, mask)
    closure = _masked_tensor_values(_env_tensor(env, "_topdown_contact_teacher_closure_fraction"), stage, mask)
    thumb_fraction = _masked_tensor_values(_env_tensor(env, "_topdown_contact_teacher_thumb_fraction"), stage, mask)
    index_fraction = _masked_tensor_values(_env_tensor(env, "_topdown_contact_teacher_index_fraction"), stage, mask)

    contact_min = _env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
    lift_min = _env_float("TOPDOWN_LIFT_SUCCESS_HEIGHT", 0.035)
    drift_max = _env_float("TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX", 0.04)
    drift_term = _env_float("CURRICULUM_BLOCK_DRIFT_THRESHOLD", 0.12)
    tilt_max = _env_float("TOPDOWN_LIFT_SUCCESS_BLOCK_TILT_MAX_DEG", 0.0)
    contact_ok = strict_t >= contact_min
    lift_ok = lift_t >= lift_min
    drift_ok = xy_drift <= drift_max
    tilt_ok = block_tilt_deg <= tilt_max if tilt_max > 0.0 else torch.ones_like(contact_ok)
    clean_lift_ok = contact_ok & lift_ok & drift_ok
    centered_upright_lift_ok = clean_lift_ok & center_ok & tilt_ok
    positive_z_vel = z_vel.clamp_min(0.0)

    return {
        "lift_stage/stage2_rate": float((stage_t >= 2).float().mean().item()),
        "lift_stage/arm_lift_latched_rate": float(lift_latched.float().mean().item()),
        "lift_stage/arm_lift_contact_counter_mean": float(lift_contact_counter.mean().item()),
        "lift_stage/arm_lift_contact_counter_max": float(lift_contact_counter.max().item()),
        "lift_stage/arm_lift_latch_signal_mean": float(lift_latch_signal.mean().item()),
        "lift_stage/teacher_lift_progress_mean": float(teacher_lift_progress.mean().item()),
        "lift_stage/teacher_lift_progress_max": float(teacher_lift_progress.max().item()),
        "lift_stage/lift_freeze_active_rate": float(lift_freeze.float().mean().item()),
        "lift_stage/arm_hold_lift_release_rate": float(lift_release.float().mean().item()),
        "lift_stage/opposed_contact_mean": float(strict_t.mean().item()),
        "lift_stage/opposed_contact_max": float(strict_t.max().item()),
        "lift_stage/opposed_contact_gate_rate": float(contact_ok.float().mean().item()),
        "lift_stage/lift_height_mean": float(lift_t.mean().item()),
        "lift_stage/lift_height_max": float(lift_t.max().item()),
        "lift_stage/lift_ge_2cm_rate": float((lift_t >= 0.02).float().mean().item()),
        "lift_stage/lift_ge_success_height_rate": float(lift_ok.float().mean().item()),
        "lift_stage/lift_ge_5cm_rate": float((lift_t >= 0.05).float().mean().item()),
        "lift_stage/lift_ge_10cm_rate": float((lift_t >= 0.10).float().mean().item()),
        "lift_stage/clean_lift_success_now_rate": float(clean_lift_ok.float().mean().item()),
        "lift_stage/centered_success_gate_rate": float(center_ok.float().mean().item()),
        "lift_stage/upright_tilt_gate_rate": float(tilt_ok.float().mean().item()),
        "lift_stage/centered_upright_lift_now_rate": float(centered_upright_lift_ok.float().mean().item()),
        "lift_stage/lift_without_opposed_contact_rate": float((lift_ok & (~contact_ok)).float().mean().item()),
        "lift_stage/opposed_contact_without_lift_rate": float((contact_ok & (~lift_ok)).float().mean().item()),
        "lift_stage/success_hold_mean": float(lift_success_hold.mean().item()),
        "lift_stage/success_hold_max": float(lift_success_hold.max().item()),
        "lift_stage/success_held_rate": float((lift_success_hold >= _env_float("TOPDOWN_LIFT_SUCCESS_HOLD_STEPS", 15.0)).float().mean().item()),
        "lift_stage/sustained_lift_grip_hold_mean": float(sustained_hold.mean().item()),
        "lift_stage/sustained_lift_grip_hold_max": float(sustained_hold.max().item()),
        "lift_stage/xy_drift_mean": float(xy_drift.mean().item()),
        "lift_stage/xy_drift_max": float(xy_drift.max().item()),
        "lift_stage/xy_drift_success_gate_rate": float(drift_ok.float().mean().item()),
        "lift_stage/xy_drift_over_success_gate_rate": float((xy_drift > drift_max).float().mean().item()),
        "lift_stage/xy_drift_over_termination_rate": float((xy_drift > drift_term).float().mean().item()),
        "lift_stage/block_z_velocity_mean": float(z_vel.mean().item()),
        "lift_stage/block_z_velocity_max": float(z_vel.max().item()),
        "lift_stage/block_positive_z_velocity_mean": float(positive_z_vel.mean().item()),
        "lift_stage/block_positive_z_velocity_rate": float((z_vel > 0.0).float().mean().item()),
        "lift_stage/block_xy_velocity_mean": float(xy_vel.mean().item()),
        "lift_stage/block_xy_velocity_max": float(xy_vel.max().item()),
        "lift_stage/block_ang_velocity_mean": float(ang_vel.mean().item()),
        "lift_stage/block_ang_velocity_max": float(ang_vel.max().item()),
        "lift_stage/block_tilt_deg_mean": float(block_tilt_deg.mean().item()),
        "lift_stage/block_tilt_deg_max": float(block_tilt_deg.max().item()),
        "lift_stage/contact_teacher_closure_mean": float(closure.mean().item()),
        "lift_stage/contact_teacher_closure_max": float(closure.max().item()),
        "lift_stage/contact_teacher_thumb_fraction_mean": float(thumb_fraction.mean().item()),
        "lift_stage/contact_teacher_index_fraction_mean": float(index_fraction.mean().item()),
    }


def _target_error_tensors(env: Any, state_machine: Any | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process for `_target_error_tensors`

    Steps:
    - Resolve inputs for `_target_error_tensors` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    stage = _env_tensor(env, "_topdown_stage", default=0.0)
    zero_delta = torch.zeros((stage.shape[0], 3), device=stage.device, dtype=torch.float32)
    if state_machine is None:
        return stage, stage, zero_delta, zero_delta
    thumb_err, index_err = _safe_call_pair(env, getattr(state_machine, "open_hand_face_distances", None))
    try:
        thumb_target, index_target = state_machine._face_targets(env)
        thumb_pos = state_machine._link_pos(env, state_machine._THUMB_LINK)
        index_pos = state_machine._link_pos(env, state_machine._INDEX_LINK)
        thumb_delta = (thumb_target - thumb_pos).detach().to(dtype=torch.float32)
        index_delta = (index_target - index_pos).detach().to(dtype=torch.float32)
    except Exception:
        thumb_delta = zero_delta
        index_delta = zero_delta
    return thumb_err, index_err, thumb_delta, index_delta


def _topdown_stage_bits(
    env         : Any,
    tensors     : Mapping[str, torch.Tensor],
    loop_state  : TrainingLoopStartupState,
    active_mask : torch.Tensor,
    state_machine: Any | None,
) -> str:
    """Process for `_topdown_stage_bits`

    Steps:
    - Resolve inputs for `_topdown_stage_bits` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    stage = tensors.get("stage", _env_tensor(env, "_topdown_stage", default=0.0))
    stage_ge1_rate = float((stage[active_mask.to(device=stage.device, dtype=torch.bool)] >= 1).float().mean().item()) if bool(active_mask.any().item()) else 0.0
    stage_ge2_rate = float((stage[active_mask.to(device=stage.device, dtype=torch.bool)] >= 2).float().mean().item()) if bool(active_mask.any().item()) else 0.0
    values = TopdownStageBitInputs(
        topdown_stage=tensor_env_int(env, "_topdown_stage", 0, -1),
        best_topdown_stage=_safe_int(loop_state.episode.best_topdown_stage, default=-1),
        reach_hold=tensor_env_int(env, "_topdown_reach_hold", 0),
        align_hold=tensor_env_int(env, "_topdown_align_hold", 0),
        contact_pose_ready=tensor_env_bit(env, "_topdown_contact_pose_ready", 0),
        contact_pose_hold=tensor_env_int(env, "_topdown_contact_pose_hold", 0),
        contact_pose_shell=tensor_env_bit(env, "_topdown_contact_pose_shell", 0),
        contact_palm_dist=tensor_env_float(env, "_topdown_contact_palm_distance", 0),
        contact_palm_height=tensor_env_float(env, "_topdown_contact_palm_height_error", 0),
        stage2_age=tensor_env_int(env, "_topdown_stage2_age", 0),
        unlock_progress=tensor_env_float(env, "_topdown_finger_unlock_progress", 0),
        effective_unlock_progress=tensor_env_float(env, "_topdown_effective_finger_unlock_progress", 0, 0.0),
        finger_arm_hold_gate=tensor_env_float(env, "_topdown_finger_arm_hold_gate", 0, 0.0),
        prehold_servo=(
            tensor_env_float(env, "_topdown_contact_teacher_prehold_servo_m", 0, 0.0)
            if torch.is_tensor(getattr(env, "_topdown_contact_teacher_prehold_servo_m", None))
            else tensor_env_float(env, "_topdown_contact_teacher_precenter_servo_m", 0, 0.0)
        ),
        align_line_z=tensor_env_float(env, "_topdown_align_line_z", 0, 0.0),
        align_servo_q=tensor_env_float(env, "_topdown_align_servo_q", 0, 0.0),
        align_servo_active=tensor_env_bit(env, "_topdown_align_servo_active", 0),
        pocket_sweep_q=tensor_env_float(env, "_topdown_pocket_sweep_q", 0, 0.0),
        pocket_score_before=tensor_env_float(env, "_topdown_pocket_score_before", 0, 0.0),
        pocket_score_after=tensor_env_float(env, "_topdown_pocket_score_after", 0, 0.0),
        stage_ge1_rate=stage_ge1_rate,
        stage_ge2_rate=stage_ge2_rate,
        contact=_safe_scalar(tensors.get("contact")),
        strict_contact=_safe_scalar(tensors.get("strict_light_contact")),
        thumb_contact=_safe_scalar(tensors.get("thumb_contact")),
        index_contact=_safe_scalar(tensors.get("index_contact")),
        align_face=_safe_scalar(tensors.get("align_face_dist")),
        align_angle=_safe_scalar(tensors.get("align_angle")),
        opposed_face=_safe_scalar(tensors.get("opposed_face")),
        lift=_safe_scalar(tensors.get("lift")),
        block_disp=_safe_scalar(tensors.get("block_disp")),
    )
    del state_machine
    return format_topdown_stage_bits(
        values,
        env=env,
        log_env_id=0,
        inpocket_arm_hold_enabled=False,
        contact_teacher_enabled=True,
    )


def _write_jsonl(path: str, row: Mapping[str, Any]) -> None:
    if not path or str(path).strip().lower() in {"off", "none", "false", "0"}:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as log_file:
        write_jsonl_row(log_file, row, flush=True)


def _masked_mean(tensor: torch.Tensor | None, mask: torch.Tensor, default: float = 0.0) -> float:
    """Process for `_masked_mean`

    Steps:
    - Resolve inputs for `_masked_mean` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not torch.is_tensor(tensor):
        return float(default)
    selected = mask.to(device=tensor.device, dtype=torch.bool).reshape(-1)
    values = tensor.detach().to(dtype=torch.float32).reshape(-1)
    if values.numel() != selected.numel() or not bool(selected.any().item()):
        return float(default)
    return float(values[selected].mean().item())


def _train_env_score(
    tensors        : Mapping[str, torch.Tensor],
    topdown_metrics: Mapping[str, float],
    active_mask    : torch.Tensor,
) -> float:
    """Process for `_train_env_score`

    Steps:
    - Resolve inputs for `_train_env_score` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    stage = tensors.get("stage")
    success = float(topdown_metrics.get("success_rate", _masked_mean(tensors.get("success"), active_mask)))
    stage2 = float(topdown_metrics.get("stage_ge2_rate", _masked_mean((stage >= 2).float() if torch.is_tensor(stage) else None, active_mask)))
    strict = _masked_mean(tensors.get("strict_light_contact"), active_mask)
    lift = max(0.0, _masked_mean(tensors.get("lift"), active_mask))
    block_disp = max(0.0, _masked_mean(tensors.get("block_disp"), active_mask))
    return success * 1000.0 + lift * 250.0 + stage2 * 25.0 + strict * 10.0 - block_disp * 5.0


def _save_train_best_checkpoint(
    context        : TrainerRuntimeContext,
    configs        : RuntimeConfigBundle,
    state          : NativeTrainerState,
    loop_state     : TrainingLoopStartupState,
    plan,
    *,
    score: float,
) -> None:
    """Process for `_save_train_best_checkpoint`

    Steps:
    - Resolve inputs for `_save_train_best_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not bool(context.args.get("save_train_env_best_checkpoint", False)):
        return
    if not math.isfinite(score):
        return
    best_score = float(loop_state.best_eval_state.get("train_env_best_score", float("-inf")))
    min_delta = float(context.args.get("train_best_checkpoint_min_delta", 1e-3) or 1e-3)
    if score <= best_score + min_delta:
        return
    components = _native_components(state)
    loop_state.best_eval_state["train_env_best_score"] = float(score)
    loop_state.best_eval_state["train_env_best_step"] = float(plan.global_step)
    checkpoint_dir = os.path.dirname(os.path.abspath(context.paths.checkpoint_path)) or "."
    best_path = os.path.join(checkpoint_dir, "train_env_best.pt")
    metadata = build_native_checkpoint_metadata(
        context,
        loop_state,
        global_step=int(plan.global_step),
        handoff_compatibility=state.get("handoff_compatibility"),
    )
    save_training_checkpoint(
        best_path,
        metadata=metadata,
        agent=components.agent,
        replay=components.replay,
        include_replay=bool(configs.checkpoint.save_replay_in_checkpoint),
        rng_state=capture_rng_state(),
        extra_fields={"train_env_best_score": float(score)},
    )
    print(
        f"native_train_best step={int(plan.global_step)} score={score:.6f} path={best_path}",
        flush=True,
    )


def _persist_native_training_log(
    context   : TrainerRuntimeContext,
    configs   : RuntimeConfigBundle,
    state     : NativeTrainerState,
    loop_state: TrainingLoopStartupState,
    plan,
) -> None:
    """Process for `_persist_native_training_log`

    Steps:
    - Resolve inputs for `_persist_native_training_log` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    env = state.get("env")
    step_result = _last_step_result(state)
    if env is None or step_result is None:
        return
    components = _native_components(state)
    state_machine = _state_machine_module()
    tensors = _topdown_tensors(env, state_machine)
    active_mask = _step_active_mask(step_result, env)
    reward_tensor = getattr(step_result, "reward_tensor", None)
    done_flags = _step_done_flags(step_result, env)
    term_means = per_term_reward_means(env)
    topdown_metrics = _topdown_metrics(env, tensors, active_mask)
    lift_stage_metrics = _lift_stage_metrics(env, tensors, active_mask, state_machine)
    assist_mix, assist_arm_mix, assist_finger_mix = _assist_values(state, configs)
    reward0 = _safe_scalar(reward_tensor, default=0.0)
    success_flags = termination_term_flags(env, "success")
    block_drift_flags = termination_term_flags(env, "block_drift")
    done0 = bool(done_flags.reshape(-1)[0].item()) if torch.is_tensor(done_flags) and done_flags.numel() else False
    row: dict[str, Any] = {
        "mode"          : "topdown_curriculum_summary",
        "global_step"   : int(plan.global_step),
        "episode_idx"   : _episode_idx(loop_state),
        "episode_step"  : _episode_step(env, loop_state),
        "training_phase": _phase_name_from_context(context, configs, int(plan.global_step)),
        "action_source" : _action_source(state),
        "reward"        : reward0,
        "done"          : int(done0),
        "done_envs"     : int(done_flags.to(dtype=torch.bool).sum().item()) if torch.is_tensor(done_flags) else 0,
        "success": (
            int(bool(success_flags.reshape(-1)[0].item()))
            if success_flags.numel()
            else int(_safe_scalar(tensors.get("success")) >= 0.5)
        ),
        "block_drift"          : int(bool(block_drift_flags.reshape(-1)[0].item())) if block_drift_flags.numel() else 0,
        "replay_size"          : int(plan.replay_size),
        "num_added"            : int(plan.num_added),
        "transitions_collected": int(loop_state.transitions_collected),
        "assist_mix"           : assist_mix,
        "assist_arm_mix"       : assist_arm_mix,
        "assist_finger_mix"    : assist_finger_mix,
        "stage"                : _safe_int(tensors.get("stage"), default=-1),
        "contact"              : _safe_scalar(tensors.get("contact")),
        "strict"               : _safe_scalar(tensors.get("strict_light_contact")),
        "thumb_contact"        : _safe_scalar(tensors.get("thumb_contact")),
        "index_contact"        : _safe_scalar(tensors.get("index_contact")),
        "lift"                 : _safe_scalar(tensors.get("lift")),
        "block_disp"           : _safe_scalar(tensors.get("block_disp")),
        "align"                : _safe_scalar(tensors.get("align_face_dist")),
        "align_angle"          : _safe_scalar(tensors.get("align_angle")),
        "finger_unlock"        : _safe_scalar(tensors.get("finger_unlock")),
        "reward_terms"         : term_means,
        "topdown"              : topdown_curriculum_state(env, topdown_curriculum_task=True, env_id=0),
        "topdown_metrics"      : topdown_metrics,
        "lift_stage_metrics"   : lift_stage_metrics,
    }
    _write_jsonl(context.paths.log_jsonl, row)
    if bool(getattr(plan, "should_log", False)) and not configs.eval.play:
        _save_train_best_checkpoint(
            context,
            configs,
            state,
            loop_state,
            plan,
            score=_train_env_score(tensors, topdown_metrics, active_mask),
        )

    writer = components.tensorboard_writer
    if writer is None:
        return
    events = []
    events.extend(
        finite_scalar_events(
            "train",
            {
                "reward"           : reward0,
                "replay_size"      : int(plan.replay_size),
                "num_added"        : int(plan.num_added),
                "done_envs"        : row["done_envs"],
                "assist_mix"       : row["assist_mix"],
                "assist_arm_mix"   : row["assist_arm_mix"],
                "assist_finger_mix": row["assist_finger_mix"],
            },
            int(plan.global_step),
        )
    )
    events.extend(train_env_metric_events(tensors, active_env_mask=active_mask, global_step=int(plan.global_step)))
    events.extend(topdown_metric_events(topdown_metrics, global_step=int(plan.global_step)))
    events.extend(
        ScalarEvent(str(name), float(value), int(plan.global_step))
        for name, value in lift_stage_metrics.items()
        if not math.isnan(float(value))
    )
    events.extend(reward_term_events(term_means, global_step=int(plan.global_step)))
    events.extend(
        update_info_events(
            loop_state.last_update_info,
            actor_update_info=loop_state.last_actor_update_info,
            global_step=int(plan.global_step),
        )
    )
    write_scalar_events(writer, events)
    if hasattr(writer, "flush"):
        writer.flush()


def native_log_event_line(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    plan,                                  # Param: precomputed plan object consumed by this helper
) -> str:
    """Format a topdown native loop progress event."""
    env = state.get("env")
    step_result = _last_step_result(state)
    if env is None or step_result is None:
        return _compact_native_step_line(context, loop_state, plan)

    try:
        state_machine = _state_machine_module()
        tensors = _topdown_tensors(env, state_machine)
        thumb_err, index_err, thumb_delta, index_delta = _target_error_tensors(env, state_machine)
        reward = getattr(step_result, "reward_tensor", None)
        done_flags = _step_done_flags(step_result, env)
        active_mask = _step_active_mask(step_result, env)
        term_means = per_term_reward_means(env)
        assist_mix, assist_arm_mix, assist_finger_mix = _assist_values(state, configs)
        selection = _last_action_selection(state)
        diagnostics = getattr(selection, "diagnostics", None)
        update_bits = format_update_bits(
            loop_state.last_update_info,
            actor_update_info=loop_state.last_actor_update_info,
            actor_teacher_arm_mse=float(getattr(diagnostics, "actor_teacher_arm_mse", math.nan)),
            actor_teacher_finger_mse=float(getattr(diagnostics, "actor_teacher_finger_mse", math.nan)),
        )
        success_flags = termination_term_flags(env, "success")
        block_drift_flags = termination_term_flags(env, "block_drift")
        done0 = bool(done_flags.reshape(-1)[0].item()) if torch.is_tensor(done_flags) and done_flags.numel() else False
        done_bits = format_topdown_done_bits(
            success=bool(success_flags.reshape(-1)[0].item()) if success_flags.numel() else bool(_safe_scalar(tensors.get("success")) >= 0.5),
            block_drift=bool(block_drift_flags.reshape(-1)[0].item()) if block_drift_flags.numel() else False,
            done=done0,
        )
        stage_bits = _topdown_stage_bits(env, tensors, loop_state, active_mask, state_machine)
        summary = ProgressLineSummary(
            global_step=int(plan.global_step),
            episode_idx=_episode_idx(loop_state),
            episode_step=_episode_step(env, loop_state),
            phase_name=_phase_name_from_context(context, configs, int(plan.global_step)),
            action_source=_action_source(state),
            reward=_safe_scalar(reward, default=0.0),
            tip=max(_safe_scalar(thumb_err), _safe_scalar(index_err)),
            palm=_safe_scalar(_safe_call_tensor(env, getattr(state_machine, "palm_distance", None), default=0.0)) if state_machine is not None else 0.0,
            palm_height_error=_safe_scalar(_safe_call_tensor(env, getattr(state_machine, "palm_height_error", None), default=0.0)) if state_machine is not None else 0.0,
            orient_deg=(
                _safe_scalar(torch.rad2deg(_safe_call_tensor(env, getattr(state_machine, "palm_drop_axis_error_rad", None), default=0.0)))
                if state_machine is not None
                else 0.0
            ),
            thumb_err=_safe_scalar(thumb_err),
            idx_err=_safe_scalar(index_err),
            thumb_target_delta=tuple(float(v) for v in thumb_delta.reshape(-1, 3)[0].tolist()),
            index_target_delta=tuple(float(v) for v in index_delta.reshape(-1, 3)[0].tolist()),
            done_envs=int(done_flags.to(dtype=torch.bool).sum().item()) if torch.is_tensor(done_flags) else 0,
            geometry_frame="live_state",
            replay_size=int(plan.replay_size),
            assist_mix=assist_mix,
            assist_arm_mix=assist_arm_mix,
            assist_finger_mix=assist_finger_mix,
            stage_bits=stage_bits,
            done_bits=done_bits,
            update_bits=update_bits,
            reward_term_bits=format_reward_term_bits(term_means),
        )
        return format_progress_line(summary)
    except Exception as exc:
        return _compact_native_step_line(context, loop_state, plan, log_error=repr(exc))


def trace_native_log_event(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    plan,                                                            # Param: precomputed plan object consumed by this helper
    *,
    callbacks: NativeLogEventCallbacks = NativeLogEventCallbacks(),  # Param: input value used as callbacks
) -> str:
    """Trace one native progress event and return the line

    Steps:
    - Resolve inputs for `trace_native_log_event` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    line = native_log_event_line(context, configs, state, loop_state, plan)
    if callbacks.trace_fn is not None:
        callbacks.trace_fn(line)
    env = state.get("env")
    step_result = _last_step_result(state)
    if env is not None and step_result is not None:
        try:
            _persist_native_training_log(context, configs, state, loop_state, plan)
        except Exception as exc:
            if callbacks.trace_fn is not None:
                callbacks.trace_fn(f"native_log_persist_error error={repr(exc)[:160]}")
    return line


def native_log_event(
    callbacks: NativeLogEventCallbacks = NativeLogEventCallbacks(),  # Param: input value used as callbacks
) -> PlanFn:
    """Build a PlanFn for native progress logging"""

    def _event(context, configs, state, loop_state, plan):
        trace_native_log_event(context, configs, state, loop_state, plan, callbacks=callbacks)

    return _event


def _save_scheduled_native_checkpoint(
    *,
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    components: NativeTrainingComponents,  # Param: input value used as components
    callbacks : NativeCheckpointEventCallbacks,  # Param: integer input for callbacks
) -> Callable[[ScheduledCheckpointJob], Mapping[str, object]]:
    def _save(job: ScheduledCheckpointJob) -> Mapping[str, object]:
        """Process for `_save`

        Steps:
        - Resolve inputs for `_save` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        metadata = build_native_checkpoint_metadata(
            context,
            loop_state,
            global_step=job.global_step,
            handoff_compatibility=state.get("handoff_compatibility"),
        )
        path = context.paths.checkpoint_path if job.dest_path is None else job.dest_path
        payload = callbacks.save_checkpoint_fn(
            path,
            metadata=metadata,
            agent=components.agent,
            replay=components.replay,
            include_replay=job.include_replay,
            rng_state=capture_rng_state(),
        )
        if callbacks.trace_fn is not None:
            callbacks.trace_fn(
                f"native_checkpoint label={job.label} step={int(job.global_step)} path={path}"
            )
        return payload

    return _save


def run_native_checkpoint_event(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    plan,                                                                          # Param: precomputed plan object consumed by this helper
    *,
    callbacks: NativeCheckpointEventCallbacks = NativeCheckpointEventCallbacks(),  # Param: integer input for callbacks
) -> None:
    """Run scheduled native checkpoint jobs for one loop plan"""
    components = _native_components(state)
    save_fn = _save_scheduled_native_checkpoint(
        context=context,
        state=state,
        loop_state=loop_state,
        components=components,
        callbacks=callbacks,
    )
    run_scheduled_checkpoint_plan(
        plan.checkpoint_plan,
        save_fn,
        remove_path_fn=callbacks.remove_path_fn,
    )
    handoff_path = str(configs.checkpoint.handoff_checkpoint_path or "").strip()
    handoff_key = "_native_handoff_checkpoint_saved"
    handoff_saved = bool(loop_state.best_eval_state.get(handoff_key, 0.0))
    if handoff_path and not handoff_saved and int(plan.global_step) >= int(configs.counts.rl_phase_start_steps):
        save_fn(
            ScheduledCheckpointJob(
                label="handoff_checkpoint",
                global_step=int(plan.global_step),
                dest_path=handoff_path,
                include_replay=True,
            )
        )
        loop_state.best_eval_state[handoff_key] = 1.0
        if configs.checkpoint.stop_after_handoff_checkpoint:
            loop_state.skip_training_after_handoff_reuse = True


def native_checkpoint_event(
    callbacks: NativeCheckpointEventCallbacks = NativeCheckpointEventCallbacks(),  # Param: integer input for callbacks
) -> PlanFn:
    """Build a PlanFn for native scheduled checkpoint jobs"""

    def _event(context, configs, state, loop_state, plan):
        run_native_checkpoint_event(context, configs, state, loop_state, plan, callbacks=callbacks)

    return _event


def run_native_eval_event(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    plan,                                  # Param: precomputed plan object consumed by this helper
    *,
    callbacks: NativeEvalEventCallbacks,   # Param: input value used as callbacks
) -> tuple[NativeEvalResult, ...]:
    """Run configured native eval episodes for one loop plan

    Steps:
    - Resolve inputs for `run_native_eval_event` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    components = _native_components(state)
    gate_config = components.td3_config.gate_config
    results: list[NativeEvalResult] = []
    episode_count = configs.eval.play_episodes if configs.eval.play else configs.eval.eval_episodes
    for episode_idx in range(max(0, int(episode_count))):
        base_config = NativeEvalConfig(
            obs_keys=context.obs_keys,
            global_step=int(plan.global_step),
            eval_episode_idx=episode_idx,
            max_steps=configs.eval.eval_steps,
            teacher_assist_mix=configs.eval.eval_teacher_assist_mix,
            num_arm=gate_config.num_arm,
            num_fingers=gate_config.num_fingers,
        )
        config = (
            callbacks.eval_config_fn(context, configs, state, loop_state, plan, episode_idx, base_config)
            if callbacks.eval_config_fn is not None
            else base_config
        )
        result = run_native_eval_episode(
            config,
            callbacks.eval_callbacks_fn(context, configs, state, loop_state, episode_idx),
        )
        if callbacks.result_fn is not None:
            callbacks.result_fn(result)
        results.append(result)
    if callbacks.post_eval_fn is not None:
        callbacks.post_eval_fn(context, configs, state, loop_state, plan, tuple(results))
    return tuple(results)


def native_eval_event(callbacks: NativeEvalEventCallbacks) -> PlanFn:
    """Build a PlanFn for native scheduled eval episodes"""

    def _event(context, configs, state, loop_state, plan):
        run_native_eval_event(context, configs, state, loop_state, plan, callbacks=callbacks)

    return _event
