"""

Env attr conversion for native contact-teacher parts

File map:

NativeContactTeacherAttrConfig:                   Contact-teacher attr names and action layout
native_contact_teacher_attr_config_from_runtime:  Build contact-teacher attr config from runtime config
_device:                                          Handle device logic
_num_envs:                                        Handle num envs logic
_scalar_attr:                                     Handle scalar attr logic
_vector_attr:                                     Handle vector attr logic
_bool_tensor_attr:                                Handle bool tensor attr logic
_episode_step:                                    Handle episode step logic
_safe_state_machine_tensors:                      Handle safe state machine tensors logic
_write_attr:                                      Handle write attr logic
_contact_attrs_have_signal:                       Handle contact attrs have signal logic
_env_float:                                       Handle env float logic
_update_missing_contact_teacher_attrs:            Populate env contact-teacher attrs when native owns the teacher state
contact_teacher_finger_fraction_matrix:           Build per-finger closure fractions from thumb index middle attrs
contact_teacher_parts_from_env_attrs:             Build native contact-teacher parts from env attrs
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from ..actions.action_space import compute_teacher_finger_reduced_in_current_mode
from ..core.configs import RuntimeConfigBundle
from ..teacher.teacher_actions import TopdownContactTeacherParts
from ..teacher.contact_teacher import (
    apply_hold_fractions_on_new_latch,
    contact_descent_from_missing,
    contact_teacher_ready_mask,
    ensure_contact_teacher_state_attrs,
    missing_tip_servo,
    update_contact_latches,
)


@dataclass(frozen=True)
class NativeContactTeacherAttrConfig:
    """Contact-teacher attr names and action layout"""

    prefix            : str   = "_topdown_contact_teacher_"  # Field: string prefix value used by native contact teacher attr config
    num_arm           : int   = 6  # Field: number of arm action dimensions in the active layout
    num_fingers       : int   = 7  # Field: number of finger action dimensions in the active layout
    max_fraction      : float = 1.0  # Field: floating-point max fraction value used by native contact teacher attr config
    middle_scale      : float = 0.0  # Field: multiplier applied to middle terms
    use_middle_teacher: bool  = False  # Field: boolean value indicating the use middle teacher state for native contact teacher attr config


def native_contact_teacher_attr_config_from_runtime(
    configs: RuntimeConfigBundle,      # Param: typed runtime config bundle used to derive this plan
    *,
    use_middle_teacher: bool = False,  # Param: boolean input selecting whether middle teacher is used
    num_arm           : int  = 6,  # Param: number of arm action dimensions in the active layout
    num_fingers       : int  = 7,  # Param: number of finger action dimensions in the active layout
) -> NativeContactTeacherAttrConfig:
    """Build contact-teacher attr config from runtime config"""
    return NativeContactTeacherAttrConfig(
        num_arm=int(num_arm),
        num_fingers=int(num_fingers),
        max_fraction=max(float(configs.teacher.topdown_contact_teacher_max_fraction), 0.0),
        middle_scale=max(float(configs.teacher.topdown_contact_teacher_middle_scale), 0.0),
        use_middle_teacher=bool(use_middle_teacher),
    )


def _device(env: object, fallback: torch.Tensor) -> torch.device:
    value = getattr(env, "device", None)
    return torch.device(value) if value is not None else fallback.device


def _num_envs(env: object, fallback: torch.Tensor) -> int:
    return int(getattr(env, "num_envs", int(fallback.shape[0])))


def _scalar_attr(
    env : object,  # Param: environment or backend object used for runtime calls
    name: str,  # Param: attribute, field, or option name being resolved
    *,
    default : float,  # Param: fallback value used when the input omits or rejects a setting
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device,  # Param: torch device where tensors are read or allocated
) -> torch.Tensor:
    value = getattr(env, name, None)
    if torch.is_tensor(value) and tuple(value.shape) == (num_envs,):
        return value.to(device=device, dtype=torch.float32)
    return torch.full((num_envs,), float(default), dtype=torch.float32, device=device)


def _vector_attr(
    env : object,  # Param: environment or backend object used for runtime calls
    name: str,  # Param: attribute, field, or option name being resolved
    *,
    width   : int,  # Param: integer input for width
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device,  # Param: torch device where tensors are read or allocated
) -> torch.Tensor:
    value = getattr(env, name, None)
    if torch.is_tensor(value) and tuple(value.shape) == (num_envs, width):
        return value.to(device=device, dtype=torch.float32)
    return torch.zeros((num_envs, width), dtype=torch.float32, device=device)


def _bool_tensor_attr(
    env : object,  # Param: environment/backend object used for runtime calls
    name: str,  # Param: attribute, field, or option name being resolved
    *,
    default : bool,
    num_envs: int,
    device  : torch.device,
) -> torch.Tensor:
    value = getattr(env, name, None)
    if torch.is_tensor(value) and tuple(value.shape) == (num_envs,):
        return value.to(device=device, dtype=torch.bool)
    return torch.full((num_envs,), bool(default), dtype=torch.bool, device=device)


def _episode_step(env: object, *, num_envs: int, device: torch.device) -> torch.Tensor:
    value = getattr(env, "episode_length_buf", None)
    if torch.is_tensor(value) and tuple(value.shape) == (num_envs,):
        return value.to(device=device, dtype=torch.float32)
    value = getattr(env, "episode_step", None)
    if torch.is_tensor(value) and tuple(value.shape) == (num_envs,):
        return value.to(device=device, dtype=torch.float32)
    return torch.zeros((num_envs,), dtype=torch.float32, device=device)


def _safe_state_machine_tensors(env: object, *, num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    zeros = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    zero_vec = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    values: dict[str, torch.Tensor] = {
        "close_gate": zeros.clone(),
        "thumb_contact": zeros.clone(),
        "index_contact": zeros.clone(),
        "middle_contact": zeros.clone(),
        "thumb_delta": zero_vec.clone(),
        "index_delta": zero_vec.clone(),
        "middle_delta": zero_vec.clone(),
    }
    try:
        from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp import state_machine
    except Exception:
        return values
    try:
        values["close_gate"] = state_machine.finger_xyz_block_center_gate(
            env,
            write_diagnostics=True,
        ).to(device=device, dtype=torch.float32).reshape(-1)
    except Exception:
        pass
    for key, fn_name in (
        ("thumb_contact", "thumb_contact_strength"),
        ("index_contact", "index_contact_strength"),
        ("middle_contact", "middle_contact_strength"),
    ):
        try:
            values[key] = getattr(state_machine, fn_name)(env).to(device=device, dtype=torch.float32).reshape(-1)
        except Exception:
            pass
    try:
        block_pos, _ = state_machine._block_pose(env)
        thumb_pos, index_pos, middle_pos = state_machine._active_finger_points(env)
        block_pos = block_pos.to(device=device, dtype=torch.float32)
        values["thumb_delta"] = block_pos - thumb_pos.to(device=device, dtype=torch.float32)
        values["index_delta"] = block_pos - index_pos.to(device=device, dtype=torch.float32)
        values["middle_delta"] = block_pos - middle_pos.to(device=device, dtype=torch.float32)
    except Exception:
        pass
    return values


def _write_attr(env: object, name: str, value: torch.Tensor) -> torch.Tensor:
    detached = value.detach().clone()
    setattr(env, name, detached)
    return detached


def _contact_attrs_have_signal(env: object, prefix: str, *, num_envs: int) -> bool:
    for suffix in ("thumb_fraction", "index_fraction", "closure_fraction", "descent_z", "tip_servo"):
        value = getattr(env, f"{prefix}{suffix}", None)
        if not torch.is_tensor(value):
            continue
        if value.shape[0] != int(num_envs):
            continue
        try:
            if bool(value.detach().abs().max().item() > 1.0e-8):
                return True
        except RuntimeError:
            continue
    return False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _update_missing_contact_teacher_attrs(
    *,
    env        : object,
    configs    : RuntimeConfigBundle,
    config     : NativeContactTeacherAttrConfig,
    num_envs   : int,
    device     : torch.device,
    prefix     : str,
) -> None:
    """Populate env contact-teacher attrs when native owns the teacher state."""
    step = _episode_step(env, num_envs=num_envs, device=device)
    state = ensure_contact_teacher_state_attrs(env, step, prefix=prefix)
    stage = getattr(env, "_topdown_stage", None)
    stage_t = stage.to(device=device) if torch.is_tensor(stage) and tuple(stage.shape) == (num_envs,) else None
    pose_ready = getattr(env, "_topdown_contact_pose_ready", None)
    pose_ready_t = (
        pose_ready.to(device=device, dtype=torch.bool)
        if torch.is_tensor(pose_ready) and tuple(pose_ready.shape) == (num_envs,)
        else None
    )
    sm = _safe_state_machine_tensors(env, num_envs=num_envs, device=device)
    close_gate = sm["close_gate"].clamp(0.0, 1.0)
    ready = contact_teacher_ready_mask(
        num_envs=num_envs,
        device=device,
        stage=stage_t,
        proximity=close_gate,
        pose_ready=pose_ready_t,
        proximity_threshold=0.05,
    )
    ready = ready | (close_gate > 0.0)
    max_fraction = max(float(config.max_fraction), 0.0)
    start_fraction = max(float(configs.teacher.topdown_contact_teacher_start_fraction), 0.0)
    close_rate = max(float(configs.teacher.topdown_contact_teacher_close_rate), 0.0)
    middle_active = torch.full((num_envs,), bool(config.use_middle_teacher), dtype=torch.bool, device=device)
    prev_thumb = state["thumb_fraction"].to(device=device, dtype=torch.float32)
    prev_index = state["index_fraction"].to(device=device, dtype=torch.float32)
    prev_middle = state["middle_fraction"].to(device=device, dtype=torch.float32)
    ready_f = ready.to(dtype=torch.float32) * close_gate
    starter = torch.full_like(prev_thumb, min(start_fraction, max_fraction))
    thumb = torch.where(ready, torch.maximum(prev_thumb, starter) + close_rate * ready_f, torch.zeros_like(prev_thumb))
    index = torch.where(ready, torch.maximum(prev_index, starter) + close_rate * ready_f, torch.zeros_like(prev_index))
    middle = torch.where(
        ready & middle_active,
        torch.maximum(prev_middle, starter) + close_rate * ready_f,
        torch.zeros_like(prev_middle),
    )
    thumb = thumb.clamp(0.0, max_fraction)
    index = index.clamp(0.0, max_fraction)
    middle = middle.clamp(0.0, max_fraction)

    latches = update_contact_latches(
        close_progress_ready=ready,
        thumb_contact=sm["thumb_contact"],
        index_contact=sm["index_contact"],
        middle_contact=sm["middle_contact"],
        prev_thumb_latched=state["thumb_latched"],
        prev_index_latched=state["index_latched"],
        prev_middle_latched=state["middle_latched"],
        middle_active=middle_active,
        threshold=0.08,
    )
    thumb_hold, index_hold, middle_hold = apply_hold_fractions_on_new_latch(
        thumb_fraction=thumb,
        index_fraction=index,
        middle_fraction=middle,
        newly_thumb_latched=latches.newly_thumb_latched,
        newly_index_latched=latches.newly_index_latched,
        newly_middle_latched=latches.newly_middle_latched,
        thumb_hold_fraction=state["thumb_hold_fraction"].to(device=device, dtype=torch.float32),
        index_hold_fraction=state["index_hold_fraction"].to(device=device, dtype=torch.float32),
        middle_hold_fraction=state["middle_hold_fraction"].to(device=device, dtype=torch.float32),
        hold_extra=0.04,
        max_fraction=max_fraction,
        hold_max_fraction=max_fraction,
    )

    thumb_missing = ready & (sm["thumb_contact"] < 0.08)
    index_missing = ready & (sm["index_contact"] < 0.08)
    middle_missing = ready & middle_active & (sm["middle_contact"] < 0.08)
    thumb_z_gap = sm["thumb_delta"][:, 2].clamp_min(0.0)
    index_z_gap = sm["index_delta"][:, 2].clamp_min(0.0)
    middle_z_gap = sm["middle_delta"][:, 2].clamp_min(0.0)
    required = torch.maximum(thumb, index)
    descent = contact_descent_from_missing(
        descent_ready=ready,
        thumb_missing=thumb_missing,
        index_missing=index_missing,
        middle_missing=middle_missing,
        thumb_z_gap=thumb_z_gap,
        index_z_gap=index_z_gap,
        middle_z_gap=middle_z_gap,
        required_closure=required,
        base_descent=max(_env_float("TOPDOWN_CONTACT_TEACHER_DESCENT_Z", 0.08), 0.0),
        extra_descent=max(_env_float("TOPDOWN_CONTACT_TEACHER_MISSING_CONTACT_EXTRA_DESCENT", 0.005), 0.0),
        z_target=0.0,
        z_full=0.06,
        min_closure=start_fraction,
        full_closure=max(max_fraction, start_fraction + 1.0e-6),
    )
    tip_servo = missing_tip_servo(
        thumb_err_vec=sm["thumb_delta"],
        index_err_vec=sm["index_delta"],
        thumb_missing=thumb_missing,
        index_missing=index_missing,
        z_need=descent.z_need,
        gain=0.5,
        max_m=0.05,
    )
    tip_servo = tip_servo * ready.to(dtype=tip_servo.dtype).unsqueeze(-1)
    servo_m = torch.linalg.norm(tip_servo, dim=1)
    xy_offset = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    inward = torch.full(
        (num_envs,),
        max(_env_float("TOPDOWN_CONTACT_TEACHER_INWARD_M", 0.0), 0.0),
        dtype=torch.float32,
        device=device,
    )
    inward = inward * ready.to(dtype=torch.float32)

    _write_attr(env, f"{prefix}thumb_fraction", thumb)
    _write_attr(env, f"{prefix}index_fraction", index)
    _write_attr(env, f"{prefix}middle_fraction", middle)
    _write_attr(env, f"{prefix}closure_fraction", required)
    _write_attr(env, f"{prefix}thumb_latched", latches.thumb_latched)
    _write_attr(env, f"{prefix}index_latched", latches.index_latched)
    _write_attr(env, f"{prefix}middle_latched", latches.middle_latched)
    _write_attr(env, f"{prefix}thumb_hold_fraction", thumb_hold)
    _write_attr(env, f"{prefix}index_hold_fraction", index_hold)
    _write_attr(env, f"{prefix}middle_hold_fraction", middle_hold)
    _write_attr(env, f"{prefix}descent_z", descent.descent)
    _write_attr(env, f"{prefix}descent_z_need", descent.z_need)
    _write_attr(env, f"{prefix}descent_closure_gate", descent.closure_gate)
    _write_attr(env, f"{prefix}xy_offset", xy_offset)
    _write_attr(env, f"{prefix}inward_m", inward)
    _write_attr(env, f"{prefix}tip_servo", tip_servo)
    _write_attr(env, f"{prefix}tip_servo_m", servo_m)
    _write_attr(env, f"{prefix}precenter_servo_m", servo_m)
    _write_attr(env, f"{prefix}prehold_servo_m", servo_m)
    _write_attr(env, f"{prefix}center_servo_m", servo_m)
    _write_attr(env, f"{prefix}ready", ready)
    _write_attr(env, f"{prefix}finger_ready", ready)
    _write_attr(env, f"{prefix}center_gate", close_gate > 0.0)
    _write_attr(env, f"{prefix}finger_close_gate", close_gate)
    _write_attr(env, f"{prefix}descent_ready", ready)
    _write_attr(env, f"{prefix}descent_start_gate", ready & (descent.descent > 0.0))
    _write_attr(env, f"{prefix}thumb_missing", thumb_missing.to(dtype=torch.float32))
    _write_attr(env, f"{prefix}index_missing", index_missing.to(dtype=torch.float32))
    _write_attr(env, f"{prefix}middle_missing", middle_missing.to(dtype=torch.float32))
    setattr(env, "_native_contact_teacher_generated_attrs", True)


def contact_teacher_finger_fraction_matrix(
    *,
    thumb_fraction : torch.Tensor,  # Param: tensor input carrying thumb fraction values
    index_fraction : torch.Tensor,  # Param: tensor input carrying index fraction values
    middle_fraction: torch.Tensor,  # Param: tensor input carrying middle fraction values
    config         : NativeContactTeacherAttrConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Build per-finger closure fractions from thumb index middle attrs

    Steps:
    - Resolve inputs for `contact_teacher_finger_fraction_matrix` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    num_envs = int(thumb_fraction.shape[0])
    max_fraction = max(float(config.max_fraction), 0.0)
    fingers = torch.zeros(
        (num_envs, int(config.num_fingers)),
        dtype=torch.float32,
        device=thumb_fraction.device,
    )
    fingers[:, 0:3] = thumb_fraction.clamp(0.0, max_fraction).unsqueeze(-1)
    fingers[:, 3:5] = index_fraction.clamp(0.0, max_fraction).unsqueeze(-1)
    middle_scale = max(float(config.middle_scale), 0.0)
    if int(config.num_fingers) >= 7 and middle_scale > 0.0:
        source = middle_fraction if bool(config.use_middle_teacher) else index_fraction
        fingers[:, 5:7] = (source * middle_scale).clamp(0.0, max_fraction).unsqueeze(-1)
    return fingers


def contact_teacher_parts_from_env_attrs(
    *,
    env           : object,  # Param: environment or backend object used for runtime calls
    mapped_indices: torch.Tensor,  # Param: tensor input carrying mapped indices values
    mapped_scales : torch.Tensor,  # Param: tensor input carrying mapped scales values
    configs       : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    attr_config   : NativeContactTeacherAttrConfig | None = None,  # Param: input value used as attr config
) -> TopdownContactTeacherParts:
    """Build native contact-teacher parts from env attrs

    Steps:
    - Resolve inputs for `contact_teacher_parts_from_env_attrs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    config = (
        native_contact_teacher_attr_config_from_runtime(configs)
        if attr_config is None
        else attr_config
    )
    device = _device(env, mapped_indices)
    num_envs = _num_envs(env, mapped_indices)
    prefix = config.prefix
    if bool(getattr(env, "_native_contact_teacher_generated_attrs", False)) or not _contact_attrs_have_signal(
        env,
        prefix,
        num_envs=num_envs,
    ):
        _update_missing_contact_teacher_attrs(
            env=env,
            configs=configs,
            config=config,
            num_envs=num_envs,
            device=device,
            prefix=prefix,
        )
    thumb = _scalar_attr(
        env,
        f"{prefix}thumb_fraction",
        default=0.0,
        num_envs=num_envs,
        device=device,
    )
    index = _scalar_attr(
        env,
        f"{prefix}index_fraction",
        default=0.0,
        num_envs=num_envs,
        device=device,
    )
    middle = _scalar_attr(
        env,
        f"{prefix}middle_fraction",
        default=0.0,
        num_envs=num_envs,
        device=device,
    )
    fraction = contact_teacher_finger_fraction_matrix(
        thumb_fraction=thumb,
        index_fraction=index,
        middle_fraction=middle,
        config=config,
    )
    finger_action = compute_teacher_finger_reduced_in_current_mode(
        env,
        mapped_indices.to(device=device),
        mapped_scales.to(device=device),
        fraction,
        num_arm=config.num_arm,
        num_fingers=config.num_fingers,
        finger_action_mode=configs.teacher.finger_action_mode,
        finger_delta_scale=configs.teacher.finger_delta_scale,
    )
    closure_attr = getattr(env, f"{prefix}closure_fraction", None)
    if torch.is_tensor(closure_attr) and tuple(closure_attr.shape) == (num_envs,):
        closure = closure_attr.to(device=device, dtype=torch.float32)
    else:
        closure = torch.maximum(thumb, index)
        if bool(config.use_middle_teacher):
            closure = torch.maximum(closure, middle)
    return TopdownContactTeacherParts(
        finger_action=finger_action,
        closure_fraction=closure,
        descent=_scalar_attr(
            env,
            f"{prefix}descent_z",
            default=0.0,
            num_envs=num_envs,
            device=device,
        ),
        xy_offset=_vector_attr(
            env,
            f"{prefix}xy_offset",
            width=2,
            num_envs=num_envs,
            device=device,
        ),
        inward=_scalar_attr(
            env,
            f"{prefix}inward_m",
            default=0.0,
            num_envs=num_envs,
            device=device,
        ),
        tip_servo=_vector_attr(
            env,
            f"{prefix}tip_servo",
            width=3,
            num_envs=num_envs,
            device=device,
        ),
    )
