"""
Flat-observation action gates for replay and TD3 targets

This module provides helper functions to apply action gating based on flat policy observations, used by replay and TD3 target actions
The gates implemented here include contact finger open gating, curriculum unlock gating, and align-task open-hand gating. 
The configuration for these gates is encapsulated in the `ActionGateConfig` dataclass, which allows for flexible control over the 
gating behavior based on the task and curriculum state.

topdown_xyz_preload_fraction:               Clamp topdown XYZ preload fraction to the action range
finger_unlock_requires_arm_hold_enabled:    Return whether finger unlock requires arm hold
finger_unlock_requires_center_enabled:      Return whether finger unlock requires center readiness
topdown_finger_close_gate_mode:             Normalize the configured live finger-close gate mode
topdown_finger_center_gate_from_attr:       Read a topdown finger-center bool gate from an env attr
topdown_finger_center_action_gate:          Read the latched pre-contact center gate
topdown_finger_center_live_gate:            Read the live pre-contact center gate
topdown_finger_xyz_close_gate:              Read the geometry-only finger close gate
preserve_xyz_close_after_lift_latch:        Force xyz close permission after the lift latch fires  
topdown_finger_close_gate:                  Return the configured finger close gate as a float tensor
_finger_bounds:                             Identify the finger action column bounds in the active layout
_replace_fingers:                           Replace finger columns in an action tensor with new values
apply_contact_finger_close_cap:             Clamp contact-stage finger close actions to a preload band
apply_contact_finger_gate_tensor:           Scale finger actions by a smooth unlock gate
contact_finger_unlock_gate_from_flat_obs:   Read contact finger-unlock gate from flat policy observations
apply_contact_finger_open_from_flat_obs:    Apply observation-derived contact finger gate
apply_contact_finger_open_until_ready:      Apply live contact finger gate until readiness unlocks closure
scale_finger_columns:                       Scale only finger action columns
apply_middle_index_mirror:                  Copy index finger commands onto middle finger commands when configured
apply_align_open_hand_action:               Zero finger columns for the alignment-only open-hand task
topdown_finger_preload_floor_like:          Return per-finger preload floors for topdown grasp fingers
gate_finger_columns_above_preload:          Keep fixed preload and gate closure above that preload
curriculum_finger_unlock_from_flat_obs:     Read curriculum finger-unlock progress from flat policy observations
apply_curriculum_finger_unlock_from_flat_obs: Apply replay-observation finger unlock gating
apply_curriculum_finger_unlock_live:          Apply live curriculum finger unlock from explicit tensors
apply_replay_action_gates_from_flat_obs:      Apply observation-derived gates used by replay and target actions


"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionGateConfig:
    """Configuration for import-safe action gating"""

    num_arm                             : int  # number of arm action dimensions in the active layout
    num_fingers                         : int  # number of finger action dimensions in the active layout
    align_task                          : bool       = False  # boolean value indicating the align task state for action gate config
    contact_task                        : bool       = False  # boolean value indicating the contact task state for action gate config
    topdown_curriculum                  : bool       = True  # boolean value indicating the topdown curriculum state for action gate config
    contact_finger_close_cap            : float      = 0.70  # floating-point contact finger close cap value used by action gate config
    contact_unlock_obs_col              : int | None = None  # integer contact unlock obs col value tracked by action gate config
    curriculum_unlock_obs_col           : int | None = None  # integer curriculum unlock obs col value tracked by action gate config
    stage_one_hot_obs_col               : int | None = None  # integer stage one hot obs col value tracked by action gate config
    contact_unlock_gate_threshold       : float      = 0.5  # threshold/tolerance used when evaluating contact unlock gate threshold
    contact_unlock_gate_start           : float      = 0.20  # floating-point contact unlock gate start value used by action gate config
    mirror_middle_to_index              : bool       = False  # index identifying the mirror middle to entry
    three_finger_centering              : bool       = False  # boolean value indicating the three finger centering state for action gate config
    topdown_contact_teacher_middle_scale: float      = 1.0  # multiplier applied to topdown contact teacher middle terms
    finger_action_mode                  : str        = "absolute"  # configured interpretation of finger action columns
    finger_close_gate_mode              : str        = "center"  # string finger close gate mode value used by action gate config
    finger_xyz_preload_fraction         : float      = 0.20  # floating-point finger xyz preload fraction value used by action gate config


@dataclass(frozen=True)
class LiveFingerUnlockConfig:
    """Configuration for live curriculum finger unlock"""

    contact_teacher                  : bool  = False  # boolean value indicating the contact teacher state for live finger unlock config
    contact_teacher_bypass_unlock    : bool  = False  # boolean value indicating the contact teacher bypass unlock state for live finger unlock config
    contact_teacher_start_fraction   : float = 0.0  # floating-point contact teacher start fraction value used by live finger unlock config
    contact_teacher_arm_hold_fallback: bool  = True  # boolean value indicating the contact teacher arm hold fallback state for live finger unlock config
    finger_unlock_requires_arm_hold  : bool  = False  # boolean value indicating the finger unlock requires arm hold state for live finger unlock config
    finger_unlock_requires_center    : bool  = False  # boolean value indicating the finger unlock requires center state for live finger unlock config


@dataclass(frozen=True)
class LiveFingerUnlockResult:
    """Result of live curriculum finger unlock gating"""

    action            : torch.Tensor  # environment action tensor selected for the step
    effective_progress: torch.Tensor | None  # tensor containing effective progress values for batched env rows
    arm_hold_gate     : torch.Tensor | None  # tensor containing arm hold gate values for batched env rows


def _bool_config_value(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def topdown_xyz_preload_fraction(value: object = 0.20) -> float:
    """Clamp topdown XYZ preload fraction to the action range"""
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        fraction = 0.20
    return min(max(fraction, 0.0), 1.0)


def finger_unlock_requires_arm_hold_enabled(value: object = False) -> bool:
    """Return whether finger unlock requires arm hold"""
    return _bool_config_value(value, False)


def finger_unlock_requires_center_enabled(value: object = False) -> bool:
    """Return whether finger unlock requires center readiness"""
    return _bool_config_value(value, False)


def topdown_finger_close_gate_mode(value: object = "center") -> str:
    """Normalize the configured live finger-close gate mode"""
    return str(value).strip().lower()


def topdown_finger_center_gate_from_attr(
    env,                                 # Param: environment or backend object used for runtime calls
    attr_name: str,                      # Param: string input for attr name
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    requires_center        : bool,  # Param: boolean input controlling requires center
    ensure_stage_updated=None,           # Param: input value used as ensure stage updated
    write_attr_name: str | None = None,  # Param: string input for write attr name
) -> torch.Tensor:
    """Read a topdown finger-center bool gate from an env attr"""
    shape = (int(env.num_envs),)
    if not bool(topdown_curriculum_task) or not bool(requires_center):
        gate = torch.ones(shape, dtype=torch.bool, device=env.device)
    else:
        if ensure_stage_updated is not None:
            ensure_stage_updated(env)
        source = getattr(env, attr_name, None)
        if torch.is_tensor(source) and source.shape == shape:
            gate = source.to(device=env.device, dtype=torch.bool)
        else:
            gate = torch.zeros(shape, dtype=torch.bool, device=env.device)
    if write_attr_name is not None:
        setattr(env, write_attr_name, gate.detach().clone())
    return gate


def topdown_finger_center_action_gate(
    env,                            # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    requires_center        : bool,  # Param: boolean input controlling requires center
    ensure_stage_updated=None,      # Param: input value used as ensure stage updated
) -> torch.Tensor:
    """Read the latched pre-contact center gate"""
    return topdown_finger_center_gate_from_attr(
        env,
        "_topdown_finger_center_ready",
        topdown_curriculum_task=topdown_curriculum_task,
        requires_center=requires_center,
        ensure_stage_updated=ensure_stage_updated,
        write_attr_name="_finger_unlock_center_gate",
    )


def topdown_finger_center_live_gate(
    env,                            # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    requires_center        : bool,  # Param: boolean input controlling requires center
    ensure_stage_updated=None,      # Param: input value used as ensure stage updated
) -> torch.Tensor:
    """Read the live pre-contact center gate"""
    return topdown_finger_center_gate_from_attr(
        env,
        "_topdown_finger_center_live",
        topdown_curriculum_task=topdown_curriculum_task,
        requires_center=requires_center,
        ensure_stage_updated=ensure_stage_updated,
        write_attr_name="_finger_unlock_center_live_gate",
    )


def topdown_finger_xyz_close_gate(
    env,                            # Param: environment or backend object used for runtime calls
    *,
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    xyz_gate_fn=None,               # Param: callback used to compute or fetch xyz gate
) -> torch.Tensor:
    """Read the geometry-only finger close gate"""
    shape = (int(env.num_envs),)
    if not bool(topdown_curriculum_task):
        return torch.ones(shape, dtype=torch.float32, device=env.device)
    if xyz_gate_fn is None:
        return torch.zeros(shape, dtype=torch.float32, device=env.device)
    return xyz_gate_fn(env).to(device=env.device, dtype=torch.float32)


def preserve_xyz_close_after_lift_latch(
    close_gate  : torch.Tensor,  # Param: tensor input carrying close gate values
    lift_latched: torch.Tensor | None,  # Param: mask selecting env rows with lift latch active
) -> torch.Tensor:
    """Force xyz close permission after the lift latch fires"""
    if lift_latched is None:
        return close_gate
    latched = lift_latched.to(device=close_gate.device, dtype=torch.bool)
    if latched.shape != close_gate.shape:
        return close_gate
    return torch.where(latched, torch.ones_like(close_gate), close_gate)


def topdown_finger_close_gate(
    env,                            # Param: environment or backend object used for runtime calls
    *,
    mode                   : str,  # Param: string input for mode
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    requires_center        : bool,  # Param: boolean input controlling requires center
    xyz_gate_fn=None,               # Param: callback used to compute or fetch xyz gate
    ensure_stage_updated=None,      # Param: input value used as ensure stage updated
) -> torch.Tensor:
    """Return the configured finger close gate as a float tensor"""
    normalized = topdown_finger_close_gate_mode(mode)
    if normalized in ("off", "none", "disabled"):
        gate = torch.ones((int(env.num_envs),), dtype=torch.float32, device=env.device)
    elif normalized in ("xyz_front", "xyz"):
        gate = topdown_finger_xyz_close_gate(
            env,
            topdown_curriculum_task=topdown_curriculum_task,
            xyz_gate_fn=xyz_gate_fn,
        )
    else:
        gate = topdown_finger_center_action_gate(
            env,
            topdown_curriculum_task=topdown_curriculum_task,
            requires_center=requires_center,
            ensure_stage_updated=ensure_stage_updated,
        ).to(dtype=torch.float32)
    env._topdown_finger_close_gate = gate.detach().clone()
    return gate


def _finger_bounds(action: torch.Tensor, config: ActionGateConfig) -> tuple[int, int] | None:
    if action.shape[-1] == config.num_fingers:
        return 0, config.num_fingers
    end = config.num_arm + config.num_fingers
    if action.shape[-1] < end:
        return None
    return config.num_arm, end


def _replace_fingers(action: torch.Tensor, fingers: torch.Tensor, config: ActionGateConfig) -> torch.Tensor:
    """Process for `_replace_fingers`

    Steps:
    - Resolve inputs for `_replace_fingers` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    if start == 0 and end == action.shape[-1]:
        return fingers
    return torch.cat((action[..., :start], fingers, action[..., end:]), dim=-1)


def apply_contact_finger_close_cap(action: torch.Tensor, config: ActionGateConfig) -> torch.Tensor:
    """Clamp contact-stage finger close actions to a preload band

    Steps:
    - Resolve inputs for `apply_contact_finger_close_cap` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not config.contact_task:
        return action
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    capped = torch.minimum(action[..., start:end], torch.full_like(action[..., start:end], config.contact_finger_close_cap))
    return _replace_fingers(action, capped, config)


def apply_contact_finger_gate_tensor(
    action     : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    unlock_gate: torch.Tensor,  # Param: tensor input carrying unlock gate values
    config     : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Scale finger actions by a smooth unlock gate

    Steps:
    - Resolve inputs for `apply_contact_finger_gate_tensor` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not config.contact_task:
        return action
    span = max(config.contact_unlock_gate_threshold - config.contact_unlock_gate_start, 1.0e-6)
    gate_scale = torch.clamp((unlock_gate - config.contact_unlock_gate_start) / span, 0.0, 1.0)
    gate_scale = gate_scale * gate_scale * (3.0 - 2.0 * gate_scale)
    if gate_scale.dim() == 1:
        gate_scale = gate_scale.unsqueeze(-1)
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    return _replace_fingers(action, action[..., start:end] * gate_scale, config)


def contact_finger_unlock_gate_from_flat_obs(
    flat_obs: torch.Tensor,  # Param: tensor input carrying flat obs values
    config  : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor | None:
    """Read contact finger-unlock gate from flat policy observations"""
    if not config.contact_task:
        return None
    col = config.contact_unlock_obs_col
    if col is None or col >= flat_obs.shape[-1]:
        return torch.zeros((flat_obs.shape[0],), device=flat_obs.device, dtype=flat_obs.dtype)
    return flat_obs[:, col].clamp(0.0, 1.0)


def apply_contact_finger_open_from_flat_obs(
    action  : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    flat_obs: torch.Tensor,  # Param: tensor input carrying flat obs values
    config  : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Apply observation-derived contact finger gate"""
    unlock_gate = contact_finger_unlock_gate_from_flat_obs(flat_obs, config)
    if unlock_gate is None:
        return action
    return apply_contact_finger_gate_tensor(
        action,
        unlock_gate.to(device=action.device, dtype=action.dtype),
        config,
    )


def apply_contact_finger_open_until_ready(
    action: torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    config: ActionGateConfig,  # Param: configuration object used by this helper
    *,
    unlock_gate : torch.Tensor | None = None,  # Param: tensor input carrying unlock gate values
    disable_gate: bool                = False,  # Param: boolean input controlling disable gate
) -> torch.Tensor:
    """Apply live contact finger gate until readiness unlocks closure"""
    if config.align_task or not config.contact_task or bool(disable_gate):
        return action
    gate = (
        torch.zeros((action.shape[0],), device=action.device, dtype=action.dtype)
        if unlock_gate is None
        else unlock_gate.to(device=action.device, dtype=action.dtype).reshape(-1)
    )
    return apply_contact_finger_gate_tensor(action, gate, config)


def scale_finger_columns(action: torch.Tensor, scale: torch.Tensor, config: ActionGateConfig) -> torch.Tensor:
    """Scale only finger action columns

    Steps:
    - Resolve inputs for `scale_finger_columns` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1)
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    return _replace_fingers(action, action[..., start:end] * scale, config)


def apply_middle_index_mirror(action: torch.Tensor, config: ActionGateConfig) -> torch.Tensor:
    """Copy index finger commands onto middle finger commands when configured

    Steps:
    - Resolve inputs for `apply_middle_index_mirror` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not config.mirror_middle_to_index or not config.topdown_curriculum:
        return action
    bounds = _finger_bounds(action, config)
    if bounds is None or config.num_fingers < 7:
        return action
    start, _ = bounds
    mirrored = action.clone()
    mirrored[..., start + 5 : start + 7] = mirrored[..., start + 3 : start + 5]
    return mirrored


def apply_align_open_hand_action(action: torch.Tensor, config: ActionGateConfig) -> torch.Tensor:
    """Zero finger columns for the alignment-only open-hand task

    Steps:
    - Resolve inputs for `apply_align_open_hand_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not config.align_task:
        return action
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    if start == 0 and end == action.shape[-1]:
        return torch.zeros_like(action)
    fingers = torch.zeros_like(action[..., start:end])
    return _replace_fingers(action, fingers, config)


def topdown_finger_preload_floor_like(
    fingers: torch.Tensor,  # Param: tensor input carrying fingers values
    floor  : float,  # Param: floating-point input for floor
    config : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Return per-finger preload floors for topdown grasp fingers

    Steps:
    - Resolve inputs for `topdown_finger_preload_floor_like` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    floor_tensor = torch.zeros_like(fingers)
    floor_value = max(float(floor), 0.0)
    if floor_value <= 0.0:
        return floor_tensor
    floor_tensor[..., 0 : min(5, fingers.shape[-1])] = floor_value
    middle_scale = max(float(config.topdown_contact_teacher_middle_scale), 0.0)
    if config.three_finger_centering and middle_scale > 0.0 and fingers.shape[-1] >= 7:
        floor_tensor[..., 5:7] = floor_value * middle_scale
    return floor_tensor


def gate_finger_columns_above_preload(
    action       : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    gate         : torch.Tensor,  # Param: tensor input carrying gate values
    preload_floor: float,  # Param: floating-point input for preload floor
    config       : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Keep fixed preload and gate closure above that preload

    Steps:
    - Resolve inputs for `gate_finger_columns_above_preload` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if config.finger_action_mode != "absolute":
        return scale_finger_columns(action, gate, config)
    if gate.dim() == 1:
        gate = gate.unsqueeze(-1)
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    fingers = action[..., start:end]
    floor = topdown_finger_preload_floor_like(fingers, preload_floor, config)
    gated = floor + torch.clamp(fingers - floor, min=0.0) * gate
    return _replace_fingers(action, gated, config)


def curriculum_finger_unlock_from_flat_obs(
    flat_obs: torch.Tensor,  # Param: tensor input carrying flat obs values
    config  : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor | None:
    """Read curriculum finger-unlock progress from flat policy observations"""
    if not config.topdown_curriculum:
        return None
    col = config.curriculum_unlock_obs_col
    if col is None or col >= flat_obs.shape[-1]:
        return None
    return flat_obs[:, col].clamp(0.0, 1.0)


def apply_curriculum_finger_unlock_from_flat_obs(
    action  : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    flat_obs: torch.Tensor,  # Param: tensor input carrying flat obs values
    config  : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Apply replay-observation finger unlock gating

    Steps:
    - Resolve inputs for `apply_curriculum_finger_unlock_from_flat_obs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    progress = curriculum_finger_unlock_from_flat_obs(flat_obs, config)
    if progress is None:
        return apply_middle_index_mirror(action, config)
    progress = progress.to(device=action.device, dtype=action.dtype)
    if topdown_finger_close_gate_mode(config.finger_close_gate_mode) in ("xyz_front", "xyz"):
        preload_floor = topdown_xyz_preload_fraction(config.finger_xyz_preload_fraction)
    else:
        preload_floor = 0.0
    if preload_floor > 0.0:
        return apply_middle_index_mirror(
            gate_finger_columns_above_preload(
                action,
                progress,
                preload_floor,
                config,
            ),
            config,
        )
    return apply_middle_index_mirror(scale_finger_columns(action, progress, config), config)


def _gate_like_action_batch(action: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return torch.ones((action.shape[0],), device=action.device, dtype=action.dtype)
    return gate.to(device=action.device, dtype=action.dtype).reshape(-1)


def apply_curriculum_finger_unlock_live(
    action     : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    config     : ActionGateConfig,  # Param: configuration object used by this helper
    live_config: LiveFingerUnlockConfig,  # Param: input value used as live config
    *,
    progress         : torch.Tensor | None = None,  # Param: tensor input carrying progress values
    center_gate      : torch.Tensor | None = None,  # Param: tensor input carrying center gate values
    close_gate       : torch.Tensor | None = None,  # Param: tensor input carrying close gate values
    arm_hold_active  : torch.Tensor | None = None,  # Param: mask or boolean input marking arm hold as active
    arm_hold_valid   : torch.Tensor | None = None,  # Param: tensor input carrying arm hold valid values
    arm_hold_fallback: torch.Tensor | None = None,  # Param: tensor input carrying arm hold fallback values
    lift_latched     : torch.Tensor | None = None,  # Param: mask selecting env rows with lift latch active
) -> LiveFingerUnlockResult:
    """Apply live curriculum finger unlock from explicit tensors

    Steps:
    - Resolve inputs for `apply_curriculum_finger_unlock_live` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not config.topdown_curriculum:
        return LiveFingerUnlockResult(action=action, effective_progress=None, arm_hold_gate=None)
    if live_config.contact_teacher and live_config.contact_teacher_bypass_unlock:
        return LiveFingerUnlockResult(
            action=apply_middle_index_mirror(action, config),
            effective_progress=None,
            arm_hold_gate=None,
        )

    mode = topdown_finger_close_gate_mode(config.finger_close_gate_mode)
    if mode in ("xyz_front", "xyz"):
        close = _gate_like_action_batch(action, close_gate).clamp(0.0, 1.0)
        close = preserve_xyz_close_after_lift_latch(close, lift_latched)
        preload_floor = topdown_xyz_preload_fraction(config.finger_xyz_preload_fraction)
        effective = preload_floor + (1.0 - preload_floor) * close
        if preload_floor > 0.0:
            gated = gate_finger_columns_above_preload(action, close, preload_floor, config)
        else:
            gated = scale_finger_columns(action, close, config)
        return LiveFingerUnlockResult(
            action=apply_middle_index_mirror(gated, config),
            effective_progress=effective.detach().clone(),
            arm_hold_gate=None,
        )

    if progress is None:
        if not live_config.finger_unlock_requires_center:
            return LiveFingerUnlockResult(
                action=apply_middle_index_mirror(action, config),
                effective_progress=None,
                arm_hold_gate=None,
            )
        progress_t = _gate_like_action_batch(action, center_gate).clamp(0.0, 1.0)
    else:
        progress_t = progress.to(device=action.device, dtype=action.dtype).reshape(-1).clamp(0.0, 1.0)

    if live_config.finger_unlock_requires_arm_hold:
        if (
            torch.is_tensor(arm_hold_active)
            and torch.is_tensor(arm_hold_valid)
            and arm_hold_active.shape == progress_t.shape
            and arm_hold_valid.shape == progress_t.shape
        ):
            arm_hold_gate = (
                arm_hold_active.to(device=action.device, dtype=torch.bool)
                & arm_hold_valid.to(device=action.device, dtype=torch.bool)
            ).to(dtype=action.dtype)
        else:
            arm_hold_gate = torch.zeros_like(progress_t)
        progress_t = progress_t * arm_hold_gate
    else:
        arm_hold_gate = torch.ones_like(progress_t)

    if (
        live_config.contact_teacher
        and live_config.contact_teacher_arm_hold_fallback
        and torch.is_tensor(arm_hold_fallback)
        and arm_hold_fallback.shape == progress_t.shape
    ):
        progress_t = torch.maximum(progress_t, arm_hold_fallback.to(device=action.device, dtype=action.dtype))

    close = _gate_like_action_batch(action, close_gate).clamp(0.0, 1.0)
    progress_t = (progress_t * close).clamp(0.0, 1.0)
    preload_floor = max(float(live_config.contact_teacher_start_fraction), 0.0) if live_config.contact_teacher else 0.0
    if preload_floor > 0.0:
        effective = preload_floor + (1.0 - preload_floor) * progress_t
        gated = gate_finger_columns_above_preload(action, progress_t, preload_floor, config)
    else:
        effective = progress_t
        gated = scale_finger_columns(action, progress_t, config)
    return LiveFingerUnlockResult(
        action=apply_middle_index_mirror(gated, config),
        effective_progress=effective.detach().clone(),
        arm_hold_gate=arm_hold_gate.detach().clone(),
    )


def apply_replay_action_gates_from_flat_obs(
    action  : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    flat_obs: torch.Tensor,  # Param: tensor input carrying flat obs values
    config  : ActionGateConfig,  # Param: configuration object used by this helper
    *,
    align_open_hand_action: Callable[[torch.Tensor], torch.Tensor] | None = None,  # Param: callback used to compute or fetch align open hand action
) -> torch.Tensor:
    """Apply observation-derived gates used by replay and target actions

    Steps:
    - Resolve inputs for `apply_replay_action_gates_from_flat_obs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    gated = apply_contact_finger_close_cap(action, config)
    gated = apply_contact_finger_open_from_flat_obs(gated, flat_obs, config)
    if align_open_hand_action is not None:
        gated = align_open_hand_action(gated)
    else:
        gated = apply_align_open_hand_action(gated, config)
    gated = apply_curriculum_finger_unlock_from_flat_obs(gated, flat_obs, config)
    return gated


def objective_action_from_gate_mode(
    raw_action: torch.Tensor,  # Param: tensor input carrying raw action values
    flat_obs  : torch.Tensor,  # Param: tensor input carrying flat obs values
    mode      : str,  # Param: string input for mode
    config    : ActionGateConfig,  # Param: configuration object used by this helper
    *,
    align_open_hand_action: Callable[[torch.Tensor], torch.Tensor] | None = None,  # Param: callback used to compute or fetch align open hand action
) -> torch.Tensor:
    """Return the action used by an update objective

    Steps:
    - Resolve inputs for `objective_action_from_gate_mode` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if mode == "raw":
        return raw_action
    gated = apply_replay_action_gates_from_flat_obs(
        raw_action,
        flat_obs,
        config,
        align_open_hand_action=align_open_hand_action,
    )
    if mode == "straight_through":
        return raw_action + (gated - raw_action).detach()
    if mode != "env":
        raise ValueError(f"unknown action gate mode: {mode!r}")
    return gated


def make_per_dim_noise_sigma(
    action_shape: torch.Size,  # Param: input value used as action shape
    arm_sigma   : float,  # Param: floating-point input for arm sigma
    finger_sigma: float,  # Param: floating-point input for finger sigma
    device      : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype       : torch.dtype,  # Param: torch dtype used when converting or allocating tensors
    config      : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Return a per-dimension noise scale vector"""
    last_dim = action_shape[-1]
    sigma = torch.full((last_dim,), float(arm_sigma), device=device, dtype=dtype)
    if last_dim >= config.num_arm + config.num_fingers and finger_sigma > 0.0:
        sigma[config.num_arm : config.num_arm + config.num_fingers] = float(finger_sigma)
    elif last_dim == config.num_fingers and finger_sigma > 0.0:
        sigma[:] = float(finger_sigma)
    return sigma


def add_assist_action_noise(
    action      : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    arm_sigma   : float,  # Param: floating-point input for arm sigma
    finger_sigma: float,  # Param: floating-point input for finger sigma
    noise_clip  : float,  # Param: floating-point input for noise clip
    config      : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Perturb assist actions while keeping clean teacher labels separate

    Steps:
    - Resolve inputs for `add_assist_action_noise` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if arm_sigma <= 0.0 and finger_sigma <= 0.0:
        return action
    sigma = make_per_dim_noise_sigma(
        action.shape,
        arm_sigma=float(max(0.0, arm_sigma)),
        finger_sigma=float(max(0.0, finger_sigma)),
        device=action.device,
        dtype=action.dtype,
        config=config,
    )
    noise = torch.randn_like(action) * sigma
    if noise_clip > 0.0:
        noise = noise.clamp(-float(noise_clip), float(noise_clip))
    return (action + noise).clamp(-1.0, 1.0)


def add_post_unlock_finger_noise(
    action      : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    flat_obs    : torch.Tensor | None,  # Param: tensor input carrying flat obs values
    finger_sigma: float,  # Param: floating-point input for finger sigma
    noise_clip  : float,  # Param: floating-point input for noise clip
    config      : ActionGateConfig,  # Param: configuration object used by this helper
) -> torch.Tensor:
    """Add finger-only Gaussian noise after the curriculum unlock gate

    Steps:
    - Resolve inputs for `add_post_unlock_finger_noise` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if finger_sigma <= 0.0 or not config.topdown_curriculum:
        return action
    bounds = _finger_bounds(action, config)
    if bounds is None:
        return action
    start, end = bounds
    finger_action = action[..., start:end]
    noise = torch.randn_like(finger_action) * float(finger_sigma)
    if noise_clip > 0.0:
        noise = noise.clamp(-float(noise_clip), float(noise_clip))
    col = config.stage_one_hot_obs_col
    if flat_obs is not None and col is not None and col + 2 < flat_obs.shape[-1]:
        stage2_gate = flat_obs[..., col + 2].to(dtype=finger_action.dtype)
        noise = noise * stage2_gate.unsqueeze(-1)
    new_action = action.clone()
    new_action[..., start:end] = action[..., start:end] + noise
    return new_action.clamp(-1.0, 1.0)
