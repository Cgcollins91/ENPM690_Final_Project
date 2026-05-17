"""

Topdown contact teacher readiness and state helpers

File map:

CONTACT_TEACHER_ZERO_FLOAT_ATTRS:    Define contact teacher zero float attrs constant
CONTACT_TEACHER_BOOL_ATTRS:          Define contact teacher bool attrs constant
CONTACT_TEACHER_MINUS_ONE_ATTRS:     Define contact teacher minus one attrs constant
CONTACT_TEACHER_ATTR_PREFIX:         Define contact teacher attr prefix constant
contact_teacher_enabled:             Return whether topdown contact teacher is enabled
contact_teacher_ready_mask:          Return rows where contact teacher may close fingers
wrist_yaw_release_gate:              Return rows close enough in Y to release wrist yaw
initial_contact_teacher_state:       Return default topdown contact teacher state tensors
contact_teacher_attr_name:           Return the env attr name for one contact teacher state tensor
reset_contact_teacher_state_rows:    Return contact teacher state with reset rows restored to defaults
ensure_contact_teacher_state_attrs:  Ensure env-owned contact teacher state attrs exist and reset rows
apply_preload_fraction:              Apply minimum preload fraction to ready fingers
preload_one_sided_reject_mask:       Return one-sided preload contact reject rows
ContactLatchUpdate:                  Updated per-finger contact latch state
update_contact_latches:              Update thumb index and middle contact latches
apply_hold_fractions_on_new_latch:   Snapshot per-finger hold fractions on new contact
clamp_servo:                         Clamp per-env servo vector norm
ContactDescent:                      Contact teacher descent result
contact_descent_from_missing:        Compute descent distance from missing-tip state
missing_tip_servo:                   Return weighted missing-tip servo vector
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


CONTACT_TEACHER_ZERO_FLOAT_ATTRS = (
    "thumb_fraction",
    "index_fraction",
    "middle_fraction",
    "descent_ready_age",
    "preload_recovery_clear_age",
    "preload_recovery_no_contact_age",
)
CONTACT_TEACHER_BOOL_ATTRS = (
    "thumb_latched",
    "index_latched",
    "middle_latched",
    "descent_started",
    "preload_recovery_active",
    "preload_recovery_unload_active",
)
CONTACT_TEACHER_MINUS_ONE_ATTRS = (
    "thumb_hold_fraction",
    "index_hold_fraction",
    "middle_hold_fraction",
    "thumb_lift_freeze_fraction",
    "index_lift_freeze_fraction",
    "middle_lift_freeze_fraction",
)
CONTACT_TEACHER_ATTR_PREFIX = "_topdown_contact_teacher_"


def contact_teacher_enabled(*, requested: bool, topdown_curriculum_task: bool) -> bool:
    """Return whether topdown contact teacher is enabled"""
    return bool(requested) and bool(topdown_curriculum_task)


def contact_teacher_ready_mask(
    *,
    num_envs           : int,  # Param: number of parallel environment rows represented
    device             : torch.device | str,  # Param: torch device where tensors are read or allocated
    stage              : torch.Tensor | None = None,  # Param: tensor input carrying stage values
    proximity          : torch.Tensor | None = None,  # Param: tensor input carrying proximity values
    pose_ready         : torch.Tensor | None = None,  # Param: mask or boolean input marking pose as ready
    proximity_threshold: float               = 0.95,  # Param: cutoff used when evaluating proximity
) -> torch.Tensor:
    """Return rows where contact teacher may close fingers"""
    if torch.is_tensor(stage):
        ready = stage.to(device=device) >= 2
    elif torch.is_tensor(proximity):
        ready = proximity.to(device=device, dtype=torch.float32) >= float(proximity_threshold)
    else:
        ready = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    if torch.is_tensor(pose_ready):
        ready = ready | pose_ready.to(device=device, dtype=torch.bool)
    return ready


def wrist_yaw_release_gate(
    *,
    palm_y     : torch.Tensor,  # Param: palm Y coordinate used for wrist-yaw release gating
    block_pos  : torch.Tensor,  # Param: block position tensor used as the geometric reference
    release_tol: float,  # Param: maximum Y error allowed before wrist yaw is released
) -> torch.Tensor:
    """
    Return rows close enough in Y to release wrist yaw
    """

    return torch.abs(palm_y.to(device=block_pos.device) - block_pos[:, 1]) <= float(release_tol)


def initial_contact_teacher_state(
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> dict[str, torch.Tensor]:

    """Return default topdown contact teacher state tensors

    Steps:
    - Resolve inputs for `initial_contact_teacher_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """

    shape = (int(num_envs),)
    state: dict[str, torch.Tensor] = {}
    for name in CONTACT_TEACHER_ZERO_FLOAT_ATTRS:
        state[name] = torch.zeros(shape, dtype=torch.float32, device=device)
    for name in CONTACT_TEACHER_BOOL_ATTRS:
        state[name] = torch.zeros(shape, dtype=torch.bool, device=device)
    for name in CONTACT_TEACHER_MINUS_ONE_ATTRS:
        state[name] = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    return state


def contact_teacher_attr_name(name: str, prefix: str = CONTACT_TEACHER_ATTR_PREFIX) -> str:
    """Return the env attr name for one contact teacher state tensor"""
    return f"{prefix}{name}"


def reset_contact_teacher_state_rows(
    state     : dict[str, torch.Tensor],  # Param: mutable or immutable runtime state read by this helper
    reset_mask: torch.Tensor,  # Param: boolean mask selecting reset rows
) -> dict[str, torch.Tensor]:
    """Return contact teacher state with reset rows restored to defaults

    Steps:
    - Resolve inputs for `reset_contact_teacher_state_rows` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    out = {key: value.clone() for key, value in state.items()}
    mask = reset_mask.to(dtype=torch.bool)
    for name in CONTACT_TEACHER_ZERO_FLOAT_ATTRS:
        if name in out:
            out[name][mask] = 0.0
    for name in CONTACT_TEACHER_BOOL_ATTRS:
        if name in out:
            out[name][mask] = False
    for name in CONTACT_TEACHER_MINUS_ONE_ATTRS:
        if name in out:
            out[name][mask] = -1.0
    return out


def ensure_contact_teacher_state_attrs(
    env,                                        # Param: environment or backend object used for runtime calls
    episode_step: torch.Tensor,                 # Param: per-env step count inside the current episode
    *,
    prefix: str = CONTACT_TEACHER_ATTR_PREFIX,  # Param: string input for prefix
) -> dict[str, torch.Tensor]:
    """Ensure env-owned contact teacher state attrs exist and reset rows

    Steps:
    - Resolve inputs for `ensure_contact_teacher_state_attrs` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    defaults = initial_contact_teacher_state(num_envs=env.num_envs, device=env.device)
    current: dict[str, torch.Tensor] = {}
    shape = (int(env.num_envs),)
    for name, default_value in defaults.items():
        attr_name = contact_teacher_attr_name(name, prefix)
        value = getattr(env, attr_name, None)
        if not torch.is_tensor(value) or value.shape != shape:
            value = default_value.clone()
            setattr(env, attr_name, value)
        current[name] = value.to(device=env.device)

    reset_mask = episode_step.to(device=env.device, dtype=torch.float32) <= 1.0
    updated = reset_contact_teacher_state_rows(current, reset_mask)
    for name, value in updated.items():
        setattr(env, contact_teacher_attr_name(name, prefix), value)
    return updated


def apply_preload_fraction(
    *,
    thumb_fraction  : torch.Tensor,  # Param: tensor input carrying thumb fraction values
    index_fraction  : torch.Tensor,  # Param: tensor input carrying index fraction values
    middle_fraction : torch.Tensor,  # Param: tensor input carrying middle fraction values
    preload_ready   : torch.Tensor,  # Param: mask or boolean input marking preload as ready
    middle_active   : torch.Tensor,  # Param: mask or boolean input marking middle as active
    preload_fraction: float,  # Param: floating-point input for preload fraction
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply minimum preload fraction to ready fingers

    Steps:
    - Resolve inputs for `apply_preload_fraction` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ready = preload_ready.to(device=thumb_fraction.device, dtype=torch.bool)
    preload = torch.full_like(thumb_fraction, float(preload_fraction))
    thumb = torch.where(ready & (thumb_fraction < preload), preload, thumb_fraction)
    index = torch.where(ready & (index_fraction < preload), preload, index_fraction)
    middle_ready = ready & middle_active.to(device=thumb_fraction.device, dtype=torch.bool)
    middle = torch.where(middle_ready & (middle_fraction < preload), preload, middle_fraction)
    return thumb, index, middle


def preload_one_sided_reject_mask(
    *,
    thumb_fraction    : torch.Tensor,  # Param: tensor input carrying thumb fraction values
    index_fraction    : torch.Tensor,  # Param: tensor input carrying index fraction values
    thumb_contact     : torch.Tensor,  # Param: tensor input carrying thumb contact values
    index_contact     : torch.Tensor,  # Param: tensor input carrying index contact values
    prev_thumb_latched: torch.Tensor,  # Param: tensor input carrying prev thumb latched values
    prev_index_latched: torch.Tensor,  # Param: tensor input carrying prev index latched values
    threshold         : float,  # Param: cutoff used by the comparison or gate
    preload_fraction  : float,  # Param: floating-point input for preload fraction
    eps               : float = 1.0e-6,  # Param: small tolerance used to avoid unstable equality comparisons
) -> torch.Tensor:
    """Return one-sided preload contact reject rows

    Steps:
    - Resolve inputs for `preload_one_sided_reject_mask` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    no_prior_pinch_latch = ~(prev_thumb_latched.to(dtype=torch.bool) | prev_index_latched.to(dtype=torch.bool))
    both_at_preload = (
        (thumb_fraction <= (float(preload_fraction) + float(eps)))
        & (index_fraction <= (float(preload_fraction) + float(eps)))
    )
    live_thumb_contact = thumb_contact >= float(threshold)
    live_index_contact = index_contact >= float(threshold)
    return no_prior_pinch_latch & both_at_preload & (live_thumb_contact ^ live_index_contact)


@dataclass(frozen=True)
class ContactLatchUpdate:
    """Updated per-finger contact latch state"""

    thumb_latched       : torch.Tensor  # per-env latch state for thumb contact
    index_latched       : torch.Tensor  # per-env latch state for index-finger contact
    middle_latched      : torch.Tensor  # per-env latch state for middle-finger contact
    newly_thumb_latched : torch.Tensor  # mask for thumb contacts that latched on this update
    newly_index_latched : torch.Tensor  # mask for index contacts that latched on this update
    newly_middle_latched: torch.Tensor  # mask for middle-finger contacts that latched on this update
    all_required_latched: torch.Tensor  # mask where every required contact latch is active
    one_sided_latched   : torch.Tensor  # mask where only one pinch-side contact latch is active


def update_contact_latches(
    *,
    close_progress_ready: torch.Tensor,  # Param: mask or boolean input marking close progress as ready
    thumb_contact       : torch.Tensor,  # Param: tensor input carrying thumb contact values
    index_contact       : torch.Tensor,  # Param: tensor input carrying index contact values
    middle_contact      : torch.Tensor,  # Param: tensor input carrying middle contact values
    prev_thumb_latched  : torch.Tensor,  # Param: tensor input carrying prev thumb latched values
    prev_index_latched  : torch.Tensor,  # Param: tensor input carrying prev index latched values
    prev_middle_latched : torch.Tensor,  # Param: tensor input carrying prev middle latched values
    middle_active       : torch.Tensor,  # Param: mask or boolean input marking middle as active
    threshold           : float,  # Param: cutoff used by the comparison or gate
) -> ContactLatchUpdate:
    """Update thumb index and middle contact latches

    Steps:
    - Resolve inputs for `update_contact_latches` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ready = close_progress_ready.to(dtype=torch.bool)
    thumb_for_hold = ready & (thumb_contact >= float(threshold))
    index_for_hold = ready & (index_contact >= float(threshold))
    middle_for_hold = middle_active.to(dtype=torch.bool) & ready & (middle_contact >= float(threshold))
    prev_thumb = prev_thumb_latched.to(dtype=torch.bool)
    prev_index = prev_index_latched.to(dtype=torch.bool)
    prev_middle = prev_middle_latched.to(dtype=torch.bool)
    newly_thumb = thumb_for_hold & (~prev_thumb)
    newly_index = index_for_hold & (~prev_index)
    newly_middle = middle_for_hold & (~prev_middle)
    thumb_latched = prev_thumb | thumb_for_hold
    index_latched = prev_index | index_for_hold
    middle_latched = torch.where(
        middle_active.to(dtype=torch.bool),
        prev_middle | middle_for_hold,
        torch.zeros_like(prev_middle),
    )
    all_required = thumb_latched & index_latched
    all_required = torch.where(middle_active.to(dtype=torch.bool), all_required & middle_latched, all_required)
    return ContactLatchUpdate(
        thumb_latched=thumb_latched,
        index_latched=index_latched,
        middle_latched=middle_latched,
        newly_thumb_latched=newly_thumb,
        newly_index_latched=newly_index,
        newly_middle_latched=newly_middle,
        all_required_latched=all_required,
        one_sided_latched=thumb_latched ^ index_latched,
    )


def apply_hold_fractions_on_new_latch(
    *,
    thumb_fraction      : torch.Tensor,  # Param: tensor input carrying thumb fraction values
    index_fraction      : torch.Tensor,  # Param: tensor input carrying index fraction values
    middle_fraction     : torch.Tensor,  # Param: tensor input carrying middle fraction values
    newly_thumb_latched : torch.Tensor,  # Param: tensor input carrying newly thumb latched values
    newly_index_latched : torch.Tensor,  # Param: tensor input carrying newly index latched values
    newly_middle_latched: torch.Tensor,  # Param: tensor input carrying newly middle latched values
    thumb_hold_fraction : torch.Tensor,  # Param: tensor input carrying thumb hold fraction values
    index_hold_fraction : torch.Tensor,  # Param: tensor input carrying index hold fraction values
    middle_hold_fraction: torch.Tensor,  # Param: tensor input carrying middle hold fraction values
    hold_extra          : float,  # Param: floating-point input for hold extra
    max_fraction        : float,  # Param: floating-point input for max fraction
    hold_max_fraction   : float,  # Param: floating-point input for hold max fraction
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Snapshot per-finger hold fractions on new contact

    Steps:
    - Resolve inputs for `apply_hold_fractions_on_new_latch` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    cap = min(float(max_fraction), float(hold_max_fraction))
    thumb_new = torch.clamp(thumb_fraction + float(hold_extra), 0.0, cap)
    index_new = torch.clamp(index_fraction + float(hold_extra), 0.0, cap)
    middle_new = torch.clamp(middle_fraction + float(hold_extra), 0.0, cap)
    return (
        torch.where(newly_thumb_latched, thumb_new, thumb_hold_fraction),
        torch.where(newly_index_latched, index_new, index_hold_fraction),
        torch.where(newly_middle_latched, middle_new, middle_hold_fraction),
    )


def clamp_servo(vec: torch.Tensor, max_m: float) -> torch.Tensor:
    """Clamp per-env servo vector norm"""
    if float(max_m) <= 0.0:
        return torch.zeros_like(vec)
    norm = torch.linalg.norm(vec, dim=1, keepdim=True)
    return vec * torch.clamp(float(max_m) / norm.clamp_min(1.0e-6), max=1.0)


@dataclass(frozen=True)
class ContactDescent:
    """Contact teacher descent result"""

    descent     : torch.Tensor  # tensor containing descent values for batched env rows
    z_need      : torch.Tensor  # tensor containing z need values for batched env rows
    closure_gate: torch.Tensor  # tensor containing closure gate values for batched env rows


def contact_descent_from_missing(
    *,
    descent_ready   : torch.Tensor,  # Param: mask or boolean input marking descent as ready
    thumb_missing   : torch.Tensor,  # Param: tensor input carrying thumb missing values
    index_missing   : torch.Tensor,  # Param: tensor input carrying index missing values
    middle_missing  : torch.Tensor,  # Param: tensor input carrying middle missing values
    thumb_z_gap     : torch.Tensor,  # Param: tensor input carrying thumb z gap values
    index_z_gap     : torch.Tensor,  # Param: tensor input carrying index z gap values
    middle_z_gap    : torch.Tensor,  # Param: tensor input carrying middle z gap values
    required_closure: torch.Tensor,  # Param: tensor input carrying required closure values
    base_descent    : float,  # Param: floating-point input for base descent
    extra_descent   : float,  # Param: floating-point input for extra descent
    z_target        : float,  # Param: target value for z
    z_full          : float,  # Param: floating-point input for z full
    min_closure     : float,  # Param: floating-point input for min closure
    full_closure    : float,  # Param: floating-point input for full closure
    uses_z_need     : bool = True,  # Param: boolean input controlling uses z need
) -> ContactDescent:
    """Compute descent distance from missing-tip state

    Steps:
    - Resolve inputs for `contact_descent_from_missing` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ready = descent_ready.to(device=thumb_z_gap.device, dtype=torch.bool)
    zeros = torch.zeros_like(thumb_z_gap)
    missing_tip_z = torch.maximum(
        torch.maximum(
            torch.where(thumb_missing, thumb_z_gap, zeros),
            torch.where(index_missing, index_z_gap, zeros),
        ),
        torch.where(middle_missing, middle_z_gap, zeros),
    )
    z_den = max(float(z_full) - float(z_target), 1.0e-4)
    z_need = torch.clamp((missing_tip_z - float(z_target)) / z_den, 0.0, 1.0)
    z_need = torch.where(ready, z_need, torch.zeros_like(z_need))
    descent_scale = z_need if bool(uses_z_need) else ready.to(dtype=torch.float32)
    if float(min_closure) > 0.0:
        if float(full_closure) > float(min_closure) + 1.0e-6:
            closure_gate = torch.clamp(
                (required_closure - float(min_closure)) / (float(full_closure) - float(min_closure)),
                0.0,
                1.0,
            )
        else:
            closure_gate = (required_closure >= float(min_closure)).to(dtype=torch.float32)
    else:
        closure_gate = torch.ones_like(required_closure, dtype=torch.float32)
    either_missing = thumb_missing | index_missing | middle_missing
    raw = torch.full_like(thumb_z_gap, float(base_descent)) + either_missing.to(dtype=torch.float32) * float(extra_descent)
    descent = torch.where(ready, raw * descent_scale * closure_gate, torch.zeros_like(raw))
    return ContactDescent(descent=descent, z_need=z_need, closure_gate=closure_gate)


def missing_tip_servo(
    *,
    thumb_err_vec: torch.Tensor,  # Param: tensor input carrying thumb err vec values
    index_err_vec: torch.Tensor,  # Param: tensor input carrying index err vec values
    thumb_missing: torch.Tensor,  # Param: tensor input carrying thumb missing values
    index_missing: torch.Tensor,  # Param: tensor input carrying index missing values
    z_need       : torch.Tensor,  # Param: tensor input carrying z need values
    gain         : float,  # Param: floating-point input for gain
    max_m        : float,  # Param: floating-point input for max m
) -> torch.Tensor:
    """Return weighted missing-tip servo vector

    Steps:
    - Resolve inputs for `missing_tip_servo` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    thumb_weight = thumb_missing.to(device=thumb_err_vec.device, dtype=torch.float32).unsqueeze(-1)
    index_weight = index_missing.to(device=thumb_err_vec.device, dtype=torch.float32).unsqueeze(-1)
    weight_sum = (thumb_weight + index_weight).clamp_min(1.0)
    servo = (thumb_err_vec * thumb_weight + index_err_vec * index_weight) / weight_sum
    servo = servo * max(float(gain), 0.0)
    servo[:, 2] = servo[:, 2] * z_need.to(device=servo.device, dtype=servo.dtype)
    return clamp_servo(servo, max_m)
