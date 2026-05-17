"""

Teacher arm target shaping helpers

File map:

scalar_or_tensor_1d:         Return a per-env tensor from a scalar or tensor input
descent_active_mask:         Return rows with active vertical contact descent
prehover_active_mask:        Return rows still in explicit prehover
inward_amount_from_closure:  Return inward offset distance for topdown arm target
apply_inward_offset:         Apply inward offset to target position
apply_planar_bias:           Apply world XY bias and per-env XY offset
apply_vertical_descent:      Apply vertical contact descent to target position
LiftProgress:                Smoothed lift and nominal-Z blend progress
lift_progress_from_latch:    Return smoothed lift progress from latch timing
freeze_lift_target_pose:     Apply latched lift target XY and Z captures
BlockXYStabilizer:           Block XY stabilizer correction and active mask
block_xy_stabilizer:         Return block drift stabilizer correction
apply_block_xy_stabilizer:   Subtract block drift stabilizer from target XY
descent_tip_servo:           Return descent-gated fingertip servo correction
clamp_vector_norm:           Clamp each vector row to max norm
prehold_tip_servo:           Return prehold IK fingertip servo vector
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .teacher_closure import smoothstep01


def scalar_or_tensor_1d(
    value: float | torch.Tensor,  # Param: input value normalized or converted by this helper
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype   : torch.dtype,  # Param: torch dtype used when converting or allocating tensors
) -> torch.Tensor:
    """Return a per-env tensor from a scalar or tensor input"""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.full((int(num_envs),), float(value), device=device, dtype=dtype)


def descent_active_mask(
    descent: float | torch.Tensor | None,  # Param: tensor input carrying descent values
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype   : torch.dtype = torch.float32,  # Param: torch dtype used when converting or allocating tensors
) -> torch.Tensor:
    """Return rows with active vertical contact descent"""
    if descent is None:
        return torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    descent_t = scalar_or_tensor_1d(descent, num_envs=num_envs, device=device, dtype=dtype)
    return torch.clamp(descent_t, min=0.0) > 1.0e-6


def prehover_active_mask(
    *,
    explicit_prehover: bool,  # Param: boolean input controlling explicit prehover
    descent_started  : torch.Tensor,  # Param: tensor input carrying descent started values
) -> torch.Tensor:
    """Return rows still in explicit prehover"""
    if not bool(explicit_prehover):
        return torch.zeros_like(descent_started, dtype=torch.bool)
    return ~descent_started.to(dtype=torch.bool)


def inward_amount_from_closure(
    closure_fraction: float | torch.Tensor,              # Param: tensor input carrying closure fraction values
    *,
    num_envs       : int,  # Param: number of parallel environment rows represented
    device         : torch.device | str,  # Param: torch device where tensors are read or allocated
    dtype          : torch.dtype,  # Param: torch dtype used when converting or allocating tensors
    contact_inward : float | torch.Tensor | None = None,  # Param: tensor input carrying contact inward values
    base_scale     : float                       = 0.006,  # Param: multiplier applied to base
    prehover_active: torch.Tensor | None         = None,  # Param: mask selecting env rows currently in prehover mode
) -> torch.Tensor:
    """Return inward offset distance for topdown arm target

    Steps:
    - Resolve inputs for `inward_amount_from_closure` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    closure_t = scalar_or_tensor_1d(closure_fraction, num_envs=num_envs, device=device, dtype=dtype)
    inward = float(base_scale) * closure_t
    if contact_inward is not None:
        inward = inward + torch.clamp(
            scalar_or_tensor_1d(contact_inward, num_envs=num_envs, device=device, dtype=dtype),
            min=0.0,
        )
    if prehover_active is not None:
        inward = torch.where(prehover_active.to(device=device, dtype=torch.bool), torch.zeros_like(inward), inward)
    return inward


def apply_inward_offset(
    target_pos   : torch.Tensor,  # Param: tensor input carrying target pos values
    inward_dir   : torch.Tensor,  # Param: directory path for inward
    inward_amount: torch.Tensor,  # Param: tensor input carrying inward amount values
    *,
    vertical_only: bool = False,  # Param: boolean input controlling vertical only
) -> torch.Tensor:
    """Apply inward offset to target position"""
    direction = inward_dir.to(device=target_pos.device, dtype=target_pos.dtype)
    if bool(vertical_only):
        direction = torch.zeros_like(direction)
        direction[:, 2] = -1.0
    return target_pos + direction * inward_amount.to(device=target_pos.device, dtype=target_pos.dtype).unsqueeze(-1)


def apply_planar_bias(
    target_pos: torch.Tensor,               # Param: tensor input carrying target pos values
    *,
    bias_x   : float               = 0.0,  # Param: floating-point input for bias x
    bias_y   : float               = 0.0,  # Param: floating-point input for bias y
    xy_offset: torch.Tensor | None = None,  # Param: tensor input carrying xy offset values
) -> torch.Tensor:
    """Apply world XY bias and per-env XY offset

    Steps:
    - Resolve inputs for `apply_planar_bias` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    out = target_pos.clone()
    if float(bias_x) != 0.0:
        out[:, 0] = out[:, 0] + float(bias_x)
    if float(bias_y) != 0.0:
        out[:, 1] = out[:, 1] + float(bias_y)
    if xy_offset is not None:
        offset = xy_offset.to(device=out.device, dtype=out.dtype).reshape(out.shape[0], 2)
        out[:, :2] = out[:, :2] + offset
    return out


def apply_vertical_descent(
    target_pos: torch.Tensor,  # Param: tensor input carrying target pos values
    descent   : float | torch.Tensor,  # Param: tensor input carrying descent values
) -> torch.Tensor:
    """Apply vertical contact descent to target position"""
    out = target_pos.clone()
    descent_t = scalar_or_tensor_1d(
        descent,
        num_envs=out.shape[0],
        device=out.device,
        dtype=out.dtype,
    )
    out[:, 2] = out[:, 2] - torch.clamp(descent_t, min=0.0)
    return out


@dataclass(frozen=True)
class LiftProgress:
    """Smoothed lift and nominal-Z blend progress"""

    steps_since_latch: torch.Tensor  # Field: tensor containing steps since latch values for batched env rows
    lift_progress    : torch.Tensor  # Field: tensor containing lift progress values for batched env rows
    nominal_z_blend  : torch.Tensor  # Field: tensor containing nominal z blend values for batched env rows


def lift_progress_from_latch(
    *,
    episode_step         : torch.Tensor,  # Param: per-env step count inside the current episode
    latch_step           : torch.Tensor,  # Param: step count used for latch step
    grip_settle_steps    : float,  # Param: step count used for grip settle steps
    lift_ramp_steps      : float,  # Param: step count used for lift ramp steps
    nominal_z_blend_steps: float = 0.0,  # Param: step count used for nominal z blend steps
    nominal_z_blend_delay: float = 0.0,  # Param: floating-point input for nominal z blend delay
) -> LiftProgress:
    """Return smoothed lift progress from latch timing

    Steps:
    - Resolve inputs for `lift_progress_from_latch` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    step = episode_step.to(dtype=torch.float32)
    latch = latch_step.to(device=step.device, dtype=torch.float32)
    steps_since_latch = torch.where(latch >= 0.0, step - latch, torch.full_like(latch, -1.0))

    if float(nominal_z_blend_steps) > 0.0:
        z_blend_linear = torch.clamp(
            (steps_since_latch - float(nominal_z_blend_delay)) / float(nominal_z_blend_steps),
            0.0,
            1.0,
        )
        z_blend = smoothstep01(z_blend_linear)
        lift_start_delay = max(float(grip_settle_steps), float(nominal_z_blend_delay) + float(nominal_z_blend_steps))
    else:
        z_blend = torch.zeros_like(steps_since_latch)
        lift_start_delay = float(grip_settle_steps)

    lift_linear = torch.clamp(
        (steps_since_latch - lift_start_delay) / max(float(lift_ramp_steps), 1.0),
        0.0,
        1.0,
    )
    return LiftProgress(
        steps_since_latch=steps_since_latch,
        lift_progress=smoothstep01(lift_linear),
        nominal_z_blend=z_blend,
    )


def freeze_lift_target_pose(
    target_pos: torch.Tensor,                      # Param: tensor input carrying target pos values
    *,
    lift_latched    : torch.Tensor,  # Param: mask selecting env rows with lift latch active
    target_xy       : torch.Tensor | None = None,  # Param: tensor input carrying target xy values
    target_base_z   : torch.Tensor | None = None,  # Param: tensor input carrying target base z values
    target_nominal_z: torch.Tensor | None = None,  # Param: tensor input carrying target nominal z values
    nominal_z_blend : torch.Tensor | None = None,  # Param: tensor input carrying nominal z blend values
) -> torch.Tensor:
    """Apply latched lift target XY and Z captures

    Steps:
    - Resolve inputs for `freeze_lift_target_pose` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    out = target_pos.clone()
    latched = lift_latched.to(device=out.device, dtype=torch.bool)
    if target_xy is not None:
        xy = target_xy.to(device=out.device, dtype=out.dtype)
        out[:, :2] = torch.where(latched.unsqueeze(-1), xy, out[:, :2])
    if target_base_z is not None:
        base_z = target_base_z.to(device=out.device, dtype=out.dtype)
        next_z = base_z
        if target_nominal_z is not None and nominal_z_blend is not None:
            nominal_z = target_nominal_z.to(device=out.device, dtype=out.dtype)
            blend = nominal_z_blend.to(device=out.device, dtype=out.dtype)
            next_z = base_z + (nominal_z - base_z) * blend
        out[:, 2] = torch.where(latched, next_z, out[:, 2])
    return out


@dataclass(frozen=True)
class BlockXYStabilizer:
    """Block XY stabilizer correction and active mask"""

    correction: torch.Tensor  # Field: tensor containing correction values for batched env rows
    active    : torch.Tensor  # Field: whether this configuration or runtime path is active


def block_xy_stabilizer(
    *,
    block_xy      : torch.Tensor,  # Param: tensor input carrying block xy values
    block_xy_latch: torch.Tensor,  # Param: tensor input carrying block xy latch values
    lift_latched  : torch.Tensor,  # Param: mask selecting env rows with lift latch active
    gain          : float,  # Param: floating-point input for gain
    max_m         : float,  # Param: floating-point input for max m
) -> BlockXYStabilizer:
    """Return block drift stabilizer correction

    Steps:
    - Resolve inputs for `block_xy_stabilizer` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    latched = lift_latched.to(device=block_xy.device, dtype=torch.bool)
    if float(gain) <= 0.0 or float(max_m) <= 0.0:
        zeros = torch.zeros_like(block_xy)
        return BlockXYStabilizer(correction=zeros, active=torch.zeros_like(latched, dtype=torch.bool))
    drift = block_xy - block_xy_latch.to(device=block_xy.device, dtype=block_xy.dtype)
    correction = drift * float(gain)
    norm = torch.linalg.norm(correction, dim=1, keepdim=True).clamp_min(1.0e-6)
    correction = correction * torch.clamp(float(max_m) / norm, max=1.0)
    correction = torch.where(latched.unsqueeze(-1), correction, torch.zeros_like(correction))
    active = latched & (torch.linalg.norm(correction, dim=1) > 1.0e-5)
    return BlockXYStabilizer(correction=correction, active=active)


def apply_block_xy_stabilizer(
    target_pos: torch.Tensor,  # Param: tensor input carrying target pos values
    stabilizer: BlockXYStabilizer,  # Param: input value used as stabilizer
) -> torch.Tensor:
    """Subtract block drift stabilizer from target XY"""
    out = target_pos.clone()
    out[:, :2] = out[:, :2] - stabilizer.correction.to(device=out.device, dtype=out.dtype)
    return out


def descent_tip_servo(
    *,
    tip_servo           : torch.Tensor,  # Param: tensor input carrying tip servo values
    descent_active      : torch.Tensor,  # Param: mask or boolean input marking descent as active
    xy_max_m            : float,  # Param: floating-point input for xy max m
    keep_one_sided_z    : torch.Tensor | None = None,  # Param: tensor input carrying keep one sided z values
    both_missing        : torch.Tensor | None = None,  # Param: tensor input carrying both missing values
    both_missing_z_max_m: float               = 0.0,  # Param: floating-point input for both missing z max m
) -> torch.Tensor:
    """Return descent-gated fingertip servo correction

    Steps:
    - Resolve inputs for `descent_tip_servo` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    servo = tip_servo.clone()
    if float(xy_max_m) > 0.0:
        descent_servo = servo.clone()
        descent_servo[:, 2] = 0.0
        xy_norm = torch.linalg.norm(descent_servo[:, :2], dim=1, keepdim=True).clamp_min(1.0e-6)
        descent_servo[:, :2] = descent_servo[:, :2] * torch.clamp(float(xy_max_m) / xy_norm, max=1.0)
    else:
        descent_servo = torch.zeros_like(servo)
    if keep_one_sided_z is not None:
        keep_z = keep_one_sided_z.to(device=servo.device, dtype=torch.bool)
        descent_servo[:, 2] = torch.where(keep_z, servo[:, 2], descent_servo[:, 2])
    if both_missing is not None and float(both_missing_z_max_m) > 0.0:
        both = both_missing.to(device=servo.device, dtype=torch.bool)
        downward_z = torch.clamp(servo[:, 2], min=-float(both_missing_z_max_m), max=0.0)
        descent_servo[:, 2] = torch.where(both, downward_z, descent_servo[:, 2])
    active = descent_active.to(device=servo.device, dtype=torch.bool)
    return torch.where(active.unsqueeze(-1), descent_servo, servo)


def clamp_vector_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clamp each vector row to max norm"""
    if float(max_norm) <= 0.0:
        return torch.zeros_like(vec)
    norm = torch.linalg.norm(vec, dim=1, keepdim=True)
    return vec * torch.clamp(float(max_norm) / norm.clamp_min(1.0e-6), max=1.0)


def prehold_tip_servo(
    *,
    thumb_pos   : torch.Tensor,  # Param: tensor input carrying thumb pos values
    index_pos   : torch.Tensor,  # Param: tensor input carrying index pos values
    thumb_target: torch.Tensor,  # Param: target value for thumb
    index_target: torch.Tensor,  # Param: target value for index
    enabled     : torch.Tensor,  # Param: tensor input carrying enabled values
    gain        : float,  # Param: floating-point input for gain
    max_m       : float,  # Param: floating-point input for max m
    z_scale     : float               = 1.0,  # Param: multiplier applied to z
    z_gate      : torch.Tensor | None = None,  # Param: tensor input carrying z gate values
) -> torch.Tensor:
    """Return prehold IK fingertip servo vector

    Steps:
    - Resolve inputs for `prehold_tip_servo` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    dtype = thumb_pos.dtype
    device = thumb_pos.device
    thumb_err = thumb_target.to(device=device, dtype=dtype) - thumb_pos
    index_err = index_target.to(device=device, dtype=dtype) - index_pos
    servo = 0.5 * (thumb_err + index_err) * max(float(gain), 0.0)
    servo[:, 2] = servo[:, 2] * float(z_scale)
    if z_gate is not None:
        gate = z_gate.to(device=device, dtype=torch.bool)
        servo[:, 2] = torch.where(gate, servo[:, 2], torch.zeros_like(servo[:, 2]))
    active = enabled.to(device=device, dtype=torch.bool)
    servo = torch.where(active.unsqueeze(-1), servo, torch.zeros_like(servo))
    return clamp_vector_norm(servo, max_m)
