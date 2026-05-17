"""

Arm lift latch tensor-state helpers

This module provides helper functions and data structures for tracking arm lift latch state and captured lift targets in tensors,
used by the training loop and environment wrapper

Arm lift latch state includes latched, latch step, contact counter, latch update step, latch signal, first touch, and reset mask tensors

Captured lift targets include target xy, target base z, target nominal z, block xy latch, and target captured tensors

File map:

ArmLiftLatchState:                Updated arm lift latch tensors
ArmLiftTargetCapture:             Captured lift target tensors
initial_arm_lift_latch_state:     Create default arm lift latch state tensors
update_arm_lift_latch_tensors:    Update arm lift latch counters and first-touch state
initial_arm_lift_target_capture:  Create default lift target capture tensors
capture_arm_lift_targets:         Capture lift target XY Z and block latch XY on selected rows
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ArmLiftLatchState:
    """Updated arm lift latch tensors"""

    latched          : torch.Tensor  # per-env latch mask or aggregate latch state
    latch_step       : torch.Tensor  # step count used for latch step scheduling or reporting
    contact_counter  : torch.Tensor  # tensor containing contact counter values for batched env rows
    latch_update_step: torch.Tensor  # step count used for latch update step scheduling or reporting
    latch_signal     : torch.Tensor  # tensor containing latch signal values for batched env rows
    first_touch      : torch.Tensor  # tensor containing first touch values for batched env rows
    reset_mask       : torch.Tensor  # boolean mask selecting reset rows for arm lift latch state


@dataclass(frozen=True)
class ArmLiftTargetCapture:
    """Captured lift target tensors"""

    target_xy       : torch.Tensor  # tensor containing target xy values for batched env rows
    target_base_z   : torch.Tensor  # tensor containing target base z values for batched env rows
    target_nominal_z: torch.Tensor  # tensor containing target nominal z values for batched env rows
    block_xy_latch  : torch.Tensor  # tensor containing block xy latch values for batched env rows
    target_captured : torch.Tensor  # tensor containing target captured values for batched env rows


def initial_arm_lift_latch_state(
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> ArmLiftLatchState:
    """Create default arm lift latch state tensors"""
    return ArmLiftLatchState(
        latched=torch.zeros(num_envs, dtype=torch.bool, device=device),
        latch_step=torch.full((num_envs,), -1.0, dtype=torch.float32, device=device),
        contact_counter=torch.zeros(num_envs, dtype=torch.int32, device=device),
        latch_update_step=torch.full((num_envs,), -1.0, dtype=torch.float32, device=device),
        latch_signal=torch.zeros(num_envs, dtype=torch.float32, device=device),
        first_touch=torch.zeros(num_envs, dtype=torch.bool, device=device),
        reset_mask=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def update_arm_lift_latch_tensors(
    *,
    episode_step     : torch.Tensor,  # Param: per-env step count inside the current episode
    latched          : torch.Tensor,  # Param: tensor input carrying latched values
    latch_step       : torch.Tensor,  # Param: step count used for latch step
    contact_counter  : torch.Tensor,  # Param: tensor input carrying contact counter values
    latch_update_step: torch.Tensor,  # Param: step count used for latch update step
    contact_signal   : torch.Tensor,  # Param: tensor input carrying contact signal values
    touch_now        : torch.Tensor,  # Param: tensor input carrying touch now values
    hold_steps       : int,  # Param: step count used for hold steps
) -> ArmLiftLatchState:
    """Update arm lift latch counters and first-touch state

    Steps:
    - Resolve inputs for `update_arm_lift_latch_tensors` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    device = episode_step.device
    step = episode_step.to(dtype=torch.float32)
    latched_next = latched.to(device=device, dtype=torch.bool).clone()
    latch_step_next = latch_step.to(device=device, dtype=torch.float32).clone()
    counter_next = contact_counter.to(device=device, dtype=torch.int32).clone()
    update_step_next = latch_update_step.to(device=device, dtype=torch.float32).clone()
    contact_signal_next = contact_signal.to(device=device, dtype=torch.float32).detach().clone()
    touch = touch_now.to(device=device, dtype=torch.bool)

    reset_mask = step <= 1.0
    if bool(reset_mask.any().item()):
        latched_next[reset_mask] = False
        latch_step_next[reset_mask] = -1.0
        counter_next[reset_mask] = 0
        update_step_next[reset_mask] = -1.0

    already_updated = update_step_next == step
    touch_unlatched = touch & (~latched_next) & (~already_updated)
    counter_next = torch.where(
        already_updated,
        counter_next,
        torch.where(
            touch_unlatched,
            counter_next + 1,
            torch.zeros_like(counter_next),
        ),
    )
    first_touch = touch_unlatched & (counter_next >= max(int(hold_steps), 1))
    latch_step_next[first_touch] = step[first_touch]
    latched_next[first_touch] = True
    update_step_next = torch.where(
        reset_mask,
        torch.full_like(update_step_next, -1.0),
        torch.where(already_updated, update_step_next, step),
    )
    return ArmLiftLatchState(
        latched=latched_next.detach().clone(),
        latch_step=latch_step_next.detach().clone(),
        contact_counter=counter_next.detach().clone(),
        latch_update_step=update_step_next.detach().clone(),
        latch_signal=contact_signal_next,
        first_touch=first_touch.detach().clone(),
        reset_mask=reset_mask.detach().clone(),
    )


def initial_arm_lift_target_capture(
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> ArmLiftTargetCapture:
    """Create default lift target capture tensors"""
    return ArmLiftTargetCapture(
        target_xy=torch.zeros((num_envs, 2), dtype=torch.float32, device=device),
        target_base_z=torch.zeros(num_envs, dtype=torch.float32, device=device),
        target_nominal_z=torch.zeros(num_envs, dtype=torch.float32, device=device),
        block_xy_latch=torch.zeros((num_envs, 2), dtype=torch.float32, device=device),
        target_captured=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def capture_arm_lift_targets(
    *,
    previous     : ArmLiftTargetCapture,  # Param: input value used as previous
    capture_mask : torch.Tensor,  # Param: boolean mask selecting capture rows
    target_pos   : torch.Tensor,  # Param: tensor input carrying target pos values
    block_xy     : torch.Tensor,  # Param: tensor input carrying block xy values
    freeze_pos   : torch.Tensor | None = None,  # Param: tensor input carrying freeze pos values
    use_actual_xy: bool                = True,  # Param: boolean input selecting whether actual xy is used
    use_actual_z : bool                = True,  # Param: boolean input selecting whether actual z is used
) -> ArmLiftTargetCapture:
    """Capture lift target XY Z and block latch XY on selected rows

    Steps:
    - Resolve inputs for `capture_arm_lift_targets` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    mask = capture_mask.to(device=target_pos.device, dtype=torch.bool)
    target = target_pos.detach().to(dtype=torch.float32)
    actual = freeze_pos.detach().to(device=target.device, dtype=torch.float32) if torch.is_tensor(freeze_pos) else None
    xy_source = actual if bool(use_actual_xy) and actual is not None else target
    z_source = actual if bool(use_actual_z) and actual is not None else target

    target_xy = previous.target_xy.to(device=target.device, dtype=torch.float32).clone()
    target_base_z = previous.target_base_z.to(device=target.device, dtype=torch.float32).clone()
    target_nominal_z = previous.target_nominal_z.to(device=target.device, dtype=torch.float32).clone()
    block_xy_latch = previous.block_xy_latch.to(device=target.device, dtype=torch.float32).clone()
    target_captured = previous.target_captured.to(device=target.device, dtype=torch.bool).clone()

    if bool(mask.any().item()):
        target_xy[mask] = xy_source[mask, :2]
        target_base_z[mask] = z_source[mask, 2]
        target_nominal_z[mask] = target[mask, 2]
        block_xy_latch[mask] = block_xy.detach().to(device=target.device, dtype=torch.float32)[mask]
        target_captured[mask] = True

    return ArmLiftTargetCapture(
        target_xy=target_xy.detach().clone(),
        target_base_z=target_base_z.detach().clone(),
        target_nominal_z=target_nominal_z.detach().clone(),
        block_xy_latch=block_xy_latch.detach().clone(),
        target_captured=target_captured.detach().clone(),
    )
