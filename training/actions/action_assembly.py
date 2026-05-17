"""

Policy-level to environment-reduced action assembly helpers


closure_fraction_from_curl:             Return normalized IK closure fraction from finger curl
apply_live_action_gates:                Apply live env gates used immediately before env step
assemble_policy_controlled_env_action:  Assemble env action when the policy controls arm and fingers
assemble_ik_controlled_env_action:      Assemble env action when IK supplies arm dims and policy supplies fingers
assemble_env_reduced_action:            Dispatch env action assembly for policy and IK arm controllers
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .action_gates import ActionGateConfig, apply_contact_finger_close_cap
from .action_space import convert_finger_delta_to_reduced, expand_reduced_action


ActionTransform = Callable[[torch.Tensor], torch.Tensor]


def closure_fraction_from_curl(curl: torch.Tensor, closure_scale: float) -> torch.Tensor:
    """Return normalized IK closure fraction from finger curl"""
    return torch.clamp(curl / max(float(closure_scale), 1.0e-6), 0.0, 1.0)


def apply_live_action_gates(
    action: torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    config: ActionGateConfig,  # Param: configuration object used by this helper
    *,
    contact_finger_open_until_ready: ActionTransform | None = None,  # Param: mask or boolean input marking contact finger open until as ready
    align_open_hand_action         : ActionTransform | None = None,  # Param: input value used as align open hand action
) -> torch.Tensor:
    """Apply live env gates used immediately before env step"""
    gated = apply_contact_finger_close_cap(action, config)
    if contact_finger_open_until_ready is not None:
        gated = contact_finger_open_until_ready(gated)
    if align_open_hand_action is not None:
        gated = align_open_hand_action(gated)
    return gated


def assemble_policy_controlled_env_action(
    policy_level_action: torch.Tensor,  # Param: tensor input carrying policy level action values
    config             : ActionGateConfig,  # Param: configuration object used by this helper
    *,
    contact_finger_open_until_ready: ActionTransform | None = None,  # Param: mask or boolean input marking contact finger open until as ready
    align_open_hand_action         : ActionTransform | None = None,  # Param: input value used as align open hand action
) -> torch.Tensor:
    """Assemble env action when the policy controls arm and fingers"""
    return apply_live_action_gates(
        policy_level_action,
        config,
        contact_finger_open_until_ready=contact_finger_open_until_ready,
        align_open_hand_action=align_open_hand_action,
    )


def assemble_ik_controlled_env_action(
    policy_finger_action: torch.Tensor,  # Param: tensor input carrying policy finger action values
    arm_reduced_action  : torch.Tensor,  # Param: tensor input carrying arm reduced action values
    config              : ActionGateConfig,  # Param: configuration object used by this helper
    *,
    finger_action_mode: str,                                         # Param: mode string selecting the finger action behavior
    env=None,                                                        # Param: environment or backend object used for runtime calls
    mapped_indices                 : torch.Tensor | None    = None,  # Param: tensor input carrying mapped indices values
    mapped_scales                  : torch.Tensor | None    = None,  # Param: tensor input carrying mapped scales values
    finger_delta_scale             : float                  = 0.05,  # Param: multiplier applied to finger delta
    contact_finger_open_until_ready: ActionTransform | None = None,  # Param: mask or boolean input marking contact finger open until as ready
    align_open_hand_action         : ActionTransform | None = None,  # Param: input value used as align open hand action
) -> torch.Tensor:
    """Assemble env action when IK supplies arm dims and policy supplies fingers"""
    if finger_action_mode == "delta":
        if env is None or mapped_indices is None or mapped_scales is None:
            raise ValueError("delta finger assembly requires env mapped_indices and mapped_scales")
        finger_reduced = convert_finger_delta_to_reduced(
            env,
            policy_finger_action,
            mapped_indices,
            mapped_scales,
            num_arm=config.num_arm,
            finger_delta_scale=finger_delta_scale,
        )
    elif finger_action_mode == "absolute":
        finger_reduced = policy_finger_action
    else:
        raise ValueError(f"unknown finger_action_mode: {finger_action_mode!r}")

    env_reduced = torch.cat([arm_reduced_action, finger_reduced], dim=-1).clamp(-1.0, 1.0)
    return apply_live_action_gates(
        env_reduced,
        config,
        contact_finger_open_until_ready=contact_finger_open_until_ready,
        align_open_hand_action=align_open_hand_action,
    )


def assemble_env_reduced_action(
    policy_level_action: torch.Tensor,  # Param: tensor input carrying policy level action values
    config             : ActionGateConfig,  # Param: configuration object used by this helper
    *,
    arm_controller    : str,  # Param: string input for arm controller
    finger_action_mode: str                 = "absolute",  # Param: mode string selecting the finger action behavior
    arm_reduced_action: torch.Tensor | None = None,  # Param: tensor input carrying arm reduced action values
    env=None,                                                        # Param: environment or backend object used for runtime calls
    mapped_indices                 : torch.Tensor | None    = None,  # Param: tensor input carrying mapped indices values
    mapped_scales                  : torch.Tensor | None    = None,  # Param: tensor input carrying mapped scales values
    finger_delta_scale             : float                  = 0.05,  # Param: multiplier applied to finger delta
    contact_finger_open_until_ready: ActionTransform | None = None,  # Param: mask or boolean input marking contact finger open until as ready
    align_open_hand_action         : ActionTransform | None = None,  # Param: input value used as align open hand action
) -> torch.Tensor:
    """Dispatch env action assembly for policy and IK arm controllers

    Steps:
    - Resolve inputs for `assemble_env_reduced_action` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if arm_controller != "ik":
        env_reduced = assemble_policy_controlled_env_action(
            policy_level_action,
            config,
            contact_finger_open_until_ready=contact_finger_open_until_ready,
            align_open_hand_action=align_open_hand_action,
        )
    else:
        if arm_reduced_action is None:
            raise ValueError("ik action assembly requires arm_reduced_action")
        env_reduced = assemble_ik_controlled_env_action(
            policy_level_action,
            arm_reduced_action,
            config,
            finger_action_mode=finger_action_mode,
            env=env,
            mapped_indices=mapped_indices,
            mapped_scales=mapped_scales,
            finger_delta_scale=finger_delta_scale,
            contact_finger_open_until_ready=contact_finger_open_until_ready,
            align_open_hand_action=align_open_hand_action,
        )
    action_manager = getattr(env, "action_manager", None)
    action_dim = int(getattr(action_manager, "total_action_dim", env_reduced.shape[-1]))
    if env_reduced.shape[-1] == action_dim:
        return env_reduced
    if mapped_indices is None or mapped_scales is None:
        raise ValueError(
            "full Isaac action assembly requires mapped_indices and mapped_scales "
            f"for reduced_dim={env_reduced.shape[-1]} action_dim={action_dim}"
        )
    return expand_reduced_action(env_reduced, action_dim, mapped_indices, mapped_scales)
