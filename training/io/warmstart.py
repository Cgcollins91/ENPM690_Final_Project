"""

Warm-start helpers for checkpointed actor policies

Checkpoint warm-starting functions for copying compatible parameters from a source actor into a wider target actor,
copying observation stats, and building frozen policy teachers from checkpoint actors. These are used to initialize
training runs with pretrained policies when the architecture has changed in a compatible way, such as by adding new
joints to the arm or increasing hidden layer widths.

File map:

ActorWarmStartConfig:                        Action mapping used when copying a source actor into a wider actor
_checkpoint_or_load:                         Handle checkpoint or load logic
checkpoint_actor_state:                      Return the actor state from a trainer checkpoint
validate_checkpoint_arm_scales:              Validate source arm action scales when metadata is present
warm_start_actor_from_checkpoint:            Copy a source actor into the compatible prefix of a target actor
_copy_output_rows_by_joint:                  Copy source output rows into the target tensor based on matching joint names, and zero out unmatched rows in the target tensor
warm_start_obs_stats_from_checkpoint:        Copy source observation stats into the target prefix
make_frozen_policy_teacher_from_checkpoint:  Build a frozen policy teacher from a checkpoint actor
frozen_teacher_action_columns:               Return source action columns that correspond to current arm joints
frozen_teacher_obs_stats:                    Return frozen teacher observation mean and std tensors when present
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os

import torch

from .checkpoint_io import load_training_checkpoint
from ..model.networks import Actor, FrozenPhase1PolicyTeacher
from ..model.normalization import RunningTensorMoments


@dataclass(frozen=True)
class ActorWarmStartConfig:
    """Action mapping used when copying a source actor into a wider actor"""

    arm_joint_names    : tuple[str, ...]                     # Field: ordered names used to resolve arm joint attributes
    expected_arm_scales: tuple[float, ...] = ()              # Field: floating-point expected arm scales value used by actor warm start config
    first_weight_suffix: str               = "net.0.weight"  # Field: string first weight suffix value used by actor warm start config
    final_weight_suffix: str               = "net.6.weight"  # Field: string final weight suffix value used by actor warm start config
    final_bias_suffix  : str               = "net.6.bias"    # Field: string final bias suffix value used by actor warm start config


def _checkpoint_or_load(
    checkpoint_path: str | os.PathLike[str] | None,  # Param: base checkpoint path used for scheduled save decisions
    checkpoint     : Mapping[str, object] | None,    # Param: checkpoint payload or path being loaded or saved
) -> Mapping[str, object]:
    if checkpoint is not None:
        return checkpoint
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpoint is not provided")
    return load_training_checkpoint(checkpoint_path, map_location="cpu")


def checkpoint_actor_state(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    """Return the actor state from a trainer checkpoint

    Steps:
    - Resolve inputs for `checkpoint_actor_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    agent_state = checkpoint.get("agent")
    if not isinstance(agent_state, Mapping):
        raise RuntimeError("checkpoint is missing agent state")
    actor_state = agent_state.get("actor")
    if not isinstance(actor_state, Mapping):
        raise RuntimeError("checkpoint is missing agent actor state")
    return actor_state


def validate_checkpoint_arm_scales(
    checkpoint: Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    config    : ActorWarmStartConfig,  # Param: configuration object used by this helper
) -> None:
    """Validate source arm action scales when metadata is present

    Steps:
    - Resolve inputs for `validate_checkpoint_arm_scales` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    source_joints = tuple(checkpoint.get("reduced_action_joints", ()))
    source_scales = tuple(float(x) for x in checkpoint.get("reduced_action_scales", ()))
    if not config.expected_arm_scales or not source_joints or not source_scales:
        return
    if len(config.arm_joint_names) != len(config.expected_arm_scales):
        raise RuntimeError("arm_joint_names and expected_arm_scales must have equal length")
    if len(source_joints) != len(source_scales):
        raise RuntimeError("checkpoint reduced action joints and scales have mismatched lengths")
    expected = dict(zip(config.arm_joint_names, config.expected_arm_scales))
    for joint_name, expected_scale in expected.items():
        if joint_name not in source_joints:
            continue
        actual_scale = source_scales[source_joints.index(joint_name)]
        if not math.isclose(actual_scale, expected_scale, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(
                "checkpoint arm action scale mismatch for warm-start: "
                f"{joint_name} checkpoint={actual_scale} current={expected_scale}"
            )


def warm_start_actor_from_checkpoint(
    actor: Actor,                                           # Param: input value used as actor
    *,
    config         : ActorWarmStartConfig,                  # Param: configuration object used by this helper
    checkpoint_path: str | os.PathLike[str] | None = None,  # Param: base checkpoint path used for scheduled save decisions
    checkpoint     : Mapping[str, object] | None   = None,  # Param: checkpoint payload or path being loaded or saved
) -> dict[str, int]:
    """Copy a source actor into the compatible prefix of a target actor

    The source actor is loaded from the checkpoint and must have a compatible architecture where all hidden layers match in width
    and all input and output dimensions are at least as large as the target actor. T
    he source actor parameters are copied into the target actor where they fit, and any unmatched input or output dimensions
    in the target actor are zeroed out. This allows warm-starting with a pretrained checkpoint even when the architecture has
    changed in a compatible way, such as by adding new joints to the arm or increasing hidden layer widths.

    Steps:
    - Resolve inputs for `warm_start_actor_from_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ckpt        = _checkpoint_or_load(checkpoint_path, checkpoint)
    actor_state = checkpoint_actor_state(ckpt)
    validate_checkpoint_arm_scales(ckpt, config)

    source_joints = tuple(ckpt.get("reduced_action_joints", ()))
    device        = next(actor.parameters()).device
    own_state     = actor.state_dict()
    applied       = {"hidden_layers": 0, "input_cols_copied": 0, "output_rows_copied": 0}

    # Copy matching parameters from the source actor into the target actor, and zero out unmatched input or output dimensions in the target actor
    # Hidden layers must match in width, input and output layers can be wider in the target actor but not narrower, and any unmatched dimensions
    # in the target actor are zeroed out to preserve the behavior of the copied parameters and prevent uninitialized values from affecting the
    # policy at the start of training
    for key, source_tensor in actor_state.items():
        if key not in own_state or not torch.is_tensor(source_tensor):
            continue
        target_tensor = own_state[key]
        if target_tensor.shape == source_tensor.shape:
            target_tensor.copy_(source_tensor.to(device))
            applied["hidden_layers"] += 1
        elif key.endswith(config.first_weight_suffix) and target_tensor.ndim == 2:
            rows = min(source_tensor.shape[0], target_tensor.shape[0])
            cols = min(source_tensor.shape[1], target_tensor.shape[1])
            target_tensor[:rows, :cols].copy_(source_tensor[:rows, :cols].to(device))
            if target_tensor.shape[1] > cols:
                target_tensor[:, cols:].zero_()
            applied["input_cols_copied"] = int(cols)
        elif key.endswith(config.final_weight_suffix) and target_tensor.ndim == 2:
            rows_copied = _copy_output_rows_by_joint(
                target_tensor,
                source_tensor,
                source_joints=source_joints,
                arm_joint_names=config.arm_joint_names,
                device=device,
            )
            applied["output_rows_copied"] = rows_copied
        elif key.endswith(config.final_bias_suffix) and target_tensor.ndim == 1:
            _copy_output_rows_by_joint(
                target_tensor,
                source_tensor,
                source_joints=source_joints,
                arm_joint_names=config.arm_joint_names,
                device=device,
            )

    actor.load_state_dict(own_state)
    return applied


def _copy_output_rows_by_joint(
    target_tensor: torch.Tensor,      # Param: tensor containing target values
    source_tensor: torch.Tensor,       # Param: tensor containing source values
    *,
    source_joints  : tuple[str, ...],  # Param: string input for source joints
    arm_joint_names: tuple[str, ...],  # Param: ordered candidate names used to resolve arm joint
    device         : torch.device,     # Param: torch device where tensors are read or allocated
) -> int:
    """
    Copy source output rows into the target tensor based on matching joint names, and zero out unmatched rows in the target tensor
    """

    if source_joints and len(source_joints) == source_tensor.shape[0]:
        rows_copied = 0
        for dst_row, joint_name in enumerate(arm_joint_names):
            if dst_row >= target_tensor.shape[0] or joint_name not in source_joints:
                continue
            src_row = source_joints.index(joint_name)
            target_tensor[dst_row].copy_(source_tensor[src_row].to(device))
            rows_copied += 1
        return rows_copied
    rows = min(source_tensor.shape[0], len(arm_joint_names), target_tensor.shape[0])
    target_tensor[:rows].copy_(source_tensor[:rows].to(device))
    return int(rows)


def warm_start_obs_stats_from_checkpoint(
    obs_stats: RunningTensorMoments,                        # Param: input value used as obs stats
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,  # Param: base checkpoint path used for scheduled save decisions
    checkpoint     : Mapping[str, object] | None   = None,  # Param: checkpoint payload or path being loaded or saved
) -> dict[str, float | int]:
    """Copy source observation stats into the target prefix

    The source observation stats are loaded from the checkpoint and must have a compatible architecture where the observation dimension
    is at least as large as the target. The source observation stats are copied into the target where they fit, and any unmatched dimensions
    in the target are left unchanged. This allows warm-starting with pretrained checkpoint observation stats even when the architecture has
    changed in a compatible way, such as by adding new observation dimensions.

    Steps:
    - Resolve inputs for `warm_start_obs_stats_from_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    ckpt = _checkpoint_or_load(checkpoint_path, checkpoint)
    agent_state = ckpt.get("agent", {}) or {}
    p1_stats = agent_state.get("obs_stats") if isinstance(agent_state, Mapping) else None
    applied: dict[str, float | int] = {"dims_copied": 0, "source_count_mean": 0.0}
    if not isinstance(p1_stats, Mapping) or "mean" not in p1_stats or "var" not in p1_stats:
        return applied
    p1_mean = p1_stats["mean"].detach().float().cpu()
    p1_var = p1_stats["var"].detach().float().cpu().clamp(min=1e-8)
    dims = min(int(p1_mean.shape[0]), int(obs_stats.mean.shape[0]))
    obs_stats.mean[:dims] = p1_mean[:dims]
    obs_stats.var[:dims] = p1_var[:dims]
    raw_count = p1_stats.get("count")
    if raw_count is not None and dims > 0:
        if torch.is_tensor(raw_count):
            count_tensor = raw_count.detach().float().cpu()
            if count_tensor.shape == obs_stats.count.shape:
                obs_stats.count[:dims] = count_tensor[:dims]
            else:
                obs_stats.count[:dims] = float(count_tensor.reshape(-1)[0].item())
        else:
            obs_stats.count[:dims] = float(raw_count)
    applied["dims_copied"] = int(dims)
    if dims > 0:
        applied["source_count_mean"] = float(obs_stats.count[:dims].mean().item())
    return applied


def make_frozen_policy_teacher_from_checkpoint(
    checkpoint: Mapping[str, object],           # Param: checkpoint payload or path being loaded or saved
    *,
    current_obs_keys     : tuple[str, ...],     # Param: ordered mapping keys used to resolve current obs
    arm_joint_names      : tuple[str, ...],     # Param: ordered candidate names used to resolve arm joint
    device               : torch.device | str,  # Param: torch device where tensors are read or allocated
    obs_norm_eps_default : float = 1e-5,        # Param: floating-point input for obs norm eps default
    obs_norm_clip_default: float = 10.0,        # Param: floating-point input for obs norm clip default
) -> FrozenPhase1PolicyTeacher:
    """Build a frozen policy teacher from a checkpoint actor

    Frozen policy teachers are used to provide consistent teacher actions for distillation when the privileged critic is enabled,
    and they require a compatible architecture where the actor hidden layers match in width and the observation and action dimensions
    are at least as large as the current teacher. The checkpoint actor parameters are loaded into a frozen actor, and any compatible
    observation stats are also loaded. This allows using a pretrained checkpoint actor as a frozen teacher even when the architecture
    has changed in a compatible way, such as by adding new joints to the arm or increasing hidden layer widths.

    Steps:
    - Resolve inputs for `make_frozen_policy_teacher_from_checkpoint` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    actor_state = checkpoint_actor_state(checkpoint)
    first_weight = actor_state.get("mlp.net.0.weight")
    final_bias = actor_state.get("mlp.net.6.bias")
    if not torch.is_tensor(first_weight) or not torch.is_tensor(final_bias):
        raise RuntimeError("checkpoint actor does not match the expected MLP layout")
    hidden_dim = int(first_weight.shape[0])
    obs_dim = int(first_weight.shape[1])
    action_dim = int(final_bias.shape[0])
    target_device = torch.device(device)
    actor = Actor(obs_dim, hidden_dim, action_dim).to(target_device)
    actor.load_state_dict({key: value.to(target_device) for key, value in actor_state.items()})
    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)

    source_obs_keys = tuple(checkpoint.get("policy_obs_keys", checkpoint.get("obs_keys", ())))
    if source_obs_keys and tuple(current_obs_keys[: len(source_obs_keys)]) != source_obs_keys:
        raise RuntimeError(
            "checkpoint policy_obs_keys must match the current observation prefix for frozen teacher: "
            f"checkpoint={source_obs_keys} current_prefix={tuple(current_obs_keys[: len(source_obs_keys)])}"
        )

    action_cols = frozen_teacher_action_columns(
        checkpoint,
        action_dim=action_dim,
        arm_joint_names=arm_joint_names,
    )
    obs_mean, obs_std = frozen_teacher_obs_stats(
        checkpoint,
        obs_dim=obs_dim,
        device=target_device,
    )
    ckpt_args = checkpoint.get("args", {}) or {}
    ckpt_args_map = ckpt_args if isinstance(ckpt_args, Mapping) else {}
    return FrozenPhase1PolicyTeacher(
        actor=actor,
        obs_dim=obs_dim,
        action_cols=action_cols,
        obs_mean=obs_mean,
        obs_std=obs_std,
        obs_norm_eps=float(ckpt_args_map.get("obs_norm_eps", obs_norm_eps_default)),
        obs_norm_clip=float(ckpt_args_map.get("obs_norm_clip", obs_norm_clip_default)),
        obs_keys=source_obs_keys,
    )


def frozen_teacher_action_columns(
    checkpoint: Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    *,
    action_dim     : int,  # Param: integer input for action dim
    arm_joint_names: tuple[str, ...],  # Param: ordered candidate names used to resolve arm joint
) -> tuple[int, ...]:
    """Return source action columns that correspond to current arm joints"""
    source_joints = tuple(checkpoint.get("reduced_action_joints", ()))
    if source_joints:
        missing = [joint_name for joint_name in arm_joint_names if joint_name not in source_joints]
        if missing:
            raise RuntimeError(f"checkpoint is missing required arm joints for frozen teacher: {missing}")
        return tuple(source_joints.index(joint_name) for joint_name in arm_joint_names)
    if int(action_dim) == len(arm_joint_names):
        return tuple(range(len(arm_joint_names)))
    raise RuntimeError(
        "checkpoint has no reduced_action_joints metadata and its action width "
        f"{action_dim} is not arm width {len(arm_joint_names)}"
    )


def frozen_teacher_obs_stats(
    checkpoint: Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    *,
    obs_dim: int,  # Param: integer input for obs dim
    device : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return frozen teacher observation mean and std tensors when present

    Steps:
    - Resolve inputs for `frozen_teacher_obs_stats` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    agent_state = checkpoint.get("agent", {}) or {}
    ckpt_args = checkpoint.get("args", {}) or {}
    ckpt_args_map = ckpt_args if isinstance(ckpt_args, Mapping) else {}
    p1_stats = agent_state.get("obs_stats") if isinstance(agent_state, Mapping) else None
    if p1_stats is None:
        if bool(ckpt_args_map.get("observation_normalization", False)):
            raise RuntimeError(
                "checkpoint args say observation_normalization=True but agent.obs_stats is absent"
            )
        return None, None
    mean = p1_stats.get("mean")
    var = p1_stats.get("var")
    if mean is None or var is None:
        raise RuntimeError("checkpoint obs_stats is missing mean or var")
    if int(mean.shape[0]) < int(obs_dim) or int(var.shape[0]) < int(obs_dim):
        raise RuntimeError(
            "checkpoint obs_stats width is smaller than actor obs width: "
            f"mean={tuple(mean.shape)} var={tuple(var.shape)} actor_obs_dim={obs_dim}"
        )
    target_device = torch.device(device)
    obs_mean = mean[:obs_dim].detach().float().to(target_device)
    obs_std = torch.sqrt(var[:obs_dim].detach().float().clamp(min=1.0e-8)).to(target_device)
    return obs_mean, obs_std
