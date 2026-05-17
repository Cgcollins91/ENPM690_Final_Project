"""

Checkpoint compatibility helpers with explicit runtime inputs

File map:

checkpoint_backend:                          Return the learner backend encoded by a checkpoint
ensure_checkpoint_backend_compatible:        Raise when checkpoint backend and active backend differ
checkpoint_arg_flag:                         Read a boolean checkpoint compatibility flag
checkpoint_uses_observation_normalization:   Return whether checkpoint used observation normalization
ensure_checkpoint_obs_norm_compatible:       Reject incompatible observation-normalization settings
checkpoint_uses_reward_normalization:        Return whether checkpoint used reward normalization
ensure_checkpoint_reward_norm_compatible:    Reject incompatible reward-normalization settings
checkpoint_obs_stats_width:                  Return checkpoint policy obs_stats width when present
checkpoint_priv_obs_stats_width:             Return checkpoint privileged obs_stats width when present
checkpoint_actor_input_width:                Infer the first actor linear layer input width
checkpoint_critic_obs_width:                 Infer critic observation width from the first critic layer
checkpoint_action_semantics:                 Return action-semantics fields from a checkpoint
action_semantics_from_specs:                 Build current action semantics from resolved action specs
validate_checkpoint_action_semantics:        Raise when checkpoint action semantics differ from current semantics
ensure_checkpoint_policy_schema_compatible:  Fail early when loading incompatible actor checkpoints
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..actions.action_space import ReducedActionSpec


def checkpoint_backend(ckpt: Mapping[str, object]) -> str:
    """Return the learner backend encoded by a checkpoint

    Steps:
    - Resolve inputs for `checkpoint_backend` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if ckpt.get("upstream_fasttd3"):
        return "upstream_fasttd3"
    agent_state = ckpt.get("agent")
    if isinstance(agent_state, Mapping) and agent_state.get("td3_backend"):
        return str(agent_state["td3_backend"])
    if ckpt.get("td3_backend"):
        return str(ckpt["td3_backend"])
    return "custom"


def ensure_checkpoint_backend_compatible(
    ckpt: Mapping[str, object],  # Param: string input for ckpt
    *,
    active_backend: str,  # Param: string input for active backend
    context       : str,  # Param: runtime context carrying validated trainer settings
) -> None:
    """Raise when checkpoint backend and active backend differ"""
    checkpoint_value = checkpoint_backend(ckpt)
    if checkpoint_value != str(active_backend):
        raise RuntimeError(
            f"{context} backend mismatch: checkpoint backend={checkpoint_value!r}, "
            f"active td3_backend={active_backend!r}"
        )


def checkpoint_arg_flag(ckpt: Mapping[str, object], key: str) -> bool | None:
    """Read a boolean checkpoint compatibility flag"""
    ckpt_args = ckpt.get("args", {}) or {}
    if isinstance(ckpt_args, Mapping):
        if key in ckpt_args:
            return bool(ckpt_args[key])
        return None
    if hasattr(ckpt_args, key):
        return bool(getattr(ckpt_args, key))
    return None


def checkpoint_uses_observation_normalization(ckpt: Mapping[str, object]) -> bool:
    """Return whether checkpoint used observation normalization"""
    arg_flag = checkpoint_arg_flag(ckpt, "observation_normalization")
    if arg_flag is not None:
        return arg_flag
    agent_state = ckpt.get("agent", {}) or {}
    return isinstance(agent_state, Mapping) and "obs_stats" in agent_state


def ensure_checkpoint_obs_norm_compatible(
    ckpt                            : Mapping[str, object],              # Param: string input for ckpt
    *,
    active_observation_normalization: bool,  # Param: boolean input controlling active observation normalization
    context                         : str,  # Param: runtime context carrying validated trainer settings
) -> None:
    """Reject incompatible observation-normalization settings"""
    ckpt_uses_obs_norm = checkpoint_uses_observation_normalization(ckpt)
    if ckpt_uses_obs_norm and not active_observation_normalization:
        raise RuntimeError(
            f"{context} was saved with observation normalization; rerun with --observation-normalization"
        )
    if active_observation_normalization and not ckpt_uses_obs_norm:
        raise RuntimeError(
            f"{context} was saved without observation normalization stats, "
            "but this run has --observation-normalization enabled"
        )


def checkpoint_uses_reward_normalization(ckpt: Mapping[str, object]) -> bool:
    """Return whether checkpoint used reward normalization"""
    arg_flag = checkpoint_arg_flag(ckpt, "reward_normalization")
    if arg_flag is not None:
        return arg_flag
    agent_state = ckpt.get("agent", {}) or {}
    return isinstance(agent_state, Mapping) and "reward_stats" in agent_state


def ensure_checkpoint_reward_norm_compatible(
    ckpt                       : Mapping[str, object], # Param: string input for ckpt
    *,
    active_reward_normalization: bool,                 # Param: boolean input controlling active reward normalization
    context                    : str,                  # Param: runtime context carrying validated trainer settings
) -> None:
    """Reject incompatible reward-normalization settings"""
    if checkpoint_uses_reward_normalization(ckpt) and not active_reward_normalization:
        raise RuntimeError(
            f"{context} was saved with reward normalization; rerun with --reward-normalization"
        )


def checkpoint_obs_stats_width(agent_state: Mapping[str, object]) -> int | None:
    """Return checkpoint policy obs_stats width when present

    Steps:
    - Resolve inputs for `checkpoint_obs_stats_width` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    stats = agent_state.get("obs_stats") if isinstance(agent_state, Mapping) else None
    if not isinstance(stats, Mapping):
        return None
    mean = stats.get("mean")
    if torch.is_tensor(mean) and mean.ndim >= 1:
        return int(mean.shape[0])
    return None


def checkpoint_priv_obs_stats_width(agent_state: Mapping[str, object]) -> int | None:
    """Return checkpoint privileged obs_stats width when present

    Steps:
    - Resolve inputs for `checkpoint_priv_obs_stats_width` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    stats = agent_state.get("priv_obs_stats") if isinstance(agent_state, Mapping) else None
    if not isinstance(stats, Mapping):
        return None
    mean = stats.get("mean")
    if torch.is_tensor(mean) and mean.ndim >= 1:
        return int(mean.shape[0])
    return None


def checkpoint_actor_input_width(agent_state: Mapping[str, object]) -> int | None:
    """Infer the first actor linear layer input width

    Steps:
    - Resolve inputs for `checkpoint_actor_input_width` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    actor_state = agent_state.get("actor") if isinstance(agent_state, Mapping) else None
    if not isinstance(actor_state, Mapping):
        return None
    preferred = (
        "mlp.net.0.weight",
        "net.0.weight",
        "actor.net.0.weight",
        "trunk.0.weight",
    )
    for key in preferred:
        weight = actor_state.get(key)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return int(weight.shape[1])
    for key, weight in actor_state.items():
        if str(key).endswith(".weight") and torch.is_tensor(weight) and weight.ndim == 2:
            return int(weight.shape[1])
    return None


def checkpoint_critic_obs_width(agent_state: Mapping[str, object], action_dim: int) -> int | None:
    """Infer critic observation width from the first critic layer"""
    if not isinstance(agent_state, Mapping):
        return None
    for state_name in ("critic1", "critic", "critic_target", "critic1_target"):
        critic_state = agent_state.get(state_name)
        if not isinstance(critic_state, Mapping):
            continue
        preferred = (
            "mlp.net.0.weight",
            "net.0.weight",
            "q1_model.0.weight",
            "q1.0.weight",
            "critic.0.weight",
        )
        for key in preferred:
            weight = critic_state.get(key)
            if torch.is_tensor(weight) and weight.ndim == 2:
                width = int(weight.shape[1]) - int(action_dim)
                return width if width > 0 else None
    return None


def checkpoint_action_semantics(
    ckpt: Mapping[str, object],              # Param: string input for ckpt
    *,
    default_arm_controller: str = "policy",  # Param: string input for default arm controller
) -> dict[str, object]:
    """Return action-semantics fields from a checkpoint"""
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_args_map = ckpt_args if isinstance(ckpt_args, Mapping) else {}
    return {
        "arm_controller": ckpt_args_map.get(
            "arm_controller", ckpt.get("arm_controller", default_arm_controller)
        ),
        "finger_action_mode": ckpt_args_map.get("finger_action_mode", "absolute"),
        "finger_delta_scale": ckpt_args_map.get("finger_delta_scale", 0.05),
        "policy_action_joints": tuple(
            ckpt.get("policy_action_joints", ckpt.get("reduced_action_joints", ()))
        ),
        "policy_action_scales": tuple(
            float(x) for x in ckpt.get("policy_action_scales", ckpt.get("reduced_action_scales", ()))
        ),
        "env_action_joints": tuple(
            ckpt.get("env_reduced_action_joints", ckpt.get("reduced_action_joints", ()))
        ),
        "env_action_scales": tuple(
            float(x) for x in ckpt.get("env_reduced_action_scales", ckpt.get("reduced_action_scales", ()))
        ),
    }


def action_semantics_from_specs(
    *,
    arm_controller    : str,  # Param: string input for arm controller
    finger_action_mode: str,  # Param: mode string selecting the finger action behavior
    finger_delta_scale: float,  # Param: multiplier applied to finger delta
    policy_action_spec: ReducedActionSpec,  # Param: input value used as policy action spec
    env_action_spec   : ReducedActionSpec,  # Param: input value used as env action spec
) -> dict[str, object]:
    """Build current action semantics from resolved action specs"""
    return {
        "arm_controller"      : str(arm_controller),
        "finger_action_mode"  : str(finger_action_mode),
        "finger_delta_scale"  : float(finger_delta_scale),
        "policy_action_joints": tuple(policy_action_spec.joint_names),
        "policy_action_scales": tuple(float(x) for x in policy_action_spec.scales),
        "env_action_joints"   : tuple(env_action_spec.joint_names),
        "env_action_scales"   : tuple(float(x) for x in env_action_spec.scales),
    }


def validate_checkpoint_action_semantics(
    ckpt: Mapping[str, object],               # Param: string input for ckpt
    *,
    current_semantics: Mapping[str, object],  # Param: string input for current semantics
    context          : str,  # Param: runtime context carrying validated trainer settings
) -> None:
    """Raise when checkpoint action semantics differ from current semantics"""
    ckpt_semantics = checkpoint_action_semantics(
        ckpt,
        default_arm_controller=str(current_semantics.get("arm_controller", "policy")),
    )
    for key, current in current_semantics.items():
        ckpt_value = ckpt_semantics[key]
        if ckpt_value != current:
            raise RuntimeError(
                f"{context} args mismatch: {key} checkpoint={ckpt_value!r} current={current!r}"
            )


def ensure_checkpoint_policy_schema_compatible(
    ckpt: Mapping[str, object],                    # Param: string input for ckpt
    *,
    obs_schema_version              : int,  # Param: integer input for obs schema version
    removed_obs_keys                : tuple[str, ...],  # Param: ordered mapping keys used to resolve removed obs
    active_observation_normalization: bool,  # Param: boolean input controlling active observation normalization
    obs_keys                        : tuple[str, ...],  # Param: ordered mapping keys used to resolve obs
    obs_dim                         : int,  # Param: integer input for obs dim
    priv_obs_dim                    : int,  # Param: integer input for priv obs dim
    action_dim                      : int,  # Param: integer input for action dim
    context                         : str,  # Param: runtime context carrying validated trainer settings
    require_privileged_agent_state  : bool = False,  # Param: boolean input controlling require privileged agent state
) -> None:
    """Fail early when loading incompatible actor checkpoints

    Steps:
    - Resolve inputs for `ensure_checkpoint_policy_schema_compatible` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    agent_state = ckpt.get("agent", {})
    if not isinstance(agent_state, Mapping):
        agent_state = {}
    ckpt_obs_keys = tuple(ckpt.get("policy_obs_keys", ckpt.get("obs_keys", ())))
    ckpt_schema = ckpt.get("obs_schema_version")
    ckpt_priv_obs_dim = ckpt.get("priv_obs_dim")
    actor_width = checkpoint_actor_input_width(agent_state)
    critic_obs_width = checkpoint_critic_obs_width(agent_state, action_dim)
    stats_width = checkpoint_obs_stats_width(agent_state)
    priv_stats_width = checkpoint_priv_obs_stats_width(agent_state)
    expected_critic_obs_width = int(obs_dim) + int(priv_obs_dim)

    mismatch_reasons: list[str] = []
    if ckpt_schema is not None and int(ckpt_schema) != int(obs_schema_version):
        mismatch_reasons.append(
            f"obs_schema_version checkpoint={ckpt_schema} current={obs_schema_version}"
        )
    if ckpt_obs_keys and ckpt_obs_keys != obs_keys:
        mismatch_reasons.append("policy_obs_keys changed")
    if actor_width is not None and actor_width != int(obs_dim):
        mismatch_reasons.append(f"actor input width checkpoint={actor_width} current={obs_dim}")
    if critic_obs_width is not None and critic_obs_width != expected_critic_obs_width:
        mismatch_reasons.append(
            "critic observation width checkpoint="
            f"{critic_obs_width} current={expected_critic_obs_width}"
        )
    if stats_width is not None and stats_width != int(obs_dim):
        mismatch_reasons.append(f"obs_stats width checkpoint={stats_width} current={obs_dim}")
    if require_privileged_agent_state and int(priv_obs_dim) > 0 and ckpt_priv_obs_dim is None:
        mismatch_reasons.append(
            f"priv_obs_dim missing from checkpoint; current={priv_obs_dim}"
        )
    if ckpt_priv_obs_dim is not None and int(ckpt_priv_obs_dim) != int(priv_obs_dim):
        mismatch_reasons.append(
            f"priv_obs_dim checkpoint={ckpt_priv_obs_dim} current={priv_obs_dim}"
        )
    if (
        require_privileged_agent_state
        and active_observation_normalization
        and int(priv_obs_dim) > 0
        and priv_stats_width is None
    ):
        mismatch_reasons.append(
            "priv_obs_stats missing while privileged critic observation normalization is enabled"
        )
    if priv_stats_width is not None and int(priv_stats_width) != int(priv_obs_dim):
        mismatch_reasons.append(
            f"priv_obs_stats width checkpoint={priv_stats_width} current={priv_obs_dim}"
        )

    if mismatch_reasons:
        raise RuntimeError(
            f"{context} is incompatible with observation schema v{obs_schema_version}: "
            + "; ".join(mismatch_reasons)
            + "; removed actor-visible keys: "
            + ", ".join(removed_obs_keys)
        )
