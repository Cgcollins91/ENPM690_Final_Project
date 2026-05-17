"""

Training phase and resume override helpers


This module provides helper functions and data structures for applying training phase overrides based on the current training step,
and applying resume schedule overrides when resuming from a checkpoint with force-dagger requested, used by the main training loop
at startup and after each environment step.

File map:

PhaseOverrideAgent:                       Agent surface needed by phase override application
PhaseOverrideResult:                      Outcome of a phase override attempt
ResumeDaggerOverrideResult:               Outcome of force-dagger resume schedule overrides
RL_PHASE_INT_OVERRIDES:                   Define rl phase int overrides constant
RL_PHASE_FLOAT_OVERRIDES:                 Define rl phase float overrides constant
RL_PHASE_TEACHER_BC_COMPONENT_OVERRIDES:  Define rl phase teacher bc component overrides constant
phase_float_override_is_set:              Return whether a float override uses its set sentinel
training_phase_for_step:                  Return the requested phase name and id
_int_arg:                                 Handle int arg logic
_float_arg:                               Handle float arg logic
_bool_arg:                                Handle bool arg logic
apply_rl_phase_overrides:                 Apply one-shot RL phase overrides to args and agent
apply_force_dagger_resume_overrides:      Apply DAgger resume schedule overrides when requested
apply_default_teacher_bc_decay:           Apply default teacher BC decay derived from BC-only steps
mixed_n_step_schedule_notice:             Return the mixed n-step notice when the RL phase changes horizon
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol


class PhaseOverrideAgent(Protocol):
    """Agent surface needed by phase override application"""

    train_step : int  # step count used for train step scheduling or reporting

    def set_optimizer_lrs(self, *, actor_lr: float | None = None, critic_lr: float | None = None) -> None:
        """Set optimizer learning rates"""
        ...

    def reset_critic_optimizers(self) -> None:
        """Reset critic optimizers"""
        ...

    def sync_target_networks(self) -> None:
        """Copy online weights into target weights"""
        ...


@dataclass(frozen=True)
class PhaseOverrideResult:
    """Outcome of a phase override attempt"""

    switched  : bool  # boolean value indicating the switched state for phase override result
    phase_name: str  # string phase name value used by phase override result
    phase_id  : int  # integer phase id value tracked by phase override result
    changed   : tuple[str, ...]  # string changed value used by phase override result
    message   : str | None = None  # human-readable status or error detail


@dataclass(frozen=True)
class ResumeDaggerOverrideResult:
    """Outcome of force-dagger resume schedule overrides"""

    applied: bool  # boolean value indicating the applied state for resume dagger override result
    changed: tuple[str, ...]  # string changed value used by resume dagger override result
    message: str | None = None  # human-readable status or error detail


RL_PHASE_INT_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("updates_per_step", "rl_updates_per_step"),
    ("n_step", "rl_n_step"),
    ("policy_delay", "rl_policy_delay"),
    ("policy_assist_decay_steps", "rl_policy_assist_decay_steps"),
    ("policy_assist_decay_start_steps", "rl_policy_assist_decay_start_steps"),
    ("teacher_bc_decay_steps", "rl_teacher_bc_decay_steps"),
)
RL_PHASE_FLOAT_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("gamma", "rl_gamma"),
    ("tau", "rl_tau"),
    ("actor_lr", "rl_actor_lr"),
    ("critic_lr", "rl_critic_lr"),
    ("target_q_clip", "rl_target_q_clip"),
    ("critic_grad_clip", "rl_critic_grad_clip"),
    ("actor_pre_tanh_l2", "rl_actor_pre_tanh_l2"),
    ("exploration_noise", "rl_exploration_noise"),
    ("exploration_noise_finger", "rl_exploration_noise_finger"),
    ("policy_noise", "rl_policy_noise"),
    ("policy_noise_finger", "rl_policy_noise_finger"),
    ("noise_clip", "rl_noise_clip"),
    ("policy_assist_mix", "rl_policy_assist_mix"),
    ("policy_assist_mix_floor", "rl_policy_assist_mix_floor"),
    ("teacher_bc_weight", "rl_teacher_bc_weight"),
)
RL_PHASE_TEACHER_BC_COMPONENT_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("teacher_bc_arm_weight", "rl_teacher_bc_arm_weight"),
    ("teacher_bc_finger_weight", "rl_teacher_bc_finger_weight"),
)


def phase_float_override_is_set(override_name: str, value: float) -> bool:
    """Return whether a float override uses its set sentinel"""
    if override_name in ("rl_teacher_bc_arm_weight", "rl_teacher_bc_finger_weight"):
        return float(value) > -2.0
    return float(value) >= 0.0


def training_phase_for_step(
    *,
    global_step              : int,  # Param: current absolute training step
    rl_phase_start_steps     : int,  # Param: step count used for rl phase start steps
    force_dagger_after_resume: bool = False,  # Param: boolean input controlling force dagger after resume
) -> tuple[str, int]:
    """Return the requested phase name and id"""
    if int(rl_phase_start_steps) < 0:
        return "single", 0
    if force_dagger_after_resume:
        return "dagger", 0
    if int(global_step) >= int(rl_phase_start_steps):
        return "rl", 1
    return "dagger", 0


def _int_arg(args: MutableMapping[str, Any], name: str, default: int = -1) -> int:
    return int(args.get(name, default))


def _float_arg(args: MutableMapping[str, Any], name: str, default: float = -1.0) -> float:
    return float(args.get(name, default))


def _bool_arg(args: MutableMapping[str, Any], name: str, default: bool = False) -> bool:
    return bool(args.get(name, default))


def apply_rl_phase_overrides(
    args : MutableMapping[str, Any],  # Param: argument mapping or namespace read and updated by this helper
    agent: PhaseOverrideAgent,  # Param: TD3 agent whose networks, optimizers, or stats are used
    *,
    global_step: int,                # Param: current absolute training step
) -> PhaseOverrideResult:
    """Apply one-shot RL phase overrides to args and agent

    Steps:
    - Resolve inputs for `apply_rl_phase_overrides` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    phase_name, phase_id = training_phase_for_step(
        global_step=global_step,
        rl_phase_start_steps=_int_arg(args, "rl_phase_start_steps"),
        force_dagger_after_resume=_bool_arg(args, "force_dagger_after_resume"),
    )
    if phase_name == "single":
        args["_training_phase_name"] = "single"
        args["_training_phase_id"] = 0
        return PhaseOverrideResult(False, "single", 0, ())
    if args.get("_training_phase_name") == phase_name:
        return PhaseOverrideResult(False, phase_name, phase_id, ())

    args["_training_phase_name"] = phase_name
    args["_training_phase_id"] = phase_id
    if phase_name != "rl":
        message = (
            "training_phase_start "
            f"step={int(global_step)} phase=dagger "
            f"rl_phase_start={_int_arg(args, 'rl_phase_start_steps')} "
            f"force_dagger_after_resume={int(_bool_arg(args, 'force_dagger_after_resume'))}"
        )
        return PhaseOverrideResult(True, phase_name, phase_id, (), message)

    old_actor_lr = _float_arg(args, "actor_lr", 0.0)
    old_critic_lr = _float_arg(args, "critic_lr", 0.0)
    changed: list[str] = []

    for target_name, override_name in RL_PHASE_INT_OVERRIDES:
        value = _int_arg(args, override_name)
        if value >= 0:
            if target_name in ("n_step", "policy_delay") and value < 1:
                raise RuntimeError(f"{override_name} must be >= 1 when set; got {value}")
            args[target_name] = value
            changed.append(f"{target_name}={value}")

    for target_name, override_name in RL_PHASE_FLOAT_OVERRIDES:
        value = _float_arg(args, override_name)
        if phase_float_override_is_set(override_name, value):
            args[target_name] = value
            changed.append(f"{target_name}={value:g}")

    for target_name, override_name in RL_PHASE_TEACHER_BC_COMPONENT_OVERRIDES:
        value = _float_arg(args, override_name, -2.0)
        if phase_float_override_is_set(override_name, value):
            args[target_name] = value
            changed.append(f"{target_name}={value:g}")

    policy_bc_relabel = _int_arg(args, "rl_policy_bc_relabel")
    if policy_bc_relabel >= 0:
        args["policy_bc_relabel"] = policy_bc_relabel
        changed.append(f"policy_bc_relabel={policy_bc_relabel}")

    new_actor_lr = _float_arg(args, "actor_lr", old_actor_lr)
    new_critic_lr = _float_arg(args, "critic_lr", old_critic_lr)
    agent_config = getattr(agent, "config", None)
    if agent_config is not None:
        config_updates = {
            "gamma"                   : _float_arg(args, "gamma", getattr(agent_config, "gamma", 0.995)),
            "tau"                     : _float_arg(args, "tau", getattr(agent_config, "tau", 0.005)),
            "policy_delay"            : _int_arg(args, "policy_delay", getattr(agent_config, "policy_delay", 2)),
            "policy_noise"            : _float_arg(args, "policy_noise", getattr(agent_config, "policy_noise", 0.0)),
            "policy_noise_finger"     : _float_arg(args, "policy_noise_finger", getattr(agent_config, "policy_noise_finger", 0.0)),
            "noise_clip"              : _float_arg(args, "noise_clip", getattr(agent_config, "noise_clip", 0.0)),
            "exploration_noise"       : _float_arg(args, "exploration_noise", getattr(agent_config, "exploration_noise", 0.0)),
            "exploration_noise_finger": _float_arg(args, "exploration_noise_finger", getattr(agent_config, "exploration_noise_finger", 0.0)),
            "target_q_clip"           : _float_arg(args, "target_q_clip", getattr(agent_config, "target_q_clip", 0.0)),
            "critic_grad_clip"        : _float_arg(args, "critic_grad_clip", getattr(agent_config, "critic_grad_clip", 0.0)),
            "actor_pre_tanh_l2"       : _float_arg(args, "actor_pre_tanh_l2", getattr(agent_config, "actor_pre_tanh_l2", 0.0)),
            "teacher_bc_weight"       : _float_arg(args, "teacher_bc_weight", getattr(agent_config, "teacher_bc_weight", 0.0)),
            "teacher_bc_arm_weight"   : _float_arg(args, "teacher_bc_arm_weight", getattr(agent_config, "teacher_bc_arm_weight", -1.0)),
            "teacher_bc_finger_weight": _float_arg(args, "teacher_bc_finger_weight", getattr(agent_config, "teacher_bc_finger_weight", -1.0)),
            "teacher_bc_decay_steps"  : _int_arg(args, "teacher_bc_decay_steps", getattr(agent_config, "teacher_bc_decay_steps", 0)),
            "bc_only_steps"           : _int_arg(args, "bc_only_steps", getattr(agent_config, "bc_only_steps", 0)),
            "active_n_step"           : _int_arg(args, "n_step", getattr(agent_config, "active_n_step", 1)),
            "active_updates_per_step" : _int_arg(args, "updates_per_step", getattr(agent_config, "active_updates_per_step", 1)),
        }
        for field_name, field_value in config_updates.items():
            if hasattr(agent_config, field_name):
                setattr(agent_config, field_name, field_value)
    agent.set_optimizer_lrs(
        actor_lr=new_actor_lr if new_actor_lr != old_actor_lr else None,
        critic_lr=new_critic_lr if new_critic_lr != old_critic_lr else None,
    )
    if _int_arg(args, "rl_reset_critic_optimizers_on_switch", 0):
        agent.reset_critic_optimizers()
        changed.append("critic_optimizers=reset")
    if _int_arg(args, "rl_sync_targets_on_switch", 0):
        agent.sync_target_networks()
        changed.append("targets=synced")
    rl_actor_freeze_steps = max(0, _int_arg(args, "rl_actor_freeze_steps", 0))
    if rl_actor_freeze_steps > 0:
        freeze_until = int(agent.train_step) + rl_actor_freeze_steps
        args["_rl_actor_freeze_until_train_step"] = freeze_until
        if agent_config is not None and hasattr(agent_config, "rl_actor_freeze_until_train_step"):
            setattr(agent_config, "rl_actor_freeze_until_train_step", freeze_until)
        changed.append(f"rl_actor_freeze_until_train_step={freeze_until}")
    else:
        args["_rl_actor_freeze_until_train_step"] = -1
        if agent_config is not None and hasattr(agent_config, "rl_actor_freeze_until_train_step"):
            setattr(agent_config, "rl_actor_freeze_until_train_step", -1)

    message = (
        "training_phase_switch "
        f"step={int(global_step)} phase=rl replay_preserved=1 "
        f"{' '.join(changed) if changed else 'no_overrides'}"
    )
    return PhaseOverrideResult(True, phase_name, phase_id, tuple(changed), message)


def apply_force_dagger_resume_overrides(args: MutableMapping[str, Any]) -> ResumeDaggerOverrideResult:
    """Apply DAgger resume schedule overrides when requested

    Steps:
    - Resolve inputs for `apply_force_dagger_resume_overrides` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not _bool_arg(args, "force_dagger_after_resume"):
        return ResumeDaggerOverrideResult(False, ())

    changed: list[str] = []
    if _float_arg(args, "dagger_resume_policy_assist_mix") >= 0.0:
        value = _float_arg(args, "dagger_resume_policy_assist_mix")
        args["policy_assist_mix"] = value
        changed.append(f"policy_assist_mix={value:g}")
    if _float_arg(args, "dagger_resume_policy_assist_mix_floor") >= 0.0:
        value = _float_arg(args, "dagger_resume_policy_assist_mix_floor")
        args["policy_assist_mix_floor"] = value
        changed.append(f"policy_assist_mix_floor={value:g}")
    if _int_arg(args, "dagger_resume_policy_assist_decay_steps") >= 0:
        value = _int_arg(args, "dagger_resume_policy_assist_decay_steps")
        args["policy_assist_decay_steps"] = value
        changed.append(f"policy_assist_decay_steps={value}")
    message = (
        "resume_replay: force_dagger_after_resume=1 "
        f"{' '.join(changed) if changed else 'using_checkpoint_dagger_schedule'}"
    )
    return ResumeDaggerOverrideResult(True, tuple(changed), message)


def apply_default_teacher_bc_decay(args: MutableMapping[str, Any], *, explicit: bool) -> bool:
    """Apply default teacher BC decay derived from BC-only steps"""
    if explicit or _int_arg(args, "bc_only_steps", 0) <= 0:
        return False
    value = max(5000, _int_arg(args, "bc_only_steps") // 2)
    args["teacher_bc_decay_steps"] = value
    return True


def mixed_n_step_schedule_notice(args: MutableMapping[str, Any]) -> str | None:
    """Return the mixed n-step notice when the RL phase changes horizon

    Steps:
    - Resolve inputs for `mixed_n_step_schedule_notice` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if _int_arg(args, "rl_phase_start_steps") < 0:
        return None
    rl_n_step = _int_arg(args, "rl_n_step")
    n_step = _int_arg(args, "n_step")
    if rl_n_step >= 0 and rl_n_step != n_step:
        return (
            "rl_phase: preserving requested mixed n_step schedule "
            f"(base_n_step={n_step}, rl_n_step={rl_n_step})"
        )
    return None
