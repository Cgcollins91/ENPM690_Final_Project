"""

Progress logging helpers for the training loop

File map:

crossed_log_boundary:     Return whether a step batch crossed the log cadence
should_run_eval:          Return whether eval should run after the current step
_finite_float:            Handle finite float logic
format_update_bits:       Format compact TD3 update diagnostics for progress lines
format_reward_term_bits:  Format top absolute reward terms for progress lines
"""

from __future__ import annotations

from collections.abc import Mapping
import math


def crossed_log_boundary(global_step: int, num_added: int, log_every: int) -> bool:
    """Return whether a step batch crossed the log cadence"""
    if int(log_every) <= 0:
        return False
    previous_step = int(global_step) - int(num_added)
    return (int(global_step) // int(log_every)) != (previous_step // int(log_every))


def should_run_eval(
    *,
    next_eval_step: int | None,  # Param: next global step that should trigger evaluation, or None when eval is disabled
    replay_size   : int,  # Param: number of transitions currently available in replay
    batch_size    : int,  # Param: number of replay samples required for one update batch
    global_step   : int,  # Param: current absolute training step
    env0_done     : bool,  # Param: boolean input controlling env0 done
) -> bool:
    """Return whether eval should run after the current step"""
    return (
        next_eval_step is not None
        and int(replay_size) >= int(batch_size)
        and int(global_step) + 1 >= int(next_eval_step)
        and not bool(env0_done)
    )


def _finite_float(metrics: Mapping[str, object], key: str, default: float = math.nan) -> float:
    value = metrics.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def format_update_bits(
    update_info: Mapping[str, object] | None,               # Param: string input for update info
    *,
    actor_update_info       : Mapping[str, object] | None = None,  # Param: string input for actor update info
    actor_teacher_arm_mse   : float                       = math.nan,  # Param: floating-point input for actor teacher arm mse
    actor_teacher_finger_mse: float                       = math.nan,  # Param: floating-point input for actor teacher finger mse
) -> str:
    """Format compact TD3 update diagnostics for progress lines

    Steps:
    - Resolve inputs for `format_update_bits` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if update_info is None:
        return ""
    bits = (
        f" q1={_finite_float(update_info, 'q1_mean'):+.3f}"
        f"/{_finite_float(update_info, 'q1_std'):.3f}"
        f" q2={_finite_float(update_info, 'q2_mean'):+.3f}"
        f"/{_finite_float(update_info, 'q2_std'):.3f}"
        f" tgt={_finite_float(update_info, 'target_mean'):+.3f}"
        f"/{_finite_float(update_info, 'target_std'):.3f}"
        f" c1={_finite_float(update_info, 'critic1_loss'):.4f}"
        f" c2={_finite_float(update_info, 'critic2_loss'):.4f}"
    )
    target_clip_fraction = _finite_float(update_info, "target_clip_fraction", 0.0)
    if target_clip_fraction > 0.0:
        bits += f" tclip={target_clip_fraction:.2f}"
    actor_loss = _finite_float(update_info, "actor_loss")
    if not math.isnan(actor_loss):
        bits += f" actor={actor_loss:+.4f}"

    bc_source = actor_update_info if actor_update_info is not None else update_info
    bc_loss = _finite_float(bc_source, "bc_loss")
    if not math.isnan(bc_loss):
        bits += f" bc={bc_loss:.4f}@{_finite_float(bc_source, 'bc_weight'):.3f}"
        bc_arm = _finite_float(bc_source, "bc_arm_loss")
        bc_finger = _finite_float(bc_source, "bc_finger_loss")
        if not math.isnan(bc_arm):
            bits += f" bc_arm={bc_arm:.4f}"
        if not math.isnan(bc_finger):
            bits += f" bc_f={bc_finger:.4f}"
        bc_arm_w = _finite_float(bc_source, "bc_arm_weight")
        bc_finger_w = _finite_float(bc_source, "bc_finger_weight")
        if not math.isnan(bc_arm_w) or not math.isnan(bc_finger_w):
            bits += f" bc_w={bc_arm_w:.2f}/{bc_finger_w:.2f}"
    if not math.isnan(float(actor_teacher_arm_mse)):
        bits += f" a2t_arm={float(actor_teacher_arm_mse):.4f}"
    if not math.isnan(float(actor_teacher_finger_mse)):
        bits += f" a2t_f={float(actor_teacher_finger_mse):.4f}"
    pre_tanh_rms = _finite_float(update_info, "pre_tanh_rms")
    if not math.isnan(pre_tanh_rms):
        bits += f" raw_rms={pre_tanh_rms:.2f}"
    return bits


def format_reward_term_bits(term_means: Mapping[str, float], *, limit: int = 3) -> str:
    """Format top absolute reward terms for progress lines"""
    if not term_means:
        return ""
    top_terms = sorted(term_means.items(), key=lambda item: abs(float(item[1])), reverse=True)[: int(limit)]
    return " rterm=" + ",".join(f"{name[:12]}={float(value):+.2f}" for name, value in top_terms)
