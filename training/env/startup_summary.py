"""

Startup summary formatting for trainer launches

File map:

format_startup_summary:  Return startup summary lines for a resolved trainer launch
"""

from __future__ import annotations

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext


def format_startup_summary(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    *,
    eval_every    : int | None,  # Param: interval controlling how often eval runs
    handoff_digest: str | None = None,  # Param: string input for handoff digest
) -> tuple[str, ...]:
    """Return startup summary lines for a resolved trainer launch

    Steps:
    - Resolve inputs for `format_startup_summary` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    args = context.args
    lines = [
        f"task={context.task}",
        (
            f"obs_dim={context.dims.obs_dim} action_dim={context.dims.action_dim} "
            f"full_action_dim={context.dims.full_action_dim}"
        ),
        f"td3_backend={context.td3_backend}",
        f"arm_controller={context.action.arm_controller}",
        (
            f"include_wrist_roll={int(bool(args.get('include_wrist_roll', False)))} "
            f"include_waist_yaw={int(bool(args.get('include_waist_yaw', False)))} "
            f"arm_action_scale_profile={context.args.get('arm_action_scale_profile', 'side')}"
        ),
        f"teacher_arm_source={configs.teacher.teacher_arm_source}",
        f"n_step={configs.counts.n_step}",
        (
            f"observation_normalization={int(bool(args.get('observation_normalization', False)))} "
            f"reward_normalization={int(bool(args.get('reward_normalization', False)))}"
        ),
        (
            f"finger_action_mode={configs.teacher.finger_action_mode} "
            f"finger_delta_scale={configs.teacher.finger_delta_scale}"
        ),
        (
            f"actor_lr={configs.optimization.actor_lr:g} "
            f"critic_lr={configs.optimization.critic_lr:g} "
            f"gamma={configs.optimization.gamma:g} "
            f"target_q_clip={configs.optimization.target_q_clip:g} "
            f"critic_grad_clip={configs.optimization.critic_grad_clip:g}"
        ),
        (
            f"actor_q_action_gate_mode={args.get('actor_q_action_gate_mode', 'env')} "
            f"actor_bc_action_gate_mode={args.get('actor_bc_action_gate_mode', 'env')}"
        ),
        f"start_steps={configs.counts.start_steps} log_jsonl={configs.checkpoint.log_jsonl}",
        f"checkpoint_path={configs.checkpoint.checkpoint_path}",
    ]
    if eval_every is None:
        lines.append(
            f"eval_every=disabled eval_steps={configs.eval.eval_steps} "
            f"eval_episodes={configs.eval.eval_episodes}"
        )
    else:
        lines.append(
            f"eval_every={eval_every} eval_start_steps={configs.eval.eval_start_steps} "
            f"eval_steps={configs.eval.eval_steps} eval_episodes={configs.eval.eval_episodes}"
        )
    if configs.checkpoint.handoff_checkpoint_path or handoff_digest:
        lines.append(
            f"handoff_path={configs.checkpoint.handoff_checkpoint_path or 'none'} "
            f"handoff_digest={handoff_digest or 'none'}"
        )
    if configs.teacher.contact_start_mode != "reset":
        lines.append(
            f"contact_start_mode={configs.teacher.contact_start_mode} "
            f"preroll_max_steps={configs.teacher.contact_preroll_max_steps} "
            f"preroll_touch_mode={configs.teacher.contact_preroll_touch_mode}"
        )
    if configs.teacher.topdown_preroll_fraction > 0.0:
        lines.append(
            f"topdown_preroll_fraction={configs.teacher.topdown_preroll_fraction:g} "
            f"topdown_preroll_max_steps={configs.teacher.topdown_preroll_max_steps}"
        )
    return tuple(lines)
