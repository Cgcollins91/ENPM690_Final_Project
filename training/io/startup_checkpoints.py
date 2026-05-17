"""

Checkpoint startup decision rules for trainer launches

File map:

CheckpointStartupPlan:             Pure checkpoint and warm-start decisions before loading tensors
build_checkpoint_startup_plan:     Build checkpoint startup decisions from typed configs
validate_checkpoint_startup_plan:  Raise for checkpoint startup combinations rejected by the monolith
phase1_warm_start_message:         Return warm-start branch label for logging tests and future launch code
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.configs import RuntimeConfigBundle


@dataclass(frozen=True)
class CheckpointStartupPlan:
    """Pure checkpoint and warm-start decisions before loading tensors"""

    resume_requested              : bool  # boolean value indicating the resume requested state for checkpoint startup plan
    actor_init_requested          : bool  # boolean value indicating the actor init requested state for checkpoint startup plan
    phase1_requested              : bool  # boolean value indicating the phase1 requested state for checkpoint startup plan
    phase1_teacher_only           : bool  # boolean value indicating the phase1 teacher only state for checkpoint startup plan
    phase1_ik_skip                : bool  # boolean value indicating the phase1 ik skip state for checkpoint startup plan
    phase1_actor_copy_allowed     : bool  # boolean value indicating the phase1 actor copy allowed state for checkpoint startup plan
    phase1_policy_teacher_required: bool  # boolean value indicating the phase1 policy teacher required state for checkpoint startup plan
    reset_obs_stats_after_load    : bool  # boolean value indicating the reset obs stats after load state for checkpoint startup plan
    play_requested                : bool  # boolean value indicating the play requested state for checkpoint startup plan
    play_checkpoint_required      : bool  # boolean value indicating the play checkpoint required state for checkpoint startup plan
    replay_resume_requested       : bool  # boolean value indicating the replay resume requested state for checkpoint startup plan
    global_step_resume_requested  : bool  # boolean value indicating the global step resume requested state for checkpoint startup plan
    force_dagger_after_resume     : bool  # boolean value indicating the force dagger after resume state for checkpoint startup plan


def build_checkpoint_startup_plan(
    configs: RuntimeConfigBundle,     # Param: typed runtime config bundle used to derive this plan
    *,
    observation_normalization: bool,  # Param: boolean input controlling observation normalization
    reset_obs_stats_on_resume: bool,  # Param: boolean input controlling reset obs stats on resume
) -> CheckpointStartupPlan:
    """Build checkpoint startup decisions from typed configs

    Steps:
    - Resolve inputs for `build_checkpoint_startup_plan` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    args = configs
    resume_requested = bool(args.checkpoint.resume_checkpoint)
    actor_init_requested = bool(args.checkpoint.actor_init_checkpoint)
    phase1_requested = bool(args.checkpoint.phase1_checkpoint)
    phase1_teacher_only = phase1_requested and args.checkpoint.phase1_teacher_only
    phase1_ik_skip = phase1_requested and args.teacher.arm_controller == "ik"
    phase1_actor_copy_allowed = (
        phase1_requested
        and not phase1_ik_skip
        and not phase1_teacher_only
        and not resume_requested
        and not actor_init_requested
    )
    phase1_policy_teacher_required = (
        args.teacher.teacher_arm_source == "policy"
        and args.teacher.arm_controller == "policy"
    ) or args.teacher.contact_start_mode == "phase1_terminal"
    reset_obs_stats_after_load = (
        bool(observation_normalization)
        and bool(reset_obs_stats_on_resume)
        and not args.checkpoint.resume_replay
        and not args.checkpoint.resume_global_step
        and (resume_requested or actor_init_requested)
    )
    return CheckpointStartupPlan(
        resume_requested=resume_requested,
        actor_init_requested=actor_init_requested,
        phase1_requested=phase1_requested,
        phase1_teacher_only=phase1_teacher_only,
        phase1_ik_skip=phase1_ik_skip,
        phase1_actor_copy_allowed=phase1_actor_copy_allowed,
        phase1_policy_teacher_required=phase1_policy_teacher_required,
        reset_obs_stats_after_load=reset_obs_stats_after_load,
        play_requested=args.eval.play,
        play_checkpoint_required=args.eval.play and not args.eval.play_skip_checkpoint,
        replay_resume_requested=args.checkpoint.resume_replay,
        global_step_resume_requested=args.checkpoint.resume_global_step,
        force_dagger_after_resume=args.checkpoint.force_dagger_after_resume,
    )


def validate_checkpoint_startup_plan(
    plan   : CheckpointStartupPlan,  # Param: precomputed plan object consumed by this helper
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
) -> None:
    """Raise for checkpoint startup combinations rejected by the monolith

    Steps:
    - Resolve inputs for `validate_checkpoint_startup_plan` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if plan.resume_requested and plan.actor_init_requested:
        raise RuntimeError("--actor-init-checkpoint cannot be combined with --resume-checkpoint")
    if plan.phase1_policy_teacher_required and not plan.phase1_requested:
        raise RuntimeError(
            "--teacher-arm-source=policy and contact_start_mode=phase1_terminal require --phase1-checkpoint"
        )
    if configs.teacher.contact_start_mode == "phase1_terminal" and configs.teacher.arm_controller != "policy":
        raise RuntimeError("contact_start_mode=phase1_terminal requires arm_controller=policy")
    if configs.eval.play_skip_checkpoint and configs.eval.eval_teacher_assist_mix < 1.0:
        raise RuntimeError("--play-skip-checkpoint requires --eval-teacher-assist-mix=1.0")
    if configs.checkpoint.resume_replay and not plan.resume_requested:
        raise RuntimeError("--resume-replay requires --resume-checkpoint")
    if configs.checkpoint.resume_global_step and not plan.resume_requested:
        raise RuntimeError("--resume-global-step requires --resume-checkpoint")


def phase1_warm_start_message(plan: CheckpointStartupPlan) -> str | None:
    """Return warm-start branch label for logging tests and future launch code

    Steps:
    - Resolve inputs for `phase1_warm_start_message` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not plan.phase1_requested:
        return None
    if plan.phase1_ik_skip:
        return "phase1_skipped_ik"
    if plan.phase1_actor_copy_allowed:
        return "warm_start_actor"
    if plan.phase1_teacher_only and not (plan.resume_requested or plan.actor_init_requested):
        return "phase1_teacher_only"
    if plan.resume_requested:
        return "phase1_teacher_after_resume"
    if plan.actor_init_requested:
        return "phase1_teacher_after_actor_init"
    return "phase1_teacher_reference"
