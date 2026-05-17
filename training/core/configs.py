"""

Typed runtime config slices built from parsed trainer args

File map:

_bool:                        Handle bool logic
_int:                         Handle int logic
_float:                       Handle float logic
_str:                         Handle str logic
TrainingCountsConfig:         Rollout size replay and TD update cadence
OptimizationRuntimeConfig:    TD3 optimizer discount and noise settings
AssistRuntimeConfig:          Policy assist BC and DAgger schedule settings
EvalRuntimeConfig:            Eval play and logging cadence settings
CheckpointRuntimeConfig:      Checkpoint resume and handoff settings
TeacherRuntimeConfig:         Teacher action and contact pre-roll settings
RuntimeConfigBundle:          Typed config bundle for migrated trainer code
build_runtime_config_bundle:  Build typed config slices from one parsed CLI request
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .cli import TrainingCliRequest


def _bool(args: Mapping[str, object], name: str, default: bool = False) -> bool:
    value = args.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int(args: Mapping[str, object], name: str, default: int = 0) -> int:
    try:
        return int(args.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _float(args: Mapping[str, object], name: str, default: float = 0.0) -> float:
    try:
        return float(args.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _str(args: Mapping[str, object], name: str, default: str = "") -> str:
    return str(args.get(name, default))


@dataclass(frozen=True)
class TrainingCountsConfig:
    """Rollout size replay and TD update cadence"""

    num_envs        : int  # Field: number of parallel environment rows represented
    total_steps     : int  # Field: step count used for total steps scheduling or reporting
    start_steps     : int  # Field: step count used for start steps scheduling or reporting
    batch_size      : int  # Field: number of replay samples used in each update batch
    updates_per_step: int  # Field: step count used for updates per step scheduling or reporting
    n_step          : int  # Field: step count used for n step scheduling or reporting
    replay_size     : int  # Field: configured or observed replay-buffer size
    hidden_dim      : int  # Field: integer hidden dim value tracked by training counts config
    policy_delay    : int  # Field: integer policy delay value tracked by training counts config
    rl_phase_start_steps: int  # Field: global step where RL-phase overrides begin


@dataclass(frozen=True)
class OptimizationRuntimeConfig:
    """TD3 optimizer discount and noise settings"""

    gamma                   : float  # Field: discount factor used by TD3 updates
    tau                     : float  # Field: target-network interpolation factor for TD3 updates
    actor_lr                : float  # Field: actor optimizer learning rate
    critic_lr               : float  # Field: critic optimizer learning rate
    exploration_noise       : float  # Field: floating-point exploration noise value used by optimization runtime config
    exploration_noise_finger: float  # Field: floating-point exploration noise finger value used by optimization runtime config
    policy_noise            : float  # Field: floating-point policy noise value used by optimization runtime config
    policy_noise_finger     : float  # Field: floating-point policy noise finger value used by optimization runtime config
    noise_clip              : float  # Field: floating-point noise clip value used by optimization runtime config
    target_q_clip           : float  # Field: floating-point target q clip value used by optimization runtime config
    critic_grad_clip        : float  # Field: floating-point critic grad clip value used by optimization runtime config
    actor_pre_tanh_l2       : float  # Field: floating-point actor pre tanh l2 value used by optimization runtime config
    actor_freeze_steps      : int  # Field: step count used for actor freeze steps scheduling or reporting
    critic_burn_in_steps    : int  # Field: step count used for critic burn in steps scheduling or reporting


@dataclass(frozen=True)
class AssistRuntimeConfig:
    """Policy assist BC and DAgger schedule settings"""

    policy_bc_relabel              : bool  # Field: boolean value indicating the policy bc relabel state for assist runtime config
    policy_assist_mix              : float  # Field: floating-point policy assist mix value used by assist runtime config
    policy_assist_mix_floor        : float  # Field: floating-point policy assist mix floor value used by assist runtime config
    policy_assist_decay_steps      : int  # Field: step count used for policy assist decay steps scheduling or reporting
    policy_assist_decay_start_steps: int  # Field: step count used for policy assist decay start steps scheduling or reporting
    teacher_bc_weight              : float  # Field: weight applied to teacher bc terms
    teacher_bc_arm_weight          : float  # Field: weight applied to teacher bc arm terms
    teacher_bc_finger_weight       : float  # Field: weight applied to teacher bc finger terms
    teacher_bc_decay_steps         : int  # Field: step count used for teacher bc decay steps scheduling or reporting
    bc_only_steps                  : int  # Field: step count used for bc only steps scheduling or reporting
    bc_only_weight                 : float  # Field: weight applied to bc only terms
    bc_only_arm_weight             : float  # Field: weight applied to bc only arm terms
    bc_only_finger_weight          : float  # Field: weight applied to bc only finger terms
    assist_noise_arm               : float  # Field: floating-point assist noise arm value used by assist runtime config
    assist_noise_finger            : float  # Field: floating-point assist noise finger value used by assist runtime config
    assist_noise_clean_bc_target   : bool  # Field: boolean value indicating the assist noise clean bc target state for assist runtime config


@dataclass(frozen=True)
class EvalRuntimeConfig:
    """Eval play and logging cadence settings"""

    eval_every             : int  # Field: global transition interval between inline eval runs; 0 keeps automatic cadence
    eval_steps             : int  # Field: step count used for eval steps scheduling or reporting
    eval_episodes          : int  # Field: integer eval episodes value tracked by eval runtime config
    eval_start_steps       : int  # Field: step count used for eval start steps scheduling or reporting
    eval_teacher_assist_mix: float  # Field: floating-point eval teacher assist mix value used by eval runtime config
    play                   : bool  # Field: boolean value indicating the play state for eval runtime config
    play_skip_checkpoint   : bool  # Field: boolean value indicating the play skip checkpoint state for eval runtime config
    play_episodes          : int  # Field: integer play episodes value tracked by eval runtime config
    log_every              : int  # Field: integer log every value tracked by eval runtime config
    sleep                  : float  # Field: floating-point sleep value used by eval runtime config


@dataclass(frozen=True)
class CheckpointRuntimeConfig:
    """Checkpoint resume and handoff settings"""

    checkpoint_path                   : str  # Field: checkpoint file path used for load/save operations
    log_jsonl                         : str  # Field: JSONL log path or enablement flag for structured logging
    tensorboard_dir                   : str  # Field: filesystem location for tensorboard dir
    phase1_checkpoint                 : str  # Field: string phase1 checkpoint value used by checkpoint runtime config
    phase1_teacher_only               : bool  # Field: boolean value indicating the phase1 teacher only state for checkpoint runtime config
    resume_checkpoint                 : str  # Field: string resume checkpoint value used by checkpoint runtime config
    actor_init_checkpoint             : str  # Field: string actor init checkpoint value used by checkpoint runtime config
    checkpoint_every                  : int  # Field: integer checkpoint every value tracked by checkpoint runtime config
    rolling_checkpoint_every          : int  # Field: integer rolling checkpoint every value tracked by checkpoint runtime config
    rolling_checkpoint_keep           : int  # Field: integer rolling checkpoint keep value tracked by checkpoint runtime config
    save_replay_in_checkpoint         : bool  # Field: boolean value indicating the save replay in checkpoint state for checkpoint runtime config
    resume_replay                     : bool  # Field: boolean value indicating the resume replay state for checkpoint runtime config
    resume_global_step                : bool  # Field: step count used for resume global step scheduling or reporting
    force_dagger_after_resume         : bool  # Field: boolean value indicating the force dagger after resume state for checkpoint runtime config
    reset_optimizers_on_resume        : bool  # Field: boolean value indicating the reset optimizers on resume state for checkpoint runtime config
    handoff_checkpoint_path           : str  # Field: filesystem location for handoff checkpoint path
    final_handoff_checkpoint_path     : str  # Field: filesystem location for final handoff checkpoint path
    stop_after_handoff_checkpoint     : bool  # Field: boolean value indicating the stop after handoff checkpoint state for checkpoint runtime config
    allow_handoff_source_hash_mismatch: bool  # Field: boolean value indicating the allow handoff source hash mismatch state for checkpoint runtime config


@dataclass(frozen=True)
class TeacherRuntimeConfig:
    """Teacher action and contact pre-roll settings"""

    arm_controller                        : str  # Field: string arm controller value used by teacher runtime config
    teacher_arm_source                    : str  # Field: string teacher arm source value used by teacher runtime config
    finger_action_mode                    : str  # Field: configured interpretation of finger action columns
    finger_delta_scale                    : float  # Field: scale applied to finger delta action columns
    finger_curl_mode                      : str  # Field: string finger curl mode value used by teacher runtime config
    topdown_contact_teacher               : bool  # Field: boolean value indicating the topdown contact teacher state for teacher runtime config
    topdown_contact_teacher_bypass_unlock : bool  # Field: boolean value indicating the topdown contact teacher bypass unlock state for teacher runtime config
    topdown_contact_teacher_close_rate    : float  # Field: floating-point topdown contact teacher close rate value used by teacher runtime config
    topdown_contact_teacher_start_fraction: float  # Field: floating-point topdown contact teacher start fraction value used by teacher runtime config
    topdown_contact_teacher_max_fraction  : float  # Field: floating-point topdown contact teacher max fraction value used by teacher runtime config
    topdown_contact_teacher_middle_scale  : float  # Field: multiplier applied to topdown contact teacher middle terms
    contact_start_mode                    : str  # Field: string contact start mode value used by teacher runtime config
    contact_preroll_max_steps             : int  # Field: step count used for contact preroll max steps scheduling or reporting
    contact_preroll_touch_mode            : str  # Field: string contact preroll touch mode value used by teacher runtime config
    topdown_preroll_fraction              : float  # Field: floating-point topdown preroll fraction value used by teacher runtime config
    topdown_preroll_max_steps             : int  # Field: step count used for topdown preroll max steps scheduling or reporting


@dataclass(frozen=True)
class RuntimeConfigBundle:
    """Typed config bundle for migrated trainer code"""

    counts      : TrainingCountsConfig  # Field: stores counts for runtime config bundle
    optimization: OptimizationRuntimeConfig  # Field: stores optimization for runtime config bundle
    assist      : AssistRuntimeConfig  # Field: string assist value used by runtime config bundle
    eval        : EvalRuntimeConfig  # Field: stores eval for runtime config bundle
    checkpoint  : CheckpointRuntimeConfig  # Field: integer checkpoint value tracked by runtime config bundle
    teacher     : TeacherRuntimeConfig  # Field: stores teacher for runtime config bundle


def build_runtime_config_bundle(request: TrainingCliRequest) -> RuntimeConfigBundle:
    """Build typed config slices from one parsed CLI request"""
    args = request.known_args
    return RuntimeConfigBundle(
        counts=TrainingCountsConfig(
            num_envs=_int(args, "num_envs", 1),
            total_steps=_int(args, "total_steps", 100000),
            start_steps=_int(args, "start_steps", 400),
            batch_size=_int(args, "batch_size", 128),
            updates_per_step=_int(args, "updates_per_step", 4),
            n_step=_int(args, "n_step", 1),
            replay_size=_int(args, "replay_size", 200000),
            hidden_dim=_int(args, "hidden_dim", 256),
            policy_delay=_int(args, "policy_delay", 2),
            rl_phase_start_steps=_int(args, "rl_phase_start_steps", -1),
        ),
        optimization=OptimizationRuntimeConfig(
            gamma=_float(args, "gamma", 0.995),
            tau=_float(args, "tau", 0.005),
            actor_lr=_float(args, "actor_lr", 3e-4),
            critic_lr=_float(args, "critic_lr", 3e-4),
            exploration_noise=_float(args, "exploration_noise", 0.25),
            exploration_noise_finger=_float(args, "exploration_noise_finger", 0.0),
            policy_noise=_float(args, "policy_noise", 0.2),
            policy_noise_finger=_float(args, "policy_noise_finger", 0.0),
            noise_clip=_float(args, "noise_clip", 0.5),
            target_q_clip=_float(args, "target_q_clip", 0.0),
            critic_grad_clip=_float(args, "critic_grad_clip", 0.0),
            actor_pre_tanh_l2=_float(args, "actor_pre_tanh_l2", 0.0),
            actor_freeze_steps=_int(args, "actor_freeze_steps", 0),
            critic_burn_in_steps=_int(args, "critic_burn_in_steps", -1),
        ),
        assist=AssistRuntimeConfig(
            policy_bc_relabel=_bool(args, "policy_bc_relabel", False),
            policy_assist_mix=_float(args, "policy_assist_mix", 0.9),
            policy_assist_mix_floor=_float(args, "policy_assist_mix_floor", 0.0),
            policy_assist_decay_steps=_int(args, "policy_assist_decay_steps", 3000),
            policy_assist_decay_start_steps=_int(args, "policy_assist_decay_start_steps", -1),
            teacher_bc_weight=_float(args, "teacher_bc_weight", 0.5),
            teacher_bc_arm_weight=_float(args, "teacher_bc_arm_weight", -1.0),
            teacher_bc_finger_weight=_float(args, "teacher_bc_finger_weight", -1.0),
            teacher_bc_decay_steps=_int(args, "teacher_bc_decay_steps", 30000),
            bc_only_steps=_int(args, "bc_only_steps", 0),
            bc_only_weight=_float(args, "bc_only_weight", 1.0),
            bc_only_arm_weight=_float(args, "bc_only_arm_weight", -1.0),
            bc_only_finger_weight=_float(args, "bc_only_finger_weight", -1.0),
            assist_noise_arm=_float(args, "assist_noise_arm", 0.0),
            assist_noise_finger=_float(args, "assist_noise_finger", 0.0),
            assist_noise_clean_bc_target=_bool(args, "assist_noise_clean_bc_target", True),
        ),
        eval=EvalRuntimeConfig(
            eval_every=_int(args, "eval_every", 0),
            eval_steps=_int(args, "eval_steps", 500),
            eval_episodes=_int(args, "eval_episodes", 20),
            eval_start_steps=_int(args, "eval_start_steps", 0),
            eval_teacher_assist_mix=_float(args, "eval_teacher_assist_mix", 0.0),
            play=_bool(args, "play", False),
            play_skip_checkpoint=_bool(args, "play_skip_checkpoint", False),
            play_episodes=_int(args, "play_episodes", 5),
            log_every=_int(args, "log_every", 100),
            sleep=_float(args, "sleep", 0.0),
        ),
        checkpoint=CheckpointRuntimeConfig(
            checkpoint_path=request.checkpoint_path,
            log_jsonl=request.log_jsonl,
            tensorboard_dir=request.tensorboard_dir,
            phase1_checkpoint=_str(args, "phase1_checkpoint"),
            phase1_teacher_only=_bool(args, "phase1_teacher_only", False),
            resume_checkpoint=_str(args, "resume_checkpoint"),
            actor_init_checkpoint=_str(args, "actor_init_checkpoint"),
            checkpoint_every=_int(args, "checkpoint_every", 2500),
            rolling_checkpoint_every=_int(args, "rolling_checkpoint_every", 5000),
            rolling_checkpoint_keep=_int(args, "rolling_checkpoint_keep", 20),
            save_replay_in_checkpoint=_bool(args, "save_replay_in_checkpoint", False),
            resume_replay=_bool(args, "resume_replay", False),
            resume_global_step=_bool(args, "resume_global_step", False),
            force_dagger_after_resume=_bool(args, "force_dagger_after_resume", False),
            reset_optimizers_on_resume=_bool(args, "reset_optimizers_on_resume", False),
            handoff_checkpoint_path=_str(args, "handoff_checkpoint_path"),
            final_handoff_checkpoint_path=_str(args, "final_handoff_checkpoint_path"),
            stop_after_handoff_checkpoint=_bool(args, "stop_after_handoff_checkpoint", False),
            allow_handoff_source_hash_mismatch=_bool(args, "allow_handoff_source_hash_mismatch", False),
        ),
        teacher=TeacherRuntimeConfig(
            arm_controller=request.arm_controller,
            teacher_arm_source=_str(args, "teacher_arm_source", "ik"),
            finger_action_mode=request.finger_action_mode,
            finger_delta_scale=request.finger_delta_scale,
            finger_curl_mode=_str(args, "finger_curl_mode", "distance"),
            topdown_contact_teacher=_bool(args, "topdown_contact_teacher", False),
            topdown_contact_teacher_bypass_unlock=_bool(args, "topdown_contact_teacher_bypass_unlock", False),
            topdown_contact_teacher_close_rate=_float(args, "topdown_contact_teacher_close_rate", 0.025),
            topdown_contact_teacher_start_fraction=_float(args, "topdown_contact_teacher_start_fraction", 0.08),
            topdown_contact_teacher_max_fraction=_float(args, "topdown_contact_teacher_max_fraction", 0.7),
            topdown_contact_teacher_middle_scale=_float(args, "topdown_contact_teacher_middle_scale", 0.0),
            contact_start_mode=_str(args, "contact_start_mode", "reset"),
            contact_preroll_max_steps=_int(args, "contact_preroll_max_steps", 500),
            contact_preroll_touch_mode=_str(args, "contact_preroll_touch_mode", "off"),
            topdown_preroll_fraction=_float(args, "topdown_preroll_fraction", 0.0),
            topdown_preroll_max_steps=_int(args, "topdown_preroll_max_steps", 700),
        ),
    )
