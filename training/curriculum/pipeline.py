"""
Fresh adaptive-BC pipeline run with explicit full-reward inputs.


Steps:
- Start from a fresh policy under the full-reward configuration with teacher labels enabled.
- Collect 150k teacher-only replay rows first; the actor and critic do not update and assist noise is disabled.
- Run 500k noisy DAgger/BC rows next: assisted rollout actions get noise, while BC targets stay clean teacher labels.
- Keep fixed policy-assist decay disabled; adaptive assist owns the RL transition after the fixed DAgger/BC block.
- Gate RL assist decay on train-env completed episodes after the DAgger block.
- Lower assist by 0.005 when any completed episode in the metric window has strict-contact rate >= 5%.
- Raise assist by 0.010 when the completed-episode metric window has no episode above that strict-contact floor.
- Once assist reaches the floor, disable teacher BC/relabeling and run 1M policy-only RL transition rows before stopping.
- Save replay-bearing checkpoints every 250k transition rows and optionally evaluate every 500k transition rows.


"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enpm690_final_project.config.profiles import PROFILES
from enpm690_final_project.manifest import write_manifest
from enpm690_final_project.training_engine import build_plan, manifest_for_plan, run_plan


BASE_PROFILE = "teacher_dagger_upstream_fasttd3_lift02_phase1_transfer1m"
ISAAC_PYTHON = os.environ.get("ENPM690_PYTHON", sys.executable)
DESCRIPTION = "Fresh full-reward adaptive-BC/DAgger pipeline run."

TRAINING_OVERRIDE_PREFIXES = (
    "TOPDOWN_",
    "CURRICULUM_",
    "RL_",
    "BC_",
    "TEACHER_",
    "POLICY_",
    "ASSIST_",
    "ADAPTIVE_",
    "CONTACT_",
    "PHASE1_",
    "ACTOR_",
    "CRITIC_",
    "REWARD_",
    "RESET_",
)
TRAINING_OVERRIDE_KEYS = {
    "RUN_DIR",
    "CHECKPOINT_PATH",
    "RESUME_CKPT",
    "ACTOR_INIT_CKPT",
    "PHASE1_CKPT",
    "FASTTD3_REPO",
    "TD3_BACKEND",
    "TASK",
    "DEVICE",
    "SEED",
    "NUM_ENVS",
    "TOTAL_STEPS",
    "START_STEPS",
    "HEADLESS",
    "ENABLE_CAMERAS",
    "DISABLE_CAMERA_PERCEPTION",
}


@dataclass(frozen=True)
class FullRewardInputs:
    """All reward weights copied from the full-reward lift launcher."""

    reach_alignment_error_quadratic           : float = -6.0
    reach_fingertip_line_angle_quadratic      : float = -0.75
    align_alignment_error_quadratic           : float = -6.0
    align_fingertip_line_angle_quadratic      : float = -0.60
    contact_preunlock_pocket_quality          : float = 2.0
    contact_preunlock_no_contact_penalty      : float = -2.0
    contact_target_distance                   : float = -4.0
    contact_vertical_gap                      : float = -4.0
    contact_thumb_contact                     : float = 0.75
    contact_index_contact                     : float = 0.50
    contact_opposed_contact                   : float = 16.0
    contact_bilateral_contact                 : float = 6.0
    contact_bilateral_imbalance               : float = -2.0
    contact_centered_contact                  : float = 8.0
    contact_deep_shell                        : float = 5.0
    contact_one_sided                         : float = -4.0
    contact_one_sided_flip                    : float = -1.0
    contact_overforce                         : float = -2.0
    contact_pose_ready_no_contact             : float = -3.0
    contact_smooth_pose_no_contact            : float = -4.0
    contact_smooth_success_pose               : float = 6.0
    contact_smooth_success_with_contact       : float = 12.0
    contact_success_now_continuous            : float = 8.0
    light_contact_success_bonus               : float = 40.0
    stage2_floor                              : float = 0.25
    contact_finger_center_x_error_quadratic   : float = -1.0
    contact_finger_center_y_error_quadratic   : float = -1.0
    contact_alignment_degradation             : float = -12.0
    lift_height_progress                      : float = 6.0
    contact_lift_progress                     : float = 20.0
    lift_with_grip                            : float = 16.0
    centered_lift_progress                    : float = 4.0
    centered_upright_lift_bonus               : float = 8.0
    block_off_table_bonus                     : float = 30.0
    sustained_lift_grip_bonus                 : float = 10.0
    block_xy_velocity_penalty                 : float = -1.0
    block_angular_velocity_penalty            : float = -0.5
    lift_xy_drift_penalty                     : float = -6.0
    block_tilt_lift_penalty                   : float = -8.0
    uncentered_lift_penalty                   : float = -4.0
    block_drop_penalty                        : float = -10.0

    def env(self) -> dict[str, Any]:
        return {
            "CURRICULUM_W_REACH_ALIGNMENT_ERROR_QUADRATIC"      : self.reach_alignment_error_quadratic,
            "CURRICULUM_W_REACH_FINGERTIP_LINE_ANGLE_QUADRATIC" : self.reach_fingertip_line_angle_quadratic,
            "CURRICULUM_W_ALIGN_ALIGNMENT_ERROR_QUADRATIC"      : self.align_alignment_error_quadratic,
            "CURRICULUM_W_ALIGN_FINGERTIP_LINE_ANGLE_QUADRATIC" : self.align_fingertip_line_angle_quadratic,
            "CURRICULUM_W_CONTACT_PREUNLOCK_POCKET_QUALITY"     : self.contact_preunlock_pocket_quality,
            "CURRICULUM_W_CONTACT_PREUNLOCK_NO_CONTACT_PENALTY" : self.contact_preunlock_no_contact_penalty,
            "CURRICULUM_W_CONTACT_TARGET_DISTANCE"              : self.contact_target_distance,
            "CURRICULUM_W_CONTACT_VERTICAL_GAP"                 : self.contact_vertical_gap,
            "CURRICULUM_W_CONTACT_THUMB_CONTACT"                : self.contact_thumb_contact,
            "CURRICULUM_W_CONTACT_INDEX_CONTACT"                : self.contact_index_contact,
            "CURRICULUM_W_CONTACT_OPPOSED_CONTACT"              : self.contact_opposed_contact,
            "CURRICULUM_W_CONTACT_BILATERAL_CONTACT"            : self.contact_bilateral_contact,
            "CURRICULUM_W_CONTACT_BILATERAL_IMBALANCE"          : self.contact_bilateral_imbalance,
            "CURRICULUM_W_CONTACT_CENTERED_CONTACT"             : self.contact_centered_contact,
            "CURRICULUM_W_CONTACT_DEEP_SHELL"                   : self.contact_deep_shell,
            "CURRICULUM_W_CONTACT_ONE_SIDED"                    : self.contact_one_sided,
            "CURRICULUM_W_CONTACT_ONE_SIDED_FLIP"               : self.contact_one_sided_flip,
            "CURRICULUM_W_CONTACT_OVERFORCE"                    : self.contact_overforce,
            "CURRICULUM_W_CONTACT_POSE_READY_NO_CONTACT"        : self.contact_pose_ready_no_contact,
            "CURRICULUM_W_CONTACT_SMOOTH_POSE_NO_CONTACT"       : self.contact_smooth_pose_no_contact,
            "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_POSE"          : self.contact_smooth_success_pose,
            "CURRICULUM_W_CONTACT_SMOOTH_SUCCESS_WITH_CONTACT"  : self.contact_smooth_success_with_contact,
            "CURRICULUM_W_CONTACT_SUCCESS_NOW_CONTINUOUS"       : self.contact_success_now_continuous,
            "CURRICULUM_W_LIGHT_CONTACT_SUCCESS_BONUS"          : self.light_contact_success_bonus,
            "CURRICULUM_W_STAGE2_FLOOR"                         : self.stage2_floor,
            "CURRICULUM_W_CONTACT_FINGER_CENTER_X_ERROR_QUADRATIC": self.contact_finger_center_x_error_quadratic,
            "CURRICULUM_W_CONTACT_FINGER_CENTER_Y_ERROR_QUADRATIC": self.contact_finger_center_y_error_quadratic,
            "CURRICULUM_W_CONTACT_ALIGNMENT_DEGRADATION"        : self.contact_alignment_degradation,
            "CURRICULUM_W_LIFT_HEIGHT_PROGRESS"                 : self.lift_height_progress,
            "CURRICULUM_W_CONTACT_LIFT_PROGRESS"                : self.contact_lift_progress,
            "CURRICULUM_W_LIFT_WITH_GRIP"                       : self.lift_with_grip,
            "CURRICULUM_W_CENTERED_LIFT_PROGRESS"               : self.centered_lift_progress,
            "CURRICULUM_W_CENTERED_UPRIGHT_LIFT_BONUS"          : self.centered_upright_lift_bonus,
            "CURRICULUM_W_BLOCK_OFF_TABLE_BONUS"                : self.block_off_table_bonus,
            "CURRICULUM_W_SUSTAINED_LIFT_GRIP_BONUS"            : self.sustained_lift_grip_bonus,
            "CURRICULUM_W_BLOCK_XY_VELOCITY_PENALTY"            : self.block_xy_velocity_penalty,
            "CURRICULUM_W_BLOCK_ANGULAR_VELOCITY_PENALTY"       : self.block_angular_velocity_penalty,
            "CURRICULUM_W_LIFT_XY_DRIFT_PENALTY"                : self.lift_xy_drift_penalty,
            "CURRICULUM_W_BLOCK_TILT_LIFT_PENALTY"              : self.block_tilt_lift_penalty,
            "CURRICULUM_W_UNCENTERED_LIFT_PENALTY"              : self.uncentered_lift_penalty,
            "CURRICULUM_W_BLOCK_DROP_PENALTY"                   : self.block_drop_penalty,
        }


@dataclass(frozen=True)
class SuccessSafetyInputs:
    """All success, drift, and drop-gating inputs."""

    lift_success_height             : float = 0.02
    lift_success_hold_steps         : int   = 60
    lift_success_mode               : str   = "gated"
    lift_success_requires_contact   : int   = 1
    lift_success_contact_mode       : str   = "opposed"
    lift_success_contact_min        : float = 0.30
    lift_success_xy_drift_max       : float = 0.08
    lift_success_block_tilt_max_deg : float = 45.0
    block_drift_threshold           : float = 0.25
    contact_block_disp_max          : float = 0.12
    lift_terminate_drop_from_max    : float = 0.03
    lift_terminate_drop_min_peak    : float = 0.04
    lift_terminate_drop_hold_steps  : int   = 3

    def env(self) -> dict[str, Any]:
        return {
            "TOPDOWN_LIFT_SUCCESS_HEIGHT": self.lift_success_height,
            "TOPDOWN_LIFT_SUCCESS_HOLD_STEPS": self.lift_success_hold_steps,
            "TOPDOWN_LIFT_SUCCESS_MODE": self.lift_success_mode,
            "TOPDOWN_LIFT_SUCCESS_REQUIRES_CONTACT": self.lift_success_requires_contact,
            "TOPDOWN_LIFT_SUCCESS_CONTACT_MODE": self.lift_success_contact_mode,
            "TOPDOWN_LIFT_SUCCESS_CONTACT_MIN": self.lift_success_contact_min,
            "TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX": self.lift_success_xy_drift_max,
            "TOPDOWN_LIFT_SUCCESS_BLOCK_TILT_MAX_DEG": self.lift_success_block_tilt_max_deg,
            "CURRICULUM_BLOCK_DRIFT_THRESHOLD": self.block_drift_threshold,
            "CURRICULUM_CONTACT_BLOCK_DISP_MAX": self.contact_block_disp_max,
            "TOPDOWN_LIFT_TERMINATE_DROP_FROM_MAX": self.lift_terminate_drop_from_max,
            "TOPDOWN_LIFT_TERMINATE_DROP_MIN_PEAK": self.lift_terminate_drop_min_peak,
            "TOPDOWN_LIFT_TERMINATE_DROP_HOLD_STEPS": self.lift_terminate_drop_hold_steps,
        }


@dataclass(frozen=True)
class SupportOptimizationInputs:
    """Critic support and action-gating inputs from the full-reward launcher."""

    fasttd3_num_atoms        : int   = 201
    fasttd3_v_min            : float = -100.0
    fasttd3_v_max            : float = 100.0
    reward_normalization     : int   = 0
    actor_q_action_gate_mode : str   = "raw"
    actor_bc_action_gate_mode: str   = "raw"
    actor_lr                 : float = 5e-4
    critic_lr                : float = 1e-4
    rl_actor_lr              : float = 5e-4
    rl_critic_lr             : float = 1e-4
    updates_per_step         : int   = 8
    rl_updates_per_step      : int   = 20
    n_step                   : int   = 1
    rl_n_step                : int   = 3
    policy_delay             : int   = 4
    rl_policy_delay          : int   = 4
    tau                      : float = 0.005
    rl_tau                   : float = 0.005
    target_q_clip            : float = 100.0
    rl_target_q_clip         : float = 100.0
    critic_grad_clip         : float = 5.0
    rl_actor_freeze_steps    : int   = 0
    exploration_noise        : float = 0.02
    exploration_noise_finger : float = 0.03
    policy_noise             : float = 0.01
    policy_noise_finger      : float = 0.02
    noise_clip               : float = 0.05

    def env(self) -> dict[str, Any]:
        return {
            "FASTTD3_NUM_ATOMS": self.fasttd3_num_atoms,
            "FASTTD3_V_MIN": self.fasttd3_v_min,
            "FASTTD3_V_MAX": self.fasttd3_v_max,
            "REWARD_NORMALIZATION": self.reward_normalization,
            "ACTOR_Q_ACTION_GATE_MODE": self.actor_q_action_gate_mode,
            "ACTOR_BC_ACTION_GATE_MODE": self.actor_bc_action_gate_mode,
        }


@dataclass(frozen=True)
class TeacherBcInputs:
    """Teacher relabeling and BC weights."""

    bc_only_weight          : float = 10.0
    bc_only_arm_weight      : float = 10.0
    bc_only_finger_weight   : float = 4.0
    teacher_bc_weight       : float = 0.0
    teacher_bc_arm_weight   : float = 10.0
    teacher_bc_finger_weight: float = 4.0
    teacher_bc_decay_steps  : int   = 0
    policy_bc_relabel       : int   = 1
    rl_teacher_bc_weight       : float = 0.0
    rl_teacher_bc_arm_weight   : float = -1.0
    rl_teacher_bc_finger_weight: float = -1.0
    rl_teacher_bc_decay_steps  : int   = 0
    rl_policy_bc_relabel       : int   = 0


@dataclass(frozen=True)
class PipelineInputs:
    """All configurable inputs for the fresh adaptive-BC pipeline run."""

    run_dir                               : str   = "runs_training/pipeline_fullreward_adaptive_bc_r1"
    resume_from                           : str   = ""
    resume_replay                         : bool  = True
    resume_global_step                    : bool  = True
    immediate_adaptive                    : bool  = False
    isaac_python                          : str   = ISAAC_PYTHON
    base_profile                          : str   = BASE_PROFILE
    num_envs                              : int   = 300
    total_steps                           : int   = 50_000_000
    teacher_only_steps                    : int   = 150_000
    dagger_steps                          : int   = 500_000
    start_steps                           : int   = 150_000
    bc_only_steps                         : int   = 650_000
    rl_phase_start_steps                  : int   = 650_000
    replay_size                           : int   = 1_000_000
    batch_size                            : int   = 384
    assist_mix                            : float = 1.0
    assist_floor                          : float = 0.0
    assist_decay_steps                    : int   = 0
    adaptive_assist_start_steps           : int   = 650_000
    adaptive_assist_window_steps          : int   = 25_000
    adaptive_assist_bad_window_steps      : int   = 5_000
    adaptive_assist_arm_error             : float = 0.030
    adaptive_assist_finger_error          : float = 0.050
    adaptive_assist_bad_arm_error         : float = 0.060
    adaptive_assist_bad_finger_error      : float = 0.100
    adaptive_assist_step                   : float = 0.005
    adaptive_assist_recover_step           : float = 0.010
    adaptive_assist_metric_gate            : int   = 1
    adaptive_assist_strict_contact_gate    : int   = 1
    adaptive_assist_strict_contact_min_rate: float = 0.05
    adaptive_assist_metric_window_episodes : int   = 100
    adaptive_assist_baseline_episodes      : int   = 100
    adaptive_assist_metric_min_ratio       : float = 0.70
    adaptive_assist_sync_bc_weights        : int   = 0
    adaptive_assist_post_floor_steps       : int   = 1_000_000
    adaptive_assist_disable_bc_after_floor : int   = 1
    adaptive_assist_bc_only_until_decay    : int   = 0
    stop_on_adaptive_assist_floor          : bool  = True
    assist_noise_start_steps               : int   = 150_000
    assist_noise_arm                       : float = 0.01
    assist_noise_finger                    : float = 0.02
    assist_noise_clean_bc_target          : int   = 1
    checkpoint_every                      : int   = 250_000
    rolling_checkpoint_every              : int   = 250_000
    rolling_checkpoint_keep               : int   = 24
    save_replay_in_checkpoint             : bool  = True
    eval_every                            : int   = 500_000
    eval_steps                            : int   = 1000
    eval_episodes                         : int   = 1
    eval_teacher_assist_mix               : float = 0.0
    tensorboard_dir                       : str   = ""
    log_every                             : int   = 10000
    dry_run                               : bool  = False
    reward                                : FullRewardInputs          = field(default_factory = FullRewardInputs)
    success                               : SuccessSafetyInputs       = field(default_factory = SuccessSafetyInputs)
    support                               : SupportOptimizationInputs = field(default_factory = SupportOptimizationInputs)
    teacher_bc                            : TeacherBcInputs           = field(default_factory = TeacherBcInputs)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _clean_env(inputs: PipelineInputs) -> None:
    os.environ.setdefault("ENPM690_PYTHON", inputs.isaac_python)
    os.environ.pop("FASTTD3_REPO", None)
    os.environ.pop("VIRTUAL_ENV", None)
    os.environ.pop("PYTHONHOME", None)
    os.environ.pop("PYTHONPATH", None)
    for key in list(os.environ):
        if key.startswith("CONDA_"):
            os.environ.pop(key, None)
        elif key in TRAINING_OVERRIDE_KEYS or key.startswith(TRAINING_OVERRIDE_PREFIXES):
            os.environ.pop(key, None)


def _replace_arg(command: list[str], flag: str, value: Any) -> None:
    token = str(value)
    if flag in command:
        idx = command.index(flag)
        if idx + 1 < len(command) and not command[idx + 1].startswith("--"):
            command[idx + 1] = token
        else:
            command.insert(idx + 1, token)
        return
    command.extend([flag, token])


def _remove_flag(command: list[str], flag: str, *, takes_value: bool = False) -> None:
    while flag in command:
        idx = command.index(flag)
        del command[idx]
        if takes_value and idx < len(command) and not command[idx].startswith("--"):
            del command[idx]


def _add_flag(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def _set(command: list[str], env: dict[str, str], flag: str, env_key: str, value: Any) -> None:
    env[env_key] = str(value)
    _replace_arg(command, flag, value)


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineInputs()
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", default=defaults.run_dir)
    parser.add_argument("--resume-from", default=defaults.resume_from)
    parser.add_argument("--no-resume-replay", action="store_true")
    parser.add_argument("--no-resume-global-step", action="store_true")
    parser.add_argument("--immediate-adaptive", action="store_true")
    parser.add_argument("--num-envs", type=_positive_int, default=defaults.num_envs)
    parser.add_argument("--total-steps", type=_positive_int, default=defaults.total_steps)
    parser.add_argument("--teacher-only-steps", type=_nonnegative_int, default=defaults.teacher_only_steps)
    parser.add_argument("--dagger-steps", type=_nonnegative_int, default=defaults.dagger_steps)
    parser.add_argument("--post-floor-rl-steps", type=_nonnegative_int, default=defaults.adaptive_assist_post_floor_steps)
    parser.add_argument("--assist-floor", type=float, default=defaults.assist_floor)
    parser.add_argument("--assist-noise-arm", type=float, default=defaults.assist_noise_arm)
    parser.add_argument("--assist-noise-finger", type=float, default=defaults.assist_noise_finger)
    parser.add_argument("--eval-every", type=_nonnegative_int, default=defaults.eval_every)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=defaults.checkpoint_every)
    parser.add_argument("--tensorboard-dir", default=defaults.tensorboard_dir)
    parser.add_argument("--log-every", type=_positive_int, default=defaults.log_every)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _inputs_from_args(args: argparse.Namespace) -> PipelineInputs:
    defaults = PipelineInputs()
    resume_from = str(args.resume_from)
    if resume_from and not Path(resume_from).exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
    teacher_only_steps = 0 if args.immediate_adaptive else args.teacher_only_steps
    dagger_steps = 0 if args.immediate_adaptive else args.dagger_steps
    dagger_end_step = teacher_only_steps + dagger_steps
    eval_steps = 0 if args.eval_every == 0 else defaults.eval_steps
    eval_episodes = 0 if args.eval_every == 0 else defaults.eval_episodes
    return replace(
        defaults,
        run_dir                          = args.run_dir,
        resume_from                      = resume_from,
        resume_replay                    = not args.no_resume_replay,
        resume_global_step               = not args.no_resume_global_step,
        immediate_adaptive               = args.immediate_adaptive,
        num_envs                         = args.num_envs,
        total_steps                      = args.total_steps,
        teacher_only_steps               = teacher_only_steps,
        dagger_steps                     = dagger_steps,
        start_steps                      = teacher_only_steps,
        bc_only_steps                    = dagger_end_step,
        rl_phase_start_steps             = dagger_end_step,
        adaptive_assist_start_steps      = dagger_end_step,
        assist_noise_start_steps         = teacher_only_steps,
        assist_noise_arm                 = args.assist_noise_arm,
        assist_noise_finger              = args.assist_noise_finger,
        assist_floor                     = args.assist_floor,
        adaptive_assist_post_floor_steps = args.post_floor_rl_steps,
        eval_every                       = args.eval_every,
        eval_steps                       = eval_steps,
        eval_episodes                    = eval_episodes,
        tensorboard_dir                  = args.tensorboard_dir,
        checkpoint_every                 = args.checkpoint_every,
        rolling_checkpoint_every         = args.checkpoint_every,
        log_every                        = args.log_every,
        dry_run = args.dry_run,
    )


def _apply_pipeline_contract(plan, inputs: PipelineInputs):
    run_dir      = inputs.run_dir
    command      = list(plan.command)
    env          = dict(plan.env)
    support      = inputs.support
    teacher_bc   = inputs.teacher_bc
    latest       = f"{run_dir}/latest.pt"
    log_jsonl    = f"{run_dir}/log.jsonl"
    final_replay = f"{run_dir}/final_replay.pt"
    tensorboard_dir = inputs.tensorboard_dir or f"{run_dir}/tb"

    _remove_flag(command, "--resume-replay")
    _remove_flag(command, "--resume-global-step")
    _remove_flag(command, "--actor-init-checkpoint", takes_value=True)
    _remove_flag(command, "--phase1-checkpoint", takes_value=True)
    env.pop("ACTOR_INIT_CKPT", None)
    env.pop("PHASE1_CKPT", None)
    if inputs.resume_from:
        _replace_arg(command, "--resume-checkpoint", inputs.resume_from)
        if inputs.resume_replay:
            _add_flag(command, "--resume-replay")
        if inputs.resume_global_step:
            _add_flag(command, "--resume-global-step")
        env["RESUME_CKPT"] = inputs.resume_from
        env["RESUME_CHECKPOINT"] = inputs.resume_from
        env["RESUME_REPLAY"] = "1" if inputs.resume_replay else "0"
        env["RESUME_GLOBAL_STEP"] = "1" if inputs.resume_global_step else "0"
    else:
        _remove_flag(command, "--resume-checkpoint", takes_value=True)
        env.pop("RESUME_CKPT", None)
        env.pop("RESUME_CHECKPOINT", None)
        env.pop("RESUME_REPLAY", None)
        env.pop("RESUME_GLOBAL_STEP", None)

    _set(command, env, "--num-envs", "NUM_ENVS", inputs.num_envs)
    _set(command, env, "--total-steps", "TOTAL_STEPS", inputs.total_steps)
    _set(command, env, "--start-steps", "START_STEPS", inputs.start_steps)
    _set(command, env, "--bc-only-steps", "BC_ONLY_STEPS", inputs.bc_only_steps)
    _set(command, env, "--rl-phase-start-steps", "RL_PHASE_START_STEPS", inputs.rl_phase_start_steps)
    _set(command, env, "--replay-size", "REPLAY_SIZE", inputs.replay_size)
    _set(command, env, "--batch-size", "BATCH_SIZE", inputs.batch_size)
    _set(command, env, "--checkpoint-every", "CHECKPOINT_EVERY", inputs.checkpoint_every)
    _set(command, env, "--rolling-checkpoint-every", "ROLLING_CHECKPOINT_EVERY", inputs.rolling_checkpoint_every)
    _set(command, env, "--rolling-checkpoint-keep", "ROLLING_CHECKPOINT_KEEP", inputs.rolling_checkpoint_keep)
    _set(command, env, "--eval-every", "EVAL_EVERY", inputs.eval_every)
    _set(command, env, "--eval-steps", "EVAL_STEPS", inputs.eval_steps)
    _set(command, env, "--eval-episodes", "EVAL_EPISODES", inputs.eval_episodes)
    _set(command, env, "--log-every", "LOG_EVERY", inputs.log_every)
    _set(command, env, "--checkpoint-path", "CHECKPOINT_PATH", latest)
    _replace_arg(command, "--log-jsonl", log_jsonl)
    _replace_arg(command, "--final-handoff-checkpoint-path", final_replay)
    _replace_arg(command, "--tensorboard-dir", tensorboard_dir)
    env["RUN_DIR"] = run_dir
    env["LOG_JSONL"] = log_jsonl
    env["FINAL_HANDOFF_CHECKPOINT_PATH"] = final_replay
    env["TENSORBOARD_DIR"] = tensorboard_dir

    for key, value in support.env().items():
        env[key] = str(value)
    _replace_arg(command, "--fasttd3-num-atoms", support.fasttd3_num_atoms)
    _replace_arg(command, "--fasttd3-v-min", support.fasttd3_v_min)
    _replace_arg(command, "--fasttd3-v-max", support.fasttd3_v_max)
    _replace_arg(command, "--actor-q-action-gate-mode", support.actor_q_action_gate_mode)
    _replace_arg(command, "--actor-bc-action-gate-mode", support.actor_bc_action_gate_mode)
    _replace_arg(command, "--actor-lr", support.actor_lr)
    _replace_arg(command, "--critic-lr", support.critic_lr)
    _replace_arg(command, "--rl-actor-lr", support.rl_actor_lr)
    _replace_arg(command, "--rl-critic-lr", support.rl_critic_lr)
    _replace_arg(command, "--updates-per-step", support.updates_per_step)
    _replace_arg(command, "--rl-updates-per-step", support.rl_updates_per_step)
    _replace_arg(command, "--n-step", support.n_step)
    _replace_arg(command, "--rl-n-step", support.rl_n_step)
    _replace_arg(command, "--policy-delay", support.policy_delay)
    _replace_arg(command, "--rl-policy-delay", support.rl_policy_delay)
    _replace_arg(command, "--rl-sync-targets-on-switch", 1)
    _replace_arg(command, "--rl-reset-critic-optimizers-on-switch", 1)
    _replace_arg(command, "--tau", support.tau)
    _replace_arg(command, "--rl-tau", support.rl_tau)
    _replace_arg(command, "--target-q-clip", support.target_q_clip)
    _replace_arg(command, "--rl-target-q-clip", support.rl_target_q_clip)
    _replace_arg(command, "--critic-grad-clip", support.critic_grad_clip)
    _replace_arg(command, "--rl-actor-freeze-steps", support.rl_actor_freeze_steps)
    _replace_arg(command, "--exploration-noise", support.exploration_noise)
    _replace_arg(command, "--exploration-noise-finger", support.exploration_noise_finger)
    _replace_arg(command, "--rl-exploration-noise", support.exploration_noise)
    _replace_arg(command, "--rl-exploration-noise-finger", support.exploration_noise_finger)
    _replace_arg(command, "--policy-noise", support.policy_noise)
    _replace_arg(command, "--policy-noise-finger", support.policy_noise_finger)
    _replace_arg(command, "--rl-policy-noise", support.policy_noise)
    _replace_arg(command, "--rl-policy-noise-finger", support.policy_noise_finger)
    _replace_arg(command, "--noise-clip", support.noise_clip)
    _replace_arg(command, "--rl-noise-clip", support.noise_clip)
    env.update(
        {
            "ACTOR_LR": str(support.actor_lr),
            "CRITIC_LR": str(support.critic_lr),
            "RL_ACTOR_LR": str(support.rl_actor_lr),
            "RL_CRITIC_LR": str(support.rl_critic_lr),
            "UPDATES_PER_STEP": str(support.updates_per_step),
            "RL_UPDATES_PER_STEP": str(support.rl_updates_per_step),
            "N_STEP": str(support.n_step),
            "RL_N_STEP": str(support.rl_n_step),
            "POLICY_DELAY": str(support.policy_delay),
            "RL_POLICY_DELAY": str(support.rl_policy_delay),
            "TAU": str(support.tau),
            "RL_TAU": str(support.rl_tau),
            "TARGET_Q_CLIP": str(support.target_q_clip),
            "RL_TARGET_Q_CLIP": str(support.rl_target_q_clip),
            "CRITIC_GRAD_CLIP": str(support.critic_grad_clip),
            "RL_ACTOR_FREEZE_STEPS": str(support.rl_actor_freeze_steps),
            "EXPLORATION_NOISE": str(support.exploration_noise),
            "EXPLORATION_NOISE_FINGER": str(support.exploration_noise_finger),
            "RL_EXPLORATION_NOISE": str(support.exploration_noise),
            "RL_EXPLORATION_NOISE_FINGER": str(support.exploration_noise_finger),
            "POLICY_NOISE": str(support.policy_noise),
            "POLICY_NOISE_FINGER": str(support.policy_noise_finger),
            "RL_POLICY_NOISE": str(support.policy_noise),
            "RL_POLICY_NOISE_FINGER": str(support.policy_noise_finger),
            "NOISE_CLIP": str(support.noise_clip),
            "RL_NOISE_CLIP": str(support.noise_clip),
        }
    )

    _set(command, env, "--policy-assist-mix", "POLICY_ASSIST_MIX", inputs.assist_mix)
    _set(command, env, "--policy-assist-mix-floor", "POLICY_ASSIST_MIX_FLOOR", inputs.assist_floor)
    _set(command, env, "--policy-assist-decay-start-steps", "POLICY_ASSIST_DECAY_START_STEPS", inputs.rl_phase_start_steps)
    _set(command, env, "--policy-assist-decay-steps", "POLICY_ASSIST_DECAY_STEPS", inputs.assist_decay_steps)
    _set(command, env, "--rl-policy-assist-mix", "RL_POLICY_ASSIST_MIX", inputs.assist_mix)
    _set(command, env, "--rl-policy-assist-mix-floor", "RL_POLICY_ASSIST_MIX_FLOOR", inputs.assist_floor)
    _set(command, env, "--rl-policy-assist-decay-start-steps", "RL_POLICY_ASSIST_DECAY_START_STEPS", inputs.rl_phase_start_steps)
    _set(command, env, "--rl-policy-assist-decay-steps", "RL_POLICY_ASSIST_DECAY_STEPS", inputs.assist_decay_steps)
    _replace_arg(command, "--assist-noise-arm", inputs.assist_noise_arm)
    _replace_arg(command, "--assist-noise-finger", inputs.assist_noise_finger)
    _replace_arg(command, "--assist-noise-start-steps", inputs.assist_noise_start_steps)
    _replace_arg(command, "--assist-noise-clean-bc-target", inputs.assist_noise_clean_bc_target)
    env["ASSIST_NOISE_ARM"] = str(inputs.assist_noise_arm)
    env["ASSIST_NOISE_FINGER"] = str(inputs.assist_noise_finger)
    env["ASSIST_NOISE_START_STEPS"] = str(inputs.assist_noise_start_steps)
    env["ASSIST_NOISE_CLEAN_BC_TARGET"] = str(inputs.assist_noise_clean_bc_target)

    _add_flag(command, "--adaptive-policy-assist")
    env["ADAPTIVE_POLICY_ASSIST"] = "1"
    _replace_arg(command, "--adaptive-assist-start-steps", inputs.adaptive_assist_start_steps)
    _replace_arg(command, "--adaptive-assist-window-steps", inputs.adaptive_assist_window_steps)
    _replace_arg(command, "--adaptive-assist-bad-window-steps", inputs.adaptive_assist_bad_window_steps)
    _replace_arg(command, "--adaptive-assist-arm-error", inputs.adaptive_assist_arm_error)
    _replace_arg(command, "--adaptive-assist-finger-error", inputs.adaptive_assist_finger_error)
    _replace_arg(command, "--adaptive-assist-bad-arm-error", inputs.adaptive_assist_bad_arm_error)
    _replace_arg(command, "--adaptive-assist-bad-finger-error", inputs.adaptive_assist_bad_finger_error)
    _replace_arg(command, "--adaptive-assist-step", inputs.adaptive_assist_step)
    _replace_arg(command, "--adaptive-assist-recover-step", inputs.adaptive_assist_recover_step)
    _replace_arg(command, "--adaptive-assist-metric-gate", inputs.adaptive_assist_metric_gate)
    _replace_arg(command, "--adaptive-assist-strict-contact-gate", inputs.adaptive_assist_strict_contact_gate)
    _replace_arg(command, "--adaptive-assist-strict-contact-min-rate", inputs.adaptive_assist_strict_contact_min_rate)
    _replace_arg(command, "--adaptive-assist-metric-window-episodes", inputs.adaptive_assist_metric_window_episodes)
    _replace_arg(command, "--adaptive-assist-baseline-episodes", inputs.adaptive_assist_baseline_episodes)
    _replace_arg(command, "--adaptive-assist-metric-min-ratio", inputs.adaptive_assist_metric_min_ratio)
    _replace_arg(command, "--adaptive-assist-sync-bc-weights", inputs.adaptive_assist_sync_bc_weights)
    _replace_arg(command, "--adaptive-assist-post-floor-steps", inputs.adaptive_assist_post_floor_steps)
    _replace_arg(command, "--adaptive-assist-disable-bc-after-floor", inputs.adaptive_assist_disable_bc_after_floor)
    _replace_arg(command, "--adaptive-assist-bc-only-until-decay", inputs.adaptive_assist_bc_only_until_decay)
    env["ADAPTIVE_ASSIST_START_STEPS"] = str(inputs.adaptive_assist_start_steps)
    env["ADAPTIVE_ASSIST_WINDOW_STEPS"] = str(inputs.adaptive_assist_window_steps)
    env["ADAPTIVE_ASSIST_BAD_WINDOW_STEPS"] = str(inputs.adaptive_assist_bad_window_steps)
    env["ADAPTIVE_ASSIST_ARM_ERROR"] = str(inputs.adaptive_assist_arm_error)
    env["ADAPTIVE_ASSIST_FINGER_ERROR"] = str(inputs.adaptive_assist_finger_error)
    env["ADAPTIVE_ASSIST_BAD_ARM_ERROR"] = str(inputs.adaptive_assist_bad_arm_error)
    env["ADAPTIVE_ASSIST_BAD_FINGER_ERROR"] = str(inputs.adaptive_assist_bad_finger_error)
    env["ADAPTIVE_ASSIST_STEP"] = str(inputs.adaptive_assist_step)
    env["ADAPTIVE_ASSIST_RECOVER_STEP"] = str(inputs.adaptive_assist_recover_step)
    env["ADAPTIVE_ASSIST_METRIC_GATE"] = str(inputs.adaptive_assist_metric_gate)
    env["ADAPTIVE_ASSIST_STRICT_CONTACT_GATE"] = str(inputs.adaptive_assist_strict_contact_gate)
    env["ADAPTIVE_ASSIST_STRICT_CONTACT_MIN_RATE"] = str(inputs.adaptive_assist_strict_contact_min_rate)
    env["ADAPTIVE_ASSIST_METRIC_WINDOW_EPISODES"] = str(inputs.adaptive_assist_metric_window_episodes)
    env["ADAPTIVE_ASSIST_BASELINE_EPISODES"] = str(inputs.adaptive_assist_baseline_episodes)
    env["ADAPTIVE_ASSIST_METRIC_MIN_RATIO"] = str(inputs.adaptive_assist_metric_min_ratio)
    env["ADAPTIVE_ASSIST_SYNC_BC_WEIGHTS"] = str(inputs.adaptive_assist_sync_bc_weights)
    env["ADAPTIVE_ASSIST_POST_FLOOR_STEPS"] = str(inputs.adaptive_assist_post_floor_steps)
    env["ADAPTIVE_ASSIST_DISABLE_BC_AFTER_FLOOR"] = str(inputs.adaptive_assist_disable_bc_after_floor)
    env["ADAPTIVE_ASSIST_BC_ONLY_UNTIL_DECAY"] = str(inputs.adaptive_assist_bc_only_until_decay)
    if inputs.stop_on_adaptive_assist_floor:
        _add_flag(command, "--stop-on-adaptive-assist-floor")

    _replace_arg(command, "--bc-only-weight", teacher_bc.bc_only_weight)
    _replace_arg(command, "--bc-only-arm-weight", teacher_bc.bc_only_arm_weight)
    _replace_arg(command, "--bc-only-finger-weight", teacher_bc.bc_only_finger_weight)
    _replace_arg(command, "--teacher-bc-weight", teacher_bc.teacher_bc_weight)
    _replace_arg(command, "--teacher-bc-arm-weight", teacher_bc.teacher_bc_arm_weight)
    _replace_arg(command, "--teacher-bc-finger-weight", teacher_bc.teacher_bc_finger_weight)
    _replace_arg(command, "--teacher-bc-decay-steps", teacher_bc.teacher_bc_decay_steps)
    _replace_arg(command, "--rl-teacher-bc-weight", teacher_bc.rl_teacher_bc_weight)
    _replace_arg(command, "--rl-teacher-bc-arm-weight", teacher_bc.rl_teacher_bc_arm_weight)
    _replace_arg(command, "--rl-teacher-bc-finger-weight", teacher_bc.rl_teacher_bc_finger_weight)
    _replace_arg(command, "--rl-teacher-bc-decay-steps", teacher_bc.rl_teacher_bc_decay_steps)
    _replace_arg(command, "--policy-bc-relabel", teacher_bc.policy_bc_relabel)
    _replace_arg(command, "--rl-policy-bc-relabel", teacher_bc.rl_policy_bc_relabel)
    _replace_arg(command, "--eval-teacher-assist-mix", inputs.eval_teacher_assist_mix)
    env.update(
        {
            "BC_ONLY_WEIGHT": str(teacher_bc.bc_only_weight),
            "BC_ONLY_ARM_WEIGHT": str(teacher_bc.bc_only_arm_weight),
            "BC_ONLY_FINGER_WEIGHT": str(teacher_bc.bc_only_finger_weight),
            "TEACHER_BC_WEIGHT": str(teacher_bc.teacher_bc_weight),
            "TEACHER_BC_ARM_WEIGHT": str(teacher_bc.teacher_bc_arm_weight),
            "TEACHER_BC_FINGER_WEIGHT": str(teacher_bc.teacher_bc_finger_weight),
            "TEACHER_BC_DECAY_STEPS": str(teacher_bc.teacher_bc_decay_steps),
            "RL_TEACHER_BC_WEIGHT": str(teacher_bc.rl_teacher_bc_weight),
            "RL_TEACHER_BC_ARM_WEIGHT": str(teacher_bc.rl_teacher_bc_arm_weight),
            "RL_TEACHER_BC_FINGER_WEIGHT": str(teacher_bc.rl_teacher_bc_finger_weight),
            "RL_TEACHER_BC_DECAY_STEPS": str(teacher_bc.rl_teacher_bc_decay_steps),
            "POLICY_BC_RELABEL": str(teacher_bc.policy_bc_relabel),
            "RL_POLICY_BC_RELABEL": str(teacher_bc.rl_policy_bc_relabel),
        }
    )

    if inputs.save_replay_in_checkpoint:
        _add_flag(command, "--save-replay-in-checkpoint")
        env["SAVE_REPLAY_IN_CHECKPOINT"] = "1"
    else:
        _remove_flag(command, "--save-replay-in-checkpoint")
        env["SAVE_REPLAY_IN_CHECKPOINT"] = "0"

    for key, value in inputs.reward.env().items():
        env[key] = str(value)
    for key, value in inputs.success.env().items():
        env[key] = str(value)

    stdout_log = REPO_ROOT / run_dir / "stdout.log"
    return replace(plan, command=command, env=env, run_dir=run_dir, stdout_log=stdout_log)


def build_plan_for_inputs(inputs: PipelineInputs):
    _clean_env(inputs)
    os.environ["ENPM690_PYTHON"] = inputs.isaac_python
    profile = PROFILES[inputs.base_profile]()
    plan = build_plan(profile, REPO_ROOT)
    return _apply_pipeline_contract(plan, inputs)


def main(argv: list[str] | None = None) -> int:
    inputs = _inputs_from_args(build_parser().parse_args(argv))
    plan = build_plan_for_inputs(inputs)
    manifest_path = plan.stdout_log.parent / "manifest.json"
    if inputs.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_path, manifest_for_plan(plan))
        print(f"dry-run manifest: {manifest_path}")
        print("command:")
        print(" ".join(plan.command))
        return 0
    return run_plan(plan, manifest_path=manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
