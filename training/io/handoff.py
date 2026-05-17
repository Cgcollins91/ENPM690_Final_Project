"""

Replay handoff compatibility helpers with explicit inputs

File map:

HANDOFF_RUNTIME_ARG_EXCLUDES:          Define handoff runtime arg excludes constant
HANDOFF_RUNTIME_ENV_EXCLUDES:          Define handoff runtime env excludes constant
DEFAULT_HANDOFF_SOURCE_FILES:          Define default handoff source files constant
stable_json_compact:                   Serialize compatibility payloads deterministically
handoff_digest:                        Return a stable digest for handoff payloads
file_sha256_if_present:                Hash one source file when it exists
handoff_arg_is_replay_input:           Return whether an argparse key affects generated replay
handoff_env_is_replay_input:           Return whether an environment key affects generated replay
build_handoff_compatibility:           Build the replay handoff compatibility contract
handoff_compatibility_mismatch:        Return mismatch reason or None when compatible
replay_resume_compatibility_mismatch:  Return mismatch reason for loading existing checkpoint replay
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os

from ..actions.action_space import ReducedActionSpec


HANDOFF_RUNTIME_ARG_EXCLUDES = {
    "checkpoint_path",
    "log_jsonl",
    "tensorboard_dir",
    "resume_checkpoint",
    "actor_init_checkpoint",
    "save_replay_in_checkpoint",
    "resume_replay",
    "resume_global_step",
    "force_dagger_after_resume",
    "dagger_resume_policy_assist_mix",
    "dagger_resume_policy_assist_mix_floor",
    "dagger_resume_policy_assist_decay_steps",
    "allow_handoff_source_hash_mismatch",
    "handoff_checkpoint_path",
    "final_handoff_checkpoint_path",
    "stop_after_handoff_checkpoint",
    "reset_optimizers_on_resume",
    "sleep",
    "checkpoint_every",
    "rolling_checkpoint_every",
    "rolling_checkpoint_keep",
    "log_every",
    "eval_steps",
    "eval_episodes",
    "play",
    "play_episodes",
    "viewport_camera",
    "headless",
    "livestream",
    "enable_cameras",
    "xr",
    "device",
    "verbose",
    "info",
    "experience",
    "rendering_mode",
    "kit_args",
    "anim_recording_enabled",
    "anim_recording_start_time",
    "anim_recording_stop_time",
    "debug_eval_trace",
    "detect_anomaly",
    "debug_nonfinite_updates",
    "stop_on_nonfinite_update",
    "total_steps",
}

HANDOFF_RUNTIME_ENV_EXCLUDES = {
    "RUN_DIR",
    "CHECKPOINT_PATH",
    "LOG_JSONL",
    "TENSORBOARD_DIR",
    "RESUME_CKPT",
    "ACTOR_INIT_CKPT",
    "HANDOFF_CHECKPOINT_PATH",
    "FINAL_HANDOFF_CHECKPOINT_PATH",
    "CURRICULUM_ALIGN_FAILFAST_AFTER_SECONDS",
    "CURRICULUM_ALIGN_FAILFAST_THRESHOLD",
    "SAVE_REPLAY_IN_CHECKPOINT",
    "RESUME_REPLAY",
    "RESUME_GLOBAL_STEP",
    "STOP_AFTER_HANDOFF_CHECKPOINT",
    "RESET_OPTIMIZERS_ON_RESUME",
    "ENPM690_PYTHON",
    "PROJECT_ROOT",
    "HEADLESS",
    "ENABLE_CAMERAS",
    "DEVICE",
    "TOTAL_STEPS",
    "EVAL_STEPS",
    "EVAL_EPISODES",
    "CHECKPOINT_EVERY",
    "ROLLING_CHECKPOINT_EVERY",
    "ROLLING_CHECKPOINT_KEEP",
    "LOG_EVERY",
    "SLEEP",
}

DEFAULT_HANDOFF_SOURCE_FILES = (
    "training/__init__.py",
    "training/__main__.py",
    "training/actions/action_assembly.py",
    "training/actions/action_gates.py",
    "training/actions/action_history.py",
    "training/actions/action_mix.py",
    "training/actions/action_space.py",
    "training/core/arg_normalization.py",
    "training/model/agent_factory.py",
    "training/model/agents.py",
    "training/geometry/block_state.py",
    "training/state/cadence.py",
    "training/io/checkpoint_apply.py",
    "training/io/checkpoint_io.py",
    "training/io/checkpoint_schedule.py",
    "training/io/checkpoints.py",
    "training/core/cli.py",
    "training/core/cli_legacy.py",
    "training/core/configs.py",
    "training/geometry/contact_metrics.py",
    "training/teacher/contact_preroll_actions.py",
    "training/state/contact_preroll_state.py",
    "training/teacher/contact_teacher.py",
    "training/core/context.py",
    "training/teacher/curriculum_gates.py",
    "training/logging/debugging.py",
    "training/logging/diagnostics.py",
    "training/logging/drift_diagnostics.py",
    "training/core/env_config.py",
    "training/core/entrypoint.py",
    "training/core/isaac_env.py",
    "training/core/paths.py",
    "training/state/episodes.py",
    "training/logging/episode_logging.py",
    "training/state/episode_metrics.py",
    "training/state/episode_resets.py",
    "training/eval/eval_actions.py",
    "training/eval/eval_checkpoint.py",
    "training/eval/eval_preroll.py",
    "training/eval/eval_episode.py",
    "training/eval/eval_logging.py",
    "training/eval/eval_metrics.py",
    "training/eval/eval_rows.py",
    "training/eval/eval_success_overrides.py",
    "training/eval/eval_summary_rows.py",
    "training/eval/eval_tensors.py",
    "training/io/finalization.py",
    "training/geometry/geometry.py",
    "training/io/handoff.py",
    "training/geometry/inpocket.py",
    "training/env/isaac_backend.py",
    "training/env/isaac_startup.py",
    "training/geometry/ik_masks.py",
    "training/geometry/ik_servo.py",
    "training/logging/jsonl.py",
    "training/geometry/lift_latch.py",
    "training/state/loop_plan.py",
    "training/logging/log_rows.py",
    "training/state/loop_state.py",
    "training/actions/losses.py",
    "training/native/native_actions.py",
    "training/native/native_backend.py",
    "training/native/native_checkpoint_startup.py",
    "training/native/native_components.py",
    "training/native/native_loop.py",
    "training/native/native_orchestrator.py",
    "training/native/native_phase1_startup.py",
    "training/native/native_startup.py",
    "training/model/networks.py",
    "training/model/normalization.py",
    "training/env/observations.py",
    "training/state/phase_overrides.py",
    "training/core/play_mode.py",
    "training/geometry/pocket_sweep.py",
    "training/actions/policy_arm.py",
    "training/eval/post_eval_reset.py",
    "training/state/preroll.py",
    "training/env/privileged.py",
    "training/logging/progress.py",
    "training/logging/progress_lines.py",
    "training/teacher/reach_signals.py",
    "training/state/replay.py",
    "training/state/replay_batches.py",
    "training/io/replay_startup.py",
    "training/env/reset_state.py",
    "training/state/run_state.py",
    "training/core/runner.py",
    "training/core/runtime.py",
    "training/actions/schedules.py",
    "training/state/seeding.py",
    "training/logging/stage_bits.py",
    "training/env/startup.py",
    "training/io/startup_checkpoints.py",
    "training/env/startup_summary.py",
    "training/teacher/teacher_cache.py",
    "training/teacher/teacher_actions.py",
    "training/teacher/teacher_arm_controller.py",
    "training/teacher/teacher_arm_targets.py",
    "training/teacher/teacher_closure.py",
    "training/teacher/teacher_ik_state.py",
    "training/core/task_routing.py",
    "training/geometry/task_space_ik.py",
    "training/logging/tensorboard_logging.py",
    "training/logging/tensorboard_setup.py",
    "training/env/terminal_observations.py",
    "training/env/terminal_patch.py",
    "training/geometry/tip_geometry.py",
    "training/geometry/topdown_geometry.py",
    "training/env/topdown_env_adapters.py",
    "training/geometry/topdown_metrics.py",
    "training/teacher/topdown_preroll_state.py",
    "training/geometry/topdown_summary_rows.py",
    "training/state/transition_collection.py",
    "training/state/update_schedule.py",
    "training/model/upstream_fasttd3.py",
    "training/io/warmstart.py",
    "training/native/native_entrypoint.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/cgc_topdown_curriculum_env_cfg.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/__init__.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/observations.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/rewards.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/state_machine.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/terminations.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/topdown_geometry.py",
    "tasks/g1_tasks/cgc_topdown_curriculum_g1_29dof_dex3/mdp/trainer_helpers.py",
    "tasks/common_config/robot_configs.py",
    "tasks/common_event/event_manager.py",
    "tasks/common_observations/dex3_state.py",
    "tasks/common_observations/g1_29dof_state.py",
    "tasks/utils/importer.py",
    "tasks/utils/parse_cfg.py",
    "robots/unitree.py",
)


def stable_json_compact(data: object) -> str:
    """Serialize compatibility payloads deterministically"""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def handoff_digest(data: object) -> str:
    """Return a stable digest for handoff payloads"""
    return hashlib.sha256(stable_json_compact(data).encode("utf-8")).hexdigest()[:16]


def file_sha256_if_present(project_root: str, relative_path: str) -> str | None:
    """Hash one source file when it exists

    Steps:
    - Resolve inputs for `file_sha256_if_present` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    path = os.path.join(project_root, relative_path)
    if not os.path.isfile(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def handoff_arg_is_replay_input(name: str) -> bool:
    """Return whether an argparse key affects generated replay"""
    if name.startswith("_") or name in HANDOFF_RUNTIME_ARG_EXCLUDES:
        return False
    if name.startswith("rl_") and name != "rl_phase_start_steps":
        return False
    return True


def handoff_env_is_replay_input(name: str) -> bool:
    """Return whether an environment key affects generated replay

    Steps:
    - Resolve inputs for `handoff_env_is_replay_input` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if name in HANDOFF_RUNTIME_ENV_EXCLUDES:
        return False
    if name.startswith("RL_") and name != "RL_PHASE_START_STEPS":
        return False
    prefixes = (
        "TOPDOWN_",
        "CURRICULUM_",
        "BC_",
        "TEACHER_",
        "POLICY_",
        "ASSIST_",
        "CONTACT_",
        "FINGER_",
        "INPOCKET_",
        "ARM_",
        "FASTTD3_",
        "UNITREE_",
    )
    keys = {
        "TASK",
        "TD3_BACKEND",
        "SEED",
        "NUM_ENVS",
        "START_STEPS",
        "BATCH_SIZE",
        "N_STEP",
        "GAMMA",
        "TAU",
        "UPDATES_PER_STEP",
        "POLICY_DELAY",
        "ACTOR_LR",
        "CRITIC_LR",
        "TARGET_Q_CLIP",
        "CRITIC_GRAD_CLIP",
        "EXPLORATION_NOISE",
        "EXPLORATION_NOISE_FINGER",
        "POLICY_NOISE",
        "POLICY_NOISE_FINGER",
        "NOISE_CLIP",
        "OBSERVATION_NORMALIZATION",
        "REWARD_NORMALIZATION",
        "PRIVILEGED_CRITIC",
        "INCLUDE_WRIST_ROLL",
        "INCLUDE_WAIST_YAW",
        "WAIST_YAW_ACTION_SCALE",
        "DISABLE_CAMERA_PERCEPTION",
    }
    return name in keys or name.startswith(prefixes)


def build_handoff_compatibility(
    *,
    project_root        : str,  # Param: root directory for project
    task                : str,  # Param: string input for task
    td3_backend         : str,  # Param: string input for td3 backend
    rl_phase_start_steps: int,  # Param: step count used for rl phase start steps
    obs_schema_version  : int,  # Param: integer input for obs schema version
    obs_keys            : tuple[str, ...],  # Param: ordered mapping keys used to resolve obs
    obs_dim             : int,  # Param: integer input for obs dim
    action_dim          : int,  # Param: integer input for action dim
    priv_obs_dim        : int,  # Param: integer input for priv obs dim
    policy_action_spec  : ReducedActionSpec,  # Param: input value used as policy action spec
    env_action_spec     : ReducedActionSpec,  # Param: input value used as env action spec
    args                : Mapping[str, object],  # Param: argument mapping or namespace read and updated by this helper
    env                 : Mapping[str, str],  # Param: environment or backend object used for runtime calls
    source_files        : Sequence[str] = DEFAULT_HANDOFF_SOURCE_FILES,  # Param: string input for source files
) -> dict[str, object]:
    """Build the replay handoff compatibility contract

    Steps:
    - Resolve inputs for `build_handoff_compatibility` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    args_payload = {
        key: value
        for key, value in sorted(args.items())
        if handoff_arg_is_replay_input(key)
    }
    env_payload = {
        key: value
        for key, value in sorted(env.items())
        if handoff_env_is_replay_input(key)
    }
    source_hashes = {
        path: digest
        for path in source_files
        if (digest := file_sha256_if_present(project_root, path)) is not None
    }
    payload: dict[str, object] = {
        "version"             : 1,
        "obs_schema_version"  : int(obs_schema_version),
        "task"                : task,
        "td3_backend"         : td3_backend,
        "rl_phase_start_steps": int(rl_phase_start_steps),
        "obs_keys"            : tuple(obs_keys),
        "obs_dim"             : int(obs_dim),
        "action_dim"          : int(action_dim),
        "priv_obs_dim"        : int(priv_obs_dim),
        "policy_action_joints": tuple(policy_action_spec.joint_names),
        "policy_action_scales": tuple(float(x) for x in policy_action_spec.scales),
        "env_action_joints"   : tuple(env_action_spec.joint_names),
        "env_action_scales"   : tuple(float(x) for x in env_action_spec.scales),
        "args"                : args_payload,
        "env"                 : env_payload,
        "source_hashes"       : source_hashes,
    }
    payload["digest"] = handoff_digest({k: v for k, v in payload.items() if k != "digest"})
    return payload


def handoff_compatibility_mismatch(
    ckpt   : Mapping[str, object],  # Param: string input for ckpt
    current: Mapping[str, object],  # Param: string input for current
    *,
    ignore_source_hashes: bool = False,  # Param: boolean input controlling ignore source hashes
) -> str | None:
    """Return mismatch reason or None when compatible

    Steps:
    - Resolve inputs for `handoff_compatibility_mismatch` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    saved = ckpt.get("handoff_compatibility")
    if not isinstance(saved, Mapping):
        return "checkpoint has no handoff_compatibility block"
    if saved.get("digest") == current.get("digest"):
        return None
    for section in (
        "task",
        "obs_schema_version",
        "td3_backend",
        "rl_phase_start_steps",
        "obs_keys",
        "obs_dim",
        "action_dim",
        "priv_obs_dim",
        "policy_action_joints",
        "policy_action_scales",
        "env_action_joints",
        "env_action_scales",
    ):
        if saved.get(section) != current.get(section):
            return f"{section} changed"
    for section in ("args", "env", "source_hashes"):
        saved_section = saved.get(section, {})
        current_section = current.get(section, {})
        if saved_section != current_section:
            if section == "source_hashes" and ignore_source_hashes:
                continue
            if isinstance(saved_section, Mapping) and isinstance(current_section, Mapping):
                keys = sorted(set(saved_section) | set(current_section))
                for key in keys:
                    if saved_section.get(key) != current_section.get(key):
                        return f"{section}.{key} changed"
            return f"{section} changed"
    if ignore_source_hashes:
        return None
    return f"digest changed saved={saved.get('digest')} current={current.get('digest')}"


def replay_resume_compatibility_mismatch(
    ckpt   : Mapping[str, object],  # Param: string input for ckpt
    current: Mapping[str, object],  # Param: string input for current
) -> str | None:
    """Return mismatch reason for loading existing checkpoint replay"""
    saved = ckpt.get("handoff_compatibility")
    if not isinstance(saved, Mapping):
        saved = {
            "task"                : ckpt.get("task"),
            "obs_schema_version"  : ckpt.get("obs_schema_version"),
            "td3_backend"         : ckpt.get("td3_backend"),
            "obs_keys"            : ckpt.get("obs_keys", ckpt.get("policy_obs_keys")),
            "obs_dim"             : ckpt.get("obs_dim"),
            "action_dim"          : ckpt.get("policy_action_dim", len(tuple(ckpt.get("policy_action_joints", ())))),
            "priv_obs_dim"        : ckpt.get("priv_obs_dim"),
            "policy_action_joints": ckpt.get("policy_action_joints", ckpt.get("reduced_action_joints")),
            "policy_action_scales": ckpt.get("policy_action_scales", ckpt.get("reduced_action_scales")),
            "env_action_joints"   : ckpt.get("env_action_joints", ckpt.get("env_reduced_action_joints")),
            "env_action_scales"   : ckpt.get("env_action_scales", ckpt.get("env_reduced_action_scales")),
        }
    for section in (
        "task",
        "obs_schema_version",
        "td3_backend",
        "obs_keys",
        "obs_dim",
        "action_dim",
        "priv_obs_dim",
        "policy_action_joints",
        "policy_action_scales",
        "env_action_joints",
        "env_action_scales",
    ):
        if saved.get(section) != current.get(section):
            return f"{section} changed"
    return None
