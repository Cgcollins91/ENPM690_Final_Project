"""

Legacy training argument registration for the import-safe parser



_env_int:                       Retrieve an integer value from the environment, with a fallback default
_env_float:                     Retrieve a float value from the environment, with a fallback default
_has_option:                    Check if the parser already has an option registered (naive check)
_add:                           Add an argument to the parser if it doesn't already exist
_add_typed:                     Add typed arguments to the parser if they don't already exist
_add_store_true:                Add store_true arguments to the parser if they don't already exist
register_legacy_training_args:  Register monolith trainer args that are not part of the core parser
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os

from .runtime import env_flag


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    """
    Retrieve an integer value from the environment, with a fallback default.
    """
    try:
        return int(env.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    """
    Retrieve a float value from the environment, with a fallback default.
    """
    try:
        return float(env.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    """
    Check if the parser already has an option registered (naive check)
    """
    return any(option in action.option_strings for action in parser._actions)


def _add(parser: argparse.ArgumentParser, *flags: str, **kwargs: object) -> None:
    """
    Add an argument to the parser if it doesn't already exist
    """

    if any(_has_option(parser, flag) for flag in flags):
        return
    parser.add_argument(*flags, **kwargs)


def _add_typed(
    parser  : argparse.ArgumentParser,         # Param: input value used as parser
    arg_type: type,                            # Param: input value used as arg type
    items   : tuple[tuple[str, object], ...],  # Param: string input for items
) -> None:
    """
    Add typed arguments to the parser if they don't already exist
    """

    for name, default in items:
        _add(parser, f"--{name}", type=arg_type, default=default)


def _add_store_true(
    parser: argparse.ArgumentParser,  # Param: input value used as parser
    items : tuple[str, ...],          # Param: string input for items
    *,
    default: bool = False,            # Param: fallback value used when the input omits or rejects a setting
) -> None:
    """
    Add store_true arguments to the parser if they don't already exist
    """
    for name in items:
        _add(parser, f"--{name}", action="store_true", default=default)


def register_legacy_training_args(
    parser: argparse.ArgumentParser,  # Param: input value used as parser
    *,
    project_root: str,  # Param: root directory for project
    env         : Mapping[str, str],  # Param: environment or backend object used for runtime calls
) -> None:
    """Register monolith trainer args that are not part of the core parser

    Steps:
    - Resolve inputs for `register_legacy_training_args` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    _add_typed(
        parser,
        int,
        (
            ("hidden-dim", 256),
            ("replay-size", 200000),
            ("actor-freeze-steps", 0),
            ("critic-burn-in-steps", -1),
            ("rl-updates-per-step", -1),
            ("rl-n-step", -1),
            ("rl-policy-delay", -1),
            ("rl-policy-assist-decay-steps", -1),
            ("rl-policy-assist-decay-start-steps", -1),
            ("rl-teacher-bc-decay-steps", -1),
            ("rl-actor-freeze-steps", 0),
            ("policy-assist-decay-steps", 3000),
            ("policy-assist-decay-start-steps", _env_int(env, "POLICY_ASSIST_DECAY_START_STEPS", -1)),
            ("policy-assist-arm-decay-steps", -1),
            ("policy-assist-finger-decay-steps", -1),
            ("teacher-bc-decay-steps", 30000),
            ("bc-only-steps", 0),
            ("assist-noise-start-steps", _env_int(env, "ASSIST_NOISE_START_STEPS", 0)),
            ("finger-curl-start", 24),
            ("finger-curl-duration", 24),
            ("contact-preroll-max-steps", 500),
            ("contact-preroll-eval-retries", 0),
            ("contact-handoff-action-smooth-steps", 0),
            ("topdown-preroll-max-steps", 700),
            ("log-every", 100),
            ("checkpoint-every", 2500),
            ("rolling-checkpoint-every", 5000),
            ("rolling-checkpoint-keep", 20),
            ("eval-every", _env_int(env, "EVAL_EVERY", 0)),
            ("eval-steps", 500),
            ("eval-episodes", 20),
            ("eval-start-steps", 0),
            ("play-episodes", 5),
            ("fasttd3-num-atoms", _env_int(env, "FASTTD3_NUM_ATOMS", 51)),
            ("fasttd3-critic-hidden-dim", _env_int(env, "FASTTD3_CRITIC_HIDDEN_DIM", 1024)),
            ("fasttd3-actor-hidden-dim", _env_int(env, "FASTTD3_ACTOR_HIDDEN_DIM", 512)),
            ("dagger-resume-policy-assist-decay-steps", -1),
            ("adaptive-assist-start-steps", -1),
            ("adaptive-assist-window-steps", 500),
            ("adaptive-assist-bad-window-steps", 50),
            ("adaptive-assist-sync-bc-weights", _env_int(env, "ADAPTIVE_ASSIST_SYNC_BC_WEIGHTS", 1)),
            ("adaptive-assist-post-floor-steps", _env_int(env, "ADAPTIVE_ASSIST_POST_FLOOR_STEPS", 0)),
            ("adaptive-assist-disable-bc-after-floor", _env_int(env, "ADAPTIVE_ASSIST_DISABLE_BC_AFTER_FLOOR", 0)),
            ("adaptive-assist-bc-only-until-decay", _env_int(env, "ADAPTIVE_ASSIST_BC_ONLY_UNTIL_DECAY", 0)),
            ("adaptive-assist-metric-gate", _env_int(env, "ADAPTIVE_ASSIST_METRIC_GATE", 0)),
            ("adaptive-assist-strict-contact-gate", _env_int(env, "ADAPTIVE_ASSIST_STRICT_CONTACT_GATE", 0)),
            ("adaptive-assist-metric-window-episodes", _env_int(env, "ADAPTIVE_ASSIST_METRIC_WINDOW_EPISODES", 100)),
            ("adaptive-assist-baseline-episodes", _env_int(env, "ADAPTIVE_ASSIST_BASELINE_EPISODES", 100)),
        ),
    )
    _add_typed(
        parser,
        float,
        (
            ("policy-noise", 0.2),
            ("noise-clip", 0.5),
            ("exploration-noise", 0.25),
            ("exploration-noise-finger", 0.0),
            ("policy-noise-finger", 0.0),
            ("rl-gamma", -1.0),
            ("rl-tau", -1.0),
            ("rl-actor-lr", -1.0),
            ("rl-critic-lr", -1.0),
            ("rl-target-q-clip", -1.0),
            ("rl-critic-grad-clip", -1.0),
            ("rl-actor-pre-tanh-l2", -1.0),
            ("rl-exploration-noise", -1.0),
            ("rl-exploration-noise-finger", -1.0),
            ("rl-policy-noise", -1.0),
            ("rl-policy-noise-finger", -1.0),
            ("rl-noise-clip", -1.0),
            ("rl-policy-assist-mix", -1.0),
            ("rl-policy-assist-mix-floor", -1.0),
            ("rl-teacher-bc-weight", -1.0),
            ("rl-teacher-bc-arm-weight", -2.0),
            ("rl-teacher-bc-finger-weight", -2.0),
            ("policy-assist-mix", 0.9),
            ("policy-assist-mix-floor", 0.0),
            ("policy-assist-arm-mix", -1.0),
            ("policy-assist-arm-mix-floor", -1.0),
            ("policy-assist-finger-mix", -1.0),
            ("policy-assist-finger-mix-floor", -1.0),
            ("teacher-bc-weight", 0.5),
            ("teacher-bc-arm-weight", -1.0),
            ("teacher-bc-finger-weight", -1.0),
            ("bc-only-weight", 1.0),
            ("bc-only-arm-weight", -1.0),
            ("bc-only-finger-weight", -1.0),
            ("warmup-teacher-noise", 0.03),
            ("assist-noise-arm", 0.0),
            ("assist-noise-finger", 0.0),
            ("ik-target-z-offset", 0.0),
            ("ik-damping", 0.05),
            ("ik-max-joint-step", 0.05),
            ("ik-approach-standoff", 0.05),
            ("contact-finger-close-cap", 0.7),
            ("topdown-contact-teacher-contact-threshold", 0.08),
            ("topdown-contact-teacher-close-rate", 0.025),
            ("topdown-contact-teacher-start-fraction", 0.08),
            ("topdown-contact-teacher-max-fraction", 0.7),
            ("topdown-contact-teacher-middle-scale", 0.0),
            ("topdown-contact-teacher-descent-z", 0.045),
            ("topdown-contact-teacher-missing-contact-extra-descent", 0.01),
            ("topdown-contact-teacher-inward-m", 0.035),
            ("topdown-contact-teacher-missing-contact-extra-inward", 0.02),
            ("topdown-contact-teacher-tip-servo-gain", 0.65),
            ("topdown-contact-teacher-tip-servo-max-m", 0.08),
            ("topdown-contact-teacher-policy-arm-ik-mix", 0.0),
            ("topdown-contact-teacher-policy-arm-ik-gate-palm-max", 0.09),
            ("topdown-contact-teacher-policy-arm-ik-gate-height-max", 0.04),
            ("topdown-contact-teacher-policy-arm-ik-gate-align-max", 0.2),
            ("topdown-contact-teacher-policy-arm-ik-gate-opp-min", 0.5),
            ("topdown-contact-teacher-missing-thumb-x-nudge", 0.0),
            ("topdown-contact-teacher-missing-thumb-y-nudge", 0.0),
            ("topdown-contact-teacher-missing-index-x-nudge", 0.0),
            ("topdown-contact-teacher-missing-index-y-nudge", 0.0),
            ("palm-xy-bias-x", 0.0),
            ("palm-xy-bias-y", 0.0),
            ("contact-preroll-palm-tolerance", 0.09),
            ("contact-preroll-height-tolerance", 0.05),
            ("contact-preroll-orient-deg", 25.0),
            ("contact-preroll-unlock-gate", 0.35),
            ("contact-preroll-align-face-tolerance", 0.0),
            ("contact-preroll-ik-descend-z", 0.04),
            ("contact-handoff-action-max-delta", 0.0),
            ("topdown-preroll-fraction", 0.0),
            ("topdown-preroll-unlock-progress", 1.0),
            ("obs-norm-eps", 1e-4),
            ("obs-norm-clip", 10.0),
            ("reward-norm-eps", 1e-4),
            ("reward-norm-clip", 10.0),
            ("actor-pre-tanh-l2", 0.0),
            ("target-q-clip", 0.0),
            ("critic-grad-clip", 0.0),
            ("teacher-lift-z", 0.08),
            ("contact-teacher-hold-fraction-cap", 0.4),
            ("teacher-lift-ramp-steps", 60.0),
            ("dagger-resume-policy-assist-mix", -1.0),
            ("dagger-resume-policy-assist-mix-floor", -1.0),
            ("sleep", 0.0),
            ("eval-teacher-assist-mix", 0.0),
            ("fasttd3-v-min", _env_float(env, "FASTTD3_V_MIN", -5.0)),
            ("fasttd3-v-max", _env_float(env, "FASTTD3_V_MAX", 0.0)),
            ("fasttd3-init-scale", _env_float(env, "FASTTD3_INIT_SCALE", 0.01)),
            ("fasttd3-weight-decay", _env_float(env, "FASTTD3_WEIGHT_DECAY", 0.0)),
            ("fasttd3-std-min", _env_float(env, "FASTTD3_STD_MIN", 0.001)),
            ("fasttd3-std-max", _env_float(env, "FASTTD3_STD_MAX", 0.4)),
            ("adaptive-assist-arm-error", 0.03),
            ("adaptive-assist-finger-error", 0.05),
            ("adaptive-assist-bad-arm-error", 0.06),
            ("adaptive-assist-bad-finger-error", 0.10),
            ("adaptive-assist-step", 0.005),
            ("adaptive-assist-recover-step", 0.010),
            ("adaptive-assist-metric-min-ratio", _env_float(env, "ADAPTIVE_ASSIST_METRIC_MIN_RATIO", 0.70)),
            ("adaptive-assist-strict-contact-min-rate", _env_float(env, "ADAPTIVE_ASSIST_STRICT_CONTACT_MIN_RATE", 0.05)),
        ),
    )
    _add_store_true(
        parser,
        (
            "finger-noise-bypass-unlock",
            "disable-contact-finger-unlock-gate",
            "topdown-contact-teacher",
            "topdown-contact-teacher-bypass-unlock",
            "topdown-contact-teacher-policy-arm-ik-one-contact-only",
            "topdown-contact-teacher-policy-arm-ik-tight-gate",
            "disable-camera-perception",
            "soft-policy-arm-assist",
            "debug-contact-preroll",
            "observation-normalization",
            "reward-normalization",
            "detect-anomaly",
            "debug-nonfinite-updates",
            "stop-on-nonfinite-update",
            "phase1-teacher-only",
            "reset-optimizers-on-resume",
            "play",
            "play-skip-checkpoint",
            "debug-eval-trace",
            "adaptive-policy-assist",
            "stop-on-adaptive-assist-floor",
        ),
    )
    _add(parser, "--topdown-mirror-middle-to-index", action="store_true", default=env_flag("TOPDOWN_MIRROR_MIDDLE_TO_INDEX", False, env))
    _add(parser, "--save-replay-in-checkpoint",      action="store_true", default=env_flag("SAVE_REPLAY_IN_CHECKPOINT", False, env))
    _add(parser, "--resume-replay",                  action="store_true", default=env_flag("RESUME_REPLAY", False, env))
    _add(parser, "--resume-global-step",             action="store_true", default=env_flag("RESUME_GLOBAL_STEP", False, env))
    _add(parser, "--force-dagger-after-resume",      action="store_true", default=env_flag("FORCE_DAGGER_AFTER_RESUME", False, env))
    _add(parser, "--reset-obs-stats-on-resume",      type=int, default=int(env_flag("RESET_OBS_STATS_ON_RESUME", True, env)))
    _add(
        parser,
        "--allow-handoff-source-hash-mismatch",
        action="store_true",
        default=env_flag("ALLOW_HANDOFF_SOURCE_HASH_MISMATCH", False, env),
    )
    _add(parser, "--stop-after-handoff-checkpoint",         action="store_true", default=env_flag("STOP_AFTER_HANDOFF_CHECKPOINT", False, env))
    _add(parser, "--rl-policy-bc-relabel",                  type=int, default=-1, choices=(-1, 0, 1))
    _add(parser, "--rl-sync-targets-on-switch",             type=int, default=1, choices=(0, 1))
    _add(parser, "--rl-reset-critic-optimizers-on-switch",  type=int, default=1, choices=(0, 1))
    _add(parser, "--assist-noise-clean-bc-target",          type=int, default=1, choices=(0, 1))
    _add(parser, "--policy-bc-relabel",                     type=int, default=0, choices=(0, 1))
    _add(parser, "--fasttd3-use-cdq",                       type=int, default=_env_int(env, "FASTTD3_USE_CDQ", 1), choices=(0, 1))
    _add(
        parser,
        "--actor-q-action-gate-mode",
        choices=("env", "raw", "straight_through"),
        default=env.get("ACTOR_Q_ACTION_GATE_MODE", "env"),
    )
    _add(
        parser,
        "--actor-bc-action-gate-mode",
        choices=("env", "raw", "straight_through"),
        default=env.get("ACTOR_BC_ACTION_GATE_MODE", "env"),
    )
    _add(parser, "--teacher-arm-source",               type=str, default="ik",       choices=("ik", "policy"))
    _add(parser, "--finger-curl-mode",                 type=str, default="distance", choices=("time", "distance"))
    _add(parser, "--contact-start-mode",               type=str, default="reset",    choices=("reset", "phase1_terminal"))
    _add(parser, "--contact-preroll-touch-mode",       type=str, default="off",      choices=("off", "any", "both", "strict"))
    _add(parser, "--contact-preroll-touch-arm-source", type=str, default="policy",   choices=("policy", "ik"))
    _add(
        parser,
        "--viewport-camera",
        type=str,
        default="overview",
        choices=("front_camera", "left_wrist_camera", "right_wrist_camera", "world", "overview", "table_overhead", "top"),
    )
    _add(parser, "--phase1-checkpoint",             type=str, default="")
    _add(parser, "--resume-checkpoint",             type=str, default="")
    _add(parser, "--actor-init-checkpoint",         type=str, default="")
    _add(parser, "--handoff-checkpoint-path",       type=str, default=env.get("HANDOFF_CHECKPOINT_PATH", ""))
    _add(parser, "--final-handoff-checkpoint-path", type=str, default=env.get("FINAL_HANDOFF_CHECKPOINT_PATH", ""))
    _add(parser, "--fasttd3-repo",                  type=str, default=env.get("FASTTD3_REPO", ""))
    _add(parser, "--log-jsonl",                     type=str, default=os.path.join(project_root or ".", "runs", "native_training_log.jsonl"))
    _add(parser, "--checkpoint-path",               type=str, default=os.path.join(project_root or ".", "runs", "native_training_latest.pt"))
