"""Pure-RL v35 full-reward comparison phase.

This mirrors the canonical final full-reward v35 contract, but removes the
teacher, BC, DAgger relabeling, policy assist, replay resume, and adaptive
assist surfaces.  The run starts directly in RL for a 1M-row baseline.

File map:

PURE_RL_REMOVE_ARGS:  Define pure rl remove args constant
PURE_RL_ARGS:         Define pure rl args constant
PURE_RL_ENV:          Define pure rl env constant
build_phase:          Handle build phase logic
build_phases:         Handle build phases logic
"""

from __future__ import annotations

from dataclasses import replace

from training.curriculum.pipeline import (
    FullRewardInputs,
    SuccessSafetyInputs,
    SupportOptimizationInputs,
)
from training.curriculum.envs import default_phase_env
from training.curriculum.spec import PhaseSpec


PURE_RL_STEPS = 1_000_000


PURE_RL_REMOVE_ARGS: tuple[tuple[str, bool], ...] = (
    ("--native-teacher-provider", True),
    ("--teacher-arm-source", True),
    ("--teacher-lift-z", True),
    ("--teacher-lift-ramp-steps", True),
    ("--contact-teacher-hold-fraction-cap", True),
    ("--topdown-contact-teacher", False),
    ("--topdown-contact-teacher-close-rate", True),
    ("--topdown-contact-teacher-start-fraction", True),
    ("--topdown-contact-teacher-max-fraction", True),
    ("--topdown-contact-teacher-middle-scale", True),
    ("--topdown-contact-teacher-descent-z", True),
    ("--topdown-contact-teacher-missing-contact-extra-descent", True),
    ("--topdown-contact-teacher-inward-m", True),
    ("--topdown-contact-teacher-missing-contact-extra-inward", True),
    ("--native-contact-attr-parts", False),
    ("--adaptive-policy-assist", False),
    ("--stop-on-adaptive-assist-floor", False),
    ("--adaptive-assist-start-steps", True),
    ("--adaptive-assist-window-steps", True),
    ("--adaptive-assist-bad-window-steps", True),
    ("--adaptive-assist-arm-error", True),
    ("--adaptive-assist-finger-error", True),
    ("--adaptive-assist-bad-arm-error", True),
    ("--adaptive-assist-bad-finger-error", True),
    ("--adaptive-assist-step", True),
    ("--adaptive-assist-recover-step", True),
    ("--adaptive-assist-metric-gate", True),
    ("--adaptive-assist-strict-contact-gate", True),
    ("--adaptive-assist-strict-contact-min-rate", True),
    ("--adaptive-assist-metric-window-episodes", True),
    ("--adaptive-assist-baseline-episodes", True),
    ("--adaptive-assist-metric-min-ratio", True),
    ("--adaptive-assist-sync-bc-weights", True),
    ("--adaptive-assist-post-floor-steps", True),
    ("--adaptive-assist-disable-bc-after-floor", True),
    ("--adaptive-assist-bc-only-until-decay", True),
)

PURE_RL_ARGS: dict[str, str | int | float] = {
    "--native-teacher-provider"       : "none",
    "--start-steps"                   : 0,
    "--bc-only-steps"                 : 0,
    "--rl-phase-start-steps"          : 0,
    "--policy-bc-relabel"             : 0,
    "--rl-policy-bc-relabel"          : 0,
    "--bc-only-weight"                : 0.0,
    "--bc-only-arm-weight"            : -1.0,
    "--bc-only-finger-weight"         : -1.0,
    "--teacher-bc-weight"             : 0.0,
    "--teacher-bc-arm-weight"         : -1.0,
    "--teacher-bc-finger-weight"      : -1.0,
    "--teacher-bc-decay-steps"        : 0,
    "--rl-teacher-bc-weight"          : 0.0,
    "--rl-teacher-bc-arm-weight"      : -1.0,
    "--rl-teacher-bc-finger-weight"   : -1.0,
    "--rl-teacher-bc-decay-steps"     : 0,
    "--policy-assist-arm-mix"         : 0.0,
    "--policy-assist-arm-mix-floor"   : 0.0,
    "--policy-assist-finger-mix"      : 0.0,
    "--policy-assist-finger-mix-floor": 0.0,
    "--policy-assist-mix"             : 0.0,
    "--policy-assist-mix-floor"       : 0.0,
    "--policy-assist-decay-steps"     : 0, 
    "--policy-assist-decay-start-steps": 0,
    "--rl-policy-assist-mix"          : 0.0,
    "--rl-policy-assist-mix-floor"    : 0.0,
    "--rl-policy-assist-decay-steps"  : 0,
    "--rl-policy-assist-decay-start-steps": 0,
    "--assist-noise-clean-bc-target"  : 0,
    "--assist-noise-arm"              : 0.0,
    "--assist-noise-finger"           : 0.0,
    "--assist-noise-start-steps"      : 0,
    "--eval-teacher-assist-mix"       : 0.0,
    "--warmup-teacher-noise"          : 0.0,
}

PURE_RL_ENV: dict[str, str] = {
    "TOPDOWN_CONTACT_TEACHER"       : "0",
    "NATIVE_TEACHER_PROVIDER"       : "none",
    "START_STEPS"                   : "0",
    "BC_ONLY_STEPS"                 : "0",
    "RL_PHASE_START_STEPS"          : "0",
    "POLICY_BC_RELABEL"             : "0",
    "RL_POLICY_BC_RELABEL"          : "0",
    "BC_ONLY_WEIGHT"                : "0.0",
    "BC_ONLY_ARM_WEIGHT"            : "-1.0",
    "BC_ONLY_FINGER_WEIGHT"         : "-1.0",
    "TEACHER_BC_WEIGHT"             : "0.0",
    "TEACHER_BC_ARM_WEIGHT"         : "-1.0",
    "TEACHER_BC_FINGER_WEIGHT"      : "-1.0",
    "TEACHER_BC_DECAY_STEPS"        : "0",
    "RL_TEACHER_BC_WEIGHT"          : "0.0",
    "RL_TEACHER_BC_ARM_WEIGHT"      : "-1.0",
    "RL_TEACHER_BC_FINGER_WEIGHT"   : "-1.0",
    "RL_TEACHER_BC_DECAY_STEPS"     : "0",
    "POLICY_ASSIST_ARM_MIX"         : "0.0",
    "POLICY_ASSIST_ARM_MIX_FLOOR"   : "0.0",
    "POLICY_ASSIST_FINGER_MIX"      : "0.0",
    "POLICY_ASSIST_FINGER_MIX_FLOOR": "0.0",
    "POLICY_ASSIST_MIX"             : "0.0",
    "POLICY_ASSIST_MIX_FLOOR"       : "0.0",
    "POLICY_ASSIST_DECAY_STEPS"     : "0",
    "POLICY_ASSIST_DECAY_START_STEPS": "0",
    "RL_POLICY_ASSIST_MIX"          : "0.0",
    "RL_POLICY_ASSIST_MIX_FLOOR"    : "0.0",
    "RL_POLICY_ASSIST_DECAY_STEPS"  : "0",
    "RL_POLICY_ASSIST_DECAY_START_STEPS": "0",
    "ASSIST_NOISE_CLEAN_BC_TARGET"  : "0",
    "ASSIST_NOISE_ARM"              : "0.0",
    "ASSIST_NOISE_FINGER"           : "0.0",
    "ASSIST_NOISE_START_STEPS"      : "0",
    "ADAPTIVE_POLICY_ASSIST"        : "0",
    "EVAL_TEACHER_ASSIST_MIX"       : "0.0",
    "WARMUP_TEACHER_NOISE"          : "0.0",
}


def _support_args(support: SupportOptimizationInputs) -> dict[str, str | int | float]:
    return {
        "--fasttd3-num-atoms"        : support.fasttd3_num_atoms,
        "--fasttd3-v-min"           : support.fasttd3_v_min,
        "--fasttd3-v-max"           : support.fasttd3_v_max,
        "--actor-q-action-gate-mode" : support.actor_q_action_gate_mode,
        "--actor-bc-action-gate-mode": support.actor_bc_action_gate_mode,
        "--actor-lr"                : support.actor_lr,
        "--critic-lr"               : support.critic_lr,
        "--rl-actor-lr"             : support.rl_actor_lr,
        "--rl-critic-lr"            : support.rl_critic_lr,
        "--updates-per-step"        : support.updates_per_step,
        "--rl-updates-per-step"     : support.rl_updates_per_step,
        "--n-step"                  : support.n_step,
        "--rl-n-step"               : support.rl_n_step,
        "--policy-delay"            : support.policy_delay,
        "--rl-policy-delay"         : support.rl_policy_delay,
        "--rl-sync-targets-on-switch": 1,
        "--rl-reset-critic-optimizers-on-switch": 1,
        "--tau"                     : support.tau,
        "--rl-tau"                  : support.rl_tau,
        "--target-q-clip"           : support.target_q_clip,
        "--rl-target-q-clip"        : support.rl_target_q_clip,
        "--critic-grad-clip"        : support.critic_grad_clip,
        "--rl-actor-freeze-steps"   : support.rl_actor_freeze_steps,
        "--exploration-noise"       : support.exploration_noise,
        "--exploration-noise-finger": support.exploration_noise_finger,
        "--rl-exploration-noise"    : support.exploration_noise,
        "--rl-exploration-noise-finger": support.exploration_noise_finger,
        "--policy-noise"            : support.policy_noise,
        "--policy-noise-finger"     : support.policy_noise_finger,
        "--rl-policy-noise"         : support.policy_noise,
        "--rl-policy-noise-finger"  : support.policy_noise_finger,
        "--noise-clip"              : support.noise_clip,
        "--rl-noise-clip"           : support.noise_clip,
    }


def _support_env(support: SupportOptimizationInputs) -> dict[str, str]:
    return {
        **{key: str(value) for key, value in support.env().items()},
        "ACTOR_LR"                  : str(support.actor_lr),
        "CRITIC_LR"                 : str(support.critic_lr),
        "RL_ACTOR_LR"               : str(support.rl_actor_lr),
        "RL_CRITIC_LR"              : str(support.rl_critic_lr),
        "UPDATES_PER_STEP"          : str(support.updates_per_step),
        "RL_UPDATES_PER_STEP"       : str(support.rl_updates_per_step),
        "N_STEP"                    : str(support.n_step),
        "RL_N_STEP"                 : str(support.rl_n_step),
        "POLICY_DELAY"              : str(support.policy_delay),
        "RL_POLICY_DELAY"           : str(support.rl_policy_delay),
        "RL_SYNC_TARGETS_ON_SWITCH" : "1",
        "RL_RESET_CRITIC_OPTIMIZERS_ON_SWITCH": "1",
        "TAU"                       : str(support.tau),
        "RL_TAU"                    : str(support.rl_tau),
        "TARGET_Q_CLIP"             : str(support.target_q_clip),
        "RL_TARGET_Q_CLIP"          : str(support.rl_target_q_clip),
        "CRITIC_GRAD_CLIP"          : str(support.critic_grad_clip),
        "RL_ACTOR_FREEZE_STEPS"     : str(support.rl_actor_freeze_steps),
        "EXPLORATION_NOISE"         : str(support.exploration_noise),
        "EXPLORATION_NOISE_FINGER"  : str(support.exploration_noise_finger),
        "RL_EXPLORATION_NOISE"      : str(support.exploration_noise),
        "RL_EXPLORATION_NOISE_FINGER": str(support.exploration_noise_finger),
        "POLICY_NOISE"              : str(support.policy_noise),
        "POLICY_NOISE_FINGER"       : str(support.policy_noise_finger),
        "RL_POLICY_NOISE"           : str(support.policy_noise),
        "RL_POLICY_NOISE_FINGER"    : str(support.policy_noise_finger),
        "NOISE_CLIP"                : str(support.noise_clip),
        "RL_NOISE_CLIP"             : str(support.noise_clip),
    }


def build_phase(
    *,
    steps                    : int        = PURE_RL_STEPS,
    start_steps              : int        = 0,
    assist_decay_steps       : int | None = None,
    success_hold_steps       : int        = 30,
    min_success_rate         : float      = 0.80,
    min_median_lift          : float      = 0.020,
    score_drop_limit         : float      = 6.0,
    regression_patience_steps: int        = 200_000,
) -> PhaseSpec:
    del assist_decay_steps
    reward = FullRewardInputs()
    success = SuccessSafetyInputs(lift_success_hold_steps=success_hold_steps)
    support = SupportOptimizationInputs()
    env = {
        **default_phase_env(),
        **{key: str(value) for key, value in reward.env().items()},
        **{key: str(value) for key, value in success.env().items()},
        **_support_env(support),
        **PURE_RL_ENV,
    }
    args = {
        **_support_args(support),
        **PURE_RL_ARGS,
    }
    return PhaseSpec(
        name="p01_pure_rl_v35_lift02_noblock",
        description=(
            "Pure RL baseline using the canonical v35 full-reward lift02 "
            "contract; no teacher provider, BC relabeling, teacher loss, "
            "policy assist, adaptive assist, or DAgger phase."
        ),
        steps=steps,
        start_steps=start_steps,
        bc_only_steps=0,
        rl_phase_start_steps=0,
        assist_mix=0.0,
        assist_floor=0.0,
        assist_decay_steps=0,
        success_height=success.lift_success_height,
        success_hold_steps=success_hold_steps,
        min_success_rate=min_success_rate,
        min_median_lift=min_median_lift,
        max_median_disp=float("inf"),
        score_drop_limit=score_drop_limit,
        regression_patience_steps=regression_patience_steps,
        env=env,
        args=args,
        force_dagger_after_resume=False,
        reset_optimizers_on_resume=False,
        remove_args=PURE_RL_REMOVE_ARGS,
    )


def build_phases(
    *,
    steps                    : int        = PURE_RL_STEPS,
    start_steps              : int        = 0,
    success_hold_steps       : int        = 30,
    min_success_rate         : float      = 0.80,
    min_median_lift          : float      = 0.020,
    score_drop_limit         : float      = 6.0,
    regression_patience_steps: int        = 200_000,
) -> list[PhaseSpec]:
    return [
        build_phase(
            steps=steps,
            start_steps=start_steps,
            success_hold_steps=success_hold_steps,
            min_success_rate=min_success_rate,
            min_median_lift=min_median_lift,
            score_drop_limit=score_drop_limit,
            regression_patience_steps=regression_patience_steps,
        )
    ]


__all__ = ["build_phase", "build_phases"]
