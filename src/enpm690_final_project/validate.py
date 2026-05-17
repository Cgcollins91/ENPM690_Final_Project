"""
Low-cost validation for the current no-shell profile.

This module provides a simple validation function that checks whether the current environment variables
and command-line arguments match the expected values for the no-shell profile. This is intended to catch accidental misconfigurations
or deviations from the expected setup. The validation function can be called at the start of a training run to ensure that the active
configuration is correct before proceeding with the full training process.



"""

from __future__ import annotations

from .training_engine import CommandPlan


EXPECTED_ENV = {
    "TASK": "Isaac-Topdown-Curriculum-G129-Dex3-Joint",
    "TOPDOWN_LIFT_TASK": "1",
    "TOPDOWN_SOURCE_POSE_MODE": "red",
    "TOPDOWN_BLOCK_SIZE": "0.08",
    "TOPDOWN_BLOCK_MASS": "0.25",
    "TOPDOWN_CONTACT_OFFSET": "0.002",
    "ARM_CONTROLLER": "policy",
    "TEACHER_ARM_SOURCE": "ik",
    "TOPDOWN_CONTACT_TEACHER": "1",
    "POLICY_BC_RELABEL": "1",
    "RL_POLICY_BC_RELABEL": "0",
    "PRIVILEGED_CRITIC": "1",
    "REWARD_NORMALIZATION": "0",
    "RESET_OBS_STATS_ON_RESUME": "0",
}

EXPECTED_ARGS = {
    "--num-envs": "100",
    "--total-steps": "1000000",
    "--start-steps": "10000",
    "--bc-only-steps": "100000",
    "--rl-phase-start-steps": "100000",
    "--n-step": "3",
    "--gamma": "0.995",
    "--updates-per-step": "8",
    "--rl-updates-per-step": "50",
    "--rl-policy-delay": "4",
    "--rl-tau": "0.0005",
}


def _arg_value(command: list[str], flag: str) -> str | None:
    """Return the value following a scalar CLI option, if present."""
    try:
        idx = command.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(command):
        return None
    return command[idx + 1]


def validate_current_equivalence(plan: CommandPlan) -> list[str]:
    """Validate that the new profile preserves the active current run surface."""

    errors: list[str] = []
    for key, expected in EXPECTED_ENV.items():
        actual = plan.env.get(key)
        if actual != expected:
            errors.append(f"env {key}: expected {expected!r}, got {actual!r}")
    for flag, expected in EXPECTED_ARGS.items():
        actual = _arg_value(plan.command, flag)
        if actual != expected:
            errors.append(f"arg {flag}: expected {expected!r}, got {actual!r}")
    required_flags = [
        "--headless",
        "--disable-camera-perception",
        "--include-wrist-roll",
        "--topdown-contact-teacher",
        "--privileged-critic",
    ]
    for flag in required_flags:
        if flag not in plan.command:
            errors.append(f"missing flag {flag}")
    if "--no-privileged-critic" in plan.command:
        errors.append("arg --no-privileged-critic must not be present")
    return errors
