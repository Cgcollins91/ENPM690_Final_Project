"""

Pre-launch argument normalization from the monolith trainer



ArgNormalizationResult:           Normalized args plus changed field names
teacher_bc_decay_explicit:        Return whether teacher BC decay was explicitly supplied
normalize_prelaunch_args:         Apply pre-launch argparse mutations 1. If critic burn-in steps is not set, attempt to read it from the environment variable CRITIC_BURN_IN_STEPS 2. If critic burn-in steps is set, ensure actor freeze steps is at least as large 3. If bc-only steps is set and teacher BC decay steps is not explicit, set teacher BC decay steps to max(5000, bc-only steps // 2)
normalized_training_cli_request:  Return a request with monolith-compatible pre-launch arg mutations
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .cli import TrainingCliRequest
from ..state.phase_overrides import apply_force_dagger_resume_overrides


@dataclass(frozen=True)
class ArgNormalizationResult:
    """Normalized args plus changed field names"""

    args   : dict[str, object]  # parsed CLI/config arguments passed into this helper
    changed: tuple[str, ...]  # string changed value used by arg normalization result


def teacher_bc_decay_explicit(argv: Sequence[str]) -> bool:
    """Return whether teacher BC decay was explicitly supplied"""
    return any(
        arg == "--teacher-bc-decay-steps" or str(arg).startswith("--teacher-bc-decay-steps=")
        for arg in argv
    )


def normalize_prelaunch_args(
    args: Mapping[str, object],           # Param: argument mapping or namespace read and updated by this helper
    *,
    env                          : Mapping[str, str],  # Param: environment or backend object used for runtime calls
    teacher_bc_decay_was_explicit: bool,  # Param: boolean input controlling teacher bc decay was explicit
) -> ArgNormalizationResult:
    """Apply pre-launch argparse mutations
    1. If critic burn-in steps is not set, attempt to read it from the environment variable CRITIC_BURN_IN_STEPS
    2. If critic burn-in steps is set, ensure actor freeze steps is at least as large
    3. If bc-only steps is set and teacher BC decay steps is not explicit, set teacher BC decay steps to max(5000, bc-only steps // 2)

    Steps:
    - Resolve inputs for `normalize_prelaunch_args` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    normalized = dict(args)
    changed: list[str] = []

    critic_burn_in = int(normalized.get("critic_burn_in_steps", -1))
    raw_critic_burn_in = env.get("CRITIC_BURN_IN_STEPS")
    if critic_burn_in < 0 and raw_critic_burn_in not in (None, ""):
        try:
            critic_burn_in = int(raw_critic_burn_in)
        except ValueError as exc:
            raise ValueError(
                f"CRITIC_BURN_IN_STEPS must be an integer, got {raw_critic_burn_in!r}"
            ) from exc
        normalized["critic_burn_in_steps"] = critic_burn_in
        changed.append("critic_burn_in_steps")

    if critic_burn_in >= 0:
        actor_freeze = max(int(normalized.get("actor_freeze_steps", 0)), critic_burn_in)
        if actor_freeze != int(normalized.get("actor_freeze_steps", 0)):
            normalized["actor_freeze_steps"] = actor_freeze
            changed.append("actor_freeze_steps")

    bc_only_steps = int(normalized.get("bc_only_steps", 0))
    if bc_only_steps > 0 and not teacher_bc_decay_was_explicit:
        teacher_bc_decay_steps = max(5000, bc_only_steps // 2)
        if teacher_bc_decay_steps != int(normalized.get("teacher_bc_decay_steps", 30000)):
            normalized["teacher_bc_decay_steps"] = teacher_bc_decay_steps
            changed.append("teacher_bc_decay_steps")

    resume_override = apply_force_dagger_resume_overrides(normalized)
    if resume_override.applied:
        changed.extend(resume_override.changed)

    return ArgNormalizationResult(args=normalized, changed=tuple(changed))


def normalized_training_cli_request(request: TrainingCliRequest) -> TrainingCliRequest:
    """
    Return a request with monolith-compatible pre-launch arg mutations
    """
    result = normalize_prelaunch_args(
        request.known_args,
        env=request.env,
        teacher_bc_decay_was_explicit=teacher_bc_decay_explicit(request.argv),
    )
    return TrainingCliRequest(
        argv=request.argv,
        known_args=result.args,
        unknown_args=request.unknown_args,
        project_root=request.project_root,
        env=request.env,
    )
