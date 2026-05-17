"""Termination terms for the topdown reach-align-contact curriculum.

IsaacLab calls these functions once per environment step through the
termination manager.  Each function returns a boolean tensor shaped
``(env.num_envs,)`` where ``True`` means that the corresponding vectorized
environment row should end its current episode.

This file intentionally keeps the termination decisions thin: all geometric
measurements and held-success counters live in ``state_machine.py`` so rewards,
observations, and terminations share the same source of truth.
"""

from __future__ import annotations

# Environment variables let long training runs tune termination strictness
import os
# ``math`` is used for scalar degree-to-radian conversion before tensor compare
import math
# TYPE_CHECKING avoids importing IsaacLab at module import time during tests
from typing import TYPE_CHECKING

# Termination terms return torch boolean masks for vectorized env rows
import torch

# Geometry, drift, and debounced success helpers are shared with rewards
from .state_machine import (
    block_displacement,           # Full 3D block displacement from reset pose
    block_lift_height,            # Positive Z lift gained since episode start
    lift_drop_from_max_bad,       # Drop-from-best-lift failure detector
    block_tilt_angle_rad,         # Active block tilt angle in radians
    block_xy_displacement,        # Horizontal block drift from reset pose
    open_hand_alignment_error,    # Pre-contact thumb/index face-target error
    light_contact_success_held,   # Debounced Stage-2 light-contact success
    lift_success_held,            # Debounced dynamic-block lift success
)

# Keep IsaacLab imports type-only so this module remains importable in smoke tests
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Lift mode swaps the success and drift semantics from contact-only to lift-aware
_TOPDOWN_LIFT_TASK = os.environ.get("TOPDOWN_LIFT_TASK", "0") == "1"


def _env_float(
    name   : str,  # Param: environment variable name to read from process env
    default: float,  # Param: fallback value used when the variable is unset
) -> float:
    """Read a typed environment override, falling back to the supplied default.

    Unlike the state-machine helper, this intentionally lets invalid floats
    raise ``ValueError`` during startup.  Termination thresholds should fail
    loudly when a run manifest or launch script supplies malformed values.
    """
    raw = os.environ.get(name, "")  # Empty string is treated the same as unset
    if raw == "":                   # No override was supplied for this run
        return default              # Preserve the caller's mode-specific default
    return float(raw)               # Parse the override once as a Python scalar


# Block drift cutoff comes from env and is looser in lift mode because a good
# lift necessarily moves the block farther than contact-only alignment should
_BLOCK_DRIFT_THRESHOLD = _env_float(
    "CURRICULUM_BLOCK_DRIFT_THRESHOLD", # Env var controlling drift termination
    0.20 if _TOPDOWN_LIFT_TASK else 0.05, # Lift allows 20cm, contact allows 5cm
)
# Optional tilt cutoff is disabled at zero so legacy runs are not tilt-limited
_LIFT_TERMINATE_TILT_DEG = _env_float("TOPDOWN_LIFT_TERMINATE_TILT_DEG", 0.0)

# Tilt termination only starts after a small lift so tabletop settling is ignored
_LIFT_TERMINATE_TILT_HEIGHT = _env_float("TOPDOWN_LIFT_TERMINATE_TILT_HEIGHT", 0.015)

# Alignment fail-fast is disabled at zero and enabled by specifying grace seconds
_ALIGN_FAILFAST_AFTER_SECONDS = _env_float("CURRICULUM_ALIGN_FAILFAST_AFTER_SECONDS", 0.0)

# Alignment error above this value after the grace period terminates the episode
_ALIGN_FAILFAST_THRESHOLD = _env_float("CURRICULUM_ALIGN_FAILFAST_THRESHOLD", 0.8)


def light_contact_success(
    env: "ManagerBasedRLEnv", # Param: vectorized IsaacLab env that owns task state
) -> torch.Tensor:
    """Episode-terminating success: stage 2 success_now held for HOLD steps.

    The termination manager treats ``True`` as a successful terminal state.
    Contact curriculum runs end once light contact has been held for the
    configured debounce window.  Lift runs instead require the lift-specific
    success predicate, which includes height, drift, contact, and optional tilt
    checks inside ``state_machine.lift_success_held``.
    """
    if _TOPDOWN_LIFT_TASK:          # Dynamic-block lift uses the stricter lift success latch
        return lift_success_held(env) # Shape: (num_envs,), True where lift success is held
    return light_contact_success_held(env) # Shape: (num_envs,), True where contact success is held


def block_drifted(
    env: "ManagerBasedRLEnv", # Param: vectorized IsaacLab env containing block pose caches
) -> torch.Tensor:
    """Return a termination mask for blocks that moved too far from their reset pose.

    Contact-only curriculum runs use full 3D displacement because any large
    block motion before lift indicates a failed approach/contact episode.  Lift
    curriculum runs use horizontal displacement, optional lifted-block tilt,
    and drop-from-best-lift so successful vertical motion is not penalized.
    """
    if _TOPDOWN_LIFT_TASK: # Lift training has special rules that allow upward motion
        drift_threshold = _env_float(
            "CURRICULUM_BLOCK_DRIFT_THRESHOLD", # Re-read so launch-time overrides are honored
            _BLOCK_DRIFT_THRESHOLD,             # Fall back to the module default chosen above
        )
        bad = block_xy_displacement(env) > drift_threshold # True where horizontal drift is excessive
        tilt_max_deg = _env_float(
            "TOPDOWN_LIFT_TERMINATE_TILT_DEG", # Optional max tilt while the block is in the air
            _LIFT_TERMINATE_TILT_DEG,          # Zero keeps the tilt branch disabled
        )
        if tilt_max_deg > 0.0: # A positive max tilt enables lifted-block tilt termination
            tilt_height = _env_float(
                "TOPDOWN_LIFT_TERMINATE_TILT_HEIGHT", # Minimum lift before tilt is considered
                _LIFT_TERMINATE_TILT_HEIGHT,          # Default ignores tiny tabletop motion
            )
            bad_tilt = ( # Per-env mask for lifted blocks tilted past the allowed angle
                (block_lift_height(env) >= tilt_height) # Only check tilt after meaningful lift
                & (block_tilt_angle_rad(env) > math.radians(tilt_max_deg)) # Compare radians to radians
            )
            bad = bad | bad_tilt # Terminate rows that fail either drift or tilt
        bad = bad | lift_drop_from_max_bad(env) # Also terminate rows that dropped after prior lift
        return bad # Shape: (num_envs,), True where lift episode should end as failed
    return block_displacement(env) > _BLOCK_DRIFT_THRESHOLD # Contact mode fails on full 3D drift


def alignment_timeout_bad(
    env: "ManagerBasedRLEnv", # Param: vectorized IsaacLab env containing age and alignment state
) -> torch.Tensor:
    """Fail fast when pre-contact alignment remains poor after a grace period.

    The term is disabled unless ``CURRICULUM_ALIGN_FAILFAST_AFTER_SECONDS`` is
    positive. It targets the degradation mode where DAgger spends long episodes
    with the hand far from the open-hand face targets, adding low-quality
    policy-assist samples before the teacher can recover.
    """
    after_seconds = _env_float(
        "CURRICULUM_ALIGN_FAILFAST_AFTER_SECONDS", # Grace period before poor alignment may terminate
        _ALIGN_FAILFAST_AFTER_SECONDS,             # Default keeps this termination disabled
    )
    if after_seconds <= 0.0: # Nonpositive grace period means the fail-fast term is off
        return torch.zeros(  # Disabled terms must still return a correctly shaped bool mask
            env.num_envs,    # One boolean per vectorized environment row
            dtype=torch.bool, # Termination manager expects boolean tensors
            device=env.device, # Allocate on the same device as the Isaac env tensors
        )

    episode_age = getattr(env, "episode_length_buf", None) # IsaacLab per-env step counter
    if not torch.is_tensor(episode_age): # Defensive fallback for import/smoke contexts without episode age
        return torch.zeros(              # No age information means no fail-fast rows are selected
            env.num_envs,                # Keep the mask length aligned to env rows
            dtype=torch.bool,            # Boolean mask for termination manager
            device=env.device,           # Match env tensor device
        )

    step_dt = float(getattr(env, "step_dt", 0.01) or 0.01) # Seconds per env step, with a safe 10ms fallback
    min_steps = max(1, int(math.ceil(after_seconds / step_dt))) # Convert grace seconds to whole env steps
    threshold = _env_float(
        "CURRICULUM_ALIGN_FAILFAST_THRESHOLD", # Alignment-error cutoff after grace period
        _ALIGN_FAILFAST_THRESHOLD,             # Default threshold from module constant
    )
    return ( # Shape: (num_envs,), True where the hand stayed badly aligned too long
        (episode_age >= min_steps)             # Grace period has elapsed for this env row
        & (open_hand_alignment_error(env) > threshold) # Open-hand face error still exceeds cutoff
    )
