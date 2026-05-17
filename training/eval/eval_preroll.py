"""

Pure helpers for eval contact pre-roll diagnostics

File map:

contact_preroll_debug_steps:    Return pre-roll steps that should emit detailed diagnostics
EvalPrerollRetry:               Values printed when eval pre-roll retries from reset
EvalPrerollSummary:             Values printed after eval pre-roll finishes
EvalPrerollDebug:               Values printed for detailed eval pre-roll state
EvalPrerollRequest:             Static settings for eval pre-roll execution
EvalPrerollCallbacks:           Callback surface for Isaac-bound eval pre-roll stepping
contact_preroll_eval_to_start:  Run callback-driven eval pre-roll and return latest obs with steps
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


def contact_preroll_debug_steps(max_steps: int) -> tuple[int, ...]:
    """Return pre-roll steps that should emit detailed diagnostics"""
    last_step = max(0, int(max_steps) - 1)
    return tuple(sorted({0, 1, 2, 5, 10, 25, 50, 100, 200, 400, last_step}))


@dataclass(frozen=True)
class EvalPrerollRetry:
    """Values printed when eval pre-roll retries from reset"""

    global_step     : int  # Field: training step associated with this record or action
    eval_episode_idx: int  # Field: evaluation episode index for this record
    attempt         : int  # Field: integer attempt value tracked by eval preroll retry
    steps           : int  # Field: integer steps value tracked by eval preroll retry
    palm            : float  # Field: floating-point palm value used by eval preroll retry
    orient          : float  # Field: floating-point orient value used by eval preroll retry
    phase1_ready    : bool  # Field: boolean/tensor readiness state for phase1

    def line(self) -> str:
        """Return the retry diagnostic line"""
        return (
            f"eval_preroll_retry step={self.global_step:05d}:{self.eval_episode_idx:02d} "
            f"attempt={self.attempt} steps={self.steps} "
            f"palm={self.palm:.3f} orient={self.orient:.1f} phase1={int(self.phase1_ready)}"
        )


@dataclass(frozen=True)
class EvalPrerollSummary:
    """Values printed after eval pre-roll finishes"""

    global_step     : int  # Field: training step associated with this record or action
    eval_episode_idx: int  # Field: evaluation episode index for this record
    steps           : int  # Field: integer steps value tracked by eval preroll summary
    palm            : float  # Field: floating-point palm value used by eval preroll summary
    orient          : float  # Field: floating-point orient value used by eval preroll summary
    phase1_ready    : bool  # Field: boolean/tensor readiness state for phase1
    attempts        : int  # Field: integer attempts value tracked by eval preroll summary
    released        : bool  # Field: boolean value indicating the released state for eval preroll summary

    def line(self) -> str:
        """Return the final pre-roll summary line"""
        return (
            f"eval_preroll step={self.global_step:05d}:{self.eval_episode_idx:02d} "
            f"steps={self.steps} palm={self.palm:.3f} orient={self.orient:.1f} "
            f"phase1={int(self.phase1_ready)} attempts={self.attempts} released={int(self.released)}"
        )


@dataclass(frozen=True)
class EvalPrerollDebug:
    """Values printed for detailed eval pre-roll state"""

    attempt      : int  # Field: integer attempt value tracked by eval preroll debug
    step         : int  # Field: integer step value tracked by eval preroll debug
    palm         : float  # Field: floating-point palm value used by eval preroll debug
    height       : float  # Field: floating-point height value used by eval preroll debug
    orient       : float  # Field: floating-point orient value used by eval preroll debug
    align_ready  : bool  # Field: boolean/tensor readiness state for align
    release_ready: bool  # Field: boolean/tensor readiness state for release
    arm          : tuple[float, ...]  # Field: floating-point arm value used by eval preroll debug
    reduced_arm  : tuple[float, ...]  # Field: floating-point reduced arm value used by eval preroll debug
    joints       : tuple[float, ...]  # Field: floating-point joints value used by eval preroll debug

    def line(self) -> str:
        """Return the detailed pre-roll debug line"""
        arm = ",".join(f"{value:+.3f}" for value in self.arm)
        reduced_arm = ",".join(f"{value:+.3f}" for value in self.reduced_arm)
        joints = ",".join(f"{value:+.3f}" for value in self.joints)
        return (
            f"eval_preroll_debug attempt={self.attempt} step={self.step} "
            f"palm={self.palm:.3f} height={self.height:.3f} orient={self.orient:.1f} "
            f"align={int(self.align_ready)} release={int(self.release_ready)} "
            f"arm=[{arm}] reduced_arm=[{reduced_arm}] joints=[{joints}]"
        )


@dataclass(frozen=True)
class EvalPrerollRequest:
    """Static settings for eval pre-roll execution"""

    enabled           : bool  # Field: whether this optional feature path is enabled
    has_phase1_teacher: bool  # Field: boolean value indicating the has phase1 teacher state for eval preroll request
    max_steps         : int  # Field: step count used for max steps scheduling or reporting
    max_attempts      : int = 1  # Field: integer max attempts value tracked by eval preroll request


@dataclass(frozen=True)
class EvalPrerollCallbacks:
    """Callback surface for Isaac-bound eval pre-roll stepping"""

    release_mask              : Callable[[], torch.Tensor]  # Field: boolean mask selecting release rows for eval preroll callbacks
    compute_action            : Callable[[torch.Tensor, int], torch.Tensor]  # Field: callback used for the compute action operation
    stash_action              : Callable[[torch.Tensor], None]  # Field: callback used for the stash action operation
    assemble_action           : Callable[[torch.Tensor, int], torch.Tensor]  # Field: callback used for the assemble action operation
    step_env                  : Callable[[torch.Tensor], torch.Tensor]  # Field: callback used for the step env operation
    reset_env                 : Callable[[], torch.Tensor]  # Field: callback used for the reset env operation
    clear_counters            : Callable[[], None]  # Field: callback used for the clear counters operation
    simulation_should_continue: Callable[[], bool]  # Field: callback used for the simulation should continue operation


def contact_preroll_eval_to_start(
    obs_tensor: torch.Tensor,         # Param: policy observation tensor used by actor or replay logic
    *,
    request  : EvalPrerollRequest,  # Param: normalized request object passed into this helper
    callbacks: EvalPrerollCallbacks,  # Param: input value used as callbacks
) -> tuple[torch.Tensor, int]:
    """Run callback-driven eval pre-roll and return latest obs with steps

    Steps:
    - Resolve inputs for `contact_preroll_eval_to_start` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not request.enabled:
        return obs_tensor, 0
    if not request.has_phase1_teacher:
        raise RuntimeError("contact_start_mode=phase1_terminal requires --phase1-checkpoint")

    max_steps = max(0, int(request.max_steps))
    max_attempts = max(1, int(request.max_attempts))
    latest_obs = obs_tensor
    steps_taken = 0
    for attempt_idx in range(max_attempts):
        if attempt_idx > 0:
            latest_obs = callbacks.reset_env()
            callbacks.clear_counters()
        for preroll_step in range(max_steps):
            if not callbacks.simulation_should_continue():
                return latest_obs, steps_taken
            release = callbacks.release_mask()
            if bool(release.reshape(-1)[0].item()):
                return latest_obs, steps_taken
            episode_step = torch.full(
                (latest_obs.shape[0],),
                int(preroll_step),
                device=latest_obs.device,
                dtype=torch.long,
            )
            policy_action = callbacks.compute_action(latest_obs, preroll_step)
            callbacks.stash_action(policy_action)
            reduced_action = callbacks.assemble_action(policy_action, preroll_step)
            latest_obs = callbacks.step_env(reduced_action)
            steps_taken += 1
        if bool(callbacks.release_mask().reshape(-1)[0].item()):
            return latest_obs, steps_taken
    return latest_obs, steps_taken
