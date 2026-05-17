"""

Vectorized eval episode state helpers

This module provides helper functions and data structures for tracking eval episode state, computing eval outcomes,
and producing step masks, used by the evaluation loop

File map:

EvalStepMasks:      Masks produced by one eval state update
EvalEnv0Outcome:    Env-0 terminal booleans for eval trace lines
EvalEpisodeState:   Mutable eval returns and done flags
eval_env0_outcome:  Return env-0 eval outcome using monolith terminal semantics
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvalStepMasks:
    """Masks produced by one eval state update"""

    active_mask  : torch.Tensor  # boolean mask selecting active rows for eval step masks
    live_mask    : torch.Tensor  # boolean mask selecting live rows for eval step masks
    new_done_mask: torch.Tensor  # boolean mask selecting new done rows for eval step masks
    timeout_flags: torch.Tensor  # per-env timeout flags returned by the environment step
    done_flags   : torch.Tensor  # per-env done flags returned by the environment step


@dataclass(frozen=True)
class EvalEnv0Outcome:
    """Env-0 terminal booleans for eval trace lines"""

    success            : bool  # success flag or rate for the rollout/evaluation record
    off_table          : bool  # flag indicating that the block left the table/work surface
    phase15_shell_drift: bool  # boolean value indicating the phase15 shell drift state for eval env0 outcome
    block_drift        : bool  # measured block drift used by diagnostics or success checks
    timeout            : bool  # boolean value indicating the timeout state for eval env0 outcome
    done               : bool  # done flag tensor or scalar returned by the environment step


@dataclass
class EvalEpisodeState:
    """Mutable eval returns and done flags"""

    return_env              : torch.Tensor  # tensor containing return env values for batched env rows
    done_mask               : torch.Tensor  # boolean mask selecting done rows for eval episode state
    success_mask            : torch.Tensor  # boolean mask selecting success rows for eval episode state
    off_table_mask          : torch.Tensor  # boolean mask selecting off table rows for eval episode state
    phase15_shell_drift_mask: torch.Tensor  # boolean mask selecting phase15 shell drift rows for eval episode state
    block_drift_mask        : torch.Tensor  # boolean mask selecting block drift rows for eval episode state
    timeout_mask            : torch.Tensor  # boolean mask selecting timeout rows for eval episode state
    steps_env               : torch.Tensor  # tensor containing steps env values for batched env rows

    @classmethod
    def create(
        cls,
        *,
        num_envs: int,  # Param: number of parallel environment rows represented
        device  : torch.device | str,  # Param: torch device where tensors are read or allocated
    ) -> "EvalEpisodeState":
        """Create zeroed eval episode state tensors"""
        return cls(
            return_env=torch.zeros(int(num_envs), dtype=torch.float32, device=device),
            done_mask=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            success_mask=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            off_table_mask=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            phase15_shell_drift_mask=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            block_drift_mask=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            timeout_mask=torch.zeros(int(num_envs), dtype=torch.bool, device=device),
            steps_env=torch.zeros(int(num_envs), dtype=torch.long, device=device),
        )

    def update(
        self,
        *,
        reward                   : torch.Tensor,  # Param: reward tensor or scalar from the transition
        success_flags            : torch.Tensor,  # Param: flag values describing success
        off_table_flags          : torch.Tensor,  # Param: flag values describing off table
        phase15_shell_drift_flags: torch.Tensor,  # Param: flag values describing phase15 shell drift
        block_drift_flags        : torch.Tensor,  # Param: flag values describing block drift
        terminated_flags         : torch.Tensor,  # Param: flag values describing terminated
        truncated_flags          : torch.Tensor,  # Param: flag values describing truncated
        step_index               : int,  # Param: index selecting the step entry
    ) -> EvalStepMasks:
        """Update eval state after one env step

        Steps:
        - Resolve inputs for `update` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        device = self.return_env.device
        reward_t = reward.to(device=device, dtype=torch.float32).reshape(-1)
        success = success_flags.to(device=device, dtype=torch.bool)
        off_table = off_table_flags.to(device=device, dtype=torch.bool)
        phase15 = phase15_shell_drift_flags.to(device=device, dtype=torch.bool)
        block_drift = block_drift_flags.to(device=device, dtype=torch.bool)
        terminated = terminated_flags.to(device=device, dtype=torch.bool)
        truncated = truncated_flags.to(device=device, dtype=torch.bool)
        timeout = truncated & ~terminated
        done_flags = success | off_table | phase15 | block_drift | terminated | truncated
        active_mask = ~self.done_mask
        live_mask = active_mask & ~done_flags
        new_done_mask = active_mask & done_flags

        self.return_env = self.return_env + torch.where(
            active_mask,
            reward_t,
            torch.zeros_like(reward_t),
        )
        step_count = torch.full(
            self.steps_env.shape,
            int(step_index) + 1,
            dtype=torch.long,
            device=device,
        )
        self.steps_env = torch.where(active_mask, step_count, self.steps_env)
        self.success_mask |= new_done_mask & success
        self.off_table_mask |= new_done_mask & off_table
        self.phase15_shell_drift_mask |= new_done_mask & phase15
        self.block_drift_mask |= new_done_mask & block_drift
        self.timeout_mask |= new_done_mask & timeout
        self.done_mask |= new_done_mask
        return EvalStepMasks(
            active_mask=active_mask.detach().clone(),
            live_mask=live_mask.detach().clone(),
            new_done_mask=new_done_mask.detach().clone(),
            timeout_flags=timeout.detach().clone(),
            done_flags=done_flags.detach().clone(),
        )

    @property
    def remaining_active_mask(self) -> torch.Tensor:
        """Return rows that have not reached eval done"""
        return ~self.done_mask

    @property
    def all_done(self) -> bool:
        """Return whether every eval row is done"""
        return bool(self.done_mask.all().item())

    def finalize_remaining_timeouts(self, *, steps_taken: int) -> torch.Tensor:
        """Mark remaining active rows as timed out at rollout end"""
        remaining = self.remaining_active_mask
        if bool(remaining.any().item()):
            self.timeout_mask |= remaining
            self.done_mask |= remaining
            if int(steps_taken) > 0:
                step_count = torch.full(
                    self.steps_env.shape,
                    int(steps_taken),
                    dtype=torch.long,
                    device=self.steps_env.device,
                )
                self.steps_env = torch.where(remaining, step_count, self.steps_env)
        return remaining.detach().clone()


def eval_env0_outcome(
    *,
    success_flags            : torch.Tensor,  # Param: flag values describing success
    off_table_flags          : torch.Tensor,  # Param: flag values describing off table
    phase15_shell_drift_flags: torch.Tensor,  # Param: flag values describing phase15 shell drift
    block_drift_flags        : torch.Tensor,  # Param: flag values describing block drift
    terminated_flags         : torch.Tensor,  # Param: flag values describing terminated
    truncated_flags          : torch.Tensor,  # Param: flag values describing truncated
) -> EvalEnv0Outcome:
    """Return env-0 eval outcome using monolith terminal semantics

    Steps:
    - Resolve inputs for `eval_env0_outcome` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    success = bool(success_flags[0].item())
    off_table = bool(off_table_flags[0].item())
    phase15 = bool(phase15_shell_drift_flags[0].item())
    block_drift = bool(block_drift_flags[0].item())
    timeout = bool((truncated_flags[0] & ~terminated_flags[0]).item())
    done = success or off_table or phase15 or block_drift or timeout
    return EvalEnv0Outcome(
        success=success,
        off_table=off_table,
        phase15_shell_drift=phase15,
        block_drift=block_drift,
        timeout=timeout,
        done=done,
    )
