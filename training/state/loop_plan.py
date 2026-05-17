"""

Pure per-step training loop decision planning

This module provides helper functions and data structures for building a post-step plan of actions for the training loop,
without touching runtime objects, used by the main training loop after each environment step.

File map:

LoopStepPlan:          Actions the trainer loop should take after one env step
build_loop_step_plan:  Build post-step decisions without touching runtime objects
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..io.checkpoint_schedule import ScheduledCheckpointPlan, build_scheduled_checkpoint_plan
from .episode_resets import DoneEpisodeResetPlan, build_done_episode_reset_plan
from ..logging.progress import crossed_log_boundary, should_run_eval


@dataclass(frozen=True)
class LoopStepPlan:
    """Actions the trainer loop should take after one env step"""

    global_step    : int  # training step associated with this record or action
    num_added      : int  # count of added values
    replay_size    : int  # configured or observed replay-buffer size
    should_log     : bool  # boolean value indicating the should log state for loop step plan
    should_eval    : bool  # boolean value indicating the should eval state for loop step plan
    checkpoint_plan: ScheduledCheckpointPlan  # integer checkpoint plan value tracked by loop step plan
    done_reset_plan: DoneEpisodeResetPlan  # stores done reset plan for loop step plan


def build_loop_step_plan(
    *,
    global_step                   : int,  # Param: current absolute training step
    num_added                     : int,  # Param: number of env transitions added during the current collection step
    log_every                     : int,  # Param: global-step interval used to decide when progress rows are emitted
    next_eval_step                : int | None,  # Param: next global step that should trigger evaluation, or None when eval is disabled
    replay_size                   : int,  # Param: number of transitions currently available in replay
    batch_size                    : int,  # Param: number of replay samples required for one update batch
    done_flags                    : torch.Tensor,  # Param: per-env done flags returned by the latest env step
    active_env_mask               : torch.Tensor,  # Param: mask selecting env rows still active for collection or reset handling
    checkpoint_path               : str,  # Param: base checkpoint path used for scheduled save decisions
    save_replay_in_checkpoint     : bool,  # Param: whether checkpoint saves should include replay-buffer contents
    next_checkpoint_step          : int | None,  # Param: next global step for a regular checkpoint save
    checkpoint_every              : int,  # Param: global-step interval for regular checkpoint saves
    next_rolling_checkpoint_step  : int | None,  # Param: next global step for a rolling checkpoint save
    rolling_checkpoint_every      : int,  # Param: global-step interval for rolling checkpoint saves
    rolling_checkpoint_keep       : int,  # Param: maximum number of rolling checkpoints kept after pruning
    existing_checkpoint_names     : tuple[str, ...],  # Param: existing checkpoint filenames used to decide rolling-checkpoint pruning
    legacy_contact_preroll_enabled: bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled       : bool,  # Param: enables topdown-preroll reset handling
) -> LoopStepPlan:
    """Build post-step decisions without touching runtime objects"""
    done_reset_plan = build_done_episode_reset_plan(
        done_flags,
        active_env_mask,
        legacy_contact_preroll_enabled=legacy_contact_preroll_enabled,
        topdown_preroll_enabled=topdown_preroll_enabled,
    )
    return LoopStepPlan(
        global_step=int(global_step),
        num_added=max(0, int(num_added)),
        replay_size=max(0, int(replay_size)),
        should_log=crossed_log_boundary(
            global_step=global_step,
            num_added=num_added,
            log_every=log_every,
        ),
        should_eval=should_run_eval(
            next_eval_step=next_eval_step,
            replay_size=replay_size,
            batch_size=batch_size,
            global_step=global_step,
            env0_done=done_reset_plan.env0_done,
        ),
        checkpoint_plan=build_scheduled_checkpoint_plan(
            global_step=global_step,
            checkpoint_path=checkpoint_path,
            save_replay_in_checkpoint=save_replay_in_checkpoint,
            next_checkpoint_step=next_checkpoint_step,
            checkpoint_every=checkpoint_every,
            next_rolling_checkpoint_step=next_rolling_checkpoint_step,
            rolling_checkpoint_every=rolling_checkpoint_every,
            rolling_checkpoint_keep=rolling_checkpoint_keep,
            existing_checkpoint_names=existing_checkpoint_names,
        ),
        done_reset_plan=done_reset_plan,
    )
