"""

Play-mode startup helpers

This module provides helper functions and data structures for building and validating a play-mode plan based on runtime configs,
applying play-mode overrides to runtime arguments, and restoring overridden arguments after play rollouts, used by the evaluation
loop

File map:

PlayModePlan:                   Resolved play-mode settings for deterministic eval-only runs
build_play_mode_plan:           Build play-mode plan from typed configs
validate_play_mode_plan:        Raise for invalid play-mode combinations
apply_play_assist_overrides:    Apply monolith play-mode policy-assist overrides to args
play_eval_episode_override:     Set eval episodes to play episodes and return previous value
restore_eval_episode_override:  Restore eval episode count after play rollout
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
import os

from .configs import RuntimeConfigBundle


@dataclass(frozen=True)
class PlayModePlan:
    """Resolved play-mode settings for deterministic eval-only runs"""

    enabled                   : bool  # true when --play switches the run into eval-only rollout mode
    skip_checkpoint           : bool  # allows play mode to start without loading checkpoint_path
    checkpoint_path           : str  # checkpoint loaded before play rollout unless skip_checkpoint is set
    log_jsonl                 : str  # JSONL file that receives play/eval event rows
    play_episodes             : int  # number of eval episodes to run while play mode is active
    eval_teacher_assist_mix   : float  # teacher-assist ratio used to validate and configure play rollout
    forced_policy_assist_mix  : float = 0.0  # policy-assist mix override applied only during play mode
    forced_policy_assist_floor: float = 0.0  # policy-assist floor override applied only during play mode

    @property
    def log_dir(self) -> str:
        """Return log directory with script-compatible fallback"""
        # Use the JSONL parent directory, or current directory when log_jsonl is just a filename
        return os.path.dirname(self.log_jsonl) or "."


def build_play_mode_plan(configs: RuntimeConfigBundle) -> PlayModePlan:
    """Build play-mode plan from typed configs"""
    return PlayModePlan(
        enabled=configs.eval.play,
        skip_checkpoint=configs.eval.play_skip_checkpoint,
        checkpoint_path=configs.checkpoint.checkpoint_path,
        log_jsonl=configs.checkpoint.log_jsonl,
        play_episodes=configs.eval.play_episodes,
        eval_teacher_assist_mix=configs.eval.eval_teacher_assist_mix,
    )


def validate_play_mode_plan(plan: PlayModePlan) -> None:
    """Raise for invalid play-mode combinations"""
    if plan.skip_checkpoint and float(plan.eval_teacher_assist_mix) < 1.0:
        raise RuntimeError("--play-skip-checkpoint requires --eval-teacher-assist-mix=1.0")


def apply_play_assist_overrides(args: MutableMapping[str, object], plan: PlayModePlan) -> tuple[str, ...]:
    """Apply monolith play-mode policy-assist overrides to args"""
    if not plan.enabled:
        return ()
    args["policy_assist_mix"] = plan.forced_policy_assist_mix
    args["policy_assist_mix_floor"] = plan.forced_policy_assist_floor
    return ("policy_assist_mix", "policy_assist_mix_floor")


def play_eval_episode_override(args: MutableMapping[str, object], plan: PlayModePlan) -> int:
    """Set eval episodes to play episodes and return previous value"""
    previous = int(args.get("eval_episodes", 0))
    if plan.enabled:
        args["eval_episodes"] = int(plan.play_episodes)
    return previous


def restore_eval_episode_override(args: MutableMapping[str, object], previous_eval_episodes: int) -> None:
    """Restore eval episode count after play rollout"""
    args["eval_episodes"] = int(previous_eval_episodes)
