"""

Replay resume and handoff checkpoint startup helpers

File map:

ReplayStateTarget:               Replay buffer surface needed for startup restore
AgentStateTarget:                Agent surface needed for handoff reuse
ReplayResumeResult:              Outcome of replay resume from an existing checkpoint
HandoffReuseResult:              Outcome of optional handoff checkpoint reuse
replay_state_from_checkpoint:    Return replay state dict or raise with context
apply_resume_replay:             Load replay from a resume checkpoint after compatibility check
apply_handoff_checkpoint_reuse:  Try to reuse replay-inclusive handoff checkpoint state
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .handoff import handoff_compatibility_mismatch, replay_resume_compatibility_mismatch


class ReplayStateTarget(Protocol):
    """Replay buffer surface needed for startup restore"""

    size : int  # integer size value tracked by replay state target

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load replay buffer state"""
        ...


class AgentStateTarget(Protocol):
    """Agent surface needed for handoff reuse"""

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load full agent state"""
        ...


@dataclass(frozen=True)
class ReplayResumeResult:
    """Outcome of replay resume from an existing checkpoint"""

    loaded            : bool  # boolean value indicating the loaded state for replay resume result
    replay_size       : int  # configured or observed replay-buffer size
    source_global_step: int  # step count used for source global step scheduling or reporting


@dataclass(frozen=True)
class HandoffReuseResult:
    """Outcome of optional handoff checkpoint reuse"""

    reused                   : bool  # boolean value indicating the reused state for handoff reuse result
    stale                    : bool  # boolean value indicating the stale state for handoff reuse result
    unusable                 : bool  # boolean value indicating the unusable state for handoff reuse result
    reason                   : str | None  # string reason value used by handoff reuse result
    transitions_collected    : int  # number of replay transitions collected so far
    replay_size              : int  # configured or observed replay-buffer size
    skip_training_after_reuse: bool  # boolean value indicating the skip training after reuse state for handoff reuse result


def replay_state_from_checkpoint(checkpoint: Mapping[str, object], *, context: str) -> Mapping[str, object]:
    """Return replay state dict or raise with context"""
    replay_state = checkpoint.get("replay")
    if not isinstance(replay_state, Mapping):
        raise RuntimeError(f"{context} checkpoint does not contain replay state")
    return replay_state


def apply_resume_replay(
    replay                       : ReplayStateTarget,  # Param: replay buffer or replay target used for transition storage
    checkpoint                   : Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    current_handoff_compatibility: Mapping[str, object],  # Param: string input for current handoff compatibility
    *,
    compatibility_checker: Callable[[Mapping[str, object], Mapping[str, object]], str | None] = (  # Param: callback used to compute or fetch compatibility checker
        replay_resume_compatibility_mismatch
    ),
) -> ReplayResumeResult:
    """Load replay from a resume checkpoint after compatibility check"""
    mismatch = compatibility_checker(checkpoint, current_handoff_compatibility)
    if mismatch is not None:
        raise RuntimeError(
            "resume checkpoint replay is schema-incompatible: "
            f"{mismatch}. Use a checkpoint from the same task/action/observation schema."
        )
    replay.load_state_dict(replay_state_from_checkpoint(checkpoint, context="resume"))
    return ReplayResumeResult(
        loaded=True,
        replay_size=int(replay.size),
        source_global_step=max(0, int(checkpoint.get("global_step", 0))),
    )


def apply_handoff_checkpoint_reuse(
    *,
    agent                        : AgentStateTarget,  # Param: TD3 agent whose networks, optimizers, or stats are used
    replay                       : ReplayStateTarget,  # Param: replay buffer or replay target used for transition storage
    checkpoint                   : Mapping[str, object],  # Param: checkpoint payload or path being loaded or saved
    current_handoff_compatibility: Mapping[str, object],  # Param: string input for current handoff compatibility
    ignore_source_hashes         : bool,  # Param: boolean input controlling ignore source hashes
    stop_after_handoff_checkpoint: bool,  # Param: boolean input controlling stop after handoff checkpoint
    compatibility_checker        : Callable[..., str | None] = handoff_compatibility_mismatch,  # Param: callback used to compute or fetch compatibility checker
) -> HandoffReuseResult:
    """Try to reuse replay-inclusive handoff checkpoint state

    Steps:
    - Resolve inputs for `apply_handoff_checkpoint_reuse` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    mismatch = compatibility_checker(
        checkpoint,
        current_handoff_compatibility,
        ignore_source_hashes=bool(ignore_source_hashes),
    )
    if mismatch is not None:
        return HandoffReuseResult(
            reused=False,
            stale=True,
            unusable=False,
            reason=mismatch,
            transitions_collected=0,
            replay_size=0,
            skip_training_after_reuse=False,
        )
    replay_state = checkpoint.get("replay")
    if not isinstance(replay_state, Mapping):
        return HandoffReuseResult(
            reused=False,
            stale=False,
            unusable=True,
            reason="no_replay",
            transitions_collected=0,
            replay_size=0,
            skip_training_after_reuse=False,
        )
    agent_state = checkpoint.get("agent")
    if not isinstance(agent_state, Mapping):
        return HandoffReuseResult(
            reused=False,
            stale=False,
            unusable=True,
            reason="no_agent",
            transitions_collected=0,
            replay_size=0,
            skip_training_after_reuse=False,
        )
    agent.load_state_dict(agent_state)
    replay.load_state_dict(replay_state)
    return HandoffReuseResult(
        reused=True,
        stale=False,
        unusable=False,
        reason=None,
        transitions_collected=max(0, int(checkpoint.get("global_step", 0))),
        replay_size=int(replay.size),
        skip_training_after_reuse=bool(stop_after_handoff_checkpoint),
    )
