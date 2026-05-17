"""

Replay transition batch stacking helpers

File map:

ReplayTransitionBatch:            Stacked ready transitions for replay insertion
ReplayAddTarget:                  Replay-like object that accepts stacked transitions
_stack:                           Handle stack logic
stack_ready_transitions:          Stack ready n-step transition dictionaries for replay add
add_transition_batch_to_replay:   Add a stacked transition batch to replay and return inserted count
add_ready_transitions_to_replay:  Stack and add ready transitions to replay
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ReplayTransitionBatch:
    """Stacked ready transitions for replay insertion"""

    obs          : torch.Tensor  # policy observation tensor or observation payload for this transition
    action       : torch.Tensor  # environment action tensor selected for the step
    bc_action    : torch.Tensor  # behavior-cloning target action tensor
    reward       : torch.Tensor  # reward tensor or scalar produced by the environment step
    discount     : torch.Tensor  # tensor containing discount values for batched env rows
    next_obs     : torch.Tensor  # next policy observation tensor after the transition step
    terminated   : torch.Tensor  # tensor containing terminated values for batched env rows
    timeout      : torch.Tensor  # tensor containing timeout values for batched env rows
    is_teacher   : torch.Tensor  # tensor containing is teacher values for batched env rows
    priv_obs     : torch.Tensor | None  # privileged observation tensor used by critic/training code
    next_priv_obs: torch.Tensor | None  # tensor containing next priv obs values for batched env rows

    @property
    def size(self) -> int:
        """Return number of stacked transitions"""
        return int(self.obs.shape[0])


class ReplayAddTarget(Protocol):
    """Replay-like object that accepts stacked transitions"""

    def add(
        self,
        obs          : torch.Tensor,  # Param: observation payload returned by the environment or replay path
        action       : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
        bc_action    : torch.Tensor,  # Param: behavior-cloning target action
        reward       : torch.Tensor,  # Param: reward tensor or scalar from the transition
        discount     : torch.Tensor,  # Param: tensor input carrying discount values
        next_obs     : torch.Tensor,  # Param: next observation payload after the environment step
        terminated   : torch.Tensor,  # Param: tensor input carrying terminated values
        timeout      : torch.Tensor,  # Param: tensor input carrying timeout values
        is_teacher   : torch.Tensor,  # Param: boolean input indicating whether teacher is active
        priv_obs     : torch.Tensor | None = None,  # Param: tensor input carrying priv obs values
        next_priv_obs: torch.Tensor | None = None,  # Param: tensor input carrying next priv obs values
    ) -> None:
        """Add stacked transition tensors"""
        ...


def _stack(transitions: Sequence[Mapping[str, object]], key: str) -> torch.Tensor:
    return torch.stack([row[key] for row in transitions], dim=0)  # type: ignore[arg-type]


def stack_ready_transitions(
    transitions: Sequence[Mapping[str, object]],  # Param: string input for transitions
) -> ReplayTransitionBatch | None:
    """Stack ready n-step transition dictionaries for replay add"""
    if not transitions:
        return None
    priv_batch = (
        None
        if transitions[0]["priv_obs"] is None
        else torch.stack([row["priv_obs"] for row in transitions], dim=0)  # type: ignore[arg-type]
    )
    next_priv_batch = (
        None
        if transitions[0]["next_priv_obs"] is None
        else torch.stack([row["next_priv_obs"] for row in transitions], dim=0)  # type: ignore[arg-type]
    )
    return ReplayTransitionBatch(
        obs=_stack(transitions, "obs"),
        action=_stack(transitions, "action"),
        bc_action=_stack(transitions, "bc_action"),
        reward=_stack(transitions, "reward"),
        discount=_stack(transitions, "discount"),
        next_obs=_stack(transitions, "next_obs"),
        terminated=_stack(transitions, "terminated"),
        timeout=_stack(transitions, "timeout"),
        is_teacher=_stack(transitions, "is_teacher"),
        priv_obs=priv_batch,
        next_priv_obs=next_priv_batch,
    )


def add_transition_batch_to_replay(
    replay: ReplayAddTarget,  # Param: replay buffer or replay target used for transition storage
    batch : ReplayTransitionBatch | None,  # Param: input value used as batch
) -> int:
    """Add a stacked transition batch to replay and return inserted count"""
    if batch is None:
        return 0
    replay.add(
        obs=batch.obs,
        action=batch.action,
        bc_action=batch.bc_action,
        reward=batch.reward,
        discount=batch.discount,
        next_obs=batch.next_obs,
        terminated=batch.terminated,
        timeout=batch.timeout,
        is_teacher=batch.is_teacher,
        priv_obs=batch.priv_obs,
        next_priv_obs=batch.next_priv_obs,
    )
    return batch.size


def add_ready_transitions_to_replay(
    replay     : ReplayAddTarget,  # Param: replay buffer or replay target used for transition storage
    transitions: Sequence[Mapping[str, object]],  # Param: string input for transitions
) -> int:
    """Stack and add ready transitions to replay"""
    return add_transition_batch_to_replay(replay, stack_ready_transitions(transitions))
