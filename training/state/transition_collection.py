"""

Per-env transition collection helpers

File map:

TEACHER_TRANSITION_SOURCES:    Define teacher transition sources constant
TransitionCollectionResult:    Rows appended to n-step queues
is_teacher_transition_source:  Return whether a transition should be marked as teacher-driven
append_step_transitions:       Append one vectorized env step into per-env n-step queues
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .replay import make_step_transition


TEACHER_TRANSITION_SOURCES = (
    "teacher_ik",
    "teacher_policy_arm",
    "policy_assist",
    "policy_relabel",
)


@dataclass(frozen=True)
class TransitionCollectionResult:
    """Rows appended to n-step queues"""

    appended_env_ids: tuple[int, ...]  # Field: integer appended env ids value tracked by transition collection result
    skipped_env_ids : tuple[int, ...]  # Field: integer skipped env ids value tracked by transition collection result
    is_teacher_value: float  # Field: floating-point is teacher value value used by transition collection result

    @property
    def appended_count(self) -> int:
        """Return appended row count"""
        return len(self.appended_env_ids)


def is_teacher_transition_source(
    *,
    teacher_action_present: bool,  # Param: boolean input controlling teacher action present
    action_source         : str,  # Param: source selector for action
) -> bool:
    """Return whether a transition should be marked as teacher-driven"""
    return bool(teacher_action_present) and str(action_source) in TEACHER_TRANSITION_SOURCES


def append_step_transitions(
    *,
    n_step_queues              : Sequence[deque],  # Param: ordered input collection of n step queues entries
    preroll_mask_before        : torch.Tensor,  # Param: tensor input carrying preroll mask before values
    obs_tensor                 : torch.Tensor,  # Param: policy observation tensor used by actor or replay logic
    replay_action              : torch.Tensor,  # Param: tensor input carrying replay action values
    bc_action                  : torch.Tensor,  # Param: behavior-cloning target action
    reward_tensor              : torch.Tensor,  # Param: tensor containing reward values
    replay_next_obs_tensor     : torch.Tensor,  # Param: tensor containing replay next obs values
    terminated_flags           : torch.Tensor,  # Param: flag values describing terminated
    timeout_flags              : torch.Tensor,  # Param: per-env timeout flags returned by the latest env step
    priv_obs_tensor            : torch.Tensor | None,  # Param: privileged observation tensor used by critic-side logic
    replay_next_priv_obs_tensor: torch.Tensor | None,  # Param: tensor containing replay next priv obs values
    teacher_action_present     : bool,  # Param: boolean input controlling teacher action present
    action_source              : str,  # Param: source selector for action
) -> TransitionCollectionResult:
    """Append one vectorized env step into per-env n-step queues

    Steps:
    - Resolve inputs for `append_step_transitions` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    is_teacher = is_teacher_transition_source(
        teacher_action_present=teacher_action_present,
        action_source=action_source,
    )
    is_teacher_value = 1.0 if is_teacher else 0.0
    appended: list[int] = []
    skipped : list[int] = []
    preroll_mask = preroll_mask_before.to(dtype=torch.bool).reshape(-1)
    for env_id in range(int(obs_tensor.shape[0])):
        if bool(preroll_mask[env_id].item()):
            skipped.append(env_id)
            continue
        n_step_queues[env_id].append(
            make_step_transition(
                obs=obs_tensor[env_id],
                action=replay_action[env_id],
                bc_action=bc_action[env_id],
                reward=reward_tensor[env_id],
                next_obs=replay_next_obs_tensor[env_id],
                terminated=bool(terminated_flags[env_id].item()),
                timeout=bool(timeout_flags[env_id].item()),
                is_teacher=is_teacher_value,
                priv_obs=None if priv_obs_tensor is None else priv_obs_tensor[env_id],
                next_priv_obs=(
                    None
                    if replay_next_priv_obs_tensor is None
                    else replay_next_priv_obs_tensor[env_id]
                ),
            )
        )
        appended.append(env_id)
    return TransitionCollectionResult(
        appended_env_ids=tuple(appended),
        skipped_env_ids=tuple(skipped),
        is_teacher_value=is_teacher_value,
    )
