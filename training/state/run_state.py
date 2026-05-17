"""

Initial rollout-loop state assembly

File map:

PrerollRuntimeState:            Mutable contact pre-roll counters and windows
TrainingLoopStartupState:       Mutable state initialized before the rollout loop
initial_n_step_queues:          Create one n-step transition queue per env row
eval_every_from_settings:       Return automatic eval cadence or None when eval is disabled
preroll_heartbeat_every:        Return pre-roll heartbeat cadence from max step budget
initial_preroll_runtime_state:  Create pre-roll active rows and counters
initial_training_loop_state:    Create rollout-loop state that does not require Isaac objects
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from .cadence import TrainingCadence, initial_training_cadence
from .episodes import EpisodeState
from ..eval.eval_metrics import initial_best_eval_state


@dataclass
class PrerollRuntimeState:
    """Mutable contact pre-roll counters and windows"""

    active              : torch.Tensor  # Field: whether this configuration or runtime path is active
    steps               : torch.Tensor  # Field: tensor containing steps values for batched env rows
    release_steps_window: deque[int] = field(default_factory=lambda: deque(maxlen=1000))  # Field: integer release steps window value tracked by preroll runtime state
    forced_window       : deque[int] = field(default_factory=lambda: deque(maxlen=1000))  # Field: integer forced window value tracked by preroll runtime state
    release_total       : int        = 0  # Field: integer release total value tracked by preroll runtime state
    timeout_total       : int        = 0  # Field: integer timeout total value tracked by preroll runtime state
    next_summary        : int        = 1000  # Field: integer next summary value tracked by preroll runtime state
    heartbeat_every     : int        = 25  # Field: integer heartbeat every value tracked by preroll runtime state

    @property
    def active_count(self) -> int:
        """Return active pre-roll row count"""
        return int(self.active.to(dtype=torch.bool).sum().item())


@dataclass
class TrainingLoopStartupState:
    """Mutable state initialized before the rollout loop"""

    episode                          : EpisodeState  # Field: stores episode for training loop startup state
    n_step_queues                    : list[deque]  # Field: ordered collection of n step queues entries for training loop startup state
    preroll                          : PrerollRuntimeState  # Field: stores preroll for training loop startup state
    best_eval_state                  : dict[str, float]  # Field: floating-point best eval state value used by training loop startup state
    cadence                          : TrainingCadence  # Field: stores cadence for training loop startup state
    transitions_collected            : int                      = 0  # Field: number of replay transitions collected so far
    auto_handoff_loaded              : bool                     = False  # Field: boolean value indicating the auto handoff loaded state for training loop startup state
    skip_training_after_handoff_reuse: bool                     = False  # Field: boolean value indicating the skip training after handoff reuse state for training loop startup state
    last_update_info                 : dict[str, object] | None = None  # Field: string last update info value used by training loop startup state
    last_actor_update_info           : dict[str, object] | None = None  # Field: string last actor update info value used by training loop startup state


def initial_n_step_queues(num_envs: int) -> list[deque]:
    """Create one n-step transition queue per env row"""
    return [deque() for _ in range(int(num_envs))]


def eval_every_from_settings(
    *,
    num_envs          : int,  # Param: number of parallel environment rows represented
    max_episode_length: int,  # Param: integer input for max episode length
    eval_steps        : int,  # Param: step count used for eval steps
    eval_episodes     : int,  # Param: integer input for eval episodes
    eval_every        : int = 0,  # Param: explicit transition interval between eval runs; 0 keeps automatic cadence
) -> int | None:
    """Return automatic eval cadence or None when eval is disabled"""
    if int(eval_steps) <= 0 or int(eval_episodes) <= 0:
        return None
    if int(eval_every) > 0:
        return int(eval_every)
    return max(1, int(num_envs) * int(max_episode_length))


def preroll_heartbeat_every(max_steps: int) -> int:
    """Return pre-roll heartbeat cadence from max step budget"""
    return max(25, min(100, max(1, int(max_steps)) // 5))


def initial_preroll_runtime_state(
    *,
    num_envs                      : int,  # Param: number of parallel environment rows represented
    device                        : torch.device | str,  # Param: torch device where tensors are read or allocated
    legacy_contact_preroll_enabled: bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled       : bool,  # Param: enables topdown-preroll reset handling
    topdown_preroll_mask          : torch.Tensor | None = None,  # Param: boolean mask selecting topdown preroll rows
    contact_preroll_max_steps     : int,  # Param: step count used for contact preroll max steps
    topdown_preroll_max_steps     : int,  # Param: step count used for topdown preroll max steps
) -> PrerollRuntimeState:
    """Create pre-roll active rows and counters"""
    active = torch.zeros(int(num_envs), device=device, dtype=torch.bool)
    if legacy_contact_preroll_enabled:
        active[:] = True
    elif topdown_preroll_enabled:
        if topdown_preroll_mask is None:
            raise ValueError("topdown_preroll_mask is required when topdown pre-roll is enabled")
        active[:] = topdown_preroll_mask.to(device=device, dtype=torch.bool)
    max_steps = int(topdown_preroll_max_steps) if topdown_preroll_enabled else int(contact_preroll_max_steps)
    return PrerollRuntimeState(
        active=active,
        steps=torch.zeros(int(num_envs), device=device, dtype=torch.long),
        heartbeat_every=preroll_heartbeat_every(max_steps),
    )


def initial_training_loop_state(
    *,
    num_envs                      : int,  # Param: number of parallel environment rows represented
    device                        : torch.device | str,  # Param: torch device where tensors are read or allocated
    max_episode_length            : int,  # Param: integer input for max episode length
    transitions_collected         : int,  # Param: integer input for transitions collected
    eval_steps                    : int,  # Param: step count used for eval steps
    eval_episodes                 : int,  # Param: integer input for eval episodes
    eval_start_steps              : int,  # Param: step count used for eval start steps
    checkpoint_every              : int,  # Param: global-step interval for regular checkpoint saves
    rolling_checkpoint_every      : int,  # Param: global-step interval for rolling checkpoint saves
    legacy_contact_preroll_enabled: bool,  # Param: enables legacy contact-preroll reset handling
    topdown_preroll_enabled       : bool,  # Param: enables topdown-preroll reset handling
    topdown_preroll_mask          : torch.Tensor | None,  # Param: boolean mask selecting topdown preroll rows
    contact_preroll_max_steps     : int,  # Param: step count used for contact preroll max steps
    topdown_preroll_max_steps     : int,  # Param: step count used for topdown preroll max steps
    eval_every                    : int = 0,  # Param: explicit transition interval between eval runs; 0 keeps automatic cadence
) -> TrainingLoopStartupState:
    """Create rollout-loop state that does not require Isaac objects"""
    eval_every = eval_every_from_settings(
        num_envs=num_envs,
        max_episode_length=max_episode_length,
        eval_every=eval_every,
        eval_steps=eval_steps,
        eval_episodes=eval_episodes,
    )
    return TrainingLoopStartupState(
        episode=EpisodeState.create(num_envs=num_envs, device=device),
        n_step_queues=initial_n_step_queues(num_envs),
        preroll=initial_preroll_runtime_state(
            num_envs=num_envs,
            device=device,
            legacy_contact_preroll_enabled=legacy_contact_preroll_enabled,
            topdown_preroll_enabled=topdown_preroll_enabled,
            topdown_preroll_mask=topdown_preroll_mask,
            contact_preroll_max_steps=contact_preroll_max_steps,
            topdown_preroll_max_steps=topdown_preroll_max_steps,
        ),
        best_eval_state=initial_best_eval_state(),
        cadence=initial_training_cadence(
            transitions_collected=transitions_collected,
            eval_every=eval_every,
            eval_start_steps=eval_start_steps,
            checkpoint_every=checkpoint_every,
            rolling_checkpoint_every=rolling_checkpoint_every,
        ),
        transitions_collected=max(0, int(transitions_collected)),
    )
