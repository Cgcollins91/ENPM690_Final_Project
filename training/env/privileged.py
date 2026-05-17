"""

Privileged critic observation helpers provide functions for refreshing teacher state and flattening privileged observations
when a privileged critic is enabled

File map:

T:                                 Define t constant
refresh_privileged_teacher_state:  Refresh teacher state when privileged critic inputs need it
flatten_current_privileged_obs:    Compute and flatten current privileged observations when enabled
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import torch

from .observations import flatten_privileged_obs

T = TypeVar("T")


def refresh_privileged_teacher_state(
    *,
    privileged_critic      : bool,  # Param: boolean input controlling privileged critic
    topdown_curriculum_task: bool,  # Param: boolean input controlling topdown curriculum task
    compute_teacher_action : Callable[[], T],  # Param: callback used to compute or fetch compute teacher action
) -> T | None:
    """Refresh teacher state when privileged critic inputs need it"""
    if not bool(privileged_critic) or not bool(topdown_curriculum_task):
        return None
    return compute_teacher_action()


def flatten_current_privileged_obs(
    env,                                                            # Param: environment or backend object used for runtime calls
    *,
    privileged_critic: bool,  # Param: boolean input controlling privileged critic
    obs_compute      : Callable[[], Mapping[str, object]] | None = None,  # Param: callback used to compute or fetch obs compute
) -> torch.Tensor | None:
    """Compute and flatten current privileged observations when enabled"""
    if not bool(privileged_critic):
        return None
    compute = env.observation_manager.compute if obs_compute is None else obs_compute
    return flatten_privileged_obs(compute())
