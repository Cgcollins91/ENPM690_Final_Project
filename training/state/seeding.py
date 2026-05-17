"""

Deterministic seeding helpers for trainer launches

File map:

_env_bool:             Handle env bool logic
SeedConfig:            Global RNG and deterministic backend settings
seed_config_from_env:  Resolve seed settings from environment overrides
set_global_seed:       Seed Python NumPy Torch and backend deterministic settings
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import random

import numpy as np
import torch


def _env_bool(name: str, default: bool, env: Mapping[str, str] | None) -> bool:
    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class SeedConfig:
    """Global RNG and deterministic backend settings"""

    seed                         : int  # integer seed value tracked by seed config
    include_cuda                 : bool = True  # boolean value indicating the include cuda state for seed config
    seed_numpy                   : bool = True  # boolean value indicating the seed numpy state for seed config
    cudnn_benchmark              : bool = False  # boolean value indicating the cudnn benchmark state for seed config
    cudnn_deterministic          : bool = True  # boolean value indicating the cudnn deterministic state for seed config
    torch_deterministic          : bool = True  # boolean value indicating the torch deterministic state for seed config
    torch_deterministic_warn_only: bool = True  # boolean value indicating the torch deterministic warn only state for seed config


def seed_config_from_env(
    seed: int,                             # Param: random seed used for reproducible setup
    *,
    env         : Mapping[str, str] | None = None,  # Param: environment or backend object used for runtime calls
    include_cuda: bool                     = True,  # Param: boolean input controlling include cuda
) -> SeedConfig:
    """Resolve seed settings from environment overrides"""
    return SeedConfig(
        seed=int(seed),
        include_cuda=bool(include_cuda),
        seed_numpy=_env_bool("ENPM690_NUMPY_SEED", True, env),
        cudnn_benchmark=_env_bool("ENPM690_CUDNN_BENCHMARK", False, env),
        cudnn_deterministic=_env_bool("ENPM690_CUDNN_DETERMINISTIC", True, env),
        torch_deterministic=_env_bool("ENPM690_TORCH_DETERMINISTIC", True, env),
        torch_deterministic_warn_only=_env_bool("ENPM690_TORCH_DETERMINISTIC_WARN_ONLY", True, env),
    )


def set_global_seed(config: SeedConfig | int, *, include_cuda: bool | None = None) -> SeedConfig:
    """Seed Python NumPy Torch and backend deterministic settings

    Steps:
    - Resolve inputs for `set_global_seed` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    seed_config = (
        seed_config_from_env(int(config), include_cuda=True if include_cuda is None else include_cuda)
        if isinstance(config, int)
        else config
    )
    random.seed(seed_config.seed)
    if seed_config.seed_numpy:
        np.random.seed(seed_config.seed)
    torch.manual_seed(seed_config.seed)
    torch.backends.cudnn.benchmark = bool(seed_config.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(seed_config.cudnn_deterministic)
    if seed_config.torch_deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=bool(seed_config.torch_deterministic_warn_only),
        )
    else:
        torch.use_deterministic_algorithms(False)
    if seed_config.include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_config.seed)
    return seed_config
