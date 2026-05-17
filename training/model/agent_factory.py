"""

TD3 backend selection helpers

File map:

make_td3_agent:  Build the configured TD3 learner backend
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .agents import FastTD3Agent, TD3Config


def make_td3_agent(
    *,
    td3_backend     : str,  # Param: string input for td3 backend
    obs_dim         : int,  # Param: integer input for obs dim
    action_dim      : int,  # Param: integer input for action dim
    device          : torch.device | str,  # Param: torch device where tensors are read or allocated
    priv_obs_dim    : int                                                          = 0,  # Param: integer input for priv obs dim
    custom_config   : TD3Config | None                                             = None,  # Param: input value used as custom config
    upstream_factory: Callable[[int, int, torch.device | str, int], object] | None = None,  # Param: callback used to compute or fetch upstream factory
) -> object:
    """Build the configured TD3 learner backend"""
    if str(td3_backend) == "custom":
        return FastTD3Agent(
            obs_dim,
            action_dim,
            device,
            config=custom_config,
            priv_obs_dim=priv_obs_dim,
        )
    if str(td3_backend) == "upstream_fasttd3":
        if upstream_factory is None:
            raise RuntimeError("td3_backend=upstream_fasttd3 requires an upstream_factory")
        return upstream_factory(obs_dim, action_dim, device, priv_obs_dim)
    raise RuntimeError(f"unsupported td3_backend={td3_backend!r}")
