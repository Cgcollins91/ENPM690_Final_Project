"""

Fingertip to block geometry helpers

File map:

THUMB_TIP_LINK:                 Define thumb tip link constant
INDEX_TIP_LINK:                 Define index tip link constant
SENTINEL_DELTA:                 Define sentinel delta constant
_link_index:                    Handle link index logic
tip_to_block_geometry:          Return thumb and index distances and deltas to block center
tip_to_block_geometry_tensors:  Return vectorized thumb and index distances and deltas to block center
"""

from __future__ import annotations

from collections.abc import Callable

import torch


THUMB_TIP_LINK = "right_hand_thumb_2_link"
INDEX_TIP_LINK = "right_hand_index_1_link"
SENTINEL_DELTA = (-1.0, -1.0, -1.0)


def _link_index(body_names: list[str] | tuple[str, ...], link_name: str) -> int | None:
    try:
        return body_names.index(link_name)
    except ValueError:
        return None


def tip_to_block_geometry(
    env,                               # Param: environment or backend object used for runtime calls
    env_id: int = 0,                   # Param: integer input for env id
    *,
    thumb_link: str = THUMB_TIP_LINK,  # Param: string input for thumb link
    index_link: str = INDEX_TIP_LINK,  # Param: string input for index link
) -> tuple[float, float, tuple[float, float, float], tuple[float, float, float]]:
    """Return thumb and index distances and deltas to block center

    Steps:
    - Resolve inputs for `tip_to_block_geometry` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    robot = env.scene["robot"]
    body_names = robot.data.body_names
    thumb_idx = _link_index(body_names, thumb_link)
    index_idx = _link_index(body_names, index_link)
    if thumb_idx is None or index_idx is None:
        return -1.0, -1.0, SENTINEL_DELTA, SENTINEL_DELTA
    pose = robot.data.body_link_pose_w[int(env_id)]
    block_pos = env.scene["object"].data.root_pos_w[int(env_id), :3]
    thumb_delta = pose[thumb_idx, :3] - block_pos
    index_delta = pose[index_idx, :3] - block_pos
    return (
        float(torch.linalg.norm(thumb_delta).item()),
        float(torch.linalg.norm(index_delta).item()),
        tuple(float(x) for x in thumb_delta.detach().cpu().tolist()),
        tuple(float(x) for x in index_delta.detach().cpu().tolist()),
    )


def tip_to_block_geometry_tensors(
    env,                                                                 # Param: environment or backend object used for runtime calls
    *,
    block_positions   : torch.Tensor | None                     = None,  # Param: tensor input carrying block positions values
    block_positions_fn: Callable[[object], torch.Tensor] | None = None,  # Param: callback used to compute or fetch block positions
    thumb_link        : str                                     = THUMB_TIP_LINK,  # Param: string input for thumb link
    index_link        : str                                     = INDEX_TIP_LINK,  # Param: string input for index link
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return vectorized thumb and index distances and deltas to block center

    Steps:
    - Resolve inputs for `tip_to_block_geometry_tensors` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    robot = env.scene["robot"]
    body_names = robot.data.body_names
    thumb_idx = _link_index(body_names, thumb_link)
    index_idx = _link_index(body_names, index_link)
    if thumb_idx is None or index_idx is None:
        zeros = torch.zeros((env.num_envs,), device=env.device)
        zero_vec = torch.zeros((env.num_envs, 3), device=env.device)
        return zeros, zeros, zero_vec, zero_vec
    if block_positions is None and block_positions_fn is not None:
        block_positions = block_positions_fn(env)
    if block_positions is None:
        block_positions = env.scene["object"].data.root_pos_w[:, :3]
    pose = robot.data.body_link_pose_w
    thumb_delta = pose[:, thumb_idx, :3] - block_positions.to(device=pose.device, dtype=pose.dtype)
    index_delta = pose[:, index_idx, :3] - block_positions.to(device=pose.device, dtype=pose.dtype)
    return (
        torch.linalg.norm(thumb_delta, dim=-1),
        torch.linalg.norm(index_delta, dim=-1),
        thumb_delta,
        index_delta,
    )
