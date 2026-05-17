"""

Evaluation tensor formatting and fallback helpers

File map:

eval_cpu_float_list:      Return a detached CPU float list
eval_cpu_int_list:        Return a detached CPU int list
eval_cpu_vec3_list:       Return detached CPU vec3 rows
eval_bool_rate:           Return mean rate for bool-like tensor values
eval_masked_min:          Return current updated with candidate minima on masked rows
eval_masked_max:          Return current updated with candidate maxima on masked rows
eval_stage_tensor:        Return topdown stage tensor or a default env-sized tensor
eval_unlock_tensor:       Return finger unlock progress tensor or zeros
eval_scalar_attr_tensor:  Read one env-sized scalar diagnostic tensor
eval_vec3_attr_tensor:    Read one env-sized vec3 diagnostic tensor
"""

from __future__ import annotations

import torch


def eval_cpu_float_list(tensor: torch.Tensor) -> list[float]:
    """Return a detached CPU float list"""
    return [float(x) for x in tensor.detach().float().cpu().tolist()]


def eval_cpu_int_list(tensor: torch.Tensor) -> list[int]:
    """Return a detached CPU int list"""
    return [int(x) for x in tensor.detach().cpu().tolist()]


def eval_cpu_vec3_list(tensor: torch.Tensor) -> list[list[float]]:
    """Return detached CPU vec3 rows"""
    return [[float(v) for v in row[:3]] for row in tensor.detach().float().cpu().tolist()]


def eval_bool_rate(tensor: torch.Tensor) -> float:
    """Return mean rate for bool-like tensor values"""
    return float(tensor.to(dtype=torch.float32).mean().item()) if tensor.numel() > 0 else 0.0


def eval_masked_min(current: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return current updated with candidate minima on masked rows"""
    return torch.where(
        mask,
        torch.minimum(current, candidate.to(device=current.device, dtype=current.dtype)),
        current,
    )


def eval_masked_max(current: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return current updated with candidate maxima on masked rows"""
    return torch.where(
        mask,
        torch.maximum(current, candidate.to(device=current.device, dtype=current.dtype)),
        current,
    )


def eval_stage_tensor(env, default: int = -1) -> torch.Tensor:
    """Return topdown stage tensor or a default env-sized tensor"""
    value = getattr(env, "_topdown_stage", None)
    if torch.is_tensor(value) and value.shape[0] == env.num_envs:
        return value.to(device=env.device, dtype=torch.long)
    return torch.full((env.num_envs,), int(default), device=env.device, dtype=torch.long)


def eval_unlock_tensor(env) -> torch.Tensor:
    """Return finger unlock progress tensor or zeros"""
    value = getattr(env, "_topdown_finger_unlock_progress", None)
    if torch.is_tensor(value) and value.shape[0] == env.num_envs:
        return value.to(device=env.device, dtype=torch.float32)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def eval_scalar_attr_tensor(env, attr_name: str, default: float = float("nan")) -> torch.Tensor:
    """Read one env-sized scalar diagnostic tensor"""
    value = getattr(env, attr_name, None)
    if torch.is_tensor(value) and value.shape[0] == env.num_envs:
        return value.detach().reshape(env.num_envs, -1)[:, 0].to(
            device=env.device,
            dtype=torch.float32,
        )
    return torch.full((env.num_envs,), float(default), device=env.device, dtype=torch.float32)


def eval_vec3_attr_tensor(env, attr_name: str, default: float = float("nan")) -> torch.Tensor:
    """Read one env-sized vec3 diagnostic tensor"""
    value = getattr(env, attr_name, None)
    if torch.is_tensor(value) and value.shape[0] == env.num_envs:
        flat = value.detach().reshape(env.num_envs, -1)
        if flat.shape[1] >= 3:
            return flat[:, :3].to(device=env.device, dtype=torch.float32)
    return torch.full((env.num_envs, 3), float(default), device=env.device, dtype=torch.float32)
