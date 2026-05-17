"""

Native env reset and first-observation contract helpers

File map:

NativeResetPayload:                 Normalized result from env reset
NativeResetRequest:                 Inputs for native env reset contract resolution
NativeResetResult:                  Initial obs and resolved startup observation contract
normalize_native_env_reset_result:  Normalize common env reset result shapes
reset_native_env:                   Reset env and resolve the first-observation contract
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..env.startup import StartupObservationContract, resolve_startup_observation_contract


EnvResetFn = Callable[[], object]


@dataclass(frozen=True)
class NativeResetPayload:
    """Normalized result from env reset"""

    obs : Mapping[str, object]  # policy observation tensor or observation payload for this transition
    info: object  # auxiliary info mapping returned by the environment or backend


@dataclass(frozen=True)
class NativeResetRequest:
    """Inputs for native env reset contract resolution"""

    env_reset_fn      : EnvResetFn  # callback used for the env reset fn operation
    privileged_critic : bool  # boolean value indicating the privileged critic state for native reset request
    topdown_curriculum: bool = True  # boolean value indicating the topdown curriculum state for native reset request


@dataclass(frozen=True)
class NativeResetResult:
    """Initial obs and resolved startup observation contract"""

    obs        : Mapping[str, object]  # policy observation tensor or observation payload for this transition
    info       : object  # auxiliary info mapping returned by the environment or backend
    observation: StartupObservationContract  # stores observation for native reset result


def normalize_native_env_reset_result(raw: object) -> NativeResetPayload:
    """Normalize common env reset result shapes"""
    if isinstance(raw, NativeResetPayload):
        return raw
    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise TypeError(f"env reset tuple must have length 2, got {len(raw)}")
        obs, info = raw
    else:
        obs = raw
        info = {}
    if not isinstance(obs, Mapping):
        raise TypeError("env reset obs must be a mapping")
    return NativeResetPayload(obs=obs, info=info)


def reset_native_env(request: NativeResetRequest) -> NativeResetResult:
    """Reset env and resolve the first-observation contract"""
    payload = normalize_native_env_reset_result(request.env_reset_fn())
    observation = resolve_startup_observation_contract(
        payload.obs,
        privileged_critic=request.privileged_critic,
        topdown_curriculum=request.topdown_curriculum,
    )
    return NativeResetResult(obs=payload.obs, info=payload.info, observation=observation)
