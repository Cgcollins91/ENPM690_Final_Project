"""

Torch networks and finite-value diagnostics for trainer refactor modules

File map:

MLP:                                        Small fully connected backbone shared by actor and critics
Actor:                                      Policy network that maps observations to bounded reduced actions
Critic:                                     Q network used by TD3 for one state-action value estimate
FrozenPhase1PolicyTeacher:                  Frozen actor used only for policy-arm teacher actions
finite_fraction_value:                      Return the fraction of finite tensor values
finite_mean_value:                          Return the mean over finite tensor values
finite_std_value:                           Return population std over finite tensor values
module_param_finite_fraction:               Return the fraction of finite module parameters
ENV_COUNT_STATE_KEYS:                       Define env count state keys constant
load_module_state_allow_env_count_buffers:  Load module weights while ignoring env-count buffers
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small fully connected backbone shared by actor and critics"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network forward pass"""
        return self.net(x)


class Actor(nn.Module):
    """Policy network that maps observations to bounded reduced actions"""

    FINAL_LAYER_INIT = 3e-3

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        """Process for `__init__`

        Steps:
        - Resolve inputs for `__init__` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        super().__init__()
        self.mlp = MLP(obs_dim, hidden_dim, action_dim)
        final_linear = self.mlp.net[-1]
        nn.init.uniform_(final_linear.weight, -self.FINAL_LAYER_INIT, self.FINAL_LAYER_INIT)
        nn.init.uniform_(final_linear.bias, -self.FINAL_LAYER_INIT, self.FINAL_LAYER_INIT)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run the actor forward pass"""
        return torch.tanh(self.mlp(obs))

    def forward_with_raw(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded and pre-tanh action values"""
        raw = self.mlp(obs)
        return torch.tanh(raw), raw


class Critic(nn.Module):
    """Q network used by TD3 for one state-action value estimate"""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.mlp = MLP(obs_dim + action_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Run the critic forward pass"""
        return self.mlp(torch.cat([obs, action], dim=-1))


@dataclass
class FrozenPhase1PolicyTeacher:
    """Frozen actor used only for policy-arm teacher actions"""

    actor        : Actor  # actor network or actor checkpoint payload
    obs_dim      : int  # width of the policy observation vector
    action_cols  : tuple[int, ...]  # integer action cols value tracked by frozen phase1 policy teacher
    obs_mean     : torch.Tensor | None  # tensor containing obs mean values for batched env rows
    obs_std      : torch.Tensor | None  # tensor containing obs std values for batched env rows
    obs_norm_eps : float  # floating-point obs norm eps value used by frozen phase1 policy teacher
    obs_norm_clip: float  # floating-point obs norm clip value used by frozen phase1 policy teacher
    obs_keys     : tuple[str, ...]  # ordered keys used to resolve obs values

    def normalize_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize observations using stored running stats

        Steps:
        - Resolve inputs for `normalize_obs` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if obs_tensor.shape[-1] < self.obs_dim:
            raise RuntimeError(
                f"current observation width {obs_tensor.shape[-1]} is smaller than "
                f"Phase 1 actor width {self.obs_dim}"
            )
        obs = obs_tensor[:, : self.obs_dim]
        if self.obs_mean is None or self.obs_std is None:
            return obs
        normalized = (obs - self.obs_mean.to(obs.device)) / self.obs_std.to(obs.device).clamp(
            min=self.obs_norm_eps
        )
        if self.obs_norm_clip > 0.0:
            normalized = normalized.clamp(-self.obs_norm_clip, self.obs_norm_clip)
        return normalized

    def arm_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Return selected arm dimensions from the actor output"""
        with torch.no_grad():
            action = self.actor(self.normalize_obs(obs_tensor)).clamp(-1.0, 1.0)
        return action[:, list(self.action_cols)]


def finite_fraction_value(tensor: torch.Tensor) -> float:
    """Return the fraction of finite tensor values"""
    if tensor.numel() == 0:
        return math.nan
    return float(torch.isfinite(tensor).float().mean().item())


def finite_mean_value(tensor: torch.Tensor) -> float:
    """Return the mean over finite tensor values"""
    finite = torch.isfinite(tensor)
    if not bool(finite.any().item()):
        return math.nan
    return float(tensor[finite].mean().item())


def finite_std_value(tensor: torch.Tensor) -> float:
    """Return population std over finite tensor values

    Steps:
    - Resolve inputs for `finite_std_value` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    finite = torch.isfinite(tensor)
    if not bool(finite.any().item()):
        return math.nan
    values = tensor[finite]
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def module_param_finite_fraction(module: nn.Module) -> float:
    """Return the fraction of finite module parameters

    Steps:
    - Resolve inputs for `module_param_finite_fraction` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    total = 0
    finite = 0
    for param in module.parameters():
        data = param.detach()
        total += data.numel()
        finite += int(torch.isfinite(data).sum().item())
    if total == 0:
        return math.nan
    return float(finite) / float(total)


ENV_COUNT_STATE_KEYS = frozenset({"noise_scales"})


def load_module_state_allow_env_count_buffers(
    module: nn.Module,  # Param: input value used as module
    state : dict,  # Param: mutable or immutable runtime state read by this helper
    *,
    context: str,       # Param: runtime context carrying validated trainer settings
) -> None:
    """Load module weights while ignoring env-count buffers

    Steps:
    - Resolve inputs for `load_module_state_allow_env_count_buffers` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    current = module.state_dict()
    filtered = dict(state)
    ignored: list[str] = []
    for key in list(filtered):
        if key not in ENV_COUNT_STATE_KEYS:
            continue
        current_value = current.get(key)
        checkpoint_value = filtered[key]
        if (
            torch.is_tensor(current_value)
            and torch.is_tensor(checkpoint_value)
            and tuple(current_value.shape) != tuple(checkpoint_value.shape)
        ):
            ignored.append(key)
            del filtered[key]

    result = module.load_state_dict(filtered, strict=False)
    allowed_missing = {key for key in ENV_COUNT_STATE_KEYS if key in current}
    missing = set(result.missing_keys) - allowed_missing
    unexpected = set(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"{context} state_dict mismatch after env-buffer filtering: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    if ignored:
        print(
            f"{context}: ignored env-count-specific checkpoint buffer(s) {tuple(ignored)}",
            flush=True,
        )
