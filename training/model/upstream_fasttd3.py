"""

Optional upstream FastTD3 import resolution helpers

File map:

_update_timing_enabled:                 Handle update timing enabled logic
_timing_mark:                           Handle timing mark logic
UpstreamFastTD3Classes:                 Actor and critic classes loaded from upstream FastTD3
UpstreamFastTD3Config:                  Constructor/runtime settings for the upstream FastTD3 backend
fasttd3_repo_module_path:               Return the fast_td3.py module path for a checkout
load_fasttd3_classes_from_repo:         Load upstream FastTD3 classes from an explicit checkout
load_fasttd3_classes_from_environment:  Load upstream FastTD3 classes from the active Python environment
load_upstream_fasttd3_classes:          Load upstream FastTD3 classes from repo override or environment
UpstreamFastTD3Agent:                   Teacher/DAgger-compatible agent backed by upstream FastTD3 networks
make_upstream_fasttd3_agent:            Build the upstream FastTD3 agent wrapper
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import math
import os
import time

import torch
import torch.nn.functional as F

from ..actions.action_gates import (
    add_post_unlock_finger_noise,
    apply_contact_finger_close_cap,
    apply_curriculum_finger_unlock_from_flat_obs,
    make_per_dim_noise_sigma,
    objective_action_from_gate_mode,
)
from ..actions.losses import weighted_bc_loss
from ..actions.schedules import scheduled_teacher_bc_weights, teacher_bc_requested
from .agents import TD3Config
from .networks import (
    finite_fraction_value,
    finite_mean_value,
    finite_std_value,
    load_module_state_allow_env_count_buffers,
)
from .normalization import RunningScalarMoments, RunningTensorMoments


def _update_timing_enabled() -> bool:
    return os.environ.get("NATIVE_UPDATE_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}


def _timing_mark(enabled: bool, train_step: int, label: str, start: float) -> float:
    now = time.perf_counter()
    if enabled:
        print(
            f"native_update_timing train_step={int(train_step)} label={label} dt={now - start:.3f}",
            flush=True,
        )
    return now


@dataclass(frozen=True)
class UpstreamFastTD3Classes:
    """Actor and critic classes loaded from upstream FastTD3"""

    actor_cls : type  # stores actor cls for upstream fast t d3 classes
    critic_cls: type  # stores critic cls for upstream fast t d3 classes


@dataclass(frozen=True)
class UpstreamFastTD3Config:
    """Constructor/runtime settings for the upstream FastTD3 backend"""

    init_scale       : float = 0.01     # actor init scale used by upstream FastTD3
    actor_hidden_dim : int   = 512      # actor hidden width used by upstream FastTD3
    critic_hidden_dim: int   = 1024     # critic hidden width used by upstream FastTD3
    std_min          : float = 0.001    # minimum actor std used by upstream FastTD3
    std_max          : float = 0.4      # maximum actor std used by upstream FastTD3
    num_atoms        : int   = 51       # critic categorical support atom count
    v_min            : float = -5.0     # critic categorical support minimum
    v_max            : float = 0.0      # critic categorical support maximum
    weight_decay     : float = 0.0      # AdamW weight decay for upstream backend
    use_cdq          : bool  = True     # use clipped double-Q distribution selection
    num_envs         : int   = 1        # environment count for upstream actor buffers
    fasttd3_repo     : str   = ""       # optional upstream FastTD3 checkout path
    sim_type         : str   = ""       # simulator feature encoder mode used by upstream FastTD3
    sim_dimension    : int   = 64       # simulator feature width used by upstream FastTD3
    seq_len          : int   = 8        # simulator sequence length used by upstream FastTD3


def _ensure_expected_fasttd3_api(classes: UpstreamFastTD3Classes, *, source: str) -> None:
    """
    Reject FastTD3 packages other than the checkpoint-compatible GitHub API shape
    """
    try:
        actor_params = inspect.signature(classes.actor_cls.__init__).parameters
        critic_params = inspect.signature(classes.critic_cls.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"could not inspect FastTD3 API from {source}") from exc
    actor_required = {"n_obs", "n_act", "num_envs", "init_scale", "hidden_dim", "std_min", "std_max", "device"}
    critic_required = {"n_obs", "n_act", "num_atoms", "v_min", "v_max", "hidden_dim", "device"}
    simulator_params = {"sim_type", "sim_dimension", "seq_len"}
    if (
        not actor_required.issubset(actor_params)
        or not critic_required.issubset(critic_params)
        or simulator_params.intersection(actor_params)
        or simulator_params.intersection(critic_params)
    ):
        raise RuntimeError(
            "unsupported FastTD3 API detected. This project expects the checkpoint-compatible GitHub API "
            "from requirements.txt: "
            "fast_td3 @ git+https://github.com/younggyoseo/FastTD3.git@7acc3a3c739d2beaae57386407acfb29ee3928fa. "
            "Reinstall dependencies in the Isaac environment or set FASTTD3_REPO to that checkout."
        )


def fasttd3_repo_module_path(fasttd3_repo: str) -> str:
    """Return the fast_td3.py module path for a checkout"""
    repo_path = os.path.abspath(os.path.expanduser(fasttd3_repo))
    return os.path.join(repo_path, "fast_td3", "fast_td3.py")


def load_fasttd3_classes_from_repo(fasttd3_repo: str) -> UpstreamFastTD3Classes:
    """Load upstream FastTD3 classes from an explicit checkout

    Steps:
    - Resolve inputs for `load_fasttd3_classes_from_repo` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    module_path = fasttd3_repo_module_path(fasttd3_repo)
    if not os.path.isfile(module_path):
        repo_path = os.path.abspath(os.path.expanduser(fasttd3_repo))
        raise RuntimeError(
            f"FASTTD3_REPO does not look like an upstream FastTD3 checkout: {repo_path}"
        )
    spec = importlib.util.spec_from_file_location("_upstream_fasttd3_fast_td3", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream FastTD3 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = UpstreamFastTD3Classes(actor_cls=module.Actor, critic_cls=module.Critic)
    _ensure_expected_fasttd3_api(classes, source=module_path)
    return classes


def load_fasttd3_classes_from_environment() -> UpstreamFastTD3Classes:
    """Load upstream FastTD3 classes from the active Python environment"""
    try:
        module = importlib.import_module("fast_td3.fast_td3")
    except ImportError as exc:
        raise RuntimeError(
            "td3_backend=upstream_fasttd3 requires the upstream FastTD3 package. "
            "Install it in the IsaacSim Python environment with "
            "`pip install git+https://github.com/younggyoseo/FastTD3.git`, "
            "or set FASTTD3_REPO=/path/to/FastTD3."
        ) from exc
    classes = UpstreamFastTD3Classes(actor_cls=module.Actor, critic_cls=module.Critic)
    _ensure_expected_fasttd3_api(classes, source="installed fast_td3 package")
    return classes


def load_upstream_fasttd3_classes(fasttd3_repo: str = "") -> UpstreamFastTD3Classes:
    """Load upstream FastTD3 classes from repo override or environment"""
    if fasttd3_repo:
        return load_fasttd3_classes_from_repo(fasttd3_repo)
    return load_fasttd3_classes_from_environment()


class UpstreamFastTD3Agent:
    """Teacher/DAgger-compatible agent backed by upstream FastTD3 networks."""

    def __init__(
        self,
        obs_dim     : int,  # Param: integer input for obs dim
        action_dim  : int,  # Param: integer input for action dim
        device      : torch.device | str,  # Param: torch device where tensors are read or allocated
        *,
        config         : TD3Config,  # Param: shared TD3/runtime config
        upstream_config: UpstreamFastTD3Config,  # Param: upstream FastTD3 backend config
        priv_obs_dim   : int = 0,  # Param: privileged critic observation width
    ) -> None:
        """Process for `__init__`

        Steps:
        - Resolve inputs for `__init__` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        self.config = config
        self.upstream_config = upstream_config
        self.device = torch.device(device)
        self.priv_obs_dim = int(priv_obs_dim)
        critic_obs_dim = int(obs_dim) + self.priv_obs_dim
        classes = load_upstream_fasttd3_classes(upstream_config.fasttd3_repo)
        self.actor = classes.actor_cls(
            n_obs=obs_dim,
            n_act=action_dim,
            num_envs=max(1, int(upstream_config.num_envs)),
            init_scale=float(upstream_config.init_scale),
            hidden_dim=int(upstream_config.actor_hidden_dim),
            std_min=float(upstream_config.std_min),
            std_max=float(upstream_config.std_max),
            device=self.device,
        ).to(self.device)
        self.actor_target = classes.actor_cls(
            n_obs=obs_dim,
            n_act=action_dim,
            num_envs=max(1, int(upstream_config.num_envs)),
            init_scale=float(upstream_config.init_scale),
            hidden_dim=int(upstream_config.actor_hidden_dim),
            std_min=float(upstream_config.std_min),
            std_max=float(upstream_config.std_max),
            device=self.device,
        ).to(self.device)
        self.critic = classes.critic_cls(
            n_obs=critic_obs_dim,
            n_act=action_dim,
            num_atoms=int(upstream_config.num_atoms),
            v_min=float(upstream_config.v_min),
            v_max=float(upstream_config.v_max),
            hidden_dim=int(upstream_config.critic_hidden_dim),
            device=self.device,
        ).to(self.device)
        self.critic_target = classes.critic_cls(
            n_obs=critic_obs_dim,
            n_act=action_dim,
            num_atoms=int(upstream_config.num_atoms),
            v_min=float(upstream_config.v_min),
            v_max=float(upstream_config.v_max),
            hidden_dim=int(upstream_config.critic_hidden_dim),
            device=self.device,
        ).to(self.device)
        self.sync_target_networks()
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=float(config.actor_lr),
            weight_decay=float(upstream_config.weight_decay),
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=float(config.critic_lr),
            weight_decay=float(upstream_config.weight_decay),
        )
        self.critic1_opt = self.critic_opt
        self.critic2_opt = self.critic_opt
        self.obs_stats = RunningTensorMoments((obs_dim,))
        self.priv_obs_stats = RunningTensorMoments((self.priv_obs_dim,)) if self.priv_obs_dim > 0 else None
        self.reward_stats = RunningScalarMoments()
        self.train_step = 0

    def set_optimizer_lrs(self, *, actor_lr: float | None = None, critic_lr: float | None = None) -> None:
        """Set actor and critic optimizer learning rates."""
        if actor_lr is not None:
            self.config.actor_lr = float(actor_lr)
            for group in self.actor_opt.param_groups:
                group["lr"] = float(actor_lr)
        if critic_lr is not None:
            self.config.critic_lr = float(critic_lr)
            for group in self.critic_opt.param_groups:
                group["lr"] = float(critic_lr)

    def reset_critic_optimizers(self) -> None:
        """Recreate critic optimizer while preserving critic weights."""
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=float(self.config.critic_lr),
            weight_decay=float(self.upstream_config.weight_decay),
        )
        self.critic1_opt = self.critic_opt
        self.critic2_opt = self.critic_opt

    def sync_target_networks(self) -> None:
        """Hard-copy online weights into target weights."""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _critic_obs(self, obs: torch.Tensor, priv_obs: torch.Tensor | None) -> torch.Tensor:
        if self.priv_obs_dim == 0 or priv_obs is None:
            return obs
        return torch.cat([obs, self.normalize_priv_obs(priv_obs)], dim=-1)

    def update_obs_stats(self, obs: torch.Tensor) -> None:
        """Update observation normalization statistics."""
        if self.config.observation_normalization:
            self.obs_stats.update(obs)

    def update_priv_obs_stats(self, priv_obs: torch.Tensor | None) -> None:
        """Update privileged observation normalization statistics."""
        if self.config.observation_normalization and self.priv_obs_stats is not None and priv_obs is not None:
            self.priv_obs_stats.update(priv_obs)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations when enabled.

        Steps:
        - Resolve inputs for `normalize_obs` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if not self.config.observation_normalization:
            return obs
        mean = self.obs_stats.mean_tensor(obs.device)
        std = self.obs_stats.std(obs.device).clamp(min=self.config.obs_norm_eps)
        normalized = (obs - mean) / std
        if self.config.obs_norm_clip > 0.0:
            normalized = normalized.clamp(-self.config.obs_norm_clip, self.config.obs_norm_clip)
        return normalized

    def normalize_priv_obs(self, priv_obs: torch.Tensor) -> torch.Tensor:
        """Normalize privileged observations when enabled.

        Steps:
        - Resolve inputs for `normalize_priv_obs` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if not self.config.observation_normalization or self.priv_obs_stats is None:
            return priv_obs
        mean = self.priv_obs_stats.mean_tensor(priv_obs.device)
        std = self.priv_obs_stats.std(priv_obs.device).clamp(min=self.config.obs_norm_eps)
        normalized = (priv_obs - mean) / std
        if self.config.obs_norm_clip > 0.0:
            normalized = normalized.clamp(-self.config.obs_norm_clip, self.config.obs_norm_clip)
        return normalized

    def update_reward_stats(self, reward: torch.Tensor) -> None:
        """Update reward normalization statistics."""
        if self.config.reward_normalization:
            self.reward_stats.update(reward)

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize rewards when enabled.

        Steps:
        - Resolve inputs for `normalize_reward` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if not self.config.reward_normalization:
            return reward
        denom = max(self.reward_stats.std, self.config.reward_norm_eps)
        normalized = reward / denom
        if self.config.reward_norm_clip > 0.0:
            normalized = normalized.clamp(-self.config.reward_norm_clip, self.config.reward_norm_clip)
        return normalized

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Evaluate the upstream actor and optionally add exploration noise.

        Action is also passed through contact finger close caps and curriculum finger unlocks based on raw observation before being returned
        Steps:
        - Resolve inputs for `select_action` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use

        """
        with torch.no_grad():
            obs = self.normalize_obs(obs)
            action = self.actor(obs)
            if not deterministic:
                if self.config.exploration_noise_finger > 0.0:
                    sigma = make_per_dim_noise_sigma(
                        action.shape,
                        arm_sigma=self.config.exploration_noise,
                        finger_sigma=self.config.exploration_noise_finger,
                        device=action.device,
                        dtype=action.dtype,
                        config=self.config.gate_config,
                    )
                    action = action + sigma * torch.randn_like(action)
                else:
                    action = action + self.config.exploration_noise * torch.randn_like(action)
            return apply_contact_finger_close_cap(action.clamp(-1.0, 1.0), self.config.gate_config)

    def _dist_ce(self, logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        return -torch.sum(target_dist * F.log_softmax(logits, dim=1), dim=1).mean()

    def _soft_update(self, src, dst) -> None:
        for source_param, dest_param in zip(src.parameters(), dst.parameters()):
            dest_param.data.mul_(1.0 - self.config.tau).add_(self.config.tau * source_param.data)

    def update(self, replay, batch_size: int, progress_step: int | None = None) -> dict[str, float]:
        """Run one upstream FastTD3 distributional update with BC and DAgger hooks

        Steps:
        - Sample replay rows and split policy obs, privileged obs, reward, done flags, and teacher labels
        - Normalize observations and rewards before critic target construction
        - Build noisy target actions through actor target, contact close caps, unlock gates, and optional post-unlock finger noise
        - Project target value distributions with critic target and select clipped double-Q targets when enabled
        - Train both critic distributions against target distributions and clip critic gradients when configured
        - Decide whether actor update is due, frozen, BC-only, or mixed RL plus teacher BC
        - Build actor objective action and BC action through gate-specific objective routing
        - Apply BC-only loss for teacher rows or RL actor loss plus decayed teacher BC loss
        - Step actor optimizer when actor loss is valid, then soft-update actor and critic targets on policy delay cadence
        - Increment train step and return scalar diagnostics for losses, Q values, BC weights, schedules, and normalization state

         Build critic targets with actor target and distributional projection
         Target actions are built with noise, contact finger close caps, curriculum finger unlocks,
         and optional post-unlock finger noise based on raw next obs before being fed into critic target
         Target distribution is selected between critic target 1 and 2 when CDQ enabled, otherwise averaged
         Bootstrap values are set to 0 on done rows and 1 on timeout rows to implement episode truncation correctly
         Reward is normalized before critic target construction, but not clipped, to allow for unbounded reward distributions with stable normalization
         Target critic value is computed from target distribution for logging and optional TD error prioritization, but not used
         directly in loss computation to allow for proper distributional loss calculation and TD error prioritization when enabled
         BC-only updates are still applied to critic when enabled, but actor is not updated and target construction still occurs with next
         actions from actor target to allow for proper TD error prioritization and critic learning even during BC-only phases
         Critic loss is computed as cross-entropy against projected target distribution, and critic gradients are clipped when configured
         Actor loss is computed as negative Q value of actor action under critic, plus optional BC loss against teacher action when enabled and
         teacher rows are present in batch with decayed weight based on train step and configured teacher BC schedules, and actor is only updated when
         policy delay cadence is met and not frozen by train step, and actor action is passed through gate-specific objective routing to allow for
         different gating of actor Q action and BC action when needed, and actor action is also passed through contact finger close caps and
         curriculum finger unlocks based on raw observation before being fed into critic for proper credit assignment and loss calculation,
         and optional post-unlock finger noise is added to target action after caps and unlocks but before critic target when configured to
         allow for proper exploration noise application even with curriculum unlocks in place to avoid noise bypassing unlocks, and actor loss is
         only stepped when valid to allow for proper application of policy delay, actor freeze, and BC-only phases without stepping with invalid losses
         Target networks are soft-updated on policy delay cadence to allow for stable target construction with proper delay, and hard-updated at
         initialization to ensure initial sync before updates occur
         Diagnostics are collected for actor and critic losses, actor Q values, BC weights and schedules, and observation and reward
         normalization state for logging and analysis


        """
        timing_enabled = _update_timing_enabled()
        mark = time.perf_counter()
        mark = _timing_mark(timing_enabled, self.train_step, "begin", mark)
        (
            obs,
            action,
            bc_action,
            reward,
            discount,
            next_obs,
            terminated,
            timeout,
            is_teacher,
            priv_obs,
            next_priv_obs,
        ) = replay.sample(batch_size, self.device)
        mark = _timing_mark(timing_enabled, self.train_step, "sample", mark)
        raw_obs           = obs
        raw_next_obs      = next_obs
        obs               = self.normalize_obs(raw_obs)
        next_obs          = self.normalize_obs(raw_next_obs)
        reward_for_target = self.normalize_reward(reward).squeeze(-1)
        critic_obs        = self._critic_obs(obs, priv_obs)
        next_critic_obs   = self._critic_obs(next_obs, next_priv_obs)
        discount_flat     = discount.squeeze(-1)
        terminated_flat   = terminated.squeeze(-1)
        timeout_flat      = timeout.squeeze(-1)
        is_teacher_fraction_value = float((is_teacher[:, 0] > 0.5).float().mean().item())
        mark = _timing_mark(timing_enabled, self.train_step, "normalize", mark)

        with torch.no_grad():
            if self.config.policy_noise_finger > 0.0:
                sigma = make_per_dim_noise_sigma(
                    action.shape,
                    arm_sigma=self.config.policy_noise,
                    finger_sigma=self.config.policy_noise_finger,
                    device=action.device,
                    dtype=action.dtype,
                    config=self.config.gate_config,
                )
                noise = sigma * torch.randn_like(action)
            else:
                noise = torch.randn_like(action) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = apply_contact_finger_close_cap(
                (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0),
                self.config.gate_config,
            )
            next_action = apply_curriculum_finger_unlock_from_flat_obs(
                next_action,
                raw_next_obs,
                self.config.gate_config,
            )
            if self.config.finger_noise_bypass_unlock and self.config.policy_noise_finger > 0.0:
                next_action = add_post_unlock_finger_noise(
                    next_action,
                    raw_next_obs,
                    finger_sigma=self.config.policy_noise_finger,
                    noise_clip=self.config.noise_clip,
                    config=self.config.gate_config,
                )
            done_flat = torch.maximum(terminated_flat, timeout_flat)
            bootstrap = torch.where(timeout_flat > 0.5, torch.ones_like(done_flat), 1.0 - done_flat)
            q1_target_dist, q2_target_dist = self.critic_target.projection(
                next_critic_obs,
                next_action,
                reward_for_target,
                bootstrap,
                discount_flat,
            )
            q1_target_value = self.critic_target.get_value(q1_target_dist)
            q2_target_value = self.critic_target.get_value(q2_target_dist)
            if bool(self.upstream_config.use_cdq):
                target_dist = torch.where(
                    q1_target_value.unsqueeze(1) < q2_target_value.unsqueeze(1),
                    q1_target_dist,
                    q2_target_dist,
                )
                q1_target_dist = q2_target_dist = target_dist
                target_value = self.critic_target.get_value(target_dist)
            else:
                target_value = 0.5 * (q1_target_value + q2_target_value)
        mark = _timing_mark(timing_enabled, self.train_step, "target", mark)

        q1_logits, q2_logits = self.critic(critic_obs, action)
        critic1_loss = self._dist_ce(q1_logits, q1_target_dist)
        critic2_loss = self._dist_ce(q2_logits, q2_target_dist)
        critic_loss = critic1_loss + critic2_loss
        mark = _timing_mark(timing_enabled, self.train_step, "critic_forward", mark)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.config.critic_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.critic_grad_clip)
        self.critic_opt.step()
        mark = _timing_mark(timing_enabled, self.train_step, "critic_backward", mark)

        q1_value = self.critic.get_value(F.softmax(q1_logits.detach(), dim=1))
        q2_value = self.critic.get_value(F.softmax(q2_logits.detach(), dim=1))
        actor_loss_value = math.nan
        actor_q_mean_value = math.nan
        actor_q_std_value = math.nan
        actor_q_finite_fraction_value = math.nan
        actor_action_finite_fraction_value = math.nan
        actor_raw_finite_fraction_value = math.nan
        actor_updated_value = 0.0
        bc_loss_value = math.nan
        bc_arm_loss_value = math.nan
        bc_finger_loss_value = math.nan
        bc_weight_value = math.nan
        bc_arm_weight_value = math.nan
        bc_finger_weight_value = math.nan
        bc_only_active_value = 0.0
        actor_loss = None

        actor_frozen = (
            self.config.actor_freeze_steps > 0 and self.train_step < self.config.actor_freeze_steps
        ) or (self.config.rl_actor_freeze_until_train_step > self.train_step)
        bc_progress_step = self.train_step if progress_step is None else int(progress_step)
        bc_only_active = self.config.bc_only_steps > 0 and bc_progress_step < self.config.bc_only_steps
        bc_only_active_value = 1.0 if bc_only_active else 0.0
        policy_delay_due = self.train_step % max(1, int(self.config.policy_delay)) == 0
        if policy_delay_due and not actor_frozen:
            actor_updated_value = 1.0
            actor_action_raw = self.actor(obs)
            actor_q_action = objective_action_from_gate_mode(
                actor_action_raw,
                raw_obs,
                self.config.actor_q_action_gate_mode,
                self.config.gate_config,
            )
            actor_bc_action = objective_action_from_gate_mode(
                actor_action_raw,
                raw_obs,
                self.config.actor_bc_action_gate_mode,
                self.config.gate_config,
            )
            actor_raw_finite_fraction_value = finite_fraction_value(actor_action_raw)
            actor_action_finite_fraction_value = finite_fraction_value(actor_q_action)
            actor_q1_logits, actor_q2_logits = self.critic(critic_obs, actor_q_action)
            actor_q1_value = self.critic.get_value(F.softmax(actor_q1_logits, dim=1))
            actor_q2_value = self.critic.get_value(F.softmax(actor_q2_logits, dim=1))
            actor_q = (
                torch.minimum(actor_q1_value, actor_q2_value)
                if bool(self.upstream_config.use_cdq)
                else 0.5 * (actor_q1_value + actor_q2_value)
            )
            actor_q_finite_fraction_value = finite_fraction_value(actor_q)
            actor_q_mean_value = finite_mean_value(actor_q)
            actor_q_std_value = finite_std_value(actor_q)
            teacher_mask = is_teacher[:, 0] > 0.5
            if bc_only_active:
                if torch.any(teacher_mask):
                    base_bc_weight = max(0.0, float(self.config.bc_only_weight))
                    result = weighted_bc_loss(
                        actor_bc_action[teacher_mask],
                        bc_action[teacher_mask],
                        base_bc_weight,
                        self.config.bc_only_arm_weight,
                        self.config.bc_only_finger_weight,
                        num_arm=self.config.gate_config.num_arm,
                        num_fingers=self.config.gate_config.num_fingers,
                    )
                    weighted, bc_loss, bc_arm_loss, bc_finger_loss, arm_w, finger_w = result
                    actor_loss = weighted
                    bc_weight_value = max(base_bc_weight, arm_w, finger_w)
                    bc_arm_weight_value = arm_w
                    bc_finger_weight_value = finger_w
                    bc_loss_value = float(bc_loss.item())
                    if bc_arm_loss is not None:
                        bc_arm_loss_value = float(bc_arm_loss.item())
                    if bc_finger_loss is not None:
                        bc_finger_loss_value = float(bc_finger_loss.item())
                else:
                    actor_updated_value = 0.0
            else:
                actor_loss = -actor_q.mean()
                if (
                    teacher_bc_requested(
                        self.config.teacher_bc_weight,
                        self.config.teacher_bc_arm_weight,
                        self.config.teacher_bc_finger_weight,
                    )
                    and torch.any(teacher_mask)
                ):
                    base_bc_weight, arm_bc_weight, finger_bc_weight = scheduled_teacher_bc_weights(
                        progress_step=bc_progress_step,
                        base_weight=self.config.teacher_bc_weight,
                        arm_weight=self.config.teacher_bc_arm_weight,
                        finger_weight=self.config.teacher_bc_finger_weight,
                        decay_steps=self.config.teacher_bc_decay_steps,
                    )
                    if base_bc_weight > 0.0 or arm_bc_weight > 0.0 or finger_bc_weight > 0.0:
                        result = weighted_bc_loss(
                            actor_bc_action[teacher_mask],
                            bc_action[teacher_mask],
                            base_bc_weight,
                            arm_bc_weight,
                            finger_bc_weight,
                            num_arm=self.config.gate_config.num_arm,
                            num_fingers=self.config.gate_config.num_fingers,
                        )
                        weighted, bc_loss, bc_arm_loss, bc_finger_loss, arm_w, finger_w = result
                        actor_loss = actor_loss + weighted
                        bc_weight_value = max(base_bc_weight, arm_w, finger_w)
                        bc_arm_weight_value = arm_w
                        bc_finger_weight_value = finger_w
                        bc_loss_value = float(bc_loss.item())
                        if bc_arm_loss is not None:
                            bc_arm_loss_value = float(bc_arm_loss.item())
                        if bc_finger_loss is not None:
                            bc_finger_loss_value = float(bc_finger_loss.item())
            if actor_loss is not None and actor_updated_value > 0.0:
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.config.stop_on_nonfinite_update and not bool(torch.isfinite(actor_loss.detach()).all().item()):
                    raise FloatingPointError(f"nonfinite upstream FastTD3 actor_loss at train_step={self.train_step}")
                self.actor_opt.step()
                actor_loss_value = float(actor_loss.item())
        mark = _timing_mark(timing_enabled, self.train_step, "actor", mark)

        if policy_delay_due:
            if actor_loss is not None and actor_updated_value > 0.0:
                self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        mark = _timing_mark(timing_enabled, self.train_step, "target_update", mark)

        self.train_step += 1
        mark = _timing_mark(timing_enabled, self.train_step - 1, "end", mark)
        return {
            "critic1_loss"                    : float(critic1_loss.item()),
            "critic2_loss"                    : float(critic2_loss.item()),
            "actor_loss"                      : actor_loss_value,
            "actor_updated"                   : actor_updated_value,
            "actor_frozen"                    : float(actor_frozen),
            "actor_q_mean"                    : actor_q_mean_value,
            "actor_q_std"                     : actor_q_std_value,
            "actor_q_finite_fraction"         : actor_q_finite_fraction_value,
            "actor_action_finite_fraction"    : actor_action_finite_fraction_value,
            "actor_raw_finite_fraction"       : actor_raw_finite_fraction_value,
            "q1_mean"                         : float(q1_value.mean().item()),
            "q2_mean"                         : float(q2_value.mean().item()),
            "q1_std"                          : float(q1_value.std(unbiased=False).item()),
            "q2_std"                          : float(q2_value.std(unbiased=False).item()),
            "target_mean"                     : float(target_value.mean().item()),
            "target_std"                      : float(target_value.std(unbiased=False).item()),
            "target_q_raw_mean"               : float(target_value.mean().item()),
            "target_raw_mean"                 : float(target_value.mean().item()),
            "target_clip_fraction"            : 0.0,
            "target_q_clip_fraction"          : 0.0,
            "target_post_clip_fraction"       : 0.0,
            "is_teacher_fraction"             : is_teacher_fraction_value,
            "training_phase_id"               : float(self.config.training_phase_id),
            "active_n_step"                   : float(self.config.active_n_step),
            "active_updates_per_step"         : float(self.config.active_updates_per_step),
            "active_policy_delay"             : float(self.config.policy_delay),
            "active_actor_lr"                 : float(self.config.actor_lr),
            "active_critic_lr"                : float(self.config.critic_lr),
            "active_gamma"                    : float(self.config.gamma),
            "rl_actor_freeze_until_train_step": float(self.config.rl_actor_freeze_until_train_step),
            "reward_stat_denom"               : float(max(self.reward_stats.std, self.config.reward_norm_eps)),
            "reward_stat_count"               : float(self.reward_stats.count),
            "bc_loss"                         : bc_loss_value,
            "bc_arm_loss"                     : bc_arm_loss_value,
            "bc_finger_loss"                  : bc_finger_loss_value,
            "bc_weight"                       : bc_weight_value,
            "bc_arm_weight"                   : bc_arm_weight_value,
            "bc_finger_weight"                : bc_finger_weight_value,
            "bc_only_active"                  : bc_only_active_value,
            "pre_tanh_l2"                     : math.nan,
            "pre_tanh_rms"                    : math.nan,
        }

    def state_dict(self) -> dict[str, object]:
        """Serialize learnable and normalization state."""
        return {
            "td3_backend"   : "upstream_fasttd3",
            "actor"         : self.actor.state_dict(),
            "actor_target"  : self.actor_target.state_dict(),
            "critic"        : self.critic.state_dict(),
            "critic_target" : self.critic_target.state_dict(),
            "actor_opt"     : self.actor_opt.state_dict(),
            "critic_opt"    : self.critic_opt.state_dict(),
            "obs_stats"     : self.obs_stats.state_dict(),
            "priv_obs_stats": self.priv_obs_stats.state_dict() if self.priv_obs_stats is not None else None,
            "reward_stats"  : self.reward_stats.state_dict(),
            "train_step"    : self.train_step,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore this object from an upstream FastTD3 checkpoint state.

        Steps:
        - Resolve inputs for `load_state_dict` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if state.get("td3_backend") != "upstream_fasttd3":
            raise RuntimeError(
                "Cannot load a custom scalar-Q checkpoint into td3_backend=upstream_fasttd3. "
                "Start a new BC/DAgger run or load an upstream_fasttd3 checkpoint."
            )
        load_module_state_allow_env_count_buffers(
            self.actor,
            state["actor"],
            context="upstream_fasttd3 actor",
        )
        load_module_state_allow_env_count_buffers(
            self.actor_target,
            state.get("actor_target", state["actor"]),
            context="upstream_fasttd3 actor_target",
        )
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state.get("critic_target", state["critic"]))
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])
        if "obs_stats" in state:
            self.obs_stats.load_state_dict(state["obs_stats"])
        if self.priv_obs_stats is not None:
            if not isinstance(state.get("priv_obs_stats"), dict):
                raise RuntimeError("checkpoint is missing priv_obs_stats for the privileged critic")
            self.priv_obs_stats.load_state_dict(state["priv_obs_stats"])
        if "reward_stats" in state:
            self.reward_stats.load_state_dict(state["reward_stats"])
        self.train_step = int(state.get("train_step", 0))


def make_upstream_fasttd3_agent(
    obs_dim        : int,  # Param: integer input for obs dim
    action_dim     : int,  # Param: integer input for action dim
    device         : torch.device | str,  # Param: torch device where tensors are read or allocated
    priv_obs_dim   : int,  # Param: privileged critic observation width
    *,
    config         : TD3Config,  # Param: shared TD3/runtime config
    upstream_config: UpstreamFastTD3Config,  # Param: upstream FastTD3 backend config
) -> UpstreamFastTD3Agent:
    """Build the upstream FastTD3 agent wrapper."""
    return UpstreamFastTD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        config=config,
        upstream_config=upstream_config,
        priv_obs_dim=priv_obs_dim,
    )
