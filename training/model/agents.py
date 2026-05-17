"""

Config-driven TD3 agent for import-safe trainer modules

File map:

TD3Config:     Explicit TD3 hyperparameters formerly read from global args
FastTD3Agent:  TD3 actor critic bundle with explicit config and no Isaac imports
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from ..actions.action_gates import (
    ActionGateConfig,
    add_post_unlock_finger_noise,
    apply_contact_finger_close_cap,
    apply_curriculum_finger_unlock_from_flat_obs,
    apply_replay_action_gates_from_flat_obs,
    make_per_dim_noise_sigma,
    objective_action_from_gate_mode,
)
from ..actions.losses import weighted_bc_loss
from .networks import (
    Actor,
    Critic,
    finite_fraction_value,
    finite_mean_value,
    finite_std_value,
    module_param_finite_fraction,
)
from .normalization import RunningScalarMoments, RunningTensorMoments
from ..actions.schedules import scheduled_teacher_bc_weights, teacher_bc_requested


@dataclass
class TD3Config:
    """Explicit TD3 hyperparameters formerly read from global args"""

    hidden_dim                      : int              = 256  # Field: integer hidden dim value tracked by t d3 config
    actor_lr                        : float            = 3e-4  # Field: actor optimizer learning rate
    critic_lr                       : float            = 3e-4  # Field: critic optimizer learning rate
    gamma                           : float            = 0.995  # Field: discount factor used by TD3 updates
    tau                             : float            = 0.005  # Field: target-network interpolation factor for TD3 updates
    policy_delay                    : int              = 2  # Field: integer policy delay value tracked by t d3 config
    policy_noise                    : float            = 0.2  # Field: floating-point policy noise value used by t d3 config
    policy_noise_finger             : float            = 0.0  # Field: floating-point policy noise finger value used by t d3 config
    noise_clip                      : float            = 0.5  # Field: floating-point noise clip value used by t d3 config
    exploration_noise               : float            = 0.25  # Field: floating-point exploration noise value used by t d3 config
    exploration_noise_finger        : float            = 0.0  # Field: floating-point exploration noise finger value used by t d3 config
    target_q_clip                   : float            = -1.0  # Field: floating-point target q clip value used by t d3 config
    critic_grad_clip                : float            = -1.0  # Field: floating-point critic grad clip value used by t d3 config
    actor_pre_tanh_l2               : float            = 0.0  # Field: floating-point actor pre tanh l2 value used by t d3 config
    observation_normalization       : bool             = False  # Field: boolean value indicating the observation normalization state for t d3 config
    obs_norm_eps                    : float            = 1e-5  # Field: floating-point obs norm eps value used by t d3 config
    obs_norm_clip                   : float            = 10.0  # Field: floating-point obs norm clip value used by t d3 config
    reward_normalization            : bool             = False  # Field: boolean value indicating the reward normalization state for t d3 config
    reward_norm_eps                 : float            = 1e-5  # Field: floating-point reward norm eps value used by t d3 config
    reward_norm_clip                : float            = 10.0  # Field: floating-point reward norm clip value used by t d3 config
    actor_freeze_steps              : int              = 0  # Field: step count used for actor freeze steps scheduling or reporting
    rl_actor_freeze_until_train_step: int              = -1  # Field: step count used for rl actor freeze until train step scheduling or reporting
    bc_only_steps                   : int              = 0  # Field: step count used for bc only steps scheduling or reporting
    bc_only_weight                  : float            = 0.0  # Field: weight applied to bc only terms
    bc_only_arm_weight              : float            = -1.0  # Field: weight applied to bc only arm terms
    bc_only_finger_weight           : float            = -1.0  # Field: weight applied to bc only finger terms
    teacher_bc_weight               : float            = 0.0  # Field: weight applied to teacher bc terms
    teacher_bc_arm_weight           : float            = -1.0  # Field: weight applied to teacher bc arm terms
    teacher_bc_finger_weight        : float            = -1.0  # Field: weight applied to teacher bc finger terms
    teacher_bc_decay_steps          : int              = 0  # Field: step count used for teacher bc decay steps scheduling or reporting
    actor_q_action_gate_mode        : str              = "env"  # Field: string actor q action gate mode value used by t d3 config
    actor_bc_action_gate_mode       : str              = "env"  # Field: string actor bc action gate mode value used by t d3 config
    finger_noise_bypass_unlock      : bool             = False  # Field: boolean value indicating the finger noise bypass unlock state for t d3 config
    debug_nonfinite_updates         : bool             = False  # Field: boolean value indicating the debug nonfinite updates state for t d3 config
    stop_on_nonfinite_update        : bool             = False  # Field: boolean value indicating the stop on nonfinite update state for t d3 config
    training_phase_id               : int              = 0  # Field: integer training phase id value tracked by t d3 config
    active_n_step                   : int              = 1  # Field: step count used for active n step scheduling or reporting
    active_updates_per_step         : int              = 1  # Field: step count used for active updates per step scheduling or reporting
    gate_config                     : ActionGateConfig = field(  # Field: stores gate config for t d3 config
        default_factory=lambda: ActionGateConfig(num_arm=0, num_fingers=0, topdown_curriculum=False)
    )


class FastTD3Agent:
    """TD3 actor critic bundle with explicit config and no Isaac imports"""

    def __init__(
        self,
        obs_dim   : int,  # Param: integer input for obs dim
        action_dim: int,  # Param: integer input for action dim
        device    : torch.device | str,  # Param: torch device where tensors are read or allocated
        *,
        config      : TD3Config,  # Param: configuration object used by this helper
        priv_obs_dim: int = 0,  # Param: integer input for priv obs dim
    ):
        """Process for `__init__`

        Steps:
        - Resolve inputs for `__init__` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        self.config = config
        self.device = torch.device(device)
        self.priv_obs_dim = int(priv_obs_dim)
        critic_obs_dim = int(obs_dim) + self.priv_obs_dim
        self.actor = Actor(obs_dim, config.hidden_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, config.hidden_dim, action_dim).to(self.device)
        self.critic1 = Critic(critic_obs_dim, config.hidden_dim, action_dim).to(self.device)
        self.critic2 = Critic(critic_obs_dim, config.hidden_dim, action_dim).to(self.device)
        self.critic1_target = Critic(critic_obs_dim, config.hidden_dim, action_dim).to(self.device)
        self.critic2_target = Critic(critic_obs_dim, config.hidden_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_lr)
        self.obs_stats = RunningTensorMoments((obs_dim,))
        self.priv_obs_stats = RunningTensorMoments((self.priv_obs_dim,)) if self.priv_obs_dim > 0 else None
        self.reward_stats = RunningScalarMoments()
        self.train_step = 0

    def set_optimizer_lrs(self, *, actor_lr: float | None = None, critic_lr: float | None = None) -> None:
        """Set actor and critic optimizer learning rates"""
        if actor_lr is not None:
            self.config.actor_lr = float(actor_lr)
            for group in self.actor_opt.param_groups:
                group["lr"] = float(actor_lr)
        if critic_lr is not None:
            self.config.critic_lr = float(critic_lr)
            for opt in (self.critic1_opt, self.critic2_opt):
                for group in opt.param_groups:
                    group["lr"] = float(critic_lr)

    def reset_critic_optimizers(self) -> None:
        """Recreate critic optimizers while preserving weights"""
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=float(self.config.critic_lr))
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=float(self.config.critic_lr))

    def sync_target_networks(self) -> None:
        """Hard-copy online networks into target networks"""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def _critic_obs(self, obs: torch.Tensor, priv_obs: torch.Tensor | None) -> torch.Tensor:
        if self.priv_obs_dim == 0 or priv_obs is None:
            return obs
        return torch.cat([obs, self.normalize_priv_obs(priv_obs)], dim=-1)

    def update_obs_stats(self, obs: torch.Tensor) -> None:
        """Update observation normalization statistics"""
        if self.config.observation_normalization:
            self.obs_stats.update(obs)

    def update_priv_obs_stats(self, priv_obs: torch.Tensor | None) -> None:
        """Update privileged observation normalization statistics"""
        if self.config.observation_normalization and self.priv_obs_stats is not None and priv_obs is not None:
            self.priv_obs_stats.update(priv_obs)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations when enabled

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
        """Normalize privileged observations when enabled

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
        """Update reward normalization statistics"""
        if self.config.reward_normalization:
            self.reward_stats.update(reward)

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize rewards when enabled

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
        """Evaluate actor and optionally add exploration noise"""
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

    def update(self, replay, batch_size: int, progress_step: int | None = None) -> dict[str, float]:
        """Run one TD3 optimization step

        Steps:
        - Resolve inputs for `update` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
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
        raw_obs = obs
        raw_next_obs = next_obs
        obs = self.normalize_obs(raw_obs)
        next_obs = self.normalize_obs(raw_next_obs)
        reward_for_target = self.normalize_reward(reward)
        critic_obs = self._critic_obs(obs, priv_obs)
        next_critic_obs = self._critic_obs(next_obs, next_priv_obs)
        target_q_raw_mean_value = math.nan
        target_raw_mean_value = math.nan
        target_clip_fraction_value = 0.0
        target_q_clip_fraction_value = 0.0
        target_post_clip_fraction_value = 0.0
        is_teacher_fraction_value = float((is_teacher[:, 0] > 0.5).float().mean().item())

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
            target_q1 = self.critic1_target(next_critic_obs, next_action)
            target_q2 = self.critic2_target(next_critic_obs, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            target_q_raw_mean_value = float(target_q.mean().item())
            if self.config.target_q_clip > 0.0:
                q_clip = float(self.config.target_q_clip)
                q_clip_mask = target_q.abs() > q_clip
                target_q_clip_fraction_value = float(q_clip_mask.float().mean().item())
                target_q = target_q.clamp(-q_clip, q_clip)
            no_bootstrap_mask = 1.0 - terminated
            target_unclipped = reward_for_target + discount * no_bootstrap_mask * target_q
            target_raw_mean_value = float(target_unclipped.mean().item())
            if self.config.target_q_clip > 0.0:
                q_clip = float(self.config.target_q_clip)
                post_clip_mask = target_unclipped.abs() > q_clip
                target_post_clip_fraction_value = float(post_clip_mask.float().mean().item())
                target = target_unclipped.clamp(-q_clip, q_clip)
                target_clip_fraction_value = max(target_q_clip_fraction_value, target_post_clip_fraction_value)
            else:
                target = target_unclipped

        q1 = self.critic1(critic_obs, action)
        q2 = self.critic2(critic_obs, action)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)

        self.critic1_opt.zero_grad(set_to_none=True)
        critic1_loss.backward()
        if self.config.critic_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.critic_grad_clip)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        if self.config.critic_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.critic_grad_clip)
        self.critic2_opt.step()

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
        pre_tanh_l2_value = math.nan
        pre_tanh_rms_value = math.nan
        actor_loss = None

        with torch.no_grad():
            raw_monitor = self.actor.forward_with_raw(obs)[1]
            if torch.isfinite(raw_monitor).all():
                raw_sq_monitor = raw_monitor.pow(2).mean()
                pre_tanh_l2_value = float(raw_sq_monitor.item())
                pre_tanh_rms_value = float(raw_sq_monitor.sqrt().item())

        actor_frozen = (
            self.config.actor_freeze_steps > 0 and self.train_step < self.config.actor_freeze_steps
        ) or (self.config.rl_actor_freeze_until_train_step > self.train_step)
        bc_progress_step = self.train_step if progress_step is None else int(progress_step)
        bc_only_active = self.config.bc_only_steps > 0 and bc_progress_step < self.config.bc_only_steps
        bc_only_active_value = 1.0 if bc_only_active else 0.0
        policy_delay_due = self.train_step % max(1, int(self.config.policy_delay)) == 0
        if policy_delay_due and not actor_frozen:
            actor_updated_value = 1.0
            actor_action_raw, actor_raw = self.actor.forward_with_raw(obs)
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
            actor_raw_finite_fraction_value = finite_fraction_value(actor_raw)
            actor_action_finite_fraction_value = finite_fraction_value(actor_q_action)
            actor_q = self.critic1(critic_obs, actor_q_action)
            actor_q_finite_fraction_value = finite_fraction_value(actor_q)
            actor_q_mean_value = finite_mean_value(actor_q)
            actor_q_std_value = finite_std_value(actor_q)
            if self.config.target_q_clip > 0.0:
                actor_q = actor_q.clamp(-self.config.target_q_clip, self.config.target_q_clip)
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
            if self.config.actor_pre_tanh_l2 > 0.0:
                reg_loss = self.config.actor_pre_tanh_l2 * actor_raw.clamp(-20.0, 20.0).pow(2).mean()
                actor_loss = reg_loss if actor_loss is None else actor_loss + reg_loss
            actor_loss_is_finite = True if actor_loss is None else bool(torch.isfinite(actor_loss.detach()).all().item())
            if self.config.debug_nonfinite_updates and not actor_loss_is_finite:
                print(
                    "nonfinite_actor_update "
                    f"train_step={self.train_step} progress_step={progress_step} "
                    f"actor_loss={float(actor_loss.detach().item())} "
                    f"actor_q_mean={actor_q_mean_value} "
                    f"actor_q_finite_frac={actor_q_finite_fraction_value:.6f} "
                    f"actor_action_finite_frac={actor_action_finite_fraction_value:.6f} "
                    f"actor_raw_finite_frac={actor_raw_finite_fraction_value:.6f} "
                    f"obs_finite_frac={finite_fraction_value(obs):.6f} "
                    f"reward_finite_frac={finite_fraction_value(reward_for_target):.6f} "
                    f"target_finite_frac={finite_fraction_value(target):.6f} "
                    f"critic1_param_finite_frac={module_param_finite_fraction(self.critic1):.6f} "
                    f"actor_param_finite_frac={module_param_finite_fraction(self.actor):.6f}",
                    flush=True,
                )
            if actor_loss is not None and actor_updated_value > 0.0:
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.config.stop_on_nonfinite_update and not actor_loss_is_finite:
                    raise FloatingPointError(
                        f"nonfinite actor_loss at train_step={self.train_step} progress_step={progress_step}"
                    )
                self.actor_opt.step()
                actor_loss_value = float(actor_loss.item())

        if policy_delay_due:
            if actor_loss is not None and actor_updated_value > 0.0:
                self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        self.train_step += 1
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
            "q1_mean"                         : float(q1.mean().item()),
            "q2_mean"                         : float(q2.mean().item()),
            "q1_std"                          : float(q1.std(unbiased=False).item()),
            "q2_std"                          : float(q2.std(unbiased=False).item()),
            "target_mean"                     : float(target.mean().item()),
            "target_std"                      : float(target.std(unbiased=False).item()),
            "target_q_raw_mean"               : target_q_raw_mean_value,
            "target_raw_mean"                 : target_raw_mean_value,
            "target_clip_fraction"            : target_clip_fraction_value,
            "target_q_clip_fraction"          : target_q_clip_fraction_value,
            "target_post_clip_fraction"       : target_post_clip_fraction_value,
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
            "pre_tanh_l2"                     : pre_tanh_l2_value,
            "pre_tanh_rms"                    : pre_tanh_rms_value,
        }

    def _soft_update(self, src, dst) -> None:
        for source_param, dest_param in zip(src.parameters(), dst.parameters()):
            dest_param.data.mul_(1.0 - self.config.tau).add_(self.config.tau * source_param.data)

    def state_dict(self) -> dict[str, object]:
        """Serialize learnable and normalization state"""
        return {
            "actor"         : self.actor.state_dict(),
            "actor_target"  : self.actor_target.state_dict(),
            "critic1"       : self.critic1.state_dict(),
            "critic2"       : self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt"     : self.actor_opt.state_dict(),
            "critic1_opt"   : self.critic1_opt.state_dict(),
            "critic2_opt"   : self.critic2_opt.state_dict(),
            "obs_stats"     : self.obs_stats.state_dict(),
            "priv_obs_stats": self.priv_obs_stats.state_dict() if self.priv_obs_stats is not None else None,
            "reward_stats"  : self.reward_stats.state_dict(),
            "train_step"    : self.train_step,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore learnable and normalization state

        Steps:
        - Resolve inputs for `load_state_dict` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic1_opt.load_state_dict(state["critic1_opt"])
        self.critic2_opt.load_state_dict(state["critic2_opt"])
        if "obs_stats" in state:
            self.obs_stats.load_state_dict(state["obs_stats"])
        if self.priv_obs_stats is not None:
            if not isinstance(state.get("priv_obs_stats"), dict):
                raise RuntimeError("checkpoint is missing priv_obs_stats for privileged critic")
            self.priv_obs_stats.load_state_dict(state["priv_obs_stats"])
        if "reward_stats" in state:
            self.reward_stats.load_state_dict(state["reward_stats"])
        self.train_step = int(state.get("train_step", 0))
