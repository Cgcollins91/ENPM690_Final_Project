"""

Episode bookkeeping tensors for vectorized training loops


This module provides the EpisodeState dataclass for tracking episode indices, returns, steps, and best-seen metrics across parallel environment
rows in the training loop

The best-seen metrics are calculated according to the curriculum and include tip distance, phase1 palm distance, phase1 orient distance,
contact, strict contact, lift, lift with strict contact, curl, topdown stage, and topdown unlock. The training loop updates these metrics
at each step and uses them for curriculum decisions and diagnostics.

File map:

EpisodeState:  Mutable vectorized episode counters and best-seen metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EpisodeState:
    """Mutable vectorized episode counters and best-seen metrics"""

    episode_idx                  : torch.Tensor  # Field: training episode index associated with this record
    next_episode_idx             : int  # Field: index identifying the next episode entry
    step                         : torch.Tensor  # Field: tensor containing step values for batched env rows
    episode_return               : torch.Tensor  # Field: tensor containing episode return values for batched env rows
    best_tip                     : torch.Tensor  # Field: tensor containing best tip values for batched env rows
    best_phase1_palm             : torch.Tensor  # Field: tensor containing best phase1 palm values for batched env rows
    best_phase1_orient           : torch.Tensor  # Field: tensor containing best phase1 orient values for batched env rows
    best_contact                 : torch.Tensor  # Field: tensor containing best contact values for batched env rows
    best_strict_contact          : torch.Tensor  # Field: tensor containing best strict contact values for batched env rows
    best_lift                    : torch.Tensor  # Field: tensor containing best lift values for batched env rows
    best_lift_with_strict_contact: torch.Tensor  # Field: tensor containing best lift with strict contact values for batched env rows
    best_curl                    : torch.Tensor  # Field: tensor containing best curl values for batched env rows
    best_topdown_stage           : torch.Tensor  # Field: highest topdown curriculum stage reached so far
    max_topdown_unlock           : torch.Tensor  # Field: tensor containing max topdown unlock values for batched env rows

    @classmethod
    def create(
        cls,
        *,
        num_envs         : int,  # Param: number of parallel environment rows represented
        device           : torch.device | str,  # Param: torch device where tensors are read or allocated
        start_episode_idx: int = 0,  # Param: index selecting the start episode entry
    ) -> "EpisodeState":
        """Create initialized episode state tensors"""
        idx = torch.arange(
            int(start_episode_idx),
            int(start_episode_idx) + int(num_envs),
            device=device,
            dtype=torch.long,
        )
        return cls(
            episode_idx     = idx,
            next_episode_idx=int(start_episode_idx) + int(num_envs),
            step=torch.zeros(int(num_envs), device=device, dtype=torch.long),
            episode_return=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
            best_tip=torch.full((int(num_envs),), float("inf"), device=device, dtype=torch.float32),
            best_phase1_palm=torch.full((int(num_envs),), float("inf"), device=device, dtype=torch.float32),
            best_phase1_orient=torch.full((int(num_envs),), float("inf"), device=device, dtype=torch.float32),
            best_contact=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
            best_strict_contact=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
            best_lift=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
            best_lift_with_strict_contact=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
            best_curl=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
            best_topdown_stage=torch.full((int(num_envs),), -1, device=device, dtype=torch.long),
            max_topdown_unlock=torch.zeros(int(num_envs), device=device, dtype=torch.float32),
        )

    def reset_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset counters and best metrics for all or selected env rows

        Steps:
        - Resolve inputs for `reset_metrics` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if env_ids is None:
            self.step.zero_()
            self.episode_return.zero_()
            self.best_tip.fill_(float("inf"))
            self.best_phase1_palm.fill_(float("inf"))
            self.best_phase1_orient.fill_(float("inf"))
            self.best_contact.zero_()
            self.best_strict_contact.zero_()
            self.best_lift.zero_()
            self.best_lift_with_strict_contact.zero_()
            self.best_curl.zero_()
            self.best_topdown_stage.fill_(-1)
            self.max_topdown_unlock.zero_()
            return
        ids = env_ids.to(device=self.step.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self.step[ids] = 0
        self.episode_return[ids] = 0.0
        self.best_tip[ids] = float("inf")
        self.best_phase1_palm[ids] = float("inf")
        self.best_phase1_orient[ids] = float("inf")
        self.best_contact[ids] = 0.0
        self.best_strict_contact[ids] = 0.0
        self.best_lift[ids] = 0.0
        self.best_lift_with_strict_contact[ids] = 0.0
        self.best_curl[ids] = 0.0
        self.best_topdown_stage[ids] = -1
        self.max_topdown_unlock[ids] = 0.0

    def assign_new_episode_ids(self, env_ids: torch.Tensor) -> None:
        """Assign consecutive new episode ids to selected rows"""
        ids = env_ids.to(device=self.episode_idx.device, dtype=torch.long)
        for env_id in ids.tolist():
            self.episode_idx[int(env_id)] = int(self.next_episode_idx)
            self.next_episode_idx += 1

    def reset_all_with_new_ids(self) -> None:
        """Reset all rows and assign a fresh contiguous episode id range"""
        num_envs = int(self.episode_idx.shape[0])
        self.episode_idx = torch.arange(
            int(self.next_episode_idx),
            int(self.next_episode_idx) + num_envs,
            device=self.episode_idx.device,
            dtype=torch.long,
        )
        self.next_episode_idx += num_envs
        self.reset_metrics()

    def advance(self, active_mask: torch.Tensor, reward: torch.Tensor) -> None:
        """Increment active episode counters and returns"""
        mask = active_mask.to(device=self.step.device, dtype=torch.bool)
        self.step[mask] += 1
        self.episode_return[mask] += reward.to(device=self.episode_return.device, dtype=torch.float32)[mask]

    def update_bests(
        self,
        *,
        active_mask             : torch.Tensor,  # Param: boolean mask selecting active rows
        tip                     : torch.Tensor,  # Param: tensor input carrying tip values
        phase1_palm             : torch.Tensor,  # Param: tensor input carrying phase1 palm values
        phase1_orient           : torch.Tensor,  # Param: tensor input carrying phase1 orient values
        contact                 : torch.Tensor,  # Param: tensor input carrying contact values
        strict_contact          : torch.Tensor,  # Param: tensor input carrying strict contact values
        lift                    : torch.Tensor,  # Param: tensor input carrying lift values
        lift_with_strict_contact: torch.Tensor,  # Param: tensor input carrying lift with strict contact values
        curl                    : torch.Tensor,  # Param: tensor input carrying curl values
        topdown_stage           : torch.Tensor,  # Param: tensor input carrying topdown stage values
        topdown_unlock          : torch.Tensor,  # Param: tensor input carrying topdown unlock values
    ) -> None:
        """Update best-seen metrics on active rows

        Steps:
        - Resolve inputs for `update_bests` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        mask = active_mask.to(device=self.step.device, dtype=torch.bool)
        self.best_tip = torch.where(mask, torch.minimum(self.best_tip, tip.to(self.best_tip.device)), self.best_tip)
        self.best_phase1_palm = torch.where(
            mask,
            torch.minimum(self.best_phase1_palm, phase1_palm.to(self.best_phase1_palm.device)),
            self.best_phase1_palm,
        )
        self.best_phase1_orient = torch.where(
            mask,
            torch.minimum(self.best_phase1_orient, phase1_orient.to(self.best_phase1_orient.device)),
            self.best_phase1_orient,
        )
        self.best_contact = torch.where(
            mask,
            torch.maximum(self.best_contact, contact.to(self.best_contact.device)),
            self.best_contact,
        )
        self.best_strict_contact = torch.where(
            mask,
            torch.maximum(self.best_strict_contact, strict_contact.to(self.best_strict_contact.device)),
            self.best_strict_contact,
        )
        self.best_lift = torch.where(mask, torch.maximum(self.best_lift, lift.to(self.best_lift.device)), self.best_lift)
        self.best_lift_with_strict_contact = torch.where(
            mask,
            torch.maximum(
                self.best_lift_with_strict_contact,
                lift_with_strict_contact.to(self.best_lift_with_strict_contact.device),
            ),
            self.best_lift_with_strict_contact,
        )
        self.best_curl = torch.where(mask, torch.maximum(self.best_curl, curl.to(self.best_curl.device)), self.best_curl)
        self.best_topdown_stage = torch.where(
            mask,
            torch.maximum(self.best_topdown_stage, topdown_stage.to(self.best_topdown_stage.device, dtype=torch.long)),
            self.best_topdown_stage,
        )
        self.max_topdown_unlock = torch.where(
            mask,
            torch.maximum(self.max_topdown_unlock, topdown_unlock.to(self.max_topdown_unlock.device)),
            self.max_topdown_unlock,
        )

    def reset_done_rows(self, done_ids: torch.Tensor, active_done_ids: torch.Tensor | None = None) -> None:
        """Reset done rows and assign new episode ids to active done rows"""
        self.reset_metrics(done_ids)
        if active_done_ids is not None:
            self.assign_new_episode_ids(active_done_ids)
