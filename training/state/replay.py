"""

Replay buffer and n-step transition helpers

File map:

ReplayBuffer:                    Fixed-size transition store used by TD3 and DAgger updates
make_step_transition:            Normalize a single env-step into a CPU transition record
flush_ready_n_step_transitions:  Pop n-step transitions that are ready to enter replay
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch


class ReplayBuffer:
    """Fixed-size transition store used by TD3 and DAgger updates"""

    def __init__(
        self,
        capacity    : int,  # Param: integer input for capacity
        obs_dim     : int,  # Param: integer input for obs dim
        action_dim  : int,  # Param: integer input for action dim
        priv_obs_dim: int                = 0,  # Param: integer input for priv obs dim
        device      : torch.device | str = "cpu",  # Param: torch device where tensors are read or allocated
    ):
        """Process for `__init__`

        Steps:
        - Resolve inputs for `__init__` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        self.capacity = int(capacity)
        self.priv_obs_dim = int(priv_obs_dim)
        self.device = torch.device(device)
        self.obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.bc_actions = torch.empty((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.discounts = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.terminated = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.timeout = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.is_teacher = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        if self.priv_obs_dim > 0:
            self.priv_obs = torch.empty((capacity, priv_obs_dim), dtype=torch.float32, device=self.device)
            self.next_priv_obs = torch.empty((capacity, priv_obs_dim), dtype=torch.float32, device=self.device)
        else:
            self.priv_obs = None
            self.next_priv_obs = None
        self.ptr = 0
        self.size = 0

    def _store_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(device=self.device, non_blocking=True)

    def add(
        self,
        obs          : torch.Tensor,  # Param: observation payload returned by the environment or replay path
        action       : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
        bc_action    : torch.Tensor,  # Param: behavior-cloning target action
        reward       : torch.Tensor,  # Param: reward tensor or scalar from the transition
        discount     : torch.Tensor,  # Param: tensor input carrying discount values
        next_obs     : torch.Tensor,  # Param: next observation payload after the environment step
        terminated   : torch.Tensor,  # Param: tensor input carrying terminated values
        timeout      : torch.Tensor,  # Param: tensor input carrying timeout values
        is_teacher   : torch.Tensor,  # Param: boolean input indicating whether teacher is active
        priv_obs     : torch.Tensor | None = None,  # Param: tensor input carrying priv obs values
        next_priv_obs: torch.Tensor | None = None,  # Param: tensor input carrying next priv obs values
    ) -> None:
        """Append one vectorized environment step to the replay buffer

        Steps:
        - Resolve inputs for `add` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
            bc_action = bc_action.unsqueeze(0)
            reward = reward.reshape(1, -1)
            discount = discount.reshape(1, -1)
            next_obs = next_obs.unsqueeze(0)
            terminated = terminated.reshape(1, -1)
            timeout = timeout.reshape(1, -1)
            is_teacher = is_teacher.reshape(1, -1)
            if priv_obs is not None:
                priv_obs = priv_obs.unsqueeze(0)
            if next_priv_obs is not None:
                next_priv_obs = next_priv_obs.unsqueeze(0)

        batch_size = int(obs.shape[0])
        if self.priv_obs is not None:
            if priv_obs is None or next_priv_obs is None:
                raise RuntimeError(
                    "ReplayBuffer configured with priv_obs_dim="
                    f"{self.priv_obs_dim}, but add received missing privileged tensors"
                )
            expected_priv_shape = (batch_size, self.priv_obs_dim)
            if tuple(priv_obs.shape) != expected_priv_shape:
                raise RuntimeError(
                    "priv_obs shape mismatch in replay add: "
                    f"got={tuple(priv_obs.shape)} expected={expected_priv_shape}"
                )
            if tuple(next_priv_obs.shape) != expected_priv_shape:
                raise RuntimeError(
                    "next_priv_obs shape mismatch in replay add: "
                    f"got={tuple(next_priv_obs.shape)} expected={expected_priv_shape}"
                )

        indices = (torch.arange(batch_size, dtype=torch.long, device=self.device) + self.ptr) % self.capacity
        self.obs.index_copy_(0, indices, self._store_tensor(obs))
        self.actions.index_copy_(0, indices, self._store_tensor(action))
        self.bc_actions.index_copy_(0, indices, self._store_tensor(bc_action))
        self.rewards.index_copy_(0, indices, self._store_tensor(reward))
        self.discounts.index_copy_(0, indices, self._store_tensor(discount))
        self.next_obs.index_copy_(0, indices, self._store_tensor(next_obs))
        self.terminated.index_copy_(0, indices, self._store_tensor(terminated))
        self.timeout.index_copy_(0, indices, self._store_tensor(timeout))
        self.is_teacher.index_copy_(0, indices, self._store_tensor(is_teacher))
        if self.priv_obs is not None:
            self.priv_obs.index_copy_(0, indices, self._store_tensor(priv_obs))
            self.next_priv_obs.index_copy_(0, indices, self._store_tensor(next_priv_obs))
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str):
        """Sample a random minibatch from stored transitions

        Steps:
        - Resolve inputs for `sample` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if self.size <= 0:
            raise RuntimeError("cannot sample from an empty ReplayBuffer")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        priv = self.priv_obs[idx].to(device) if self.priv_obs is not None else None
        next_priv = self.next_priv_obs[idx].to(device) if self.next_priv_obs is not None else None
        return (
            self.obs[idx].to(device),
            self.actions[idx].to(device),
            self.bc_actions[idx].to(device),
            self.rewards[idx].to(device),
            self.discounts[idx].to(device),
            self.next_obs[idx].to(device),
            self.terminated[idx].to(device),
            self.timeout[idx].to(device),
            self.is_teacher[idx].to(device),
            priv,
            next_priv,
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize valid replay rows in chronological ring-buffer order"""
        if self.size <= 0:
            idx = torch.empty((0,), dtype=torch.long, device=self.device)
        elif self.size < self.capacity:
            idx = torch.arange(self.size, dtype=torch.long, device=self.device)
        else:
            idx = (torch.arange(self.size, dtype=torch.long, device=self.device) + self.ptr) % self.capacity
        state: dict[str, object] = {
            "capacity"    : self.capacity,
            "size"        : self.size,
            "ptr"         : int(self.size % self.capacity) if self.capacity > 0 else 0,
            "priv_obs_dim": self.priv_obs_dim,
            "obs"         : self.obs[idx].detach().cpu(),
            "actions"     : self.actions[idx].detach().cpu(),
            "bc_actions"  : self.bc_actions[idx].detach().cpu(),
            "rewards"     : self.rewards[idx].detach().cpu(),
            "discounts"   : self.discounts[idx].detach().cpu(),
            "next_obs"    : self.next_obs[idx].detach().cpu(),
            "terminated"  : self.terminated[idx].detach().cpu(),
            "timeout"     : self.timeout[idx].detach().cpu(),
            "is_teacher"  : self.is_teacher[idx].detach().cpu(),
        }
        if self.priv_obs is not None:
            state["priv_obs"] = self.priv_obs[idx].detach().cpu()
            state["next_priv_obs"] = self.next_priv_obs[idx].detach().cpu()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore replay rows into this buffer

        Steps:
        - Resolve inputs for `load_state_dict` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        size = int(state.get("size", 0))
        if size > self.capacity:
            raise RuntimeError(
                f"checkpoint replay size {size} exceeds current replay capacity {self.capacity}"
            )
        tensors = {
            "obs"       : self.obs,
            "actions"   : self.actions,
            "bc_actions": self.bc_actions,
            "rewards"   : self.rewards,
            "discounts" : self.discounts,
            "next_obs"  : self.next_obs,
            "terminated": self.terminated,
            "timeout"   : self.timeout,
            "is_teacher": self.is_teacher,
        }
        for name, dest in tensors.items():
            src = state.get(name)
            if not torch.is_tensor(src):
                raise RuntimeError(f"checkpoint replay missing tensor {name!r}")
            if tuple(src.shape[1:]) != tuple(dest.shape[1:]):
                raise RuntimeError(
                    f"checkpoint replay {name} shape mismatch: checkpoint={tuple(src.shape)} "
                    f"current={tuple(dest.shape)}"
                )
            if int(src.shape[0]) != size:
                raise RuntimeError(
                    f"checkpoint replay {name} row count {int(src.shape[0])} != replay size {size}"
                )
            if size > 0:
                dest[:size].copy_(src.to(device=self.device, dtype=dest.dtype))
        if self.priv_obs is not None:
            for name, dest in (("priv_obs", self.priv_obs), ("next_priv_obs", self.next_priv_obs)):
                src = state.get(name)
                if not torch.is_tensor(src):
                    raise RuntimeError(f"checkpoint replay missing tensor {name!r}")
                if tuple(src.shape[1:]) != tuple(dest.shape[1:]) or int(src.shape[0]) != size:
                    raise RuntimeError(
                        f"checkpoint replay {name} shape mismatch: checkpoint={tuple(src.shape)} "
                        f"current_rows={size} current_shape={tuple(dest.shape)}"
                    )
                if size > 0:
                    dest[:size].copy_(src.to(device=self.device, dtype=dest.dtype))
        self.size = size
        self.ptr = int(size % self.capacity) if self.capacity > 0 else 0


def make_step_transition(
    *,
    obs          : torch.Tensor,  # Param: observation payload returned by the environment or replay path
    action       : torch.Tensor,  # Param: action tensor applied to the environment or stored in replay
    bc_action    : torch.Tensor,  # Param: behavior-cloning target action
    reward       : torch.Tensor,  # Param: reward tensor or scalar from the transition
    next_obs     : torch.Tensor,  # Param: next observation payload after the environment step
    terminated   : bool,  # Param: boolean input controlling terminated
    timeout      : bool,  # Param: boolean input controlling timeout
    is_teacher   : float,  # Param: boolean input indicating whether teacher is active
    priv_obs     : torch.Tensor | None,  # Param: tensor input carrying priv obs values
    next_priv_obs: torch.Tensor | None,  # Param: tensor input carrying next priv obs values
) -> dict[str, object]:
    """Normalize a single env-step into a CPU transition record"""
    return {
        "obs"          : obs.detach().cpu(),
        "action"       : action.detach().cpu(),
        "bc_action"    : bc_action.detach().cpu(),
        "reward"       : reward.reshape(1).detach().cpu(),
        "next_obs"     : next_obs.detach().cpu(),
        "terminated"   : bool(terminated),
        "timeout"      : bool(timeout),
        "is_teacher"   : float(is_teacher),
        "priv_obs"     : None if priv_obs is None else priv_obs.detach().cpu(),
        "next_priv_obs": None if next_priv_obs is None else next_priv_obs.detach().cpu(),
    }


def flush_ready_n_step_transitions(
    queues: Sequence[deque[dict[str, object]]],  # Param: string input for queues
    *,
    gamma : float,  # Param: discount factor used for bootstrapped returns
    n_step: int,  # Param: step count used for n step
) -> list[dict[str, object]]:
    """Pop n-step transitions that are ready to enter replay

    Steps:
    - Resolve inputs for `flush_ready_n_step_transitions` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    out: list[dict[str, object]] = []
    gamma = float(gamma)
    n_step = max(1, int(n_step))

    for queue in queues:
        while queue and (len(queue) >= n_step or queue[-1]["terminated"] or queue[-1]["timeout"]):
            start = queue[0]
            reward = start["reward"].clone()
            reward.zero_()
            next_obs = start["next_obs"]
            next_priv_obs = start["next_priv_obs"]
            terminated = False
            timeout = False
            horizon = 0

            for step_idx, transition in enumerate(queue):
                reward = reward + (gamma ** step_idx) * transition["reward"]
                next_obs = transition["next_obs"]
                next_priv_obs = transition["next_priv_obs"]
                terminated = bool(transition["terminated"])
                timeout = bool(transition["timeout"])
                horizon = step_idx + 1
                if terminated or timeout or horizon >= n_step:
                    break

            queue.popleft()
            tensor_device = start["obs"].device
            out.append(
                {
                    "obs"      : start["obs"],
                    "action"   : start["action"],
                    "bc_action": start["bc_action"],
                    "reward"   : reward,
                    "discount" : torch.tensor([gamma ** horizon], dtype=torch.float32, device=tensor_device),
                    "next_obs" : next_obs,
                    "terminated": torch.tensor(
                        [1.0 if terminated else 0.0], dtype=torch.float32, device=tensor_device
                    ),
                    "timeout": torch.tensor(
                        [1.0 if timeout else 0.0], dtype=torch.float32, device=tensor_device
                    ),
                    "is_teacher"   : torch.tensor([start["is_teacher"]], dtype=torch.float32, device=tensor_device),
                    "priv_obs"     : start["priv_obs"],
                    "next_priv_obs": next_priv_obs,
                }
            )
    return out
