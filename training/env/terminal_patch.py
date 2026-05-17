"""

Terminal-observation env step patch helpers

This module provides a helper function for patching an Isaac-style environment class to expose pre-reset terminal observations
on the "terminal_observation" key of the info/extras dict returned by the step method, used by the environment wrapper installation

File map:

PATCH_ATTR:                          Define patch attr constant
install_terminal_observation_patch:  Patch an Isaac-style env class to expose pre-reset terminal observations
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .terminal_observations import clone_observation_tree


PATCH_ATTR = "_phase1_terminal_observation_patch"


def install_terminal_observation_patch(
    env_class: type,                                           # Param: input value used as env class
    *,
    patch_attr: str                  = PATCH_ATTR,  # Param: string input for patch attr
    clone_obs : Callable[[Any], Any] = clone_observation_tree,  # Param: callback used to compute or fetch clone obs
) -> bool:
    """Patch an Isaac-style env class to expose pre-reset terminal observations

    Steps:
    - Resolve inputs for `install_terminal_observation_patch` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    step_method = getattr(env_class, "step")
    if getattr(step_method, patch_attr, False):
        return False

    def _step_with_terminal_observation(self, action):
        """Process for `_step_with_terminal_observation`

        Steps:
        - Resolve inputs for `_step_with_terminal_observation` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        terminal_observation = None

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
            if len(reset_env_ids) > 0:
                terminal_observation = clone_obs(self.obs_buf)
        elif len(reset_env_ids) > 0:
            terminal_observation = clone_obs(self.observation_manager.compute())

        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.extras["terminal_observation"] = terminal_observation
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    setattr(_step_with_terminal_observation, patch_attr, True)
    setattr(env_class, "step", _step_with_terminal_observation)
    return True
