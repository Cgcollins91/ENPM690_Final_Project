"""Register the topdown reach-align-contact curriculum task with Gym.

This package owns:
- a single task ID: ``Isaac-Topdown-Curriculum-G129-Dex3-Joint``
- a self-contained scene, reward, observation, termination, and event stack
- a per-env stage state machine that routes reward shaping by curriculum stage
"""

# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import cgc_topdown_curriculum_env_cfg


gym.register(
    id="Isaac-Topdown-Curriculum-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cgc_topdown_curriculum_env_cfg.TopdownCurriculumEnvCfg,
    },
    disable_env_checker=True,
)
