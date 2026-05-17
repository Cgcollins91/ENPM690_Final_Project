# === File Documentation =====================================================
# Path: tasks/common_event/event_manager.py
# Summary: Implements event manager.
# Primary Purpose: Supports task registration and environment configuration.
# Project Component: tasks/common_event.
# Behavior Notes:
# - This file is part of the runtime/control/configuration pipeline for the project.
# - Keep edits behavior-preserving unless a related requirement explicitly requests logic changes.
# - Prefer adding tests when modifying interfaces, assumptions, or data contracts defined here.
# ============================================================================
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""

event manager

This module defines a simple event manager for registering and triggering events in the environment. 
It includes a `SimpleEvent` class for wrapping callback functions, a `MultiObjectEvent` class
for handling events that involve multiple objects, and a `BatchObjectEvent` class for more flexible multi-object event configurations.
The `SimpleEventManager` class provides a registry for events, allowing users to register events under stable names and trigger them with 
the environment context. This setup is designed to be compatible with the local task utilities and can be easily extended for more complex
event handling needs.

"""
import torch
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp


class SimpleEvent:
    """Tiny callback wrapper used by task code that expects an event-like object."""
    def __init__(self, func, params=None):
        """Initialize the instance and store the collaborators needed by later methods."""
        self.func = func
        self.params = params or {}

    def trigger(self, env):
        """Invoke the registered callback with the provided arguments."""
        return self.func(env, **self.params)


class MultiObjectEvent:
    """支持多个物体操作的事件类"""

    def __init__(self, reset_configs=None):
        """
        Args:
            reset_configs: 多物体重置配置列表，格式为:
            [
                {
                    "asset_cfg": SceneEntityCfg("red_block"),
                    "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
                    "velocity_range": {}
                },
                {
                    "asset_cfg": SceneEntityCfg("yellow_block"),
                    "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03]},
                    "velocity_range": {}
                }
            ]
        """
        self.reset_configs = reset_configs or []

    def trigger(self, env):
        """
        触发多物体重置事件
        """
        env_ids = torch.arange(env.num_envs, device=env.device)
        results = []

        for config in self.reset_configs:
            try:
                result = base_mdp.reset_root_state_uniform(
                    env=env,
                    env_ids=env_ids,
                    pose_range=config.get("pose_range", {}),
                    velocity_range=config.get("velocity_range", {}),
                    asset_cfg=config["asset_cfg"]
                )
                results.append(result)
                print(f"重置物体: {config['asset_cfg'].name}")
            except Exception as e:
                print(f"物体 {config['asset_cfg'].name} 重置失败: {e}")

        return results


class BatchObjectEvent:
    """批量物体事件类 - 更灵活的配置方式"""

    def __init__(self, object_names=None, pose_ranges=None, velocity_ranges=None):
        """
        Args:
            object_names: 物体名称列表，如 ["red_block", "yellow_block", "green_block"]
            pose_ranges: 位置范围配置，可以是:
                - 单个配置（所有物体使用相同范围）: {"x": [-0.05, 0.05], "y": [-0.05, 0.05]}
                - 字典配置（每个物体不同范围）: {"red_block": {...}, "yellow_block": {...}}
            velocity_ranges: 速度范围配置，格式同pose_ranges
        """
        self.object_names = object_names or []
        self.pose_ranges = pose_ranges or {}
        self.velocity_ranges = velocity_ranges or {}

    def trigger(self, env):
        """触发批量重置"""
        env_ids = torch.arange(env.num_envs, device=env.device)
        results = []

        for obj_name in self.object_names:
            try:
                # 获取该物体的pose_range配置
                if isinstance(self.pose_ranges, dict) and obj_name in self.pose_ranges:
                    pose_range = self.pose_ranges[obj_name]
                elif isinstance(self.pose_ranges, dict) and "x" in self.pose_ranges:
                    # 单个配置，所有物体使用相同配置
                    pose_range = self.pose_ranges
                else:
                    pose_range = {}

                # 获取该物体的velocity_range配置
                if isinstance(self.velocity_ranges, dict) and obj_name in self.velocity_ranges:
                    velocity_range = self.velocity_ranges[obj_name]
                elif isinstance(self.velocity_ranges, dict) and "linear" in self.velocity_ranges:
                    # 单个配置，所有物体使用相同配置
                    velocity_range = self.velocity_ranges
                else:
                    velocity_range = {}

                result = base_mdp.reset_root_state_uniform(
                    env=env,
                    env_ids=env_ids,
                    pose_range=pose_range,
                    velocity_range=velocity_range,
                    asset_cfg=SceneEntityCfg(obj_name)
                )
                results.append(result)
                print(f"✅ 重置物体: {obj_name}")

            except Exception as e:
                print(f"❌ 物体 {obj_name} 重置失败: {e}")

        return results


class SimpleEventManager:
    """Minimal name-to-event registry compatible with the local task utilities."""
    def __init__(self):
        """Initialize the instance and store the collaborators needed by later methods."""
        self._events = {}

    def register(self, name, event):
        """Register an event object under a stable name."""
        self._events[name] = event

    def trigger(self, name, env):
        """Invoke the registered callback with the provided arguments."""
        event = self._events.get(name)
        if event:
            return event.trigger(env)
        else:
            print(f"Event {name} not registered")

    def register_multi_object_reset(self, name, object_names, pose_ranges=None, velocity_ranges=None):
        """
        便捷方法：注册多物体重置事件

        Args:
            name           : 事件名称
            object_names   : 物体名称列表
            pose_ranges    : 位置范围配置
            velocity_ranges: 速度范围配置
        """
        event = BatchObjectEvent(
            object_names=object_names,
            pose_ranges=pose_ranges,
            velocity_ranges=velocity_ranges
        )
        self.register(name, event)