"""

Environment configuration helpers for trainer launch

File map:

DEFAULT_CAMERA_ATTRS:              Define default camera attrs constant
TouchEnvModeConfig:                Result of applying trainer mode changes to env config
should_disable_camera_perception:  Return whether camera perception should be disabled
remove_config_attr_if_present:     Set one config attribute to None when it exists and is active
configure_touch_env_for_mode:      Mutate env config for trainer controller and perception mode
viewport_camera_pose:              Return eye and target for a named viewport camera
simulation_should_continue:        Return whether the trainer should keep stepping simulation
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions.action_space import WORKSPACE_CAMERA_EYES, WORKSPACE_CAMERA_TARGET


DEFAULT_CAMERA_ATTRS = ("front_camera", "left_wrist_camera", "right_wrist_camera")


@dataclass(frozen=True)
class TouchEnvModeConfig:
    """Result of applying trainer mode changes to env config"""

    camera_perception_disabled: bool  # Field: boolean value indicating the camera perception disabled state for touch env mode config
    removed_obs_terms         : tuple[str, ...]  # Field: string removed obs terms value used by touch env mode config
    removed_scene_terms       : tuple[str, ...]  # Field: string removed scene terms value used by touch env mode config


def should_disable_camera_perception(*, disable_camera_perception: bool, arm_controller: str) -> bool:
    """Return whether camera perception should be disabled"""
    return bool(disable_camera_perception) or str(arm_controller) == "ik"


def remove_config_attr_if_present(owner: object | None, attr_name: str) -> bool:
    """Set one config attribute to None when it exists and is active"""
    if owner is None or getattr(owner, attr_name, None) is None:
        return False
    setattr(owner, attr_name, None)
    return True


def configure_touch_env_for_mode(
    env_cfg,                                               # Param: input value used as env cfg
    *,
    disable_camera_perception: bool,  # Param: boolean input controlling disable camera perception
    arm_controller           : str,  # Param: string input for arm controller
    camera_attrs             : tuple[str, ...] = DEFAULT_CAMERA_ATTRS,  # Param: string input for camera attrs
) -> TouchEnvModeConfig:
    """Mutate env config for trainer controller and perception mode

    Steps:
    - Resolve inputs for `configure_touch_env_for_mode` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    camera_disabled = should_disable_camera_perception(
        disable_camera_perception=disable_camera_perception,
        arm_controller=arm_controller,
    )
    removed_obs_terms  : list[str] = []
    removed_scene_terms: list[str] = []
    if not camera_disabled:
        return TouchEnvModeConfig(False, (), ())

    policy_obs = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if remove_config_attr_if_present(policy_obs, "red_block_perception"):
        removed_obs_terms.append("red_block_perception")

    scene_cfg = getattr(env_cfg, "scene", None)
    for attr_name in camera_attrs:
        if remove_config_attr_if_present(scene_cfg, attr_name):
            removed_scene_terms.append(attr_name)

    return TouchEnvModeConfig(
        camera_perception_disabled=True,
        removed_obs_terms=tuple(removed_obs_terms),
        removed_scene_terms=tuple(removed_scene_terms),
    )


def viewport_camera_pose(
    viewport_camera: str,                                                        # Param: string input for viewport camera
    *,
    camera_eyes: dict[str, tuple[float, float, float]] = WORKSPACE_CAMERA_EYES,  # Param: floating-point input for camera eyes
    target     : tuple[float, float, float]            = WORKSPACE_CAMERA_TARGET,  # Param: floating-point input for target
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return eye and target for a named viewport camera"""
    eye = camera_eyes.get(viewport_camera)
    if eye is None:
        return None
    return eye, target


def simulation_should_continue(*, headless: bool, app_is_running: bool) -> bool:
    """Return whether the trainer should keep stepping simulation"""
    if bool(headless):
        return True
    return bool(app_is_running)
