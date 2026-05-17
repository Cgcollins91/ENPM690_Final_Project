"""

Standalone task-family routing helpers

These functions and data structures implement task-family routing based on the task name and environment variables, used by the standalone
trainer to select task-specific behavior and routing without relying on the legacy monolith trainer's curriculum stage machinery.
The task route is resolved from the task name and environment at startup, and can be used to conditionally activate task-specific logic
such as different reward terms, observation processing, or action selection. The standalone trainer contract currently supports only a single
topdown curriculum task, but this routing machinery can be extended in the future to support additional tasks or task families as needed.

File map:

TaskRoute:                     Resolved task-family flags for the standalone trainer
resolve_task_route:            Resolve task-family routing from task name and environment
is_lift_only_task:             Return whether the legacy lift-only task is active
is_grasp_contact_task:         Return whether a legacy grasp contact task is active
is_grasp_light_contact_task:   Return whether a legacy light contact task is active
is_grasp_align_task:           Return whether a legacy align task is active
is_topdown_lift_task:          Return whether topdown lift mode is active
is_grasp_contact_family_task:  Return whether any contact-family training task is active
use_topdown_contact_chain:     Return whether contact metrics route through topdown sensors
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .runtime import SUPPORTED_TOPDOWN_TASK, env_flag


@dataclass(frozen=True)
class TaskRoute:
    """Resolved task-family flags for the standalone trainer"""

    task             : str  # Field: string task value used by task route
    topdown_lift_task: bool = False  # Field: boolean value indicating the topdown lift task state for task route

    @property
    def is_lift_only_task(self) -> bool:
        """Return whether the legacy lift-only task is active"""
        return False

    @property
    def is_grasp_contact_task(self) -> bool:
        """Return whether a legacy grasp contact task is active"""
        return False

    @property
    def is_grasp_light_contact_task(self) -> bool:
        """Return whether a legacy light contact task is active"""
        return False

    @property
    def is_grasp_align_task(self) -> bool:
        """Return whether a legacy align task is active"""
        return False

    @property
    def is_topdown_lift_task(self) -> bool:
        """Return whether topdown lift mode is active"""
        return bool(self.topdown_lift_task)

    @property
    def is_grasp_contact_family_task(self) -> bool:
        """Return whether any contact-family training task is active"""
        return self.is_grasp_contact_task or self.is_grasp_light_contact_task or self.is_grasp_align_task

    @property
    def is_topdown_curriculum_task(self) -> bool:
        """Return whether the supported topdown curriculum task is active"""
        return self.task == SUPPORTED_TOPDOWN_TASK

    @property
    def is_topdown_curriculum_lift_task(self) -> bool:
        """Return whether topdown curriculum lift mode is active"""
        return self.is_topdown_curriculum_task and self.is_topdown_lift_task

    @property
    def use_topdown_contact_chain(self) -> bool:
        """Return whether contact metrics route through topdown sensors"""
        return self.is_topdown_curriculum_task

    @property
    def contact_family_or_topdown(self) -> bool:
        """Return whether contact-stage machinery may be active"""
        return self.is_grasp_contact_family_task or self.is_topdown_curriculum_task

    def validate_supported(self) -> None:
        """Raise when task is outside the standalone trainer contract"""
        if self.task != SUPPORTED_TOPDOWN_TASK:
            raise RuntimeError(
                "ENPM690 standalone trainer supports only "
                f"{SUPPORTED_TOPDOWN_TASK}; got {self.task!r}"
            )


def resolve_task_route(
    task: str,  # Param: string input for task
    env : Mapping[str, str] | None = None,  # Param: environment or backend object used for runtime calls
) -> TaskRoute:
    """Resolve task-family routing from task name and environment"""
    return TaskRoute(
        task=str(task),
        topdown_lift_task=env_flag("TOPDOWN_LIFT_TASK", False, env),
    )


def is_lift_only_task(route: TaskRoute) -> bool:
    """Return whether the legacy lift-only task is active"""
    return route.is_lift_only_task


def is_grasp_contact_task(route: TaskRoute) -> bool:
    """Return whether a legacy grasp contact task is active"""
    return route.is_grasp_contact_task


def is_grasp_light_contact_task(route: TaskRoute) -> bool:
    """Return whether a legacy light contact task is active"""
    return route.is_grasp_light_contact_task


def is_grasp_align_task(route: TaskRoute) -> bool:
    """Return whether a legacy align task is active"""
    return route.is_grasp_align_task


def is_topdown_lift_task(route: TaskRoute) -> bool:
    """Return whether topdown lift mode is active"""
    return route.is_topdown_lift_task


def is_grasp_contact_family_task(route: TaskRoute) -> bool:
    """Return whether any contact-family training task is active"""
    return route.is_grasp_contact_family_task


def use_topdown_contact_chain(route: TaskRoute) -> bool:
    """Return whether contact metrics route through topdown sensors"""
    return route.use_topdown_contact_chain
