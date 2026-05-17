"""

Native DifferentialIK controller and Jacobian boundary

File map:

DifferentialIKControllerSpec:            Controller settings used for Isaac DifferentialIK construction
NativeDifferentialIKControllers:         Pose and position DifferentialIK controller pair
NativeJacobianSelection:                 Selected arm Jacobian slice
create_differential_ik_controller_pair:  Construct pose and position DifferentialIK controllers from injected Isaac classes
robot_jacobians:                         Read all robot Jacobians through the Isaac PhysX view boundary
select_arm_jacobian:                     Select one body Jacobian over arm joints
compute_differential_ik_joint_pos:       Run one DifferentialIK compute call and validate the tensor result
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DifferentialIKControllerSpec:
    """Controller settings used for Isaac DifferentialIK construction"""

    command_type     : str  = "pose"  # Field: string command type value used by differential i k controller spec
    use_relative_mode: bool = False  # Field: boolean value indicating the use relative mode state for differential i k controller spec
    ik_method        : str  = "dls"  # Field: string ik method value used by differential i k controller spec


@dataclass(frozen=True)
class NativeDifferentialIKControllers:
    """Pose and position DifferentialIK controller pair"""

    controller             : Any  # Field: stores controller for native differential i k controllers
    position_controller    : Any  # Field: stores position controller for native differential i k controllers
    controller_cfg         : Any  # Field: stores controller cfg for native differential i k controllers
    position_controller_cfg: Any  # Field: stores position controller cfg for native differential i k controllers


@dataclass(frozen=True)
class NativeJacobianSelection:
    """Selected arm Jacobian slice"""

    jacobian  : torch.Tensor  # Field: tensor containing jacobian values for batched env rows
    body_index: int  # Field: index identifying the body entry
    joint_ids : tuple[int, ...]  # Field: integer joint ids value tracked by native jacobian selection


def create_differential_ik_controller_pair(
    *,
    controller_cls    : type,  # Param: input value used as controller cls
    controller_cfg_cls: type,  # Param: input value used as controller cfg cls
    num_envs          : int,  # Param: number of parallel environment rows represented
    device            : torch.device | str,  # Param: torch device where tensors are read or allocated
    pose_spec         : DifferentialIKControllerSpec = DifferentialIKControllerSpec(),  # Param: input value used as pose spec
    position_spec     : DifferentialIKControllerSpec = DifferentialIKControllerSpec(command_type="position"),  # Param: input value used as position spec
) -> NativeDifferentialIKControllers:
    """Construct pose and position DifferentialIK controllers from injected Isaac classes"""
    pose_cfg = controller_cfg_cls(
        command_type=pose_spec.command_type,
        use_relative_mode=pose_spec.use_relative_mode,
        ik_method=pose_spec.ik_method,
    )
    position_cfg = controller_cfg_cls(
        command_type=position_spec.command_type,
        use_relative_mode=position_spec.use_relative_mode,
        ik_method=position_spec.ik_method,
    )
    return NativeDifferentialIKControllers(
        controller=controller_cls(pose_cfg, num_envs=int(num_envs), device=device),
        position_controller=controller_cls(position_cfg, num_envs=int(num_envs), device=device),
        controller_cfg=pose_cfg,
        position_controller_cfg=position_cfg,
    )


def robot_jacobians(robot: object) -> torch.Tensor:
    """Read all robot Jacobians through the Isaac PhysX view boundary

    Steps:
    - Resolve inputs for `robot_jacobians` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    view = getattr(robot, "root_physx_view", None)
    if view is None or not hasattr(view, "get_jacobians"):
        raise RuntimeError("robot must expose root_physx_view.get_jacobians")
    jacobians = view.get_jacobians()
    if not torch.is_tensor(jacobians):
        raise TypeError(f"get_jacobians returned {type(jacobians)!r}")
    return jacobians


def select_arm_jacobian(
    all_jacobians: torch.Tensor,   # Param: tensor input carrying all jacobians values
    *,
    body_index   : int,  # Param: index selecting the body entry
    arm_joint_ids: Sequence[int],  # Param: integer input for arm joint ids
) -> NativeJacobianSelection:
    """Select one body Jacobian over arm joints"""
    joint_ids = tuple(int(value) for value in arm_joint_ids)
    return NativeJacobianSelection(
        jacobian=all_jacobians[:, int(body_index), :, list(joint_ids)],
        body_index=int(body_index),
        joint_ids=joint_ids,
    )


def compute_differential_ik_joint_pos(
    *,
    controller: object,  # Param: input value used as controller
    ee_pos_w  : torch.Tensor,  # Param: tensor input carrying ee pos w values
    ee_quat_w : torch.Tensor,  # Param: tensor input carrying ee quat w values
    jacobian  : torch.Tensor,  # Param: tensor input carrying jacobian values
    joint_pos : torch.Tensor,  # Param: current joint-position tensor used as the IK starting point
) -> torch.Tensor:
    """Run one DifferentialIK compute call and validate the tensor result"""
    if not hasattr(controller, "compute"):
        raise TypeError("DifferentialIK controller must expose compute")
    joint_pos_des = controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
    if not torch.is_tensor(joint_pos_des):
        raise TypeError(f"DifferentialIK compute returned {type(joint_pos_des)!r}")
    return joint_pos_des
