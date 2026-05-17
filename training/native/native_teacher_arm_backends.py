"""

Native teacher arm backend adapters

File map:

CallableTeacherArmBackend:               TeacherArmBackend that delegates to a request-shaped callable
LegacyCallableTeacherArmBackend:         TeacherArmBackend for legacy env mapped_indices mapped_scales callables
EnvMethodTeacherArmBackend:              TeacherArmBackend that dispatches to an env-owned method
_float_arg:                              Handle float arg logic
_env_bool:                               Handle env bool logic
_env_int:                                Handle env int logic
_env_float:                              Handle env float logic
_env_float_list:                         Handle env float list logic
_disable_servo_after_latch:              Handle disable servo after latch logic
_tensor_1d:                              Handle tensor 1d logic
_make_ik_controller:                     Handle make ik controller logic
_jacobian_body_index:                    Handle jacobian body index logic
_robot_joint_names:                      Handle robot joint names logic
_joint_names_for_ids:                    Handle joint names for ids logic
_joint_enabled_from_env:                 Handle joint enabled from env logic
TopdownDifferentialIKTeacherArmBackend:  Native topdown teacher arm backend when the Isaac env does not expose one
ensure_teacher_arm_action:               Validate and clamp a teacher arm action tensor
ValidatingTeacherArmBackend:             TeacherArmBackend wrapper that validates output shape
build_env_teacher_arm_backend:           Build an env-method teacher arm backend
env_has_teacher_arm_method:              Return whether env exposes a configured teacher arm method
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import os

import torch

from ..actions.action_space import RIGHT_PALM_LINK
from ..env.isaac_backend import load_isaac_runtime_symbols
from ..geometry.ik_masks import (
    topdown_arm_hold_frozen_mask,
    topdown_lift_servo_correction_mask,
    topdown_prehold_position_only_mask,
)
from ..geometry.ik_servo import align_line_z_delta, tip_jacobian_joint_weights
from ..geometry.lift_latch import (
    initial_arm_lift_latch_state,
    update_arm_lift_latch_tensors,
)
from ..geometry.pocket_sweep import pocket_sweep_search
from ..geometry.tip_geometry import INDEX_TIP_LINK, THUMB_TIP_LINK
from .native_ik import robot_jacobians
from ..teacher.teacher_arm_controller import TeacherArmRequest


RequestArmFn = Callable[[TeacherArmRequest], torch.Tensor]
LegacyArmFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class CallableTeacherArmBackend:
    """TeacherArmBackend that delegates to a request-shaped callable"""

    compute_fn : RequestArmFn  # Field: callback used for the compute fn operation

    def compute_teacher_arm_reduced(self, request: TeacherArmRequest) -> torch.Tensor:
        """Return arm action from the injected callable"""
        return self.compute_fn(request)


@dataclass(frozen=True)
class LegacyCallableTeacherArmBackend:
    """TeacherArmBackend for legacy env mapped_indices mapped_scales callables"""

    compute_fn : LegacyArmFn  # Field: callback used for the compute fn operation

    def compute_teacher_arm_reduced(self, request: TeacherArmRequest) -> torch.Tensor:
        """Return arm action from a legacy-shaped callable"""
        return self.compute_fn(
            request.env,
            request.mapped_indices,
            request.mapped_scales,
            closure_fraction=request.closure_fraction,
            episode_step=request.episode_step,
            topdown_contact_descent=request.topdown_contact_descent,
            topdown_contact_xy_offset=request.topdown_contact_xy_offset,
            topdown_contact_inward=request.topdown_contact_inward,
            topdown_contact_tip_servo=request.topdown_contact_tip_servo,
        )


@dataclass(frozen=True)
class EnvMethodTeacherArmBackend:
    """TeacherArmBackend that dispatches to an env-owned method"""

    env         : object  # Field: environment/backend object used by this runtime helper
    method_names: tuple[str, ...] = (  # Field: ordered names used to resolve method attributes
        "native_teacher_arm_reduced",
        "compute_native_teacher_arm_reduced",
        "compute_teacher_arm_reduced",
    )
    legacy_call  : bool = False         # Field: boolean value indicating the legacy call state for env method teacher arm backend

    def _method(self) -> Callable[..., torch.Tensor]:
        for name in self.method_names:
            value = getattr(self.env, name, None)
            if callable(value):
                return value
        raise RuntimeError(f"env is missing teacher arm method: {self.method_names}")

    def compute_teacher_arm_reduced(self, request: TeacherArmRequest) -> torch.Tensor:
        """Return arm action from the first configured env method"""
        method = self._method()
        if self.legacy_call:
            return LegacyCallableTeacherArmBackend(method).compute_teacher_arm_reduced(request)
        return method(request)


def _float_arg(args: dict, name: str, default: float) -> float:
    try:
        return float(args.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_float_list(name: str, default: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in os.environ.get(name, default).split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            value = abs(float(cleaned))
        except ValueError:
            continue
        if value > 0.0:
            values.append(value)
    return tuple(values)


def _disable_servo_after_latch(servo_name: str | None = None) -> bool:
    default_value = _env_bool("TOPDOWN_LIFT_DISABLE_SERVO_CORRECTIONS_AFTER_LATCH", False)
    if not servo_name:
        return default_value
    return _env_bool(f"TOPDOWN_LIFT_DISABLE_{servo_name.upper()}_AFTER_LATCH", default_value)


def _tensor_1d(
    value: float | torch.Tensor,
    *,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.full((int(num_envs),), float(value), device=device, dtype=dtype)


def _make_ik_controller(controller_cls: type, cfg_cls: type, *, command_type: str, num_envs: int, device, damping: float):
    try:
        cfg = cfg_cls(
            command_type=command_type,
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(damping)},
        )
    except TypeError:
        cfg = cfg_cls(command_type=command_type, use_relative_mode=False, ik_method="dls")
    return controller_cls(cfg, num_envs=int(num_envs), device=device)


def _jacobian_body_index(robot, body_idx: int) -> int:
    return int(body_idx) - 1 if bool(getattr(robot, "is_fixed_base", False)) else int(body_idx)


def _robot_joint_names(robot) -> tuple[str, ...]:
    names = getattr(robot.data, "joint_names", None)
    if names is None:
        return tuple()
    return tuple(str(name) for name in names)


def _joint_names_for_ids(robot, joint_ids: Sequence[int]) -> tuple[str, ...]:
    names = _robot_joint_names(robot)
    if not names:
        return tuple(str(index) for index in range(len(joint_ids)))
    return tuple(names[int(index)] for index in joint_ids)


def _joint_enabled_from_env(
    name: str,
    *,
    joint_names: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    spec = os.environ.get(name, "all").strip()
    lowered = {item.strip().lower() for item in spec.split(",") if item.strip()}
    if not lowered or "all" in lowered:
        enabled = [True for _ in joint_names]
    else:
        selected = {item.strip() for item in spec.split(",") if item.strip()}
        enabled = [joint_name in selected for joint_name in joint_names]
    return torch.tensor(enabled, dtype=torch.bool, device=device)


@dataclass
class TopdownDifferentialIKTeacherArmBackend:
    """Native topdown teacher arm backend when the Isaac env does not expose one."""

    env    : object
    args   : dict
    num_arm: int

    _state: dict | None = None

    def _ensure_state(self, request: TeacherArmRequest) -> dict:
        """Process for `_ensure_state`

        Steps:
        - Resolve inputs for `_ensure_state` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        if self._state is not None:
            return self._state
        robot = self.env.scene["robot"]
        symbols = load_isaac_runtime_symbols()
        damping = _float_arg(self.args, "ik_damping", 0.05)
        arm_joint_ids = tuple(int(value) for value in request.mapped_indices[: int(self.num_arm)].tolist())
        joint_names = _joint_names_for_ids(robot, arm_joint_ids)
        body_idx = list(robot.data.body_names).index(RIGHT_PALM_LINK)
        thumb_body_idx = list(robot.data.body_names).index(THUMB_TIP_LINK)
        index_body_idx = list(robot.data.body_names).index(INDEX_TIP_LINK)
        jacobian_body_idx = _jacobian_body_index(robot, body_idx)
        self._state = {
            "controller": _make_ik_controller(
                symbols.differential_ik_controller,
                symbols.differential_ik_controller_cfg,
                command_type="pose",
                num_envs=int(getattr(self.env, "num_envs")),
                device=getattr(self.env, "device"),
                damping=damping,
            ),
            "position_controller": _make_ik_controller(
                symbols.differential_ik_controller,
                symbols.differential_ik_controller_cfg,
                command_type="position",
                num_envs=int(getattr(self.env, "num_envs")),
                device=getattr(self.env, "device"),
                damping=damping,
            ),
            "arm_joint_ids"    : arm_joint_ids,
            "arm_joint_names"  : joint_names,
            "body_idx"         : body_idx,
            "jacobian_body_idx": jacobian_body_idx,
            "contact_thumb_body_idx"        : thumb_body_idx,
            "contact_index_body_idx"        : index_body_idx,
            "contact_thumb_jacobian_body_idx": _jacobian_body_index(robot, thumb_body_idx),
            "contact_index_jacobian_body_idx": _jacobian_body_index(robot, index_body_idx),
            "align_servo_joint_weights": tip_jacobian_joint_weights(
                joint_names,
                spec=os.environ.get("TOPDOWN_PREHOLD_ALIGN_ANGLE_JOINTS", "all"),
                device=getattr(self.env, "device"),
                dtype=torch.float32,
            ),
            "pocket_sweep_joint_enabled": _joint_enabled_from_env(
                "TOPDOWN_POCKET_SWEEP_JOINTS",
                joint_names=joint_names,
                device=getattr(self.env, "device"),
            ),
            "max_step"         : _float_arg(self.args, "ik_max_joint_step", 0.05),
            "lift": initial_arm_lift_latch_state(
                num_envs=int(getattr(self.env, "num_envs")),
                device=getattr(self.env, "device"),
            ),
        }
        return self._state

    def _target_pose(self, request: TeacherArmRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Process for `_target_pose`

        Steps:
        - Resolve inputs for `_target_pose` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp import state_machine
        from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp.topdown_geometry import (
            get_topdown_target_palm_quat,
            topdown_palm_position_from_grip_target,
        )

        env = self.env
        block_pos, _ = state_machine._block_pose(env)
        target_pos = topdown_palm_position_from_grip_target(
            env,
            state_machine._grip_target_position(env),
        ).to(device=env.device, dtype=torch.float32)
        target_quat = get_topdown_target_palm_quat(env).to(device=env.device, dtype=torch.float32)

        closure = _tensor_1d(
            request.closure_fraction,
            num_envs=int(env.num_envs),
            device=env.device,
            dtype=target_pos.dtype,
        ).clamp(0.0, 1.0)
        inward_vec = block_pos.to(device=env.device, dtype=target_pos.dtype) - target_pos
        inward_dir = inward_vec / torch.linalg.norm(inward_vec, dim=1, keepdim=True).clamp_min(1.0e-6)
        inward = 0.006 * closure
        if request.topdown_contact_inward is not None:
            inward = inward + torch.clamp(
                _tensor_1d(
                    request.topdown_contact_inward,
                    num_envs=int(env.num_envs),
                    device=env.device,
                    dtype=target_pos.dtype,
                ),
                min=0.0,
            )
        target_pos = target_pos + inward_dir * inward.unsqueeze(-1)
        env._teacher_ik_topdown_inward_m = inward.detach().clone()

        if request.topdown_contact_xy_offset is not None:
            xy = request.topdown_contact_xy_offset.to(device=env.device, dtype=target_pos.dtype)
            target_pos[:, :2] = target_pos[:, :2] + xy.reshape(env.num_envs, 2)
        if request.topdown_contact_descent is not None:
            descent = torch.clamp(
                _tensor_1d(
                    request.topdown_contact_descent,
                    num_envs=int(env.num_envs),
                    device=env.device,
                    dtype=target_pos.dtype,
                ),
                min=0.0,
            )
            target_pos[:, 2] = target_pos[:, 2] - descent
        if request.topdown_contact_tip_servo is not None:
            servo = request.topdown_contact_tip_servo.to(device=env.device, dtype=target_pos.dtype)
            target_pos = target_pos + servo.reshape(env.num_envs, 3)
            env._teacher_ik_topdown_tip_servo = servo.detach().clone()
            env._teacher_ik_topdown_tip_servo_m = torch.linalg.norm(servo, dim=1).detach().clone()

        if request.episode_step is not None:
            self._apply_lift_target(target_pos, block_pos, request.episode_step, state_machine)
        return target_pos, target_quat

    def _apply_lift_target(
        self,
        target_pos: torch.Tensor,
        block_pos: torch.Tensor,
        episode_step: int | torch.Tensor,
        state_machine,
    ) -> None:
        """Process for `_apply_lift_target`

        Steps:
        - Resolve inputs for `_apply_lift_target` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        env = self.env
        state = self._state if self._state is not None else {}
        latch_state = state.get("lift")
        step = _tensor_1d(
            episode_step,
            num_envs=int(env.num_envs),
            device=env.device,
            dtype=torch.float32,
        )
        contact = state_machine.opposed_contact_strength(env) >= 0.5
        if latch_state is None:
            latch_state = initial_arm_lift_latch_state(num_envs=int(env.num_envs), device=env.device)
        updated = update_arm_lift_latch_tensors(
            episode_step=step,
            latched=latch_state.latched,
            latch_step=latch_state.latch_step,
            contact_counter=latch_state.contact_counter,
            latch_update_step=latch_state.latch_update_step,
            contact_signal=contact.to(dtype=torch.float32),
            touch_now=contact,
            hold_steps=1,
        )
        state["lift"] = updated
        env._arm_lift_latched = updated.latched.detach().clone()
        env._arm_lift_latch_step = updated.latch_step.detach().clone()
        env._arm_lift_contact_counter = updated.contact_counter.detach().clone()
        env._arm_lift_latch_signal = updated.latch_signal.detach().clone()

        settle = _float_arg(self.args, "topdown_lift_grip_settle_steps", 20.0)
        ramp = max(_float_arg(self.args, "teacher_lift_ramp_steps", 60.0), 1.0)
        lift_z = _float_arg(self.args, "teacher_lift_z", 0.08)
        since = torch.where(
            updated.latch_step >= 0.0,
            step - updated.latch_step,
            torch.full_like(step, -1.0),
        )
        lift_linear = torch.clamp((since - settle) / ramp, 0.0, 1.0)
        lift_progress = lift_linear * lift_linear * (3.0 - 2.0 * lift_linear)
        target_pos[:, 2] = target_pos[:, 2] + float(lift_z) * lift_progress
        env._teacher_ik_topdown_lift_progress = lift_progress.detach().clone()
        env._arm_lift_target_xy = target_pos[:, :2].detach().clone()
        env._arm_lift_target_base_z = target_pos[:, 2].detach().clone()
        env._arm_lift_block_xy_latch = block_pos[:, :2].detach().clone()

    def _stage_mask(self, stage_min: int, *, dtype: torch.dtype) -> torch.Tensor:
        env = self.env
        stage = getattr(env, "_topdown_stage", None)
        if torch.is_tensor(stage) and stage.shape[0] == int(env.num_envs):
            return stage.to(device=env.device) >= int(stage_min)
        return torch.ones(int(env.num_envs), dtype=torch.bool, device=env.device)

    def _clear_planar_and_tip_jacobian_diagnostics(self, dtype: torch.dtype) -> None:
        env = self.env
        zero = torch.zeros(int(env.num_envs), dtype=dtype, device=env.device)
        active = torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)
        for name in (
            "_teacher_ik_topdown_planar_align_err_xy",
            "_teacher_ik_topdown_planar_align_servo_q",
            "_teacher_ik_topdown_planar_align_servo_m",
            "_topdown_planar_align_err",
            "_topdown_planar_align_q",
        ):
            setattr(env, name, zero.detach().clone())
        for name in (
            "_teacher_ik_topdown_planar_align_servo_active",
            "_topdown_planar_align_active",
        ):
            setattr(env, name, active.detach().clone())
        for name in (
            "_teacher_ik_topdown_tip_jacobian_ik_err_before",
            "_teacher_ik_topdown_tip_jacobian_ik_err_after",
            "_teacher_ik_topdown_tip_jacobian_ik_q",
            "_topdown_tip_jac_before",
            "_topdown_tip_jac_after",
            "_topdown_tip_jac_q",
        ):
            setattr(env, name, zero.detach().clone())
        for name in (
            "_teacher_ik_topdown_tip_jacobian_ik_active",
            "_topdown_tip_jac_active",
        ):
            setattr(env, name, active.detach().clone())

    def _apply_pocket_sweep(
        self,
        *,
        state: dict,
        all_jacobians: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_pos_des: torch.Tensor,
        soft_limits: torch.Tensor,
    ) -> torch.Tensor:
        env = self.env
        zero = torch.zeros(int(env.num_envs), dtype=joint_pos_des.dtype, device=env.device)
        inactive = torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)

        def write_diag(q: torch.Tensor, before: torch.Tensor, after: torch.Tensor, active: torch.Tensor) -> None:
            for name in ("_teacher_ik_topdown_pocket_sweep_q", "_topdown_pocket_sweep_q"):
                setattr(env, name, q.detach().clone())
            for name in ("_teacher_ik_topdown_pocket_sweep_score_before", "_topdown_pocket_score_before"):
                setattr(env, name, before.detach().clone())
            for name in ("_teacher_ik_topdown_pocket_sweep_score_after", "_topdown_pocket_score_after"):
                setattr(env, name, after.detach().clone())
            setattr(env, "_teacher_ik_topdown_pocket_sweep_active", active.detach().clone())

        if not _env_bool("TOPDOWN_POCKET_SWEEP", False):
            write_diag(zero, zero, zero, inactive)
            return joint_pos_des

        active = self._stage_mask(_env_int("TOPDOWN_POCKET_SWEEP_STAGE_MIN", 2), dtype=joint_pos_des.dtype)
        active = active & (~topdown_arm_hold_frozen_mask(env, active.shape))
        active = active & (
            ~topdown_lift_servo_correction_mask(
                env,
                active.shape,
                topdown_curriculum_lift_task=True,
                disable_after_latch=_disable_servo_after_latch("POCKET_SWEEP"),
            )
        )
        step_degs = _env_float_list("TOPDOWN_POCKET_SWEEP_DEG", "4,2,1")
        joint_enabled = state["pocket_sweep_joint_enabled"].to(device=env.device, dtype=torch.bool)
        if not bool(active.any().item()) or not step_degs or not bool(joint_enabled.any().item()):
            write_diag(zero, zero, zero, active)
            return joint_pos_des

        from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp import state_machine

        robot = env.scene["robot"]
        arm_joint_ids = list(state["arm_joint_ids"])
        thumb_pos = robot.data.body_link_pose_w[:, int(state["contact_thumb_body_idx"]), :3].to(dtype=joint_pos_des.dtype)
        index_pos = robot.data.body_link_pose_w[:, int(state["contact_index_body_idx"]), :3].to(dtype=joint_pos_des.dtype)
        thumb_target, index_target = state_machine._face_targets(env)
        thumb_target = thumb_target.to(device=env.device, dtype=joint_pos_des.dtype)
        index_target = index_target.to(device=env.device, dtype=joint_pos_des.dtype)
        thumb_j = all_jacobians[
            :,
            int(state["contact_thumb_jacobian_body_idx"]),
            :3,
            arm_joint_ids,
        ].to(dtype=joint_pos_des.dtype)
        index_j = all_jacobians[
            :,
            int(state["contact_index_jacobian_body_idx"]),
            :3,
            arm_joint_ids,
        ].to(dtype=joint_pos_des.dtype)
        result = pocket_sweep_search(
            joint_pos=joint_pos,
            joint_pos_des=joint_pos_des,
            soft_limits=soft_limits,
            max_step=torch.full_like(joint_pos, max(float(state["max_step"]), 0.0)),
            active=active,
            thumb_pos=thumb_pos,
            index_pos=index_pos,
            thumb_target=thumb_target,
            index_target=index_target,
            thumb_jacobian=thumb_j,
            index_jacobian=index_j,
            step_radians=tuple(math.radians(value) for value in step_degs),
            joint_enabled=joint_enabled,
            z_weight=max(_env_float("TOPDOWN_POCKET_SWEEP_Z_WEIGHT", 0.25), 0.0),
            iters=max(_env_int("TOPDOWN_POCKET_SWEEP_ITERS", 2), 1),
        )
        q = torch.linalg.norm(result.delta_q, dim=1)
        write_diag(q, result.score_before, result.score_after, result.active)
        return result.joint_pos_des

    def _apply_align_angle_servo(
        self,
        *,
        state: dict,
        all_jacobians: torch.Tensor,
        joint_pos_des: torch.Tensor,
    ) -> torch.Tensor:
        env = self.env
        zero = torch.zeros(int(env.num_envs), dtype=joint_pos_des.dtype, device=env.device)
        inactive = torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)

        def write_diag(line_z: torch.Tensor, q: torch.Tensor, dz: torch.Tensor, active: torch.Tensor) -> None:
            for name in ("_teacher_ik_topdown_align_line_z", "_topdown_align_line_z"):
                setattr(env, name, line_z.detach().clone())
            for name in ("_teacher_ik_topdown_align_servo_q", "_topdown_align_servo_q"):
                setattr(env, name, q.detach().clone())
            setattr(env, "_teacher_ik_topdown_align_servo_dz", dz.detach().clone())
            for name in ("_teacher_ik_topdown_align_servo_active", "_topdown_align_servo_active"):
                setattr(env, name, active.detach().clone())

        if not _env_bool("TOPDOWN_PREHOLD_ALIGN_ANGLE_SERVO", False):
            write_diag(zero, zero, zero, inactive)
            return joint_pos_des
        gain = max(_env_float("TOPDOWN_PREHOLD_ALIGN_ANGLE_GAIN", 0.0), 0.0)
        max_dz = max(_env_float("TOPDOWN_PREHOLD_ALIGN_ANGLE_MAX_DZ", 0.0), 0.0)
        max_joint_step = max(_env_float("TOPDOWN_PREHOLD_ALIGN_ANGLE_MAX_JOINT_STEP", 0.0), 0.0)
        if gain <= 0.0 or max_dz <= 0.0 or max_joint_step <= 0.0:
            write_diag(zero, zero, zero, inactive)
            return joint_pos_des

        active = self._stage_mask(_env_int("TOPDOWN_PREHOLD_ALIGN_ANGLE_STAGE_MIN", 1), dtype=joint_pos_des.dtype)
        active = active & (~topdown_arm_hold_frozen_mask(env, active.shape))
        active = active & (
            ~topdown_lift_servo_correction_mask(
                env,
                active.shape,
                topdown_curriculum_lift_task=True,
                disable_after_latch=_disable_servo_after_latch("ALIGN_SERVO"),
            )
        )
        robot = env.scene["robot"]
        arm_joint_ids = list(state["arm_joint_ids"])
        thumb_pos = robot.data.body_link_pose_w[:, int(state["contact_thumb_body_idx"]), :3].to(dtype=joint_pos_des.dtype)
        index_pos = robot.data.body_link_pose_w[:, int(state["contact_index_body_idx"]), :3].to(dtype=joint_pos_des.dtype)
        line_z = index_pos[:, 2] - thumb_pos[:, 2]
        deadband = max(_env_float("TOPDOWN_PREHOLD_ALIGN_ANGLE_DEADBAND_M", 0.0), 0.0)
        active = active & (torch.abs(line_z) > deadband)
        if not bool(active.any().item()):
            write_diag(line_z, zero, zero, active)
            return joint_pos_des

        thumb_j = all_jacobians[
            :,
            int(state["contact_thumb_jacobian_body_idx"]),
            :3,
            arm_joint_ids,
        ].to(dtype=joint_pos_des.dtype)
        index_j = all_jacobians[
            :,
            int(state["contact_index_jacobian_body_idx"]),
            :3,
            arm_joint_ids,
        ].to(dtype=joint_pos_des.dtype)
        weights = state["align_servo_joint_weights"].to(device=env.device, dtype=joint_pos_des.dtype).view(1, -1)
        line_j = (index_j[:, 2, :] - thumb_j[:, 2, :]) * weights
        delta = align_line_z_delta(
            line_z=line_z,
            line_jacobian=line_j,
            active=active,
            gain=gain,
            max_dz=max_dz,
            damping=max(_env_float("TOPDOWN_PREHOLD_ALIGN_ANGLE_DAMPING", 0.02), 0.0),
            max_joint_step=max_joint_step,
        )
        desired_dz = torch.clamp(-gain * line_z, min=-max_dz, max=max_dz)
        applied_dz = torch.where(delta.active, desired_dz, torch.zeros_like(desired_dz))
        q = torch.linalg.norm(delta.delta_q, dim=1)
        write_diag(line_z, q, applied_dz, delta.active)
        return joint_pos_des + delta.delta_q

    def compute_teacher_arm_reduced(self, request: TeacherArmRequest) -> torch.Tensor:
        """Process for `compute_teacher_arm_reduced`

        Steps:
        - Resolve inputs for `compute_teacher_arm_reduced` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        state = self._ensure_state(request)
        env = self.env
        robot = env.scene["robot"]
        arm_joint_ids = state["arm_joint_ids"]
        body_idx = int(state["body_idx"])
        jacobian_body_idx = int(state["jacobian_body_idx"])

        ee_pose_w = robot.data.body_link_pose_w[:, body_idx]
        ee_pos_w = ee_pose_w[:, :3]
        ee_quat_w = ee_pose_w[:, 3:7]
        joint_pos = robot.data.joint_pos[:, list(arm_joint_ids)]
        default_joint_pos = robot.data.default_joint_pos[:, list(arm_joint_ids)]
        all_jacobians = robot_jacobians(robot)
        jacobian = all_jacobians[:, jacobian_body_idx, :, list(arm_joint_ids)]
        soft_limits = robot.data.soft_joint_pos_limits[:, list(arm_joint_ids)]
        target_pos, target_quat = self._target_pose(request)

        command = torch.cat([target_pos, target_quat], dim=-1)
        controller = state["controller"]
        controller.set_command(command, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
        joint_pos_des = controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
        position_only_mask = topdown_prehold_position_only_mask(
            env,
            enabled=_env_bool("TOPDOWN_PREHOLD_IK_POSITION_ONLY", False),
            topdown_curriculum_task=True,
            stage_min=_env_int("TOPDOWN_PREHOLD_IK_POSITION_ONLY_STAGE_MIN", 2),
            topdown_curriculum_lift_task=True,
            disable_prehold_servos_after_latch=_env_bool("TOPDOWN_LIFT_DISABLE_PREHOLD_SERVOS_AFTER_LATCH", True),
        )
        if bool(position_only_mask.any().item()):
            position_controller = state["position_controller"]
            position_controller.set_command(target_pos, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
            position_joint_pos_des = position_controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
            joint_pos_des = torch.where(
                position_only_mask.unsqueeze(-1),
                position_joint_pos_des,
                joint_pos_des,
            )
        env._teacher_ik_position_only = position_only_mask.detach().clone()
        self._clear_planar_and_tip_jacobian_diagnostics(joint_pos_des.dtype)
        joint_pos_des = self._apply_pocket_sweep(
            state=state,
            all_jacobians=all_jacobians,
            joint_pos=joint_pos,
            joint_pos_des=joint_pos_des,
            soft_limits=soft_limits,
        )
        joint_pos_des = self._apply_align_angle_servo(
            state=state,
            all_jacobians=all_jacobians,
            joint_pos_des=joint_pos_des,
        )
        joint_pos_des = torch.clamp(joint_pos_des, min=soft_limits[..., 0], max=soft_limits[..., 1])
        max_step = max(float(state["max_step"]), 0.0)
        if max_step > 0.0:
            delta = torch.clamp(joint_pos_des - joint_pos, min=-max_step, max=max_step)
            joint_pos_des = joint_pos + delta

        env._teacher_ik_target_z = target_pos[:, 2].detach().clone()
        env._teacher_ik_actual_z = ee_pos_w[:, 2].detach().clone()
        env._teacher_ik_target_z_gap = (target_pos[:, 2] - ee_pos_w[:, 2]).detach().clone()
        raw_action = joint_pos_des - default_joint_pos
        arm_scales = request.mapped_scales[: int(self.num_arm)].to(device=env.device).unsqueeze(0)
        arm_scales = torch.where(
            arm_scales.abs() < 1.0e-6,
            torch.ones_like(arm_scales),
            arm_scales,
        )
        return (raw_action / arm_scales).clamp(-1.0, 1.0)


def ensure_teacher_arm_action(
    value: torch.Tensor,         # Param: input value normalized or converted by this helper
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    num_arm : int,  # Param: number of arm action dimensions in the active layout
    device  : torch.device | str,  # Param: torch device where tensors are read or allocated
) -> torch.Tensor:
    """Validate and clamp a teacher arm action tensor"""
    if not torch.is_tensor(value):
        raise TypeError(f"teacher arm action must be a tensor, got {type(value)!r}")
    expected = (int(num_envs), int(num_arm))
    if tuple(value.shape) != expected:
        raise RuntimeError(f"teacher arm action shape mismatch: expected {expected}, got {tuple(value.shape)}")
    return value.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)


@dataclass(frozen=True)
class ValidatingTeacherArmBackend:
    """TeacherArmBackend wrapper that validates output shape"""

    backend : object  # Field: stores backend for validating teacher arm backend
    num_envs: int  # Field: number of parallel environment rows represented
    num_arm : int  # Field: number of arm action dimensions in the active layout
    device  : torch.device | str  # Field: torch device where tensor fields should live

    def compute_teacher_arm_reduced(self, request: TeacherArmRequest) -> torch.Tensor:
        """Return validated arm action from an inner backend"""
        if not hasattr(self.backend, "compute_teacher_arm_reduced"):
            raise TypeError("inner teacher arm backend must expose compute_teacher_arm_reduced")
        value = self.backend.compute_teacher_arm_reduced(request)
        return ensure_teacher_arm_action(
            value,
            num_envs=self.num_envs,
            num_arm=self.num_arm,
            device=self.device,
        )


def build_env_teacher_arm_backend(
    env: object,                                # Param: environment or backend object used for runtime calls
    *,
    method_names: Sequence[str] | None = None,  # Param: ordered candidate names used to resolve method
    legacy_call : bool                 = False,  # Param: boolean input controlling legacy call
    validate    : bool                 = False,  # Param: boolean input controlling validate
    num_arm     : int | None           = None,  # Param: number of arm action dimensions in the active layout
) -> object:
    """Build an env-method teacher arm backend"""
    backend = EnvMethodTeacherArmBackend(
        env=env,
        method_names=tuple(method_names) if method_names is not None else EnvMethodTeacherArmBackend.method_names,
        legacy_call=bool(legacy_call),
    )
    if not validate:
        return backend
    if num_arm is None:
        raise ValueError("num_arm is required when validate=True")
    return ValidatingTeacherArmBackend(
        backend=backend,
        num_envs=int(getattr(env, "num_envs")),
        num_arm=int(num_arm),
        device=getattr(env, "device", "cpu"),
    )


def env_has_teacher_arm_method(
    env: object,
    method_names: Sequence[str] | None = None,
) -> bool:
    """Return whether env exposes a configured teacher arm method."""
    names = tuple(method_names) if method_names is not None else EnvMethodTeacherArmBackend.method_names
    return any(callable(getattr(env, name, None)) for name in names)
