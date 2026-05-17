"""

Isaac env adapters for native eval loop events

File map:

NativeIsaacEvalHookConfig:                Attribute names used to discover Isaac eval hooks
_owners:                                  Handle owners logic
_get_from_owner:                          Handle get from owner logic
_find_value:                              Handle find value logic
_call_or_value:                           Handle call or value logic
_tensor_value:                            Handle tensor value logic
_callable_hook:                           Handle callable hook logic
_components:                              Handle components logic
_env:                                     Handle env logic
_select_policy_action:                    Handle select policy action logic
_terminal_flags_from_value:               Handle terminal flags from value logic
_terminal_flags_fn:                       Handle terminal flags fn logic
_env_values_fn:                           Handle env values fn logic
_reset_cache_fn:                          Handle reset cache fn logic
_eval_action_assembly:                    Handle eval action assembly logic
refresh_native_live_rollout_after_eval:   Reset env after eval and refresh the live rollout tensors
_append_eval_jsonl:                       Handle append eval jsonl logic
_append_eval_aggregate_jsonl:             Handle append eval aggregate jsonl logic
_save_native_eval_best_checkpoint:        Handle save native eval best checkpoint logic
_finalize_eval_aggregate:                 Handle finalize eval aggregate logic
build_native_isaac_eval_event_callbacks:  Build NativeEvalEventCallbacks discovered from the Isaac env
native_isaac_eval_event_callbacks:        Build NativeLiveEventCallbacks containing the Isaac eval event
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import os
from typing import Any

import torch

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..eval.eval_checkpoint import BestCheckpointJob, build_best_checkpoint_decision, run_best_checkpoint_job
from ..eval.eval_episode import EvalEpisodeState
from ..eval.eval_logging import format_eval_pass_line
from ..eval.eval_metrics import EvalAggregationOptions, aggregate_eval_summaries
from ..io.checkpoint_io import capture_rng_state, save_training_checkpoint
from .native_backend import NativeTrainerState
from .native_components import NativeTrainingComponents
from .native_eval import NativeEvalCallbacks, NativeEvalConfig, NativeEvalTerminalFlags
from .native_events import (
    NativeEvalEventCallbacks,
    NativeEvalResultFn,
    _env_tensor,
    _safe_call_tensor,
    _state_machine_module,
    _target_error_tensors,
    _topdown_tensors,
    native_eval_event,
)
from .native_finalization import build_native_checkpoint_metadata
from .native_live import NativeLiveEventCallbacks, NativeLiveRolloutState
from .native_reset import NativeResetRequest, reset_native_env
from .native_step import NativeEnvActionAssemblyConfig
from ..logging.jsonl import write_jsonl_row
from ..logging.tensorboard_logging import finite_scalar_events, write_scalar_events
from ..state.run_state import TrainingLoopStartupState


HookCallable = Callable[..., Any]
CORE_EVAL_ENV_VALUE_KEYS = {
    "return",
    "steps",
    "success",
    "off_table",
    "phase15_shell_drift",
    "block_drift",
    "timeout",
    "done",
}


@dataclass(frozen=True)
class NativeIsaacEvalHookConfig:
    """Attribute names used to discover Isaac eval hooks"""

    teacher_action_names       : tuple[str, ...] = (                                      # ordered names used to resolve teacher action attributes
        "teacher_action_fn",
        "compute_teacher_action",
        "get_teacher_action",
    )
    assemble_action_names      : tuple[str, ...] = (                                      # ordered names used to resolve assemble action attributes
        "assemble_env_action_fn",
        "assemble_env_action",
    )
    arm_reduced_action_names   : tuple[str, ...] = (                                      # ordered names used to resolve arm reduced action attributes
        "arm_reduced_action_fn",
        "current_arm_reduced_action",
        "_teacher_arm_reduced_action",
    )
    mapped_indices_names: tuple[str, ...] = ("mapped_indices", "_mapped_indices")  # source names for mapped action-column indices
    mapped_scales_names : tuple[str, ...] = ("mapped_scales", "_mapped_scales")  # source names for mapped action-column scales
    terminal_flags_names: tuple[str, ...] = (  # ordered names used to resolve terminal flags attributes
        "native_eval_terminal_flags",
        "eval_terminal_flags",
        "terminal_flags_fn",
    )
    env_values_names           : tuple[str, ...] = (                                      # ordered names used to resolve env values attributes
        "native_eval_env_values",
        "eval_env_values",
        "eval_values_fn",
    )
    reset_cache_names          : tuple[str, ...] = (                                      # ordered names used to resolve reset cache attributes
        "clear_cached_teacher_action",
        "reset_eval_cache",
    )
    use_action_assembly_config : bool = True                                              # boolean value indicating the use action assembly config state for native isaac eval hook config


def _owners(state: NativeTrainerState) -> tuple[object, ...]:
    env = state.get("env")
    return tuple(owner for owner in (state.payload, env) if owner is not None)


def _get_from_owner(owner: object, name: str) -> object | None:
    if isinstance(owner, dict):
        return owner.get(name)
    return getattr(owner, name, None)


def _find_value(owners: Sequence[object], names: Sequence[str]) -> object | None:
    for owner in owners:
        for name in names:
            value = _get_from_owner(owner, name)
            if value is not None:
                return value
    return None


def _call_or_value(value: object, *args: object) -> object:
    if not callable(value):
        return value
    try:
        return value(*args)
    except TypeError:
        return value()


def _tensor_value(value: object) -> torch.Tensor | None:
    resolved = _call_or_value(value) if callable(value) else value
    return resolved if torch.is_tensor(resolved) else None


def _callable_hook(owners: Sequence[object], names: Sequence[str]) -> HookCallable | None:
    value = _find_value(owners, names)
    return value if callable(value) else None


def _components(state: NativeTrainerState) -> NativeTrainingComponents:
    components = state.get("components")
    if not isinstance(components, NativeTrainingComponents):
        raise TypeError(f"native state components is {type(components)!r}")
    return components


def _env(state: NativeTrainerState) -> object:
    env = state.get("env")
    if env is None or not hasattr(env, "reset") or not hasattr(env, "step"):
        raise RuntimeError("native Isaac eval requires env reset and step")
    return env


def _select_policy_action(agent: object, obs: torch.Tensor) -> torch.Tensor:
    if not hasattr(agent, "select_action"):
        raise TypeError("native eval agent must expose select_action")
    try:
        return agent.select_action(obs, deterministic=True)
    except TypeError:
        return agent.select_action(obs)


def _terminal_flags_from_value(
    value: object,         # Param: input value normalized or converted by this helper
    *,
    num_envs: int,  # Param: number of parallel environment rows represented
    device  : torch.device,  # Param: torch device where tensors are read or allocated
) -> NativeEvalTerminalFlags:
    """Process for `_terminal_flags_from_value`

    Steps:
    - Resolve inputs for `_terminal_flags_from_value` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if isinstance(value, NativeEvalTerminalFlags):
        return value
    if isinstance(value, Mapping):
        def _flag(name: str) -> torch.Tensor:
            raw = value.get(name)
            if torch.is_tensor(raw):
                return raw.to(device=device, dtype=torch.bool).reshape(-1)
            return torch.zeros(num_envs, dtype=torch.bool, device=device)

        return NativeEvalTerminalFlags(
            success=_flag("success"),
            off_table=_flag("off_table"),
            phase15_shell_drift=_flag("phase15_shell_drift"),
            block_drift=_flag("block_drift"),
        )
    if isinstance(value, tuple) and len(value) == 4 and all(torch.is_tensor(item) for item in value):
        success, off_table, phase15, block_drift = value
        return NativeEvalTerminalFlags(
            success=success.to(device=device, dtype=torch.bool).reshape(-1),
            off_table=off_table.to(device=device, dtype=torch.bool).reshape(-1),
            phase15_shell_drift=phase15.to(device=device, dtype=torch.bool).reshape(-1),
            block_drift=block_drift.to(device=device, dtype=torch.bool).reshape(-1),
        )
    zeros = torch.zeros(num_envs, dtype=torch.bool, device=device)
    return NativeEvalTerminalFlags(
        success=zeros,
        off_table=zeros,
        phase15_shell_drift=zeros,
        block_drift=zeros,
    )


def _terminal_flags_fn(owners: Sequence[object], config: NativeIsaacEvalHookConfig):
    hook = _find_value(owners, config.terminal_flags_names)
    if hook is None:
        return None

    def _flags(next_obs: Mapping[str, object], info: object) -> NativeEvalTerminalFlags:
        """Process for `_flags`

        Steps:
        - Resolve inputs for `_flags` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        policy = next_obs.get("policy")
        if not isinstance(policy, Mapping):
            raise RuntimeError("native eval terminal flags need policy obs")
        first_tensor = next(iter(policy.values()))
        if not torch.is_tensor(first_tensor):
            raise TypeError("native eval policy obs values must be tensors")
        value = _call_or_value(hook, next_obs, info)
        return _terminal_flags_from_value(
            value,
            num_envs=int(first_tensor.shape[0]),
            device=first_tensor.device,
        )

    return _flags


def _env_values_fn(owners: Sequence[object], config: NativeIsaacEvalHookConfig):
    hook = _find_value(owners, config.env_values_names)
    if hook is None:
        return None

    def _values(state: EvalEpisodeState) -> Mapping[str, Sequence[object] | object]:
        value = _call_or_value(hook, state)
        if not isinstance(value, Mapping):
            raise TypeError(f"native eval env values hook returned {type(value)!r}")
        return value

    return _values


def _reset_cache_fn(owners: Sequence[object], config: NativeIsaacEvalHookConfig):
    hook = _callable_hook(owners, config.reset_cache_names)
    if hook is None:
        return None

    def _reset_cache() -> None:
        hook()

    return _reset_cache


def _eval_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class _TopdownEvalMetricCollector:
    """Collect per-env topdown eval geometry over one eval episode."""

    def __init__(self, env: object, *, trace_every: int = 100) -> None:
        self.env = env
        self.state_machine = _state_machine_module()
        self.trace_every = max(0, int(trace_every))
        self._num_envs: int | None = None
        self._device: torch.device | None = None
        self._best_high: dict[str, torch.Tensor] = {}
        self._final_high: dict[str, torch.Tensor] = {}
        self._best_low: dict[str, torch.Tensor] = {}
        self._final_low: dict[str, torch.Tensor] = {}
        self._max_values: dict[str, torch.Tensor] = {}
        self._episode_flags: dict[str, torch.Tensor] = {}
        self._source_pose_idx: torch.Tensor | None = None

    def _ensure(self, reference: torch.Tensor) -> None:
        ref = reference.reshape(-1)
        if self._num_envs is not None:
            return
        self._num_envs = int(ref.shape[0])
        self._device = ref.device

    def _zeros(self) -> torch.Tensor:
        if self._num_envs is None or self._device is None:
            raise RuntimeError("topdown eval collector is not initialized")
        return torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

    def _fill(self, value: float) -> torch.Tensor:
        if self._num_envs is None or self._device is None:
            raise RuntimeError("topdown eval collector is not initialized")
        return torch.full((self._num_envs,), float(value), dtype=torch.float32, device=self._device)

    def _tensor(self, value: object, *, default: float = 0.0) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.detach().to(dtype=torch.float32).reshape(-1)
        else:
            tensor = self._fill(default)
        if self._num_envs is not None and tensor.numel() != self._num_envs:
            out = self._fill(default)
            count = min(int(tensor.numel()), self._num_envs)
            if count > 0:
                out[:count] = tensor[:count].to(device=out.device)
            return out
        return tensor

    def _call(self, name: str, *, default: float = 0.0) -> torch.Tensor:
        if self.state_machine is None:
            return self._fill(default)
        return self._tensor(
            _safe_call_tensor(self.env, getattr(self.state_machine, name, None), default=default),
            default=default,
        )

    def _update_high(self, best_key: str, final_key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        current_best = self._best_high.setdefault(best_key, self._zeros())
        current_final = self._final_high.setdefault(final_key, self._zeros())
        self._best_high[best_key] = torch.where(mask, torch.maximum(current_best, value), current_best)
        self._final_high[final_key] = torch.where(mask, value, current_final)

    def _update_low(self, best_key: str, final_key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        current_best = self._best_low.setdefault(best_key, self._fill(float("inf")))
        current_final = self._final_low.setdefault(final_key, self._fill(float("inf")))
        self._best_low[best_key] = torch.where(mask, torch.minimum(current_best, value), current_best)
        self._final_low[final_key] = torch.where(mask, value, current_final)

    def _update_max(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        current = self._max_values.setdefault(key, self._zeros())
        self._max_values[key] = torch.where(mask, torch.maximum(current, value), current)

    def _update_flag(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        current = self._episode_flags.setdefault(key, self._zeros())
        self._episode_flags[key] = torch.where(mask, torch.maximum(current, value.to(dtype=torch.float32)), current)

    def _update_final(self, key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        current = self._final_high.setdefault(key, self._zeros())
        self._final_high[key] = torch.where(mask, value, current)

    def _trace_env0(self, *, step_index: int, metrics: Mapping[str, torch.Tensor]) -> None:
        step = int(step_index) + 1
        if self.trace_every <= 0 or step % self.trace_every != 0:
            return

        def scalar(name: str, default: float = 0.0) -> float:
            tensor = metrics.get(name)
            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                return float(default)
            return float(tensor.reshape(-1)[0].detach().item())

        print(
            "eval_env0_metrics "
            f"step={step} "
            f"source={int(scalar('source_idx', -1.0))} "
            f"stage={int(scalar('stage', -1.0))} "
            f"block=({scalar('block_x'):.3f},{scalar('block_y'):.3f}) "
            f"tip={scalar('tip'):.4f} "
            f"palm={scalar('phase1_palm_dist'):.4f} "
            f"orient_deg={scalar('phase1_orient_deg'):.2f} "
            f"align={scalar('align_face_dist'):.4f} "
            f"align_angle={scalar('align_angle'):.2f} "
            f"contact={scalar('contact'):.3f} "
            f"strict={scalar('strict_light_contact'):.3f} "
            f"thumb={scalar('thumb_contact'):.3f} "
            f"index={scalar('index_contact'):.3f} "
            f"lift={scalar('lift'):.4f} "
            f"disp={scalar('block_disp'):.4f} "
            f"tilt_deg={scalar('block_tilt_deg'):.2f} "
            f"unlock={scalar('finger_unlock'):.3f} "
            f"success={int(scalar('success') >= 0.5)}",
            flush=True,
        )

    def step(self, _state: EvalEpisodeState, masks: object, step_index: int) -> None:
        tensors = _topdown_tensors(self.env, self.state_machine)
        stage = self._tensor(tensors.get("stage"), default=-1.0)
        self._ensure(stage)
        active = getattr(masks, "active_mask", torch.ones_like(stage, dtype=torch.bool))
        mask = active.to(device=stage.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != stage.numel():
            aligned = torch.zeros_like(stage, dtype=torch.bool)
            count = min(int(mask.numel()), int(stage.numel()))
            if count > 0:
                aligned[:count] = mask[:count]
            mask = aligned

        thumb_err, index_err, _, _ = _target_error_tensors(self.env, self.state_machine)
        tip = torch.maximum(self._tensor(thumb_err, default=float("inf")), self._tensor(index_err, default=float("inf")))
        palm = self._call("palm_distance", default=float("inf"))
        orient = torch.rad2deg(self._call("palm_drop_axis_error_rad", default=float("inf")))
        block_tilt = torch.rad2deg(self._call("block_tilt_angle_rad", default=0.0))
        block_disp = self._call("block_xy_displacement", default=0.0)
        if not torch.isfinite(block_disp).all():
            block_disp = self._tensor(tensors.get("block_disp"), default=0.0)

        contact = self._tensor(tensors.get("contact"), default=0.0)
        strict = self._tensor(tensors.get("strict_light_contact"), default=0.0)
        lift = self._tensor(tensors.get("lift"), default=0.0)
        align = self._tensor(tensors.get("align_face_dist"), default=float("inf"))
        align_angle = self._tensor(tensors.get("align_angle"), default=float("inf"))
        unlock = self._tensor(tensors.get("finger_unlock"), default=0.0)
        success = self._tensor(tensors.get("success"), default=0.0)
        tilt_max = _eval_env_float("TOPDOWN_LIFT_SUCCESS_BLOCK_TILT_MAX_DEG", 0.0)
        lift_min = _eval_env_float("TOPDOWN_LIFT_SUCCESS_HEIGHT", 0.035)
        drift_max = _eval_env_float("TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX", 0.04)
        contact_min = _eval_env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
        contact_ok = strict >= contact_min
        lift_ok = lift >= lift_min
        drift_ok = block_disp <= drift_max
        tilt_ok = block_tilt <= tilt_max if tilt_max > 0.0 else torch.ones_like(lift_ok, dtype=torch.bool)

        self._update_low("best_tip", "final_tip", tip, mask)
        self._update_low("best_phase1_palm_dist", "final_phase1_palm_dist", palm, mask)
        self._update_low("best_phase1_orient_deg", "final_phase1_orient_deg", orient, mask)
        self._update_low("best_align_face_dist", "final_align_face_dist", align, mask)
        self._update_low("best_align_angle", "final_align_angle", align_angle, mask)
        self._update_low("best_block_disp", "final_block_disp", block_disp, mask)

        self._update_high("best_contact", "final_contact", contact, mask)
        self._update_high("best_both_contact", "final_both_contact", self._tensor(tensors.get("both_contact"), default=0.0), mask)
        self._update_high("best_any_contact", "final_any_contact", self._tensor(tensors.get("fingertip_contact"), default=0.0), mask)
        self._update_high("best_hand_contact", "final_hand_contact", self._tensor(tensors.get("hand_contact"), default=0.0), mask)
        self._update_high("best_thumb_contact", "final_thumb_contact", self._tensor(tensors.get("thumb_contact"), default=0.0), mask)
        self._update_high("best_index_contact", "final_index_contact", self._tensor(tensors.get("index_contact"), default=0.0), mask)
        self._update_high("best_strict_light_contact", "final_strict_light_contact", strict, mask)
        self._update_high("best_lift", "final_lift", lift, mask)
        self._update_high("best_lift_with_strict_contact", "final_lift_with_strict_contact", torch.where(contact_ok, lift, torch.zeros_like(lift)), mask)
        self._update_high("best_curl", "final_curl", self._tensor(tensors.get("curl"), default=0.0), mask)
        self._update_high("best_opposite_face", "final_opposite_face", self._tensor(tensors.get("opposed_face"), default=0.0), mask)
        self._update_high("best_topdown_stage", "final_topdown_stage", stage, mask)

        self._update_max("max_block_tilt_deg", block_tilt, mask)
        self._update_max("max_topdown_finger_unlock_progress", unlock, mask)
        self._update_final("final_block_tilt_deg", block_tilt, mask)
        self._update_final("final_thumb_contact_force_N", self._tensor(tensors.get("thumb_contact_force_N"), default=0.0), mask)
        self._update_final("final_index_contact_force_N", self._tensor(tensors.get("index_contact_force_N"), default=0.0), mask)
        self._update_flag("success", success >= 0.5, mask)
        self._update_flag("topdown_reach_pass", stage >= 1.0, mask)
        self._update_flag("clean_lift_episode", contact_ok & lift_ok & drift_ok, mask)
        self._update_flag("upright_clean_lift_episode", contact_ok & lift_ok & drift_ok & tilt_ok, mask)
        self._update_flag("lift_xy_drift_success_gate", drift_ok, mask)
        self._update_flag("lift_block_tilt_success_gate", tilt_ok, mask)
        self._update_flag("physical_success", contact_ok & lift_ok & drift_ok & tilt_ok, mask)

        source_idx = _env_tensor(self.env, "_topdown_source_pose_idx", default=-1.0, dtype=torch.long).reshape(-1)
        if source_idx.numel() == stage.numel():
            if self._source_pose_idx is None:
                self._source_pose_idx = torch.full_like(stage, -1.0)
            self._source_pose_idx = torch.where(mask, source_idx.to(device=stage.device, dtype=torch.float32), self._source_pose_idx)

        metrics = {
            **{str(key): self._tensor(value, default=0.0) for key, value in tensors.items()},
            "tip": tip,
            "phase1_palm_dist": palm,
            "phase1_orient_deg": orient,
            "block_disp": block_disp,
            "block_tilt_deg": block_tilt,
            "source_idx": source_idx.to(device=stage.device, dtype=torch.float32) if source_idx.numel() == stage.numel() else self._fill(-1.0),
        }
        self._trace_env0(step_index=step_index, metrics=metrics)

    def env_values(self, state: EvalEpisodeState) -> Mapping[str, Sequence[object] | object]:
        del state
        values: dict[str, Sequence[object] | object] = {}
        for mapping in (self._best_high, self._final_high, self._best_low, self._final_low, self._max_values, self._episode_flags):
            for key, tensor in mapping.items():
                values[key] = tensor.detach().cpu().tolist()
        if self._source_pose_idx is not None:
            values["topdown_source_pose_idx"] = [int(value) for value in self._source_pose_idx.detach().cpu().tolist()]
        return values


def _eval_action_assembly(
    *,
    owners     : Sequence[object],  # Param: ordered input collection of owners entries
    components : NativeTrainingComponents,  # Param: input value used as components
    configs    : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    env        : object,  # Param: environment or backend object used for runtime calls
    hook_config: NativeIsaacEvalHookConfig,  # Param: input value used as hook config
) -> NativeEnvActionAssemblyConfig | None:
    if not hook_config.use_action_assembly_config:
        return None
    return NativeEnvActionAssemblyConfig(
        gate_config=components.td3_config.gate_config,
        arm_controller=configs.teacher.arm_controller,
        finger_action_mode=configs.teacher.finger_action_mode,
        arm_reduced_action=_tensor_value(_find_value(owners, hook_config.arm_reduced_action_names)),
        env=env,
        mapped_indices=_tensor_value(_find_value(owners, hook_config.mapped_indices_names)),
        mapped_scales=_tensor_value(_find_value(owners, hook_config.mapped_scales_names)),
        finger_delta_scale=configs.teacher.finger_delta_scale,
    )


def refresh_native_live_rollout_after_eval(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    plan      : object,  # Param: precomputed plan object consumed by this helper
    results   : tuple[object, ...],  # Param: ordered input collection of results entries
    *,
    topdown_curriculum_obs_contract: bool = True,  # Param: boolean input controlling topdown curriculum obs contract
) -> None:
    """Reset env after eval and refresh the live rollout tensors

    Steps:
    - Resolve inputs for `refresh_native_live_rollout_after_eval` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    del configs, loop_state, plan, results
    env = _env(state)
    live_rollout = state.get("live_rollout")
    if not isinstance(live_rollout, NativeLiveRolloutState):
        raise TypeError(f"native state live_rollout is {type(live_rollout)!r}")
    reset_result = reset_native_env(
        NativeResetRequest(
            env_reset_fn=env.reset,
            privileged_critic=context.dims.priv_obs_dim > 0,
            topdown_curriculum=topdown_curriculum_obs_contract,
        )
    )
    live_rollout.obs = dict(reset_result.obs)
    live_rollout.obs_tensor = reset_result.observation.obs_tensor
    live_rollout.priv_obs_tensor = reset_result.observation.privileged_obs_tensor
    if isinstance(state.payload, dict):
        state.payload["reset_info"] = reset_result.info


def _append_eval_jsonl(context: TrainerRuntimeContext, plan: object, results: tuple[object, ...]) -> None:
    path = str(context.paths.log_jsonl or "").strip()
    if not path or path.lower() in {"off", "none", "false", "0"}:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as log_file:
        for result in results:
            summary = getattr(result, "summary", None)
            if not isinstance(summary, Mapping):
                continue
            row = {
                "mode"       : "eval_summary",
                "global_step": int(getattr(plan, "global_step", summary.get("global_step", 0))),
                **dict(summary),
            }
            write_jsonl_row(log_file, row, flush=True)


def _append_eval_aggregate_jsonl(context: TrainerRuntimeContext, aggregate: Mapping[str, object]) -> None:
    path = str(context.paths.log_jsonl or "").strip()
    if not path or path.lower() in {"off", "none", "false", "0"}:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as log_file:
        write_jsonl_row(log_file, dict(aggregate), flush=True)


def _save_native_eval_best_checkpoint(
    context   : TrainerRuntimeContext,
    state     : NativeTrainerState,
    loop_state: TrainingLoopStartupState,
    job       : BestCheckpointJob,
) -> None:
    components = _components(state)
    if not isinstance(components.agent, Mapping) and not hasattr(components.agent, "state_dict"):
        return
    metadata = build_native_checkpoint_metadata(
        context,
        loop_state,
        global_step=job.global_step,
        handoff_compatibility=state.get("handoff_compatibility"),
    )
    save_training_checkpoint(
        job.dest_path,
        metadata=metadata,
        agent=components.agent,
        replay=components.replay,
        include_replay=job.include_replay,
        rng_state=capture_rng_state(),
        extra_fields={"best_eval_state": dict(loop_state.best_eval_state)},
    )


def _finalize_eval_aggregate(
    context   : TrainerRuntimeContext,
    configs   : RuntimeConfigBundle,
    state     : NativeTrainerState,
    loop_state: TrainingLoopStartupState,
    plan      : object,
    results   : tuple[object, ...],
) -> None:
    summaries = [
        result.summary
        for result in results
        if isinstance(getattr(result, "summary", None), Mapping)
    ]
    if not summaries:
        return
    global_step = int(getattr(plan, "global_step", summaries[-1].get("global_step", 0)))
    aggregate = aggregate_eval_summaries(
        summaries,
        global_step=global_step,
        options=EvalAggregationOptions(task_kind="topdown_lift"),
    )
    _append_eval_aggregate_jsonl(context, aggregate)
    write_scalar_events(
        _components(state).tensorboard_writer,
        finite_scalar_events("eval", aggregate, global_step),
    )
    decision = build_best_checkpoint_decision(
        eval_summary=aggregate,
        best_eval_state=loop_state.best_eval_state,
        global_step=global_step,
        checkpoint_path=context.paths.checkpoint_path,
        task_kind="topdown_lift",
        save_replay_in_checkpoint=configs.checkpoint.save_replay_in_checkpoint,
    )
    loop_state.best_eval_state.update(decision.next_best_state)
    if decision.message:
        print(decision.message, flush=True)
    run_best_checkpoint_job(
        decision.job,
        lambda job: _save_native_eval_best_checkpoint(context, state, loop_state, job),
    )
    print(format_eval_pass_line(aggregate, global_step=global_step), flush=True)


def build_native_isaac_eval_event_callbacks(
    *,
    hook_config                    : NativeIsaacEvalHookConfig = NativeIsaacEvalHookConfig(),  # Param: input value used as hook config
    result_fn                      : NativeEvalResultFn | None = None,  # Param: callback used to compute or fetch result
    topdown_curriculum_obs_contract: bool                      = True,  # Param: boolean input controlling topdown curriculum obs contract
) -> NativeEvalEventCallbacks:
    """Build NativeEvalEventCallbacks discovered from the Isaac env"""

    def _callbacks(
        context    : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs    : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state      : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
        loop_state : TrainingLoopStartupState,  # Param: input value used as loop state
        episode_idx: int,  # Param: episode index associated with the record or rollout
    ) -> NativeEvalCallbacks:
        """Process for `_callbacks`

        Steps:
        - Resolve inputs for `_callbacks` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        del context, configs, loop_state, episode_idx
        env = _env(state)
        components = _components(state)
        owners = _owners(state)
        collector = _TopdownEvalMetricCollector(env)
        external_env_values = _env_values_fn(owners, hook_config)

        def _env_values(state_: EvalEpisodeState) -> Mapping[str, Sequence[object] | object]:
            values = {
                key: value
                for key, value in collector.env_values(state_).items()
                if key not in CORE_EVAL_ENV_VALUE_KEYS
            }
            if external_env_values is not None:
                values.update(dict(external_env_values(state_)))
            return values

        return NativeEvalCallbacks(
            reset_fn=env.reset,
            env_step_fn=env.step,
            select_policy_action_fn=lambda obs: _select_policy_action(components.agent, obs),
            teacher_action_fn=_callable_hook(owners, hook_config.teacher_action_names),
            terminal_flags_fn=_terminal_flags_fn(owners, hook_config),
            env_values_fn=_env_values,
            step_metrics_fn=collector.step,
            assemble_env_action_fn=_callable_hook(owners, hook_config.assemble_action_names),
            reset_cache_fn=_reset_cache_fn(owners, hook_config),
        )

    def _config(
        context    : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs    : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state      : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
        loop_state : TrainingLoopStartupState,  # Param: input value used as loop state
        plan       : object,  # Param: precomputed plan object consumed by this helper
        episode_idx: int,  # Param: episode index associated with the record or rollout
        base_config: NativeEvalConfig,  # Param: input value used as base config
    ) -> NativeEvalConfig:
        """Process for `_config`

        Steps:
        - Resolve inputs for `_config` and capture local config or state
        - Run guard branches and early exits before heavier work
        - Build intermediate tensors, records, commands, or helper objects in dependency order
        - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
        - Return computed output or leave updated state for caller use
        """
        del context, loop_state, plan, episode_idx
        env = _env(state)
        components = _components(state)
        owners = _owners(state)
        return replace(
            base_config,
            action_assembly=_eval_action_assembly(
                owners=owners,
                components=components,
                configs=configs,
                env=env,
                hook_config=hook_config,
            ),
        )

    def _post_eval(
        context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
        configs   : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
        state     : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
        loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
        plan      : object,  # Param: precomputed plan object consumed by this helper
        results   : tuple[object, ...],  # Param: ordered input collection of results entries
    ) -> None:
        _append_eval_jsonl(context, plan, results)
        _finalize_eval_aggregate(context, configs, state, loop_state, plan, results)
        refresh_native_live_rollout_after_eval(
            context,
            configs,
            state,
            loop_state,
            plan,
            results,
            topdown_curriculum_obs_contract=topdown_curriculum_obs_contract,
        )

    return NativeEvalEventCallbacks(
        eval_callbacks_fn=_callbacks,
        eval_config_fn=_config,
        result_fn=result_fn,
        post_eval_fn=_post_eval,
    )


def native_isaac_eval_event_callbacks(
    *,
    hook_config                    : NativeIsaacEvalHookConfig = NativeIsaacEvalHookConfig(),  # Param: input value used as hook config
    result_fn                      : NativeEvalResultFn | None = None,  # Param: callback used to compute or fetch result
    topdown_curriculum_obs_contract: bool                      = True,  # Param: boolean input controlling topdown curriculum obs contract
) -> NativeLiveEventCallbacks:
    """Build NativeLiveEventCallbacks containing the Isaac eval event"""
    return NativeLiveEventCallbacks(
        on_eval=native_eval_event(
            build_native_isaac_eval_event_callbacks(
                hook_config=hook_config,
                result_fn=result_fn,
                topdown_curriculum_obs_contract=topdown_curriculum_obs_contract,
            )
        )
    )
