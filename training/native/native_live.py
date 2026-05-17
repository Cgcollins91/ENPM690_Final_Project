"""

Live native loop callback wiring

File map:

NativeLiveRolloutState:                 Mutable current observation tensors for live native rollout
AdaptivePolicyAssistState:              Stateful assist schedule gated by actor-to-teacher action error
NativeTrainEpisodeAggregateState:       Stateful train-env completed-episode metric accumulator
NativeLiveHooks:                        Injected live callbacks for the native loop
NativeLiveEventCallbacks:               Optional live loop event callbacks
native_live_action_mix_config:          Build rollout action mix config from runtime settings
_components:                            Handle components logic
_default_action_source:                 Handle default action source logic
_default_mask:                          Handle default mask logic
_policy_action:                         Handle policy action logic
_native_debug_logging_enabled:          Handle native debug logging enabled logic
_int_arg:                               Handle int arg logic
_float_arg:                             Handle float arg logic
_bool_arg:                              Handle bool arg logic
_configured_assist_bounds:              Handle configured assist bounds logic
_adaptive_assist_config:                Handle adaptive assist config logic
_update_adaptive_assist:                Handle update adaptive assist logic
_should_stop_on_adaptive_assist_floor:  Handle should stop on adaptive assist floor logic
_adaptive_assist_at_base:               Return whether adaptive assist still has full teacher authority
_adaptive_teacher_bc_base_weight:       Handle adaptive teacher bc base weight logic
_sync_adaptive_actor_training:          Handle adaptive actor training mode and BC weights
collect_native_live_step:               Collect one live native env step and update runtime tensors
update_native_live_agent:               Run one live native TD update from loop callbacks
build_native_live_loop_callbacks:       Build NativeLoopCallbacks for a live env-backed rollout
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
import os

import torch

from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..eval.eval_actions import ActionProcessor
from .native_actions import NativeActionMixConfig, NativeActionSelection, select_native_rollout_action
from .native_components import NativeTrainingComponents
from .native_loop import NativeLoopCallbacks, NativeLoopStepBatch, PlanFn
from .native_step import (
    ActionAssemblyFn,
    EnvStepFn,
    NativeEnvActionAssemblyConfig,
    NativeEnvStepCallbacks,
    NativeEnvStepRequest,
    collect_native_env_step,
)
from .native_updates import NativeRolloutStatUpdate, NativeUpdateRequest, run_native_td_update, update_native_rollout_stats
from ..env.observations import flatten_privileged_obs
from ..state.run_state import TrainingLoopStartupState


TeacherActionFn = Callable[[], torch.Tensor]
PrerollActionFn = Callable[["NativeLiveRolloutState"], torch.Tensor | None]
MaskFn = Callable[["NativeLiveRolloutState"], torch.Tensor | None]
ActionAssemblyConfigFn = Callable[[], NativeEnvActionAssemblyConfig | None]
CheckpointNamesFn = Callable[[], tuple[str, ...]]
ActionSourceFn = Callable[[NativeActionSelection], str]


@dataclass
class NativeTrainEpisodeAggregateState:
    """Stateful train-env completed-episode metric accumulator."""

    steps       : torch.Tensor
    stage1_steps: torch.Tensor
    stage2_steps: torch.Tensor
    strict_steps: torch.Tensor
    contact_steps: torch.Tensor
    best_lift   : torch.Tensor
    success_seen: torch.Tensor
    lift_values : list[list[float]]


@dataclass
class NativeLiveRolloutState:
    """Mutable current observation tensors for live native rollout"""

    obs                  : dict[str, object]  # Field: policy observation tensor or observation payload for this transition
    obs_tensor           : torch.Tensor  # Field: policy observation tensor passed to the actor or replay path
    priv_obs_tensor      : torch.Tensor | None          = None  # Field: privileged observation tensor passed to critic-side logic
    last_action_selection: NativeActionSelection | None = None  # Field: stores last action selection for native live rollout state
    last_step_result     : object | None                = None  # Field: stores last step result for native live rollout state
    last_stats_rows      : int                          = 0  # Field: integer last stats rows value tracked by native live rollout state
    adaptive_assist      : "AdaptivePolicyAssistState | None" = None  # Field: state for actor-error-gated assist decay
    episode_aggregate    : NativeTrainEpisodeAggregateState | None = None  # Field: train-env completed-episode metric accumulator


@dataclass
class AdaptivePolicyAssistState:
    """Stateful assist schedule gated by actor-to-teacher action error."""

    current_mix                 : float
    good_steps                  : int                 = 0
    bad_steps                   : int                 = 0
    floor_step                  : int                 = -1
    metric_episode_steps        : torch.Tensor | None = None
    metric_episode_stage2_steps : torch.Tensor | None = None
    metric_episode_strict_steps : torch.Tensor | None = None
    metric_episode_best_lift    : torch.Tensor | None = None
    metric_teacher_stage2_rates : deque[float]        = field(default_factory=deque)
    metric_teacher_strict_rates : deque[float]        = field(default_factory=deque)
    metric_teacher_best_lifts   : deque[float]        = field(default_factory=deque)
    metric_window_stage2_rates  : deque[float]        = field(default_factory=deque)
    metric_window_strict_rates  : deque[float]        = field(default_factory=deque)
    metric_window_best_lifts    : deque[float]        = field(default_factory=deque)
    metric_baseline_logged      : bool                = False
    metric_active_bucket        : str                 = ""


@dataclass(frozen=True)
class NativeLiveHooks:
    """Injected live callbacks for the native loop"""

    env_step_fn                 : EnvStepFn  # Field: callback used for the env step fn operation
    teacher_action_fn           : TeacherActionFn | None        = None  # Field: callback used for the teacher action fn operation
    policy_processors           : Sequence[ActionProcessor]     = ()  # Field: ordered collection of policy processors entries for native live hooks
    teacher_processors          : Sequence[ActionProcessor]     = ()  # Field: ordered collection of teacher processors entries for native live hooks
    assemble_env_action_fn      : ActionAssemblyFn | None       = None  # Field: callback used for the assemble env action fn operation
    action_assembly_fn          : ActionAssemblyConfigFn | None = None  # Field: callback used for the action assembly fn operation
    preroll_action_fn           : PrerollActionFn | None        = None  # Field: callback used for the preroll action fn operation
    preroll_mask_fn             : MaskFn | None                 = None  # Field: callback used for the preroll mask fn operation
    active_env_mask_fn          : MaskFn | None                 = None  # Field: callback used for the active env mask fn operation
    existing_checkpoint_names_fn: CheckpointNamesFn | None      = None  # Field: callback used for the existing checkpoint names fn operation
    action_source_fn            : ActionSourceFn | None         = None  # Field: callback used for the action source fn operation


@dataclass(frozen=True)
class NativeLiveEventCallbacks:
    """Optional live loop event callbacks"""

    on_step_plan : PlanFn | None = None  # Field: stores on step plan for native live event callbacks
    on_log       : PlanFn | None = None  # Field: stores on log for native live event callbacks
    on_eval      : PlanFn | None = None  # Field: stores on eval for native live event callbacks
    on_checkpoint: PlanFn | None = None  # Field: stores on checkpoint for native live event callbacks
    on_done_reset: PlanFn | None = None  # Field: stores on done reset for native live event callbacks


def native_live_action_mix_config(
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    *,
    global_step: int = 0,
) -> NativeActionMixConfig:
    """Build rollout action mix config from runtime settings

    Steps:
    - Resolve inputs for `native_live_action_mix_config` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    args = context.args
    base_mix = float(args.get("policy_assist_mix", configs.assist.policy_assist_mix))
    floor = max(0.0, float(args.get("policy_assist_mix_floor", configs.assist.policy_assist_mix_floor)))
    decay_steps = int(args.get("policy_assist_decay_steps", configs.assist.policy_assist_decay_steps))
    decay_start = int(args.get("policy_assist_decay_start_steps", configs.assist.policy_assist_decay_start_steps))
    if decay_start < 0:
        decay_start = int(configs.counts.start_steps)
    if decay_steps > 0:
        progress = min(1.0, max(0.0, (int(global_step) - decay_start) / float(decay_steps)))
        assist_mix = base_mix + (floor - base_mix) * progress
        assist_mix = max(floor, min(base_mix, assist_mix)) if base_mix >= floor else min(floor, max(base_mix, assist_mix))
    else:
        assist_mix = base_mix
    assist_noise_arm = float(args.get("assist_noise_arm", configs.assist.assist_noise_arm))
    assist_noise_finger = float(args.get("assist_noise_finger", configs.assist.assist_noise_finger))
    assist_noise_start_steps = int(args.get("assist_noise_start_steps", 0))
    if int(global_step) < assist_noise_start_steps:
        assist_noise_arm = 0.0
        assist_noise_finger = 0.0
    return NativeActionMixConfig(
        assist_mix=assist_mix,
        assist_arm_mix=float(args.get("policy_assist_arm_mix", -1.0)),
        assist_finger_mix=float(args.get("policy_assist_finger_mix", -1.0)),
        global_step=int(global_step),
        start_steps=int(args.get("start_steps", configs.counts.start_steps)),
        policy_bc_relabel=bool(args.get("policy_bc_relabel", configs.assist.policy_bc_relabel)),
        bc_only_steps=int(args.get("bc_only_steps", configs.assist.bc_only_steps)),
        teacher_bc_weight=float(args.get("teacher_bc_weight", configs.assist.teacher_bc_weight)),
        teacher_bc_arm_weight=float(args.get("teacher_bc_arm_weight", configs.assist.teacher_bc_arm_weight)),
        teacher_bc_finger_weight=float(args.get("teacher_bc_finger_weight", configs.assist.teacher_bc_finger_weight)),
        assist_noise_arm=assist_noise_arm,
        assist_noise_finger=assist_noise_finger,
        assist_noise_clean_bc_target=bool(args.get("assist_noise_clean_bc_target", configs.assist.assist_noise_clean_bc_target)),
        use_policy_arm_teacher=bool(args.get("use_policy_arm_teacher", False)),
        soft_policy_arm_assist=bool(args.get("soft_policy_arm_assist", False)),
    )


def _components(startup_state) -> NativeTrainingComponents:
    components = startup_state.get("components")
    if not isinstance(components, NativeTrainingComponents):
        raise TypeError(f"native state components is {type(components)!r}")
    return components


def _default_action_source(selection: NativeActionSelection) -> str:
    if selection.teacher_action is not None:
        if bool(getattr(selection.diagnostics, "teacher_warmup", False)):
            return "teacher_ik"
        return "policy_assist" if selection.diagnostics.teacher_available else "teacher_ik"
    return "policy"


def _default_mask(runtime: NativeLiveRolloutState) -> torch.Tensor:
    return torch.zeros(runtime.obs_tensor.shape[0], dtype=torch.bool, device=runtime.obs_tensor.device)


def _policy_action(agent, obs_tensor: torch.Tensor) -> torch.Tensor:
    if not hasattr(agent, "select_action"):
        raise TypeError("native live agent must expose select_action")
    return agent.select_action(obs_tensor, deterministic=False)


def _native_debug_logging_enabled() -> bool:
    return os.environ.get("NATIVE_DEBUG_LOGGING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _int_arg(context: TrainerRuntimeContext, name: str, default: int) -> int:
    try:
        return int(context.args.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _float_arg(context: TrainerRuntimeContext, name: str, default: float) -> float:
    try:
        return float(context.args.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _bool_arg(context: TrainerRuntimeContext, name: str, default: bool = False) -> bool:
    value = context.args.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _configured_assist_bounds(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
) -> tuple[float, float]:
    base_mix = float(context.args.get("policy_assist_mix", configs.assist.policy_assist_mix))
    floor = max(0.0, float(context.args.get("policy_assist_mix_floor", configs.assist.policy_assist_mix_floor)))
    return base_mix, floor


def _adaptive_assist_config(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
    scheduled: NativeActionMixConfig,
    *,
    global_step: int,
) -> NativeActionMixConfig:
    if not _bool_arg(context, "adaptive_policy_assist", False):
        return scheduled
    state = runtime.adaptive_assist
    base_mix, _floor = _configured_assist_bounds(context, configs)
    if not isinstance(state, AdaptivePolicyAssistState):
        state = AdaptivePolicyAssistState(current_mix=base_mix)
        runtime.adaptive_assist = state
        print(
            "adaptive_assist_start "
            f"step={int(global_step)} mix={state.current_mix:.4f}",
            flush=True,
        )
    at_floor = _adaptive_assist_at_floor(context, configs, runtime)
    disable_bc_after_floor = _bool_arg(context, "adaptive_assist_disable_bc_after_floor", False)
    kwargs = {
        "assist_mix": float(state.current_mix),
        "assist_arm_mix": -1.0,
        "assist_finger_mix": -1.0,
    }
    if at_floor and disable_bc_after_floor:
        kwargs.update(
            {
                "policy_bc_relabel": False,
                "teacher_bc_weight": 0.0,
                "teacher_bc_arm_weight": -1.0,
                "teacher_bc_finger_weight": -1.0,
            }
        )
    return replace(scheduled, **kwargs)


def _adaptive_assist_metric_gate_enabled(context: TrainerRuntimeContext) -> bool:
    return _bool_arg(context, "adaptive_assist_metric_gate", False)


def _topdown_state_machine_for_metrics():
    try:
        from tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3.mdp import state_machine
    except Exception:
        return None
    return state_machine


def _metric_tensor(
    value: object,
    *,
    n: int,
    device: torch.device,
    default: float = 0.0,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32).reshape(-1)
    elif value is None:
        tensor = torch.full((n,), float(default), device=device, dtype=torch.float32)
    else:
        try:
            tensor = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
        except Exception:
            tensor = torch.full((n,), float(default), device=device, dtype=torch.float32)
    if int(tensor.numel()) == n:
        return tensor
    if int(tensor.numel()) == 1:
        return tensor.expand(n)
    if int(tensor.numel()) > n:
        return tensor[:n]
    padded = torch.full((n,), float(default), device=device, dtype=torch.float32)
    if int(tensor.numel()) > 0:
        padded[: int(tensor.numel())] = tensor
    return padded


def _metric_env_tensor(
    env: object,
    attr: str,
    *,
    n: int,
    device: torch.device,
    default: float = 0.0,
) -> torch.Tensor:
    return _metric_tensor(getattr(env, attr, None), n=n, device=device, default=default)


def _metric_call_tensor(
    env: object,
    fn,
    *,
    n: int,
    device: torch.device,
    default: float = 0.0,
) -> torch.Tensor:
    if not callable(fn):
        return torch.full((n,), float(default), device=device, dtype=torch.float32)
    try:
        value = fn(env)
    except Exception:
        value = None
    return _metric_tensor(value, n=n, device=device, default=default)


def _strict_contact_threshold(context: TrainerRuntimeContext) -> float:
    raw = os.environ.get("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", context.args.get("topdown_lift_success_contact_min", 0.30))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.30


def _episode_aggregate_metric_tensors(
    context: TrainerRuntimeContext,
    env: object | None,
    runtime: NativeLiveRolloutState,
) -> dict[str, torch.Tensor]:
    n = int(runtime.obs_tensor.shape[0])
    device = runtime.obs_tensor.device
    zero = torch.zeros((n,), device=device, dtype=torch.float32)
    if env is None:
        return {
            "stage1" : zero,
            "stage2" : zero,
            "strict" : zero,
            "contact": zero,
            "lift"   : zero,
            "success": zero,
        }
    stage = _metric_env_tensor(env, "_topdown_stage", n=n, device=device, default=0.0)
    state_machine = _topdown_state_machine_for_metrics()
    thumb = _metric_call_tensor(
        env,
        getattr(state_machine, "thumb_contact_strength", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    index = _metric_call_tensor(
        env,
        getattr(state_machine, "index_contact_strength", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    middle = _metric_call_tensor(
        env,
        getattr(state_machine, "middle_contact_strength", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    strict_raw = _metric_call_tensor(
        env,
        getattr(state_machine, "opposed_contact_strength", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    lift = _metric_call_tensor(
        env,
        getattr(state_machine, "block_lift_height", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    success = _metric_call_tensor(
        env,
        getattr(state_machine, "lift_success_held", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    contact = torch.maximum(torch.maximum(thumb, index), middle)
    return {
        "stage1" : (stage >= 1.0).to(dtype=torch.float32),
        "stage2" : (stage >= 2.0).to(dtype=torch.float32),
        "strict" : (strict_raw >= _strict_contact_threshold(context)).to(dtype=torch.float32),
        "contact": (contact > 0.0).to(dtype=torch.float32),
        "lift"   : torch.clamp(lift, min=0.0),
        "success": (success >= 0.5).to(dtype=torch.float32),
    }


def _adaptive_episode_metric_tensors(
    context: TrainerRuntimeContext,
    env: object | None,
    runtime: NativeLiveRolloutState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(runtime.obs_tensor.shape[0])
    device = runtime.obs_tensor.device
    if env is None:
        zero = torch.zeros((n,), device=device, dtype=torch.float32)
        return zero, zero, zero
    stage = _metric_env_tensor(env, "_topdown_stage", n=n, device=device, default=0.0)
    state_machine = _topdown_state_machine_for_metrics()
    strict_raw = _metric_call_tensor(
        env,
        getattr(state_machine, "opposed_contact_strength", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    lift = _metric_call_tensor(
        env,
        getattr(state_machine, "block_lift_height", None) if state_machine is not None else None,
        n=n,
        device=device,
        default=0.0,
    )
    stage2 = (stage >= 2.0).to(dtype=torch.float32)
    strict = (strict_raw >= _strict_contact_threshold(context)).to(dtype=torch.float32)
    return stage2, strict, torch.clamp(lift, min=0.0)


def _reset_adaptive_episode_metric_accumulators(
    state: AdaptivePolicyAssistState,
    *,
    n: int,
    device: torch.device,
) -> None:
    state.metric_episode_steps = torch.zeros((n,), device=device, dtype=torch.float32)
    state.metric_episode_stage2_steps = torch.zeros((n,), device=device, dtype=torch.float32)
    state.metric_episode_strict_steps = torch.zeros((n,), device=device, dtype=torch.float32)
    state.metric_episode_best_lift = torch.zeros((n,), device=device, dtype=torch.float32)


def _ensure_adaptive_episode_metric_accumulators(
    state: AdaptivePolicyAssistState,
    *,
    n: int,
    device: torch.device,
) -> None:
    steps = state.metric_episode_steps
    if (
        steps is None
        or int(steps.numel()) != n
        or steps.device != device
    ):
        _reset_adaptive_episode_metric_accumulators(state, n=n, device=device)


def _metric_median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _new_train_episode_aggregate_state(*, n: int, device: torch.device) -> NativeTrainEpisodeAggregateState:
    return NativeTrainEpisodeAggregateState(
        steps=torch.zeros((n,), device=device, dtype=torch.float32),
        stage1_steps=torch.zeros((n,), device=device, dtype=torch.float32),
        stage2_steps=torch.zeros((n,), device=device, dtype=torch.float32),
        strict_steps=torch.zeros((n,), device=device, dtype=torch.float32),
        contact_steps=torch.zeros((n,), device=device, dtype=torch.float32),
        best_lift=torch.zeros((n,), device=device, dtype=torch.float32),
        success_seen=torch.zeros((n,), device=device, dtype=torch.float32),
        lift_values=[[] for _ in range(n)],
    )


def _ensure_train_episode_aggregate_state(
    runtime: NativeLiveRolloutState,
    *,
    n: int,
    device: torch.device,
) -> NativeTrainEpisodeAggregateState:
    state = runtime.episode_aggregate
    if (
        state is None
        or int(state.steps.numel()) != n
        or state.steps.device != device
        or len(state.lift_values) != n
    ):
        state = _new_train_episode_aggregate_state(n=n, device=device)
        runtime.episode_aggregate = state
    return state


def _mean_float(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(sum(float(value) for value in values) / len(values))


def _write_train_episode_scalar_events(
    writer,
    metrics: Mapping[str, float],
    *,
    global_step: int,
) -> None:
    if writer is None:
        return
    for name, value in metrics.items():
        scalar = float(value)
        if math.isnan(scalar):
            continue
        writer.add_scalar(f"train_episode/{name}", scalar, int(global_step))
    if hasattr(writer, "flush"):
        writer.flush()


def _update_train_episode_aggregate_metrics(
    runtime: NativeLiveRolloutState,
    writer,
    *,
    done_flags: torch.Tensor,
    metric_tensors: Mapping[str, torch.Tensor],
    global_step: int,
) -> dict[str, float]:
    stage1 = metric_tensors["stage1"].detach().to(dtype=torch.float32).reshape(-1)
    n = int(stage1.numel())
    device = stage1.device
    state = _ensure_train_episode_aggregate_state(runtime, n=n, device=device)
    stage2 = metric_tensors["stage2"].detach().to(device=device, dtype=torch.float32).reshape(-1)
    strict = metric_tensors["strict"].detach().to(device=device, dtype=torch.float32).reshape(-1)
    contact = metric_tensors["contact"].detach().to(device=device, dtype=torch.float32).reshape(-1)
    lift = metric_tensors["lift"].detach().to(device=device, dtype=torch.float32).reshape(-1).clamp_min(0.0)
    success = metric_tensors["success"].detach().to(device=device, dtype=torch.float32).reshape(-1)
    done = done_flags.to(device=device, dtype=torch.bool).reshape(-1)

    state.steps += 1.0
    state.stage1_steps += stage1
    state.stage2_steps += stage2
    state.strict_steps += strict
    state.contact_steps += contact
    state.best_lift = torch.maximum(state.best_lift, lift)
    state.success_seen = torch.maximum(state.success_seen, success)
    for env_id, lift_value in enumerate(lift.detach().cpu().tolist()):
        if env_id < len(state.lift_values):
            state.lift_values[env_id].append(float(lift_value))

    if not bool(done.any().item()):
        return {}

    idx = torch.nonzero(done, as_tuple=False).reshape(-1)
    steps = state.steps.index_select(0, idx).clamp_min(1.0)
    env_ids = [int(v) for v in idx.detach().cpu().tolist()]
    stage1_rates = (state.stage1_steps.index_select(0, idx) / steps).detach().cpu().tolist()
    stage2_rates = (state.stage2_steps.index_select(0, idx) / steps).detach().cpu().tolist()
    strict_rates = (state.strict_steps.index_select(0, idx) / steps).detach().cpu().tolist()
    contact_rates = (state.contact_steps.index_select(0, idx) / steps).detach().cpu().tolist()
    best_lifts = state.best_lift.index_select(0, idx).detach().cpu().tolist()
    success_rates = state.success_seen.index_select(0, idx).detach().cpu().tolist()
    median_lifts = [
        _metric_median(state.lift_values[env_id])
        for env_id in env_ids
        if env_id < len(state.lift_values) and state.lift_values[env_id]
    ]
    metrics = {
        "completed_count"     : float(len(env_ids)),
        "stage1_rate"         : _mean_float(stage1_rates),
        "stage2_rate"         : _mean_float(stage2_rates),
        "strict_contact_rate" : _mean_float(strict_rates),
        "contact_rate"        : _mean_float(contact_rates),
        "median_lift"         : _metric_median(median_lifts),
        "best_lift"           : max((float(value) for value in best_lifts), default=math.nan),
        "success_rate"        : _mean_float(success_rates),
    }
    _write_train_episode_scalar_events(writer, metrics, global_step=global_step)
    state.steps[done] = 0.0
    state.stage1_steps[done] = 0.0
    state.stage2_steps[done] = 0.0
    state.strict_steps[done] = 0.0
    state.contact_steps[done] = 0.0
    state.best_lift[done] = 0.0
    state.success_seen[done] = 0.0
    for env_id in env_ids:
        if env_id < len(state.lift_values):
            state.lift_values[env_id].clear()
    return metrics


def _metric_gate_passes(current: float, baseline: float, ratio: float) -> bool:
    if math.isnan(current) or math.isnan(baseline):
        return False
    if abs(float(baseline)) <= 1.0e-9:
        return float(current) >= float(baseline) - 1.0e-9
    return float(current) + 1.0e-9 >= float(baseline) * float(ratio)


def _adaptive_metric_baseline_ready(
    state: AdaptivePolicyAssistState,
    *,
    min_episodes: int,
) -> bool:
    return (
        len(state.metric_teacher_stage2_rates) >= min_episodes
        and len(state.metric_teacher_strict_rates) >= min_episodes
        and len(state.metric_teacher_best_lifts) >= min_episodes
    )


def _adaptive_metric_window_ready(
    state: AdaptivePolicyAssistState,
    *,
    min_episodes: int,
) -> bool:
    return (
        len(state.metric_window_stage2_rates) >= min_episodes
        and len(state.metric_window_strict_rates) >= min_episodes
        and len(state.metric_window_best_lifts) >= min_episodes
    )


def _clear_adaptive_metric_window(state: AdaptivePolicyAssistState) -> None:
    state.metric_window_stage2_rates.clear()
    state.metric_window_strict_rates.clear()
    state.metric_window_best_lifts.clear()


def _log_adaptive_metric_baseline_if_ready(
    state: AdaptivePolicyAssistState,
    *,
    min_episodes: int,
) -> None:
    if state.metric_baseline_logged:
        return
    if not _adaptive_metric_baseline_ready(state, min_episodes=min_episodes):
        return
    state.metric_baseline_logged = True
    print(
        "adaptive_metric_baseline_ready "
        f"episodes={len(state.metric_teacher_stage2_rates)} "
        f"stage2_med={_metric_median(state.metric_teacher_stage2_rates):.4f} "
        f"strict_med={_metric_median(state.metric_teacher_strict_rates):.4f} "
        f"lift_med={_metric_median(state.metric_teacher_best_lifts):.4f}",
        flush=True,
    )


def _update_adaptive_assist_from_episode_metrics(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
    *,
    done_flags: torch.Tensor,
    stage2_step: torch.Tensor,
    strict_step: torch.Tensor,
    lift_step: torch.Tensor,
    global_step: int,
) -> None:
    if not _bool_arg(context, "adaptive_policy_assist", False):
        return
    if not _adaptive_assist_metric_gate_enabled(context):
        return
    state = runtime.adaptive_assist
    if not isinstance(state, AdaptivePolicyAssistState):
        return

    base_mix, floor = _configured_assist_bounds(context, configs)
    baseline_end = int(context.args.get("start_steps", configs.counts.start_steps))
    start_after = _int_arg(context, "adaptive_assist_start_steps", -1)
    if start_after < 0:
        start_after = baseline_end

    if int(global_step) < baseline_end:
        bucket = "teacher"
    elif int(global_step) >= int(start_after):
        bucket = "adaptive"
    else:
        bucket = "ignored"

    n = int(stage2_step.numel())
    device = stage2_step.device
    if state.metric_active_bucket != bucket:
        _reset_adaptive_episode_metric_accumulators(state, n=n, device=device)
        state.metric_active_bucket = bucket
    else:
        _ensure_adaptive_episode_metric_accumulators(state, n=n, device=device)

    if bucket == "ignored":
        state.current_mix = max(float(state.current_mix), float(base_mix))
        state.good_steps = 0
        state.bad_steps = 0
        return

    assert state.metric_episode_steps is not None
    assert state.metric_episode_stage2_steps is not None
    assert state.metric_episode_strict_steps is not None
    assert state.metric_episode_best_lift is not None

    state.metric_episode_steps += 1.0
    state.metric_episode_stage2_steps += stage2_step.to(device=device, dtype=torch.float32).reshape(-1)
    state.metric_episode_strict_steps += strict_step.to(device=device, dtype=torch.float32).reshape(-1)
    state.metric_episode_best_lift = torch.maximum(
        state.metric_episode_best_lift,
        lift_step.to(device=device, dtype=torch.float32).reshape(-1),
    )

    done = done_flags.to(device=device, dtype=torch.bool).reshape(-1)
    if bool(done.any().item()):
        idx = torch.nonzero(done, as_tuple=False).reshape(-1)
        steps = state.metric_episode_steps.index_select(0, idx).clamp_min(1.0)
        stage2_rates = (state.metric_episode_stage2_steps.index_select(0, idx) / steps).detach().cpu().tolist()
        strict_rates = (state.metric_episode_strict_steps.index_select(0, idx) / steps).detach().cpu().tolist()
        best_lifts = state.metric_episode_best_lift.index_select(0, idx).detach().cpu().tolist()
        target_stage2 = state.metric_teacher_stage2_rates if bucket == "teacher" else state.metric_window_stage2_rates
        target_strict = state.metric_teacher_strict_rates if bucket == "teacher" else state.metric_window_strict_rates
        target_lift = state.metric_teacher_best_lifts if bucket == "teacher" else state.metric_window_best_lifts
        target_stage2.extend(float(v) for v in stage2_rates)
        target_strict.extend(float(v) for v in strict_rates)
        target_lift.extend(float(v) for v in best_lifts)
        state.metric_episode_steps[done] = 0.0
        state.metric_episode_stage2_steps[done] = 0.0
        state.metric_episode_strict_steps[done] = 0.0
        state.metric_episode_best_lift[done] = 0.0

    baseline_episodes = max(1, _int_arg(context, "adaptive_assist_baseline_episodes", 100))
    _log_adaptive_metric_baseline_if_ready(state, min_episodes=baseline_episodes)
    if bucket != "adaptive":
        state.current_mix = max(float(state.current_mix), float(base_mix))
        return

    window_episodes = max(1, _int_arg(context, "adaptive_assist_metric_window_episodes", 100))
    if not _adaptive_metric_window_ready(state, min_episodes=window_episodes):
        return

    current_stage2 = _metric_median(state.metric_window_stage2_rates)
    current_strict = _metric_median(state.metric_window_strict_rates)
    current_lift = _metric_median(state.metric_window_best_lifts)
    strict_any = max((float(value) for value in state.metric_window_strict_rates), default=math.nan)
    old_mix = float(state.current_mix)
    lower_step = max(0.0, _float_arg(context, "adaptive_assist_step", 0.005))
    raise_step = max(0.0, _float_arg(context, "adaptive_assist_recover_step", max(lower_step * 2.0, lower_step)))

    if _bool_arg(context, "adaptive_assist_strict_contact_gate", False):
        min_strict = max(0.0, _float_arg(context, "adaptive_assist_strict_contact_min_rate", 0.05))
        pass_strict_any = math.isfinite(strict_any) and strict_any + 1.0e-9 >= min_strict
        if pass_strict_any and lower_step > 0.0 and old_mix > floor:
            state.current_mix = max(float(floor), old_mix - lower_step)
            if float(state.current_mix) <= float(floor) + 1e-6 and int(state.floor_step) < 0:
                state.floor_step = int(global_step)
            print(
                "adaptive_assist_lower_strict_contact "
                f"step={int(global_step)} mix={old_mix:.4f}->{state.current_mix:.4f} "
                f"strict_any={strict_any:.4f} strict_median={current_strict:.4f} "
                f"min={min_strict:.4f} episodes={len(state.metric_window_strict_rates)}",
                flush=True,
            )
        elif (not pass_strict_any) and raise_step > 0.0 and old_mix < base_mix:
            state.current_mix = min(float(base_mix), old_mix + raise_step)
            if float(state.current_mix) > float(floor) + 1e-6:
                state.floor_step = -1
            print(
                "adaptive_assist_raise_strict_contact "
                f"step={int(global_step)} mix={old_mix:.4f}->{state.current_mix:.4f} "
                f"strict_any={strict_any:.4f} strict_median={current_strict:.4f} "
                f"min={min_strict:.4f} episodes={len(state.metric_window_strict_rates)}",
                flush=True,
            )
        else:
            print(
                "adaptive_assist_hold_strict_contact "
                f"step={int(global_step)} mix={old_mix:.4f} "
                f"strict_any={strict_any:.4f} strict_median={current_strict:.4f} "
                f"min={min_strict:.4f} episodes={len(state.metric_window_strict_rates)}",
                flush=True,
            )
        _clear_adaptive_metric_window(state)
        return

    if not _adaptive_metric_baseline_ready(state, min_episodes=baseline_episodes):
        return

    ratio = max(0.0, _float_arg(context, "adaptive_assist_metric_min_ratio", 0.70))
    baseline_stage2 = _metric_median(state.metric_teacher_stage2_rates)
    baseline_strict = _metric_median(state.metric_teacher_strict_rates)
    baseline_lift = _metric_median(state.metric_teacher_best_lifts)
    pass_stage2 = _metric_gate_passes(current_stage2, baseline_stage2, ratio)
    pass_strict = _metric_gate_passes(current_strict, baseline_strict, ratio)
    pass_lift = _metric_gate_passes(current_lift, baseline_lift, ratio)

    if pass_stage2 and pass_strict and pass_lift and lower_step > 0.0 and old_mix > floor:
        state.current_mix = max(float(floor), old_mix - lower_step)
        if float(state.current_mix) <= float(floor) + 1e-6 and int(state.floor_step) < 0:
            state.floor_step = int(global_step)
        print(
            "adaptive_assist_lower_metric "
            f"step={int(global_step)} mix={old_mix:.4f}->{state.current_mix:.4f} "
            f"stage2={current_stage2:.4f}/{baseline_stage2:.4f} "
            f"strict={current_strict:.4f}/{baseline_strict:.4f} "
            f"lift={current_lift:.4f}/{baseline_lift:.4f} "
            f"episodes={len(state.metric_window_stage2_rates)} ratio={ratio:.2f}",
            flush=True,
        )
    elif (not pass_stage2 or not pass_strict or not pass_lift) and raise_step > 0.0 and old_mix < base_mix:
        state.current_mix = min(float(base_mix), old_mix + raise_step)
        if float(state.current_mix) > float(floor) + 1e-6:
            state.floor_step = -1
        print(
            "adaptive_assist_raise_metric "
            f"step={int(global_step)} mix={old_mix:.4f}->{state.current_mix:.4f} "
            f"stage2={current_stage2:.4f}/{baseline_stage2:.4f} "
            f"strict={current_strict:.4f}/{baseline_strict:.4f} "
            f"lift={current_lift:.4f}/{baseline_lift:.4f} "
            f"pass={int(pass_stage2)}/{int(pass_strict)}/{int(pass_lift)} "
            f"episodes={len(state.metric_window_stage2_rates)} ratio={ratio:.2f}",
            flush=True,
        )
    else:
        print(
            "adaptive_assist_hold_metric "
            f"step={int(global_step)} mix={old_mix:.4f} "
            f"stage2={current_stage2:.4f}/{baseline_stage2:.4f} "
            f"strict={current_strict:.4f}/{baseline_strict:.4f} "
            f"lift={current_lift:.4f}/{baseline_lift:.4f} "
            f"pass={int(pass_stage2)}/{int(pass_strict)}/{int(pass_lift)} "
            f"episodes={len(state.metric_window_stage2_rates)} ratio={ratio:.2f}",
            flush=True,
        )
    _clear_adaptive_metric_window(state)


def _update_adaptive_assist(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
    selection: NativeActionSelection,
    *,
    global_step: int,
) -> None:
    if not _bool_arg(context, "adaptive_policy_assist", False):
        return
    if _adaptive_assist_metric_gate_enabled(context):
        return
    state = runtime.adaptive_assist
    if not isinstance(state, AdaptivePolicyAssistState):
        return
    diagnostics = selection.diagnostics
    if not bool(getattr(diagnostics, "teacher_available", False)):
        state.good_steps = 0
        state.bad_steps = 0
        return

    base_mix, floor = _configured_assist_bounds(context, configs)
    start_after = _int_arg(context, "adaptive_assist_start_steps", -1)
    if start_after < 0:
        start_after = int(context.args.get("start_steps", configs.counts.start_steps))
    if int(global_step) < int(start_after):
        state.current_mix = max(float(state.current_mix), float(base_mix))
        state.good_steps = 0
        state.bad_steps = 0
        return

    arm_error = float(getattr(diagnostics, "actor_teacher_arm_mse", math.nan))
    finger_error = float(getattr(diagnostics, "actor_teacher_finger_mse", math.nan))
    if math.isnan(arm_error) or math.isnan(finger_error):
        state.good_steps = 0
        state.bad_steps = 0
        return

    arm_threshold = _float_arg(context, "adaptive_assist_arm_error", 0.03)
    finger_threshold = _float_arg(context, "adaptive_assist_finger_error", 0.05)
    bad_arm_threshold = _float_arg(context, "adaptive_assist_bad_arm_error", max(arm_threshold * 2.0, arm_threshold))
    bad_finger_threshold = _float_arg(context, "adaptive_assist_bad_finger_error", max(finger_threshold * 2.0, finger_threshold))
    lower_after = max(1, _int_arg(context, "adaptive_assist_window_steps", 500))
    raise_after = max(1, _int_arg(context, "adaptive_assist_bad_window_steps", 50))
    lower_step = max(0.0, _float_arg(context, "adaptive_assist_step", 0.005))
    raise_step = max(0.0, _float_arg(context, "adaptive_assist_recover_step", max(lower_step * 2.0, lower_step)))
    transition_rows = 1
    mixed_action = getattr(selection, "mixed_action", None)
    if isinstance(mixed_action, torch.Tensor) and int(mixed_action.ndim) > 0:
        transition_rows = max(1, int(mixed_action.shape[0]))

    if arm_error <= arm_threshold and finger_error <= finger_threshold:
        state.good_steps += transition_rows
    else:
        state.good_steps = 0

    if arm_error >= bad_arm_threshold or finger_error >= bad_finger_threshold:
        state.bad_steps += transition_rows
    else:
        state.bad_steps = 0

    old_mix = float(state.current_mix)
    if state.good_steps >= lower_after and lower_step > 0.0 and old_mix > floor:
        state.current_mix = max(float(floor), old_mix - lower_step)
        state.good_steps = 0
        state.bad_steps = 0
        if float(state.current_mix) <= float(floor) + 1e-6 and int(state.floor_step) < 0:
            state.floor_step = int(global_step)
        print(
            "adaptive_assist_lower "
            f"step={int(global_step)} mix={old_mix:.4f}->{state.current_mix:.4f} "
            f"a2t_arm={arm_error:.4f} a2t_f={finger_error:.4f} window={lower_after}",
            flush=True,
        )
        return

    if state.bad_steps >= raise_after and raise_step > 0.0 and old_mix < base_mix:
        state.current_mix = min(float(base_mix), old_mix + raise_step)
        state.good_steps = 0
        state.bad_steps = 0
        if float(state.current_mix) > float(floor) + 1e-6:
            state.floor_step = -1
        print(
            "adaptive_assist_raise "
            f"step={int(global_step)} mix={old_mix:.4f}->{state.current_mix:.4f} "
            f"a2t_arm={arm_error:.4f} a2t_f={finger_error:.4f} window={raise_after}",
            flush=True,
        )


def _adaptive_assist_at_floor(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
) -> bool:
    if not _bool_arg(context, "adaptive_policy_assist", False):
        return False
    state = runtime.adaptive_assist
    if not isinstance(state, AdaptivePolicyAssistState):
        return False
    _base_mix, floor = _configured_assist_bounds(context, configs)
    return float(state.current_mix) <= float(floor) + 1e-6


def _adaptive_assist_at_base(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
) -> bool:
    if not _bool_arg(context, "adaptive_policy_assist", False):
        return False
    state = runtime.adaptive_assist
    if not isinstance(state, AdaptivePolicyAssistState):
        return False
    base_mix, _floor = _configured_assist_bounds(context, configs)
    return float(state.current_mix) >= float(base_mix) - 1e-6


def _should_stop_on_adaptive_assist_floor(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
    loop_state: TrainingLoopStartupState,
) -> bool:
    if not _bool_arg(context, "stop_on_adaptive_assist_floor", False):
        return False
    state = runtime.adaptive_assist
    if not isinstance(state, AdaptivePolicyAssistState):
        return False
    if not _adaptive_assist_at_floor(context, configs, runtime):
        return False
    if int(state.floor_step) < 0:
        state.floor_step = int(loop_state.transitions_collected)
    post_floor_steps = max(0, _int_arg(context, "adaptive_assist_post_floor_steps", 0))
    if post_floor_steps > 0:
        elapsed = int(loop_state.transitions_collected) - int(state.floor_step)
        if elapsed < post_floor_steps:
            return False
    _base_mix, floor = _configured_assist_bounds(context, configs)
    print(
        "adaptive_assist_floor_reached "
        f"mix={float(state.current_mix):.4f} floor={float(floor):.4f} "
        f"floor_step={int(state.floor_step)} "
        f"post_floor_steps={post_floor_steps}",
        flush=True,
    )
    return True


def _adaptive_teacher_bc_base_weight(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    name: str,
    fallback: float,
) -> float:
    value = float(context.args.get(name, fallback))
    return max(0.0, value) if name == "teacher_bc_weight" else value


def _restore_configured_bc_only_settings(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    agent_config,
) -> None:
    if hasattr(agent_config, "bc_only_steps"):
        setattr(agent_config, "bc_only_steps", int(context.args.get("bc_only_steps", configs.assist.bc_only_steps)))
    if hasattr(agent_config, "bc_only_weight"):
        setattr(agent_config, "bc_only_weight", float(context.args.get("bc_only_weight", configs.assist.bc_only_weight)))
    if hasattr(agent_config, "bc_only_arm_weight"):
        setattr(
            agent_config,
            "bc_only_arm_weight",
            float(context.args.get("bc_only_arm_weight", configs.assist.bc_only_arm_weight)),
        )
    if hasattr(agent_config, "bc_only_finger_weight"):
        setattr(
            agent_config,
            "bc_only_finger_weight",
            float(context.args.get("bc_only_finger_weight", configs.assist.bc_only_finger_weight)),
        )


def _force_adaptive_bc_only_actor_update(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    loop_state: TrainingLoopStartupState,
    agent_config,
) -> None:
    progress_step = max(0, int(loop_state.transitions_collected) - 1)
    if hasattr(agent_config, "bc_only_steps"):
        setattr(agent_config, "bc_only_steps", max(progress_step + 1, 1))
    if hasattr(agent_config, "bc_only_weight"):
        setattr(
            agent_config,
            "bc_only_weight",
            max(0.0, float(context.args.get("bc_only_weight", configs.assist.bc_only_weight))),
        )
    if hasattr(agent_config, "bc_only_arm_weight"):
        setattr(
            agent_config,
            "bc_only_arm_weight",
            float(context.args.get("bc_only_arm_weight", configs.assist.bc_only_arm_weight)),
        )
    if hasattr(agent_config, "bc_only_finger_weight"):
        setattr(
            agent_config,
            "bc_only_finger_weight",
            float(context.args.get("bc_only_finger_weight", configs.assist.bc_only_finger_weight)),
        )


def _sync_adaptive_actor_training(
    context: TrainerRuntimeContext,
    configs: RuntimeConfigBundle,
    runtime: NativeLiveRolloutState,
    loop_state: TrainingLoopStartupState,
    agent,
) -> None:
    if not _bool_arg(context, "adaptive_policy_assist", False):
        return
    state = runtime.adaptive_assist
    if not isinstance(state, AdaptivePolicyAssistState):
        return
    agent_config = getattr(agent, "config", None)
    if agent_config is None:
        return

    _restore_configured_bc_only_settings(context, configs, agent_config)
    if (
        _bool_arg(context, "adaptive_assist_bc_only_until_decay", False)
        and not _adaptive_assist_at_floor(context, configs, runtime)
    ):
        _force_adaptive_bc_only_actor_update(context, configs, loop_state, agent_config)

    if (
        _bool_arg(context, "adaptive_assist_disable_bc_after_floor", False)
        and _adaptive_assist_at_floor(context, configs, runtime)
    ):
        if hasattr(agent_config, "bc_only_steps"):
            setattr(agent_config, "bc_only_steps", 0)
        if hasattr(agent_config, "bc_only_weight"):
            setattr(agent_config, "bc_only_weight", 0.0)
        if hasattr(agent_config, "bc_only_arm_weight"):
            setattr(agent_config, "bc_only_arm_weight", -1.0)
        if hasattr(agent_config, "bc_only_finger_weight"):
            setattr(agent_config, "bc_only_finger_weight", -1.0)
        if hasattr(agent_config, "teacher_bc_weight"):
            setattr(agent_config, "teacher_bc_weight", 0.0)
        if hasattr(agent_config, "teacher_bc_arm_weight"):
            setattr(agent_config, "teacher_bc_arm_weight", -1.0)
        if hasattr(agent_config, "teacher_bc_finger_weight"):
            setattr(agent_config, "teacher_bc_finger_weight", -1.0)
        if hasattr(agent_config, "teacher_bc_decay_steps"):
            setattr(agent_config, "teacher_bc_decay_steps", 0)
        return
    if not _bool_arg(context, "adaptive_assist_sync_bc_weights", True):
        return

    mix = max(0.0, min(1.0, float(state.current_mix)))
    base_weight = _adaptive_teacher_bc_base_weight(
        context,
        configs,
        "teacher_bc_weight",
        configs.assist.teacher_bc_weight,
    )
    arm_weight = _adaptive_teacher_bc_base_weight(
        context,
        configs,
        "teacher_bc_arm_weight",
        configs.assist.teacher_bc_arm_weight,
    )
    finger_weight = _adaptive_teacher_bc_base_weight(
        context,
        configs,
        "teacher_bc_finger_weight",
        configs.assist.teacher_bc_finger_weight,
    )
    if hasattr(agent_config, "teacher_bc_weight"):
        setattr(agent_config, "teacher_bc_weight", base_weight * mix)
    if hasattr(agent_config, "teacher_bc_arm_weight"):
        setattr(agent_config, "teacher_bc_arm_weight", arm_weight * mix if arm_weight >= 0.0 else -1.0)
    if hasattr(agent_config, "teacher_bc_finger_weight"):
        setattr(
            agent_config,
            "teacher_bc_finger_weight",
            finger_weight * mix if finger_weight >= 0.0 else -1.0,
        )
    if hasattr(agent_config, "teacher_bc_decay_steps"):
        setattr(agent_config, "teacher_bc_decay_steps", 0)


def collect_native_live_step(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    startup_state,                         # Param: input value used as startup state
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    runtime   : NativeLiveRolloutState,  # Param: input value used as runtime
    hooks     : NativeLiveHooks,  # Param: input value used as hooks
) -> NativeLoopStepBatch:
    """Collect one live native env step and update runtime tensors

    Steps:
    - Resolve inputs for `collect_native_live_step` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    components = _components(startup_state)
    gate_config = components.td3_config.gate_config
    first_trace = _native_debug_logging_enabled() and loop_state.transitions_collected == 0
    if first_trace:
        print("native_live_collect_begin", flush=True)
    preroll_mask = (
        hooks.preroll_mask_fn(runtime)
        if hooks.preroll_mask_fn is not None
        else loop_state.preroll.active
    )
    if preroll_mask is None:
        preroll_mask = _default_mask(runtime)
    global_step = int(loop_state.transitions_collected)
    scheduled_mix_config = native_live_action_mix_config(
        configs,
        context,
        global_step=global_step,
    )
    selection = select_native_rollout_action(
        obs_tensor=runtime.obs_tensor,
        policy_action_fn=lambda obs: _policy_action(components.agent, obs),
        policy_processors=hooks.policy_processors,
        teacher_action_fn=hooks.teacher_action_fn,
        teacher_processors=hooks.teacher_processors,
        mix_config=_adaptive_assist_config(
            context,
            configs,
            runtime,
            scheduled_mix_config,
            global_step=global_step,
        ),
        num_arm=gate_config.num_arm,
        num_fingers=gate_config.num_fingers,
        preroll_action=hooks.preroll_action_fn(runtime) if hooks.preroll_action_fn is not None else None,
        preroll_mask=preroll_mask,
    )
    _update_adaptive_assist(
        context,
        configs,
        runtime,
        selection,
        global_step=global_step,
    )
    metric_gate_enabled = _adaptive_assist_metric_gate_enabled(context)
    if metric_gate_enabled:
        env = startup_state.get("env")
        stage2_step, strict_step, lift_step = _adaptive_episode_metric_tensors(context, env, runtime)
    else:
        stage2_step = strict_step = lift_step = None
    if first_trace:
        print(
            "native_live_action_selected "
            f"teacher={int(selection.teacher_action is not None)} "
            f"policy_shape={tuple(selection.policy_action.shape)}",
            flush=True,
        )
    active_mask = hooks.active_env_mask_fn(runtime) if hooks.active_env_mask_fn is not None else None
    source_fn = hooks.action_source_fn or _default_action_source
    if first_trace:
        print("native_live_env_step_begin", flush=True)
    step_result = collect_native_env_step(
        NativeEnvStepRequest(
            obs=runtime.obs,
            obs_tensor=runtime.obs_tensor,
            priv_obs_tensor=runtime.priv_obs_tensor,
            action_selection=selection,
            preroll_mask_before=preroll_mask,
            action_source=source_fn(selection),
            replay=components.replay,
            n_step_queues=loop_state.n_step_queues,
            obs_keys=context.obs_keys,
            gamma=_float_arg(context, "gamma", configs.optimization.gamma),
            n_step=_int_arg(context, "n_step", configs.counts.n_step),
            privileged_critic=context.dims.priv_obs_dim > 0,
            active_env_mask=active_mask,
            existing_checkpoint_names=(
                hooks.existing_checkpoint_names_fn()
                if hooks.existing_checkpoint_names_fn is not None
                else ()
            ),
            action_assembly=hooks.action_assembly_fn() if hooks.action_assembly_fn is not None else None,
        ),
        NativeEnvStepCallbacks(
            env_step_fn=hooks.env_step_fn,
            assemble_env_action_fn=hooks.assemble_env_action_fn,
        ),
    )
    if first_trace:
        print(
            "native_live_env_step_end "
            f"num_added={step_result.batch.num_added} "
            f"replay_size={step_result.batch.replay_size}",
            flush=True,
        )
    episode_metric_tensors = _episode_aggregate_metric_tensors(context, startup_state.get("env"), runtime)
    _update_train_episode_aggregate_metrics(
        runtime,
        components.tensorboard_writer,
        done_flags=step_result.batch.done_flags,
        metric_tensors=episode_metric_tensors,
        global_step=global_step + int(step_result.batch.num_added),
    )
    if metric_gate_enabled:
        assert stage2_step is not None and strict_step is not None and lift_step is not None
        _update_adaptive_assist_from_episode_metrics(
            context,
            configs,
            runtime,
            done_flags=step_result.batch.done_flags,
            stage2_step=stage2_step,
            strict_step=strict_step,
            lift_step=lift_step,
            global_step=global_step,
        )
    runtime.last_stats_rows = update_native_rollout_stats(
        components.agent,
        NativeRolloutStatUpdate(
            obs_tensor=runtime.obs_tensor,
            priv_obs_tensor=runtime.priv_obs_tensor,
            reward_tensor=step_result.reward_tensor,
            active_env_mask=active_mask,
        ),
    )
    runtime.obs = dict(step_result.next_obs)
    runtime.obs_tensor = step_result.next_obs_tensor
    runtime.priv_obs_tensor = (
        step_result.next_priv_obs_tensor
        if context.dims.priv_obs_dim > 0
        else flatten_privileged_obs(step_result.next_obs)
    )
    runtime.last_action_selection = selection
    runtime.last_step_result = step_result
    return step_result.batch


def update_native_live_agent(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    startup_state,                         # Param: input value used as startup state
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    runtime: NativeLiveRolloutState,  # Param: input value used as runtime
) -> dict[str, object]:
    """Run one live native TD update from loop callbacks"""
    components = _components(startup_state)
    _sync_adaptive_actor_training(context, configs, runtime, loop_state, components.agent)
    return run_native_td_update(
        NativeUpdateRequest(
            agent=components.agent,
            replay=components.replay,
            batch_size=_int_arg(context, "batch_size", configs.counts.batch_size),
            progress_step=max(0, int(loop_state.transitions_collected) - 1),
        )
    )


def build_native_live_loop_callbacks(
    runtime: NativeLiveRolloutState,  # Param: input value used as runtime
    hooks  : NativeLiveHooks,  # Param: input value used as hooks
    *,
    include_updates: bool                     = True,  # Param: boolean input controlling include updates
    event_callbacks: NativeLiveEventCallbacks = NativeLiveEventCallbacks(),  # Param: input value used as event callbacks
) -> NativeLoopCallbacks:
    """Build NativeLoopCallbacks for a live env-backed rollout"""

    def _collect(context, configs, startup_state, loop_state):
        return collect_native_live_step(context, configs, startup_state, loop_state, runtime, hooks)

    def _update(context, configs, startup_state, loop_state):
        return update_native_live_agent(context, configs, startup_state, loop_state, runtime)

    def _should_stop(context, configs, startup_state, loop_state):
        return _should_stop_on_adaptive_assist_floor(context, configs, runtime, loop_state)

    def _done_reset(context, configs, startup_state, loop_state, plan):
        env = startup_state.get("env")
        clear = getattr(env, "clear_cached_teacher_action", None)
        if callable(clear):
            clear()
        if event_callbacks.on_done_reset is not None:
            event_callbacks.on_done_reset(context, configs, startup_state, loop_state, plan)

    return NativeLoopCallbacks(
        collect_step=_collect,
        update_step=_update if include_updates else None,
        on_step_plan=event_callbacks.on_step_plan,
        on_log=event_callbacks.on_log,
        on_eval=event_callbacks.on_eval,
        on_checkpoint=event_callbacks.on_checkpoint,
        on_done_reset=_done_reset,
        should_stop=_should_stop,
    )
