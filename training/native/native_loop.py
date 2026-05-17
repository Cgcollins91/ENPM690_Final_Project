"""

Callback-driven native rollout loop scaffold

File map:

NativeLoopOptions:              Runtime loop switches that are independent of Isaac objects
NativeLoopStepBatch:            Result of one vectorized env step collection
NativeLoopCallbacks:            Callback surface for native loop side effects
NativeLoopRunSummary:           Summary from a native rollout loop pass
_default_active_mask:           Handle default active mask logic
_native_debug_logging_enabled:  Handle native debug logging enabled logic
_int_arg:                       Handle int arg logic
native_loop_step_batch:         Build a normalized native loop step batch
_run_updates_if_ready:          Handle run updates if ready logic
_build_step_plan:               Handle build step plan logic
_dispatch_step_callbacks:       Handle dispatch step callbacks logic
_advance_loop_state:            Handle advance loop state logic
should_continue_native_loop:    Return whether the native rollout loop should keep stepping
run_native_rollout_loop:        Run callback-driven native rollout loop decisions
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import os
from typing import Any

import torch

from ..state.cadence import advance_periodic_step
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..state.loop_plan import LoopStepPlan, build_loop_step_plan
from .native_backend import NativeTrainerState
from ..state.run_state import TrainingLoopStartupState
from ..core.runner import TrainingRunResult
from ..state.update_schedule import (
    UpdateRunResult,
    global_step_after_replay_flush,
    run_update_steps,
    update_readiness,
)
from ..state.phase_overrides import apply_rl_phase_overrides


CollectStepFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState, TrainingLoopStartupState],
    "NativeLoopStepBatch",
]
UpdateFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState, TrainingLoopStartupState],
    Mapping[str, Any],
]
PlanFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState, TrainingLoopStartupState, LoopStepPlan],
    None,
]
StopFn = Callable[
    [TrainerRuntimeContext, RuntimeConfigBundle, NativeTrainerState, TrainingLoopStartupState],
    bool,
]


@dataclass(frozen=True)
class NativeLoopOptions:
    """Runtime loop switches that are independent of Isaac objects"""

    eval_every                    : int | None = None  # integer eval every value tracked by native loop options
    max_outer_steps               : int | None = None  # step count used for max outer steps scheduling or reporting
    legacy_contact_preroll_enabled: bool       = False  # boolean state indicating whether legacy contact preroll is enabled
    topdown_preroll_enabled       : bool       = False  # boolean state indicating whether topdown preroll is enabled


@dataclass(frozen=True)
class NativeLoopStepBatch:
    """Result of one vectorized env step collection"""

    num_added                : int  # count of added values
    replay_size              : int  # configured or observed replay-buffer size
    done_flags               : torch.Tensor  # per-env done flags returned by the environment step
    active_env_mask          : torch.Tensor  # mask selecting env rows that are still active
    existing_checkpoint_names: tuple[str, ...] = ()  # ordered names used to resolve existing checkpoint attributes


@dataclass(frozen=True)
class NativeLoopCallbacks:
    """Callback surface for native loop side effects"""

    collect_step : CollectStepFn  # step count used for collect step scheduling or reporting
    update_step  : UpdateFn | None = None  # step count used for update step scheduling or reporting
    on_step_plan : PlanFn | None   = None  # stores on step plan for native loop callbacks
    on_log       : PlanFn | None   = None  # stores on log for native loop callbacks
    on_eval      : PlanFn | None   = None  # stores on eval for native loop callbacks
    on_checkpoint: PlanFn | None   = None  # stores on checkpoint for native loop callbacks
    on_done_reset: PlanFn | None   = None  # stores on done reset for native loop callbacks
    should_stop  : StopFn | None   = None  # stores should stop for native loop callbacks


@dataclass(frozen=True)
class NativeLoopRunSummary:
    """Summary from a native rollout loop pass"""

    status               : str  # string status value used by native loop run summary
    global_step          : int  # training step associated with this record or action
    transitions_collected: int  # number of replay transitions collected so far
    outer_steps          : int  # step count used for outer steps scheduling or reporting
    update_steps         : int  # step count used for update steps scheduling or reporting
    eval_runs            : int  # integer eval runs value tracked by native loop run summary
    checkpoint_jobs      : int  # integer checkpoint jobs value tracked by native loop run summary

    def as_result(self) -> TrainingRunResult:
        """Return runner-compatible result"""
        return TrainingRunResult(
            status=self.status,
            global_step=self.global_step,
            metrics={
                "transitions_collected": self.transitions_collected,
                "outer_steps"          : self.outer_steps,
                "update_steps"         : self.update_steps,
                "eval_runs"            : self.eval_runs,
                "checkpoint_jobs"      : self.checkpoint_jobs,
            },
        )


def _default_active_mask(done_flags: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(done_flags.to(dtype=torch.bool))


def _native_debug_logging_enabled() -> bool:
    return os.environ.get("NATIVE_DEBUG_LOGGING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _int_arg(context: TrainerRuntimeContext, name: str, default: int) -> int:
    try:
        return int(context.args.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def native_loop_step_batch(
    *,
    num_added                : int,  # Param: number of env transitions added during the current collection step
    replay_size              : int,  # Param: number of transitions currently available in replay
    done_flags               : torch.Tensor,  # Param: per-env done flags returned by the latest env step
    active_env_mask          : torch.Tensor | None = None,  # Param: mask selecting env rows still active for collection or reset handling
    existing_checkpoint_names: tuple[str, ...]     = (),  # Param: existing checkpoint filenames used to decide rolling-checkpoint pruning
) -> NativeLoopStepBatch:
    """Build a normalized native loop step batch"""
    done = done_flags.to(dtype=torch.bool).reshape(-1)
    active = _default_active_mask(done) if active_env_mask is None else active_env_mask
    return NativeLoopStepBatch(
        num_added=max(0, int(num_added)),
        replay_size=max(0, int(replay_size)),
        done_flags=done,
        active_env_mask=active.to(device=done.device, dtype=torch.bool).reshape(-1),
        existing_checkpoint_names=tuple(existing_checkpoint_names),
    )


def _run_updates_if_ready(
    *,
    context      : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs      : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    startup_state: NativeTrainerState,  # Param: input value used as startup state
    loop_state   : TrainingLoopStartupState,  # Param: input value used as loop state
    batch        : NativeLoopStepBatch,  # Param: input value used as batch
    callbacks    : NativeLoopCallbacks,  # Param: input value used as callbacks
) -> UpdateRunResult:
    if configs.eval.play:
        return UpdateRunResult(
            update_count=0,
            last_update_info=loop_state.last_update_info,
            last_actor_update_info=loop_state.last_actor_update_info,
        )
    readiness = update_readiness(
        transitions_collected=loop_state.transitions_collected,
        start_steps=_int_arg(context, "start_steps", configs.counts.start_steps),
        replay_size=batch.replay_size,
        batch_size=_int_arg(context, "batch_size", configs.counts.batch_size),
        num_added=batch.num_added,
        updates_per_step=_int_arg(context, "updates_per_step", configs.counts.updates_per_step),
    )
    if not readiness.should_update or callbacks.update_step is None:
        return UpdateRunResult(
            update_count=0,
            last_update_info=loop_state.last_update_info,
            last_actor_update_info=loop_state.last_actor_update_info,
        )
    return run_update_steps(
        update_fn=lambda: callbacks.update_step(context, configs, startup_state, loop_state),
        update_count=readiness.update_count,
        previous_actor_update_info=loop_state.last_actor_update_info,
    )


def _build_step_plan(
    *,
    context    : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs    : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    loop_state : TrainingLoopStartupState,  # Param: input value used as loop state
    batch      : NativeLoopStepBatch,  # Param: input value used as batch
    global_step: int,  # Param: current absolute training step
    options    : NativeLoopOptions,  # Param: input value used as options
) -> LoopStepPlan:
    return build_loop_step_plan(
        global_step=global_step,
        num_added=batch.num_added,
        log_every=configs.eval.log_every,
        next_eval_step=loop_state.cadence.next_eval_step,
        replay_size=batch.replay_size,
        batch_size=configs.counts.batch_size,
        done_flags=batch.done_flags,
        active_env_mask=batch.active_env_mask,
        checkpoint_path=context.paths.checkpoint_path,
        save_replay_in_checkpoint=configs.checkpoint.save_replay_in_checkpoint,
        next_checkpoint_step=loop_state.cadence.next_checkpoint_step,
        checkpoint_every=configs.checkpoint.checkpoint_every,
        next_rolling_checkpoint_step=loop_state.cadence.next_rolling_checkpoint_step,
        rolling_checkpoint_every=configs.checkpoint.rolling_checkpoint_every,
        rolling_checkpoint_keep=configs.checkpoint.rolling_checkpoint_keep,
        existing_checkpoint_names=batch.existing_checkpoint_names,
        legacy_contact_preroll_enabled=options.legacy_contact_preroll_enabled,
        topdown_preroll_enabled=options.topdown_preroll_enabled,
    )


def _dispatch_step_callbacks(
    *,
    context      : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs      : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    startup_state: NativeTrainerState,  # Param: input value used as startup state
    loop_state   : TrainingLoopStartupState,  # Param: input value used as loop state
    plan         : LoopStepPlan,  # Param: precomputed plan object consumed by this helper
    callbacks    : NativeLoopCallbacks,  # Param: input value used as callbacks
    force_log    : bool = False,  # Param: forces one progress row outside normal cadence
) -> tuple[int, int]:
    """Process for `_dispatch_step_callbacks`

    Steps:
    - Resolve inputs for `_dispatch_step_callbacks` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if callbacks.on_step_plan is not None:
        callbacks.on_step_plan(context, configs, startup_state, loop_state, plan)
    if (plan.should_log or bool(force_log)) and callbacks.on_log is not None:
        callbacks.on_log(context, configs, startup_state, loop_state, plan)
    eval_runs = 0
    if plan.should_eval and callbacks.on_eval is not None:
        callbacks.on_eval(context, configs, startup_state, loop_state, plan)
        eval_runs = 1
    checkpoint_jobs = 0 if configs.eval.play else len(plan.checkpoint_plan.jobs)
    handoff_due = (
        bool(str(configs.checkpoint.handoff_checkpoint_path or "").strip())
        and not bool(loop_state.best_eval_state.get("_native_handoff_checkpoint_saved", 0.0))
        and int(plan.global_step) >= int(configs.counts.rl_phase_start_steps)
    )
    if not configs.eval.play and handoff_due:
        checkpoint_jobs += 1
    if checkpoint_jobs and callbacks.on_checkpoint is not None:
        callbacks.on_checkpoint(context, configs, startup_state, loop_state, plan)
    if plan.done_reset_plan.has_done and callbacks.on_done_reset is not None:
        callbacks.on_done_reset(context, configs, startup_state, loop_state, plan)
    return eval_runs, checkpoint_jobs


def _advance_loop_state(
    *,
    loop_state   : TrainingLoopStartupState,  # Param: input value used as loop state
    plan         : LoopStepPlan,  # Param: precomputed plan object consumed by this helper
    update_result: UpdateRunResult,  # Param: input value used as update result
    eval_ran     : bool,  # Param: boolean input controlling eval ran
    eval_every   : int | None,  # Param: interval controlling how often eval runs
) -> None:
    effective_eval_every = eval_every if eval_every and int(eval_every) > 0 else loop_state.cadence.eval_every
    loop_state.cadence = replace(
        loop_state.cadence,
        next_eval_step=(
            advance_periodic_step(loop_state.cadence.next_eval_step, effective_eval_every)
            if eval_ran
            else loop_state.cadence.next_eval_step
        ),
        next_checkpoint_step=plan.checkpoint_plan.next_checkpoint_step,
        next_rolling_checkpoint_step=plan.checkpoint_plan.next_rolling_checkpoint_step,
    )
    loop_state.last_update_info = update_result.last_update_info
    loop_state.last_actor_update_info = (
        update_result.last_actor_update_info
        if update_result.last_actor_update_info is not None
        else loop_state.last_actor_update_info
    )


def should_continue_native_loop(
    *,
    loop_state : TrainingLoopStartupState,  # Param: input value used as loop state
    configs    : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    outer_steps: int,  # Param: step count used for outer steps
    options    : NativeLoopOptions,  # Param: input value used as options
) -> bool:
    """Return whether the native rollout loop should keep stepping"""
    if configs.eval.play:
        return False
    if loop_state.skip_training_after_handoff_reuse:
        return False
    if options.max_outer_steps is not None and int(outer_steps) >= int(options.max_outer_steps):
        return False
    return int(loop_state.transitions_collected) < int(configs.counts.total_steps)


def run_native_rollout_loop(
    context      : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs      : RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    startup_state: NativeTrainerState,  # Param: input value used as startup state
    loop_state   : TrainingLoopStartupState,  # Param: input value used as loop state
    callbacks    : NativeLoopCallbacks,  # Param: input value used as callbacks
    *,
    options: NativeLoopOptions = NativeLoopOptions(),  # Param: input value used as options
) -> NativeLoopRunSummary:
    """Run callback-driven native rollout loop decisions

    Steps:
    - Resolve inputs for `run_native_rollout_loop` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    context.validate_supported()
    outer_steps = 0
    update_steps = 0
    eval_runs = 0
    checkpoint_jobs = 0
    global_step = global_step_after_replay_flush(loop_state.transitions_collected, 0)
    debug_logging = _native_debug_logging_enabled()
    if debug_logging:
        print(
            "native_loop_start "
            f"transitions={loop_state.transitions_collected} "
            f"total_steps={configs.counts.total_steps} "
            f"max_outer_steps={options.max_outer_steps}",
            flush=True,
        )
    if configs.eval.play:
        done_flags = torch.zeros(int(configs.counts.num_envs), device=context.device, dtype=torch.bool)
        batch = native_loop_step_batch(
            num_added=0,
            replay_size=0,
            done_flags=done_flags,
        )
        play_plan = replace(
            _build_step_plan(
                context=context,
                configs=configs,
                loop_state=loop_state,
                batch=batch,
                global_step=global_step,
                options=options,
            ),
            should_log=False,
            should_eval=True,
        )
        eval_count, checkpoint_count = _dispatch_step_callbacks(
            context=context,
            configs=configs,
            startup_state=startup_state,
            loop_state=loop_state,
            plan=play_plan,
            callbacks=callbacks,
        )
        return NativeLoopRunSummary(
            status="ok",
            global_step=global_step,
            transitions_collected=loop_state.transitions_collected,
            outer_steps=0,
            update_steps=0,
            eval_runs=eval_count,
            checkpoint_jobs=checkpoint_count,
        )

    while should_continue_native_loop(
        loop_state=loop_state,
        configs=configs,
        outer_steps=outer_steps,
        options=options,
    ):
        if debug_logging and outer_steps == 0:
            print("native_loop_collect_begin outer_step=0", flush=True)
        if callbacks.should_stop is not None and callbacks.should_stop(
            context,
            configs,
            startup_state,
            loop_state,
        ):
            break

        batch = callbacks.collect_step(context, configs, startup_state, loop_state)
        if debug_logging and outer_steps == 0:
            print(
                "native_loop_collect_end "
                f"num_added={batch.num_added if isinstance(batch, NativeLoopStepBatch) else 'bad'} "
                f"replay_size={batch.replay_size if isinstance(batch, NativeLoopStepBatch) else 'bad'}",
                flush=True,
            )
        if not isinstance(batch, NativeLoopStepBatch):
            raise TypeError(f"native collect_step returned {type(batch)!r}")

        loop_state.transitions_collected += batch.num_added
        global_step = global_step_after_replay_flush(loop_state.transitions_collected, batch.num_added)
        components = startup_state.get("components")
        agent = getattr(components, "agent", None)
        if agent is not None:
            phase_result = apply_rl_phase_overrides(context.args, agent, global_step=global_step)
            if phase_result.message:
                print(phase_result.message, flush=True)
        if outer_steps == 0 and callbacks.on_log is not None:
            first_log_plan = _build_step_plan(
                context=context,
                configs=configs,
                loop_state=loop_state,
                batch=batch,
                global_step=global_step,
                options=options,
            )
            if not first_log_plan.should_log:
                callbacks.on_log(context, configs, startup_state, loop_state, first_log_plan)
        update_result = _run_updates_if_ready(
            context=context,
            configs=configs,
            startup_state=startup_state,
            loop_state=loop_state,
            batch=batch,
            callbacks=callbacks,
        )
        plan = _build_step_plan(
            context=context,
            configs=configs,
            loop_state=loop_state,
            batch=batch,
            global_step=global_step,
            options=options,
        )
        eval_count, checkpoint_count = _dispatch_step_callbacks(
            context=context,
            configs=configs,
            startup_state=startup_state,
            loop_state=loop_state,
            plan=plan,
            callbacks=callbacks,
        )
        _advance_loop_state(
            loop_state=loop_state,
            plan=plan,
            update_result=update_result,
            eval_ran=bool(eval_count),
            eval_every=options.eval_every,
        )
        outer_steps += 1
        update_steps += update_result.update_count
        eval_runs += eval_count
        checkpoint_jobs += checkpoint_count

    if debug_logging:
        print(
            "native_loop_end "
            f"status=ok "
            f"global_step={global_step} "
            f"transitions={loop_state.transitions_collected} "
            f"outer_steps={outer_steps} "
            f"update_steps={update_steps} "
            f"eval_runs={eval_runs} "
            f"checkpoint_jobs={checkpoint_jobs}",
            flush=True,
        )
    return NativeLoopRunSummary(
        status="ok",
        global_step=global_step,
        transitions_collected=loop_state.transitions_collected,
        outer_steps=outer_steps,
        update_steps=update_steps,
        eval_runs=eval_runs,
        checkpoint_jobs=checkpoint_jobs,
    )
