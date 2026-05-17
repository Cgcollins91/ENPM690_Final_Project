"""

Native trainer final checkpoint and cleanup adapter

File map:

NativeFinalizationCallbacks:       Injected side effects for native finalization
NativeFinalizationResult:          Final checkpoint and resource cleanup outcome
_trace_noop:                       Handle trace noop logic
build_native_checkpoint_metadata:  Build native checkpoint metadata from explicit runtime state
_save_native_final_checkpoint:     Handle save native final checkpoint logic
close_native_training_resources:   Close native writer env and app when all runtime resources exist
finalize_native_training:          Run native final checkpoints and resource cleanup
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..io.checkpoint_io import TrainingCheckpointMetadata, capture_rng_state, save_training_checkpoint
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..io.finalization import (
    FinalCheckpointResult,
    ResourceCloseResult,
    build_final_checkpoint_jobs,
    close_training_resources,
    run_final_checkpoint_jobs,
)
from .native_backend import NativeTrainerState
from .native_components import NativeTrainingComponents
from ..state.run_state import TrainingLoopStartupState


NativeCheckpointSaveFn = Callable[..., Mapping[str, object]]
TraceFn = Callable[[str], None]


@dataclass(frozen=True)
class NativeFinalizationCallbacks:
    """Injected side effects for native finalization"""

    save_checkpoint_fn: NativeCheckpointSaveFn = save_training_checkpoint  # callback used for the save checkpoint fn operation
    trace_fn          : TraceFn | None         = None  # callback used for the trace fn operation


@dataclass(frozen=True)
class NativeFinalizationResult:
    """Final checkpoint and resource cleanup outcome"""

    checkpoint_result: FinalCheckpointResult | None  # integer checkpoint result value tracked by native finalization result
    close_result     : ResourceCloseResult | None  # stores close result for native finalization result


def _trace_noop(message: str) -> None:
    del message


def build_native_checkpoint_metadata(
    context   : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    loop_state: TrainingLoopStartupState,  # Param: input value used as loop state
    *,
    global_step          : int,  # Param: current absolute training step
    handoff_compatibility: Mapping[str, object] | None = None,  # Param: string input for handoff compatibility
) -> TrainingCheckpointMetadata:
    """Build native checkpoint metadata from explicit runtime state"""
    return TrainingCheckpointMetadata(
        task=context.task,
        global_step=int(global_step),
        episode_idx=loop_state.episode.episode_idx.max(),
        arm_controller=context.action.arm_controller,
        td3_backend=context.td3_backend,
        obs_schema_version=context.obs_schema_version,
        obs_keys=tuple(context.obs_keys),
        obs_dim=context.dims.obs_dim,
        priv_obs_dim=context.dims.priv_obs_dim,
        policy_action_spec=context.action.policy_action_spec,
        env_action_spec=context.action.env_action_spec,
        log_jsonl=context.paths.log_jsonl,
        args=context.args,
        handoff_compatibility=handoff_compatibility,
    )


def _save_native_final_checkpoint(
    *,
    context           : TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    metadata          : TrainingCheckpointMetadata,  # Param: integer input for metadata
    components        : NativeTrainingComponents,  # Param: input value used as components
    save_checkpoint_fn: NativeCheckpointSaveFn,  # Param: callback used to compute or fetch save checkpoint
):
    def _save(job):
        path = context.paths.checkpoint_path if job.dest_path is None else job.dest_path
        save_checkpoint_fn(
            path,
            metadata=metadata,
            agent=components.agent,
            replay=components.replay,
            include_replay=job.include_replay,
            rng_state=capture_rng_state(),
        )

    return _save


def close_native_training_resources(
    state: NativeTrainerState,                    # Param: mutable or immutable runtime state read by this helper
    *,
    components: NativeTrainingComponents | None,  # Param: input value used as components
    trace_fn  : TraceFn = _trace_noop,  # Param: callback used to compute or fetch trace
) -> ResourceCloseResult | None:
    """Close native writer env and app when all runtime resources exist

    Steps:
    - Resolve inputs for `close_native_training_resources` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    env = state.get("env")
    simulation_app = state.get("simulation_app")
    writer = None if components is None else components.tensorboard_writer
    if env is None or simulation_app is None:
        return None
    return close_training_resources(
        tensorboard_writer=writer,
        env=env,
        simulation_app=simulation_app,
        trace_fn=trace_fn,
    )


def finalize_native_training(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    state  : NativeTrainerState,  # Param: mutable or immutable runtime state read by this helper
    *,
    callbacks: NativeFinalizationCallbacks = NativeFinalizationCallbacks(),  # Param: input value used as callbacks
) -> NativeFinalizationResult:
    """Run native final checkpoints and resource cleanup

    Steps:
    - Resolve inputs for `finalize_native_training` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    print("native_finalization_start", flush=True)
    trace_fn = callbacks.trace_fn or _trace_noop
    components = state.get("components")
    loop_state = state.get("loop_state")
    checkpoint_result = None
    if (
        not configs.eval.play
        and isinstance(components, NativeTrainingComponents)
        and isinstance(loop_state, TrainingLoopStartupState)
    ):
        jobs = build_final_checkpoint_jobs(
            transitions_collected=loop_state.transitions_collected,
            save_replay_in_checkpoint=configs.checkpoint.save_replay_in_checkpoint,
            replay_present=components.replay is not None,
            final_handoff_checkpoint_path=configs.checkpoint.final_handoff_checkpoint_path,
        )
        metadata = build_native_checkpoint_metadata(
            context,
            loop_state,
            global_step=jobs[0].global_step,
            handoff_compatibility=state.get("handoff_compatibility"),
        )
        checkpoint_result = run_final_checkpoint_jobs(
            jobs,
            _save_native_final_checkpoint(
                context=context,
                metadata=metadata,
                components=components,
                save_checkpoint_fn=callbacks.save_checkpoint_fn,
            ),
            trace_fn=trace_fn,
        )
        print(
            "native_finalization_checkpoint "
            f"labels={checkpoint_result.saved_labels} "
            f"error={type(checkpoint_result.error).__name__ if checkpoint_result.error else 'none'}",
            flush=True,
        )

    close_result = close_native_training_resources(
        state,
        components=components if isinstance(components, NativeTrainingComponents) else None,
        trace_fn=trace_fn,
    )
    print("native_finalization_end", flush=True)
    return NativeFinalizationResult(
        checkpoint_result=checkpoint_result,
        close_result=close_result,
    )
