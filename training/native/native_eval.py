"""

Callback-driven native eval rollout helpers

File map:

NativeEvalTerminalFlags:  Per-env eval terminal condition flags beyond env done
NativeEvalConfig:         Static settings for one native eval episode
NativeEvalCallbacks:      Injected eval rollout side effects
NativeEvalResult:         Result from one native eval episode rollout
_policy_obs:              Handle policy obs logic
_zero_terminal_flags:     Handle zero terminal flags logic
_default_env_values:      Handle default env values logic
run_native_eval_episode:  Run one callback-driven native eval episode
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..eval.eval_actions import ActionProcessor, EvalActionDiagnostics, TeacherActionFn, select_eval_action
from ..eval.eval_episode import EvalEpisodeState, EvalStepMasks
from ..eval.eval_summary_rows import build_eval_summary_from_env_values
from .native_reset import EnvResetFn, normalize_native_env_reset_result
from .native_step import (
    EnvStepFn,
    NativeEnvActionAssemblyConfig,
    NativeEnvStepCallbacks,
    assemble_native_env_action,
    normalize_native_env_step_result,
)
from ..env.observations import flatten_policy_obs


PolicyActionFn = Callable[[torch.Tensor], torch.Tensor]
TerminalFlagFn = Callable[[Mapping[str, object], object], "NativeEvalTerminalFlags"]
EvalEnvValuesFn = Callable[[EvalEpisodeState], Mapping[str, Sequence[object] | object]]
ResetCacheFn = Callable[[], None]
EvalStepMetricsFn = Callable[[EvalEpisodeState, EvalStepMasks, int], None]


@dataclass(frozen=True)
class NativeEvalTerminalFlags:
    """Per-env eval terminal condition flags beyond env done"""

    success            : torch.Tensor  # success flag or rate for the rollout/evaluation record
    off_table          : torch.Tensor  # flag indicating that the block left the table/work surface
    phase15_shell_drift: torch.Tensor  # tensor containing phase15 shell drift values for batched env rows
    block_drift        : torch.Tensor  # measured block drift used by diagnostics or success checks


@dataclass(frozen=True)
class NativeEvalConfig:
    """Static settings for one native eval episode"""

    obs_keys          : Sequence[str]  # ordered keys used to resolve obs values
    global_step       : int  # training step associated with this record or action
    eval_episode_idx  : int  # evaluation episode index for this record
    max_steps         : int  # step count used for max steps scheduling or reporting
    teacher_assist_mix: float                                = 0.0  # floating-point teacher assist mix value used by native eval config
    num_arm           : int                                  = 0  # number of arm action dimensions in the active layout
    num_fingers       : int                                  = 0  # number of finger action dimensions in the active layout
    action_assembly   : NativeEnvActionAssemblyConfig | None = None  # stores action assembly for native eval config


@dataclass(frozen=True)
class NativeEvalCallbacks:
    """Injected eval rollout side effects"""

    reset_fn               : EnvResetFn  # callback used for the reset fn operation
    env_step_fn            : EnvStepFn  # callback used for the env step fn operation
    select_policy_action_fn: PolicyActionFn  # callback used for the select policy action fn operation
    teacher_action_fn      : TeacherActionFn | None                        = None  # callback used for the teacher action fn operation
    terminal_flags_fn      : TerminalFlagFn | None                         = None  # callback used for the terminal flags fn operation
    env_values_fn          : EvalEnvValuesFn | None                        = None  # callback used for the env values fn operation
    step_metrics_fn        : EvalStepMetricsFn | None                      = None  # callback used to collect per-step eval metrics
    policy_processors      : Sequence[ActionProcessor]                     = ()  # ordered collection of policy processors entries for native eval callbacks
    teacher_processors     : Sequence[ActionProcessor]                     = ()  # ordered collection of teacher processors entries for native eval callbacks
    assemble_env_action_fn : Callable[[torch.Tensor], torch.Tensor] | None = None  # callback used for the assemble env action fn operation
    reset_cache_fn         : ResetCacheFn | None                           = None  # callback used for the reset cache fn operation


@dataclass(frozen=True)
class NativeEvalResult:
    """Result from one native eval episode rollout"""

    summary    : dict[str, object]  # string summary value used by native eval result
    state      : EvalEpisodeState  # stores state for native eval result
    steps_taken: int  # integer steps taken value tracked by native eval result
    diagnostics: tuple[EvalActionDiagnostics, ...]  # structured diagnostic values captured with the result


def _policy_obs(obs: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    policy = obs.get("policy")
    if not isinstance(policy, Mapping):
        raise KeyError("eval observation is missing policy observations")
    return policy  # type: ignore[return-value]


def _zero_terminal_flags(num_envs: int, device: torch.device) -> NativeEvalTerminalFlags:
    zeros = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    return NativeEvalTerminalFlags(
        success=zeros,
        off_table=zeros,
        phase15_shell_drift=zeros,
        block_drift=zeros,
    )


def _default_env_values(state: EvalEpisodeState) -> dict[str, Sequence[object]]:
    return {
        "return"             : state.return_env.detach().cpu().tolist(),
        "steps"              : state.steps_env.detach().cpu().tolist(),
        "success"            : state.success_mask.detach().cpu().tolist(),
        "off_table"          : state.off_table_mask.detach().cpu().tolist(),
        "phase15_shell_drift": state.phase15_shell_drift_mask.detach().cpu().tolist(),
        "block_drift"        : state.block_drift_mask.detach().cpu().tolist(),
        "timeout"            : state.timeout_mask.detach().cpu().tolist(),
        "done"               : state.done_mask.detach().cpu().tolist(),
    }


def run_native_eval_episode(
    config   : NativeEvalConfig,  # Param: configuration object used by this helper
    callbacks: NativeEvalCallbacks,  # Param: input value used as callbacks
) -> NativeEvalResult:
    """Run one callback-driven native eval episode

    Steps:
    - Resolve inputs for `run_native_eval_episode` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if callbacks.reset_cache_fn is not None:
        callbacks.reset_cache_fn()
    reset_payload = normalize_native_env_reset_result(callbacks.reset_fn())
    obs = reset_payload.obs
    obs_tensor = flatten_policy_obs(_policy_obs(obs), config.obs_keys)
    num_envs = int(obs_tensor.shape[0])
    state = EvalEpisodeState.create(num_envs=num_envs, device=obs_tensor.device)
    diagnostics: list[EvalActionDiagnostics] = []
    steps_taken = 0

    for step_index in range(max(0, int(config.max_steps))):
        action_result = select_eval_action(
            obs_tensor=obs_tensor,
            select_policy_action_fn=callbacks.select_policy_action_fn,
            policy_processors=callbacks.policy_processors,
            teacher_action_fn=callbacks.teacher_action_fn,
            teacher_processors=callbacks.teacher_processors,
            teacher_assist_mix=config.teacher_assist_mix,
            num_arm=config.num_arm,
            num_fingers=config.num_fingers,
        )
        diagnostics.append(action_result.diagnostics)
        env_action = assemble_native_env_action(
            action_result.action,
            callbacks=NativeEnvStepCallbacks(
                env_step_fn=callbacks.env_step_fn,
                assemble_env_action_fn=callbacks.assemble_env_action_fn,
            ),
            assembly=config.action_assembly,
        )
        payload = normalize_native_env_step_result(
            callbacks.env_step_fn(env_action),
            device=obs_tensor.device,
        )
        flags = (
            callbacks.terminal_flags_fn(payload.next_obs, payload.info)
            if callbacks.terminal_flags_fn is not None
            else _zero_terminal_flags(num_envs, obs_tensor.device)
        )
        step_masks = state.update(
            reward=payload.reward,
            success_flags=flags.success,
            off_table_flags=flags.off_table,
            phase15_shell_drift_flags=flags.phase15_shell_drift,
            block_drift_flags=flags.block_drift,
            terminated_flags=payload.terminated,
            truncated_flags=payload.timeout,
            step_index=step_index,
        )
        if callbacks.step_metrics_fn is not None:
            callbacks.step_metrics_fn(state, step_masks, step_index)
        steps_taken = step_index + 1
        obs = payload.next_obs
        obs_tensor = flatten_policy_obs(_policy_obs(obs), config.obs_keys)
        if state.all_done:
            break

    state.finalize_remaining_timeouts(steps_taken=steps_taken)
    env_values = _default_env_values(state)
    if callbacks.env_values_fn is not None:
        env_values.update(dict(callbacks.env_values_fn(state)))
    summary = build_eval_summary_from_env_values(
        global_step=config.global_step,
        eval_episode_idx=config.eval_episode_idx,
        env_values=env_values,
    )
    return NativeEvalResult(
        summary=summary,
        state=state,
        steps_taken=steps_taken,
        diagnostics=tuple(diagnostics),
    )
