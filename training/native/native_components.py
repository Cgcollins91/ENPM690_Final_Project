"""

Native trainer runtime component assembly

File map:

NativeComponentFactories:              Factories used to construct native trainer runtime objects
NativeStartupCheckpoints:              Checkpoint payloads requested by startup options
NativeTrainingComponents:              Runtime components needed by the native training loop
_bool_arg:                             Handle bool arg logic
_float_arg:                            Handle float arg logic
_int_arg:                              Handle int arg logic
_str_arg:                              Handle str arg logic
native_action_gate_config:             Build action-gate config from explicit runtime context
td3_config_from_runtime:               Build custom TD3 config from typed runtime settings
upstream_fasttd3_config_from_runtime:  Build upstream FastTD3 constructor config from runtime args
upstream_agent_factory_from_runtime:   Return the default upstream FastTD3 agent factory for native runs
load_requested_startup_checkpoints:    Load checkpoint payloads requested for native startup
create_native_training_components:     Create agent replay TensorBoard and startup checkpoint components
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from typing import Any

from ..actions.action_gates import ActionGateConfig, topdown_xyz_preload_fraction
from ..model.agent_factory import make_td3_agent
from ..model.agents import TD3Config
from ..model.upstream_fasttd3 import UpstreamFastTD3Config, make_upstream_fasttd3_agent
from ..io.checkpoint_io import load_training_checkpoint, load_training_checkpoint_mmap
from ..core.configs import RuntimeConfigBundle
from ..core.context import TrainerRuntimeContext
from ..state.replay import ReplayBuffer
from ..io.startup_checkpoints import (
    CheckpointStartupPlan,
    build_checkpoint_startup_plan,
    validate_checkpoint_startup_plan,
)
from ..logging.tensorboard_setup import TensorBoardPlan, create_tensorboard_writer, resolve_tensorboard_plan


AgentFactory = Callable[..., object]
ReplayFactory = Callable[..., object]
CheckpointLoader = Callable[[str], Mapping[str, object]]


@dataclass(frozen=True)
class NativeComponentFactories:
    """Factories used to construct native trainer runtime objects"""

    agent_factory         : AgentFactory                                     = make_td3_agent  # stores agent factory for native component factories
    replay_factory        : ReplayFactory                                    = ReplayBuffer  # stores replay factory for native component factories
    checkpoint_loader     : CheckpointLoader                                 = load_training_checkpoint  # integer checkpoint loader value tracked by native component factories
    summary_writer_cls    : type | None                                      = None  # stores summary writer cls for native component factories
    upstream_agent_factory: Callable[[int, int, object, int], object] | None = None  # callback used for the upstream agent factory operation


@dataclass(frozen=True)
class NativeStartupCheckpoints:
    """Checkpoint payloads requested by startup options"""

    resume    : Mapping[str, object] | None = None  # string resume value used by native startup checkpoints
    actor_init: Mapping[str, object] | None = None  # string actor init value used by native startup checkpoints
    phase1    : Mapping[str, object] | None = None  # string phase1 value used by native startup checkpoints
    handoff   : Mapping[str, object] | None = None  # checkpoint loaded for replay handoff reuse
    play      : Mapping[str, object] | None = None  # checkpoint loaded for play/eval mode


@dataclass(frozen=True)
class NativeTrainingComponents:
    """Runtime components needed by the native training loop"""

    agent             : object  # stores agent for native training components
    replay            : object  # stores replay for native training components
    tensorboard_plan  : TensorBoardPlan  # stores tensorboard plan for native training components
    tensorboard_writer: object | None  # stores tensorboard writer for native training components
    checkpoint_plan   : CheckpointStartupPlan  # integer checkpoint plan value tracked by native training components
    checkpoints       : NativeStartupCheckpoints  # integer checkpoints value tracked by native training components
    td3_config        : TD3Config  # stores td3 config for native training components


def _bool_arg(context: TrainerRuntimeContext, name: str, default: bool = False) -> bool:
    value = context.args.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _float_arg(context: TrainerRuntimeContext, name: str, default: float) -> float:
    try:
        return float(context.args.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _int_arg(context: TrainerRuntimeContext, name: str, default: int) -> int:
    try:
        return int(context.args.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _str_arg(context: TrainerRuntimeContext, name: str, default: str) -> str:
    return str(context.args.get(name, default))


def native_action_gate_config(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
) -> ActionGateConfig:
    """Build action-gate config from explicit runtime context"""
    action_dim = len(context.action.policy_action_spec.joint_names)
    num_fingers = min(7, action_dim)
    num_arm = max(0, action_dim - num_fingers)
    return ActionGateConfig(
        num_arm=num_arm,
        num_fingers=num_fingers,
        topdown_curriculum=True,
        contact_unlock_obs_col=170,
        curriculum_unlock_obs_col=170,
        stage_one_hot_obs_col=166,
        contact_unlock_gate_threshold=_float_arg(context, "contact_unlock_gate_threshold", 0.5),
        contact_unlock_gate_start=_float_arg(context, "contact_unlock_gate_start", 0.20),
        mirror_middle_to_index=_bool_arg(context, "topdown_mirror_middle_to_index", False),
        three_finger_centering=_bool_arg(context, "topdown_three_finger_centering", False),
        topdown_contact_teacher_middle_scale=configs.teacher.topdown_contact_teacher_middle_scale,
        finger_action_mode=configs.teacher.finger_action_mode,
        finger_close_gate_mode=_str_arg(context, "topdown_finger_close_gate_mode", "center"),
        finger_xyz_preload_fraction=topdown_xyz_preload_fraction(
            context.args.get("topdown_xyz_preload_fraction", 0.20)
        ),
    )


def td3_config_from_runtime(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
) -> TD3Config:
    """Build custom TD3 config from typed runtime settings"""
    return TD3Config(
        hidden_dim=configs.counts.hidden_dim,
        actor_lr=configs.optimization.actor_lr,
        critic_lr=configs.optimization.critic_lr,
        gamma=configs.optimization.gamma,
        tau=configs.optimization.tau,
        policy_delay=configs.counts.policy_delay,
        policy_noise=configs.optimization.policy_noise,
        policy_noise_finger=configs.optimization.policy_noise_finger,
        noise_clip=configs.optimization.noise_clip,
        exploration_noise=configs.optimization.exploration_noise,
        exploration_noise_finger=configs.optimization.exploration_noise_finger,
        target_q_clip=configs.optimization.target_q_clip,
        critic_grad_clip=configs.optimization.critic_grad_clip,
        actor_pre_tanh_l2=configs.optimization.actor_pre_tanh_l2,
        observation_normalization=_bool_arg(context, "observation_normalization", False),
        obs_norm_eps=_float_arg(context, "obs_norm_eps", 1e-4),
        obs_norm_clip=_float_arg(context, "obs_norm_clip", 10.0),
        reward_normalization=_bool_arg(context, "reward_normalization", False),
        reward_norm_eps=_float_arg(context, "reward_norm_eps", 1e-4),
        reward_norm_clip=_float_arg(context, "reward_norm_clip", 10.0),
        actor_freeze_steps=configs.optimization.actor_freeze_steps,
        bc_only_steps=configs.assist.bc_only_steps,
        bc_only_weight=configs.assist.bc_only_weight,
        bc_only_arm_weight=configs.assist.bc_only_arm_weight,
        bc_only_finger_weight=configs.assist.bc_only_finger_weight,
        teacher_bc_weight=configs.assist.teacher_bc_weight,
        teacher_bc_arm_weight=configs.assist.teacher_bc_arm_weight,
        teacher_bc_finger_weight=configs.assist.teacher_bc_finger_weight,
        teacher_bc_decay_steps=configs.assist.teacher_bc_decay_steps,
        actor_q_action_gate_mode=_str_arg(context, "actor_q_action_gate_mode", "env"),
        actor_bc_action_gate_mode=_str_arg(context, "actor_bc_action_gate_mode", "env"),
        finger_noise_bypass_unlock=_bool_arg(context, "finger_noise_bypass_unlock", False),
        debug_nonfinite_updates=_bool_arg(context, "debug_nonfinite_updates", False),
        stop_on_nonfinite_update=_bool_arg(context, "stop_on_nonfinite_update", False),
        active_n_step=configs.counts.n_step,
        active_updates_per_step=configs.counts.updates_per_step,
        gate_config=native_action_gate_config(context, configs),
    )


def upstream_fasttd3_config_from_runtime(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
) -> UpstreamFastTD3Config:
    """Build upstream FastTD3 constructor config from runtime args."""
    return UpstreamFastTD3Config(
        init_scale=_float_arg(context, "fasttd3_init_scale", 0.01),
        actor_hidden_dim=_int_arg(context, "fasttd3_actor_hidden_dim", 512),
        critic_hidden_dim=_int_arg(context, "fasttd3_critic_hidden_dim", 1024),
        std_min=_float_arg(context, "fasttd3_std_min", 0.001),
        std_max=_float_arg(context, "fasttd3_std_max", 0.4),
        num_atoms=_int_arg(context, "fasttd3_num_atoms", 51),
        v_min=_float_arg(context, "fasttd3_v_min", -5.0),
        v_max=_float_arg(context, "fasttd3_v_max", 0.0),
        weight_decay=_float_arg(context, "fasttd3_weight_decay", 0.0),
        use_cdq=_bool_arg(context, "fasttd3_use_cdq", True),
        num_envs=configs.counts.num_envs,
        fasttd3_repo=_str_arg(context, "fasttd3_repo", ""),
    )


def upstream_agent_factory_from_runtime(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    td3_config: TD3Config,  # Param: shared TD3/runtime config
) -> Callable[[int, int, object, int], object]:
    """Return the default upstream FastTD3 agent factory for native runs."""
    upstream_config = upstream_fasttd3_config_from_runtime(context, configs)

    def _factory(obs_dim: int, action_dim: int, device: object, priv_obs_dim: int) -> object:
        return make_upstream_fasttd3_agent(
            obs_dim,
            action_dim,
            device,
            priv_obs_dim,
            config=td3_config,
            upstream_config=upstream_config,
        )

    return _factory


def load_requested_startup_checkpoints(
    configs: RuntimeConfigBundle,                         # Param: typed runtime config bundle used to derive this plan
    *,
    loader: CheckpointLoader = load_training_checkpoint,  # Param: integer input for loader
) -> NativeStartupCheckpoints:
    """Load checkpoint payloads requested for native startup"""

    def _load(path: str) -> Mapping[str, object] | None:
        return loader(path) if path else None

    def _load_optional(path: str) -> Mapping[str, object] | None:
        return loader(path) if path and os.path.isfile(path) else None

    def _load_play(path: str) -> Mapping[str, object] | None:
        if not path:
            return None
        if loader is load_training_checkpoint:
            return load_training_checkpoint_mmap(path)
        return loader(path)

    return NativeStartupCheckpoints(
        resume=_load(configs.checkpoint.resume_checkpoint),
        actor_init=_load(configs.checkpoint.actor_init_checkpoint),
        phase1=_load(configs.checkpoint.phase1_checkpoint),
        handoff=_load_optional(configs.checkpoint.handoff_checkpoint_path),
        play=_load_play(configs.checkpoint.checkpoint_path) if configs.eval.play and not configs.eval.play_skip_checkpoint else None,
    )


def create_native_training_components(
    context: TrainerRuntimeContext,  # Param: runtime context carrying validated trainer settings
    configs: RuntimeConfigBundle,  # Param: typed runtime config bundle used to derive this plan
    *,
    factories   : NativeComponentFactories = NativeComponentFactories(),  # Param: input value used as factories
    tb_available: bool                     = False,  # Param: boolean input controlling tb available
) -> NativeTrainingComponents:
    """Create agent replay TensorBoard and startup checkpoint components

    Steps:
    - Resolve inputs for `create_native_training_components` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    context.validate_supported()
    summary_writer_cls = factories.summary_writer_cls
    resolved_tb_available = bool(tb_available)
    if summary_writer_cls is None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            SummaryWriter = None
        if SummaryWriter is not None:
            summary_writer_cls = SummaryWriter
            resolved_tb_available = True

    log_jsonl = str(context.paths.log_jsonl or "").strip()
    if log_jsonl and log_jsonl.lower() not in {"off", "none", "false", "0"}:
        os.makedirs(os.path.dirname(log_jsonl) or ".", exist_ok=True)
        with open(log_jsonl, "w", encoding="utf-8"):
            pass

    observation_normalization = _bool_arg(context, "observation_normalization", False)
    reset_obs_stats_on_resume = _bool_arg(context, "reset_obs_stats_on_resume", True)
    checkpoint_plan = build_checkpoint_startup_plan(
        configs,
        observation_normalization=observation_normalization,
        reset_obs_stats_on_resume=reset_obs_stats_on_resume,
    )
    validate_checkpoint_startup_plan(checkpoint_plan, configs)
    td3_config = td3_config_from_runtime(context, configs)
    upstream_agent_factory = factories.upstream_agent_factory
    if upstream_agent_factory is None and str(context.td3_backend) == "upstream_fasttd3":
        upstream_agent_factory = upstream_agent_factory_from_runtime(context, configs, td3_config)
    agent = factories.agent_factory(
        td3_backend=context.td3_backend,
        obs_dim=context.dims.obs_dim,
        action_dim=context.dims.action_dim,
        device=context.device,
        priv_obs_dim=context.dims.priv_obs_dim,
        custom_config=td3_config,
        upstream_factory=upstream_agent_factory,
    )
    replay_size = 0 if configs.eval.play else configs.counts.replay_size
    replay = factories.replay_factory(
        replay_size,
        context.dims.obs_dim,
        context.dims.action_dim,
        context.dims.priv_obs_dim,
        context.device,
    )
    tensorboard_plan = resolve_tensorboard_plan(
        tensorboard_dir=configs.checkpoint.tensorboard_dir,
        log_jsonl=context.paths.log_jsonl,
        tb_available=resolved_tb_available,
    )
    writer = create_tensorboard_writer(tensorboard_plan, summary_writer_cls)
    checkpoints = load_requested_startup_checkpoints(
        configs,
        loader=factories.checkpoint_loader,
    )
    return NativeTrainingComponents(
        agent=agent,
        replay=replay,
        tensorboard_plan=tensorboard_plan,
        tensorboard_writer=writer,
        checkpoint_plan=checkpoint_plan,
        checkpoints=checkpoints,
        td3_config=td3_config,
    )
