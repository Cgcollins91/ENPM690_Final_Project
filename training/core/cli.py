"""

Import-safe CLI request parsing for training entrypoints

This module defines a TrainingCliRequest dataclass that encapsulates the core CLI values needed before Isaac startup,
along with helper functions to parse and normalize these values. The goal is to isolate the import of argparse and
any related mutations to this module, allowing other parts of the training code to operate on a clean, normalized request
object without direct dependency on argparse or the exact CLI structure.
This also serves as a single place to apply any necessary mutations to the parsed CLI values before they are used by the
rest of the training code, ensuring consistency across different entrypoints and compatibility paths.


The main components of this module are:
- TrainingCliRequest:        A dataclass that holds the parsed CLI values and provides properties for easy access
- build_training_cli_parser: A function that constructs the argparse.ArgumentParser with the expected arguments
- parse_training_cli:        A function that takes raw argv and returns a TrainingCliRequest with the parsed values
- normalize_prelaunch_args:  A function that applies any necessary mutations to the parsed arguments

The TrainingCliRequest class also includes a validate_supported method to check if the requested task is supported,
and an action_layout method to resolve the action layout based on the parsed CLI values. The build_static_runtime_context
function can be used to create a TrainerRuntimeContext from a TrainingCliRequest, which can then be used by the rest of the training code.

File map:

TrainingCliRequest:            Core CLI values needed before Isaac startup
_path_default:                 Handle path default logic
build_training_cli_parser:     Build the import-safe core parser used before compatibility launch
parse_training_cli:            Parse core CLI values while preserving the exact forwarded argv
build_static_runtime_context:  Build an import-safe context from CLI values known before env creation
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os

from ..actions.action_space import (
    ActionLayout,
    ActionLayoutOptions,
    OBS_SCHEMA_VERSION,
    TOPDOWN_POLICY_OBS_DIM,
    TOPDOWN_POLICY_OBS_KEYS,
    TOPDOWN_PRIVILEGED_OBS_DIM,
    resolve_action_layout,
)
from .context import (
    TrainerDimensions,
    TrainerPaths,
    TrainerRuntimeContext,
    build_action_context,
)
from .cli_legacy import register_legacy_training_args
from .paths import project_root_from_training_package
from .runtime import SUPPORTED_TOPDOWN_TASK, TopdownTaskRuntime, env_flag


@dataclass(frozen=True)
class TrainingCliRequest:
    """Core CLI values needed before Isaac startup"""

    argv        : tuple[str, ...]      # Field: raw command-line argument list for parser entrypoints
    known_args  : Mapping[str, object] # Field: string known args value used by training cli request
    unknown_args: tuple[str, ...]      # Field: string unknown args value used by training cli request
    project_root: str                  # Field: filesystem location for project root
    env         : Mapping[str, str] = field(default_factory=dict)  # Field: environment/backend object used by this runtime helper

    @property
    def task(self) -> str:
        """ Return requested task name """
        return str(self.known_args["task"])

    @property
    def td3_backend(self) -> str:
        """ Return requested TD3 backend """
        return str(self.known_args["td3_backend"])

    @property
    def seed(self) -> int:
        """ Return requested seed """
        return int(self.known_args["seed"])

    @property
    def device(self) -> str:
        """ Return requested device """
        return str(self.known_args["device"])

    @property
    def checkpoint_path(self) -> str:
        """ Return requested checkpoint path """
        return str(self.known_args["checkpoint_path"])

    @property
    def log_jsonl(self) -> str:
        """ Return requested JSONL log path """
        return str(self.known_args["log_jsonl"])

    @property
    def tensorboard_dir(self) -> str:
        """ Return requested TensorBoard directory value """
        return str(self.known_args["tensorboard_dir"])

    @property
    def arm_controller(self) -> str:
        """ Return requested arm controller mode """
        return str(self.known_args["arm_controller"])

    @property
    def finger_action_mode(self) -> str:
        """ Return requested finger action mode """
        return str(self.known_args["finger_action_mode"])

    @property
    def finger_delta_scale(self) -> float:
        """ Return requested finger delta scale value for delta finger action mode """
        return float(self.known_args["finger_delta_scale"])

    @property
    def privileged_critic(self) -> bool:
        """ Return whether privileged critic observations are enabled based on CLI value """
        return bool(self.known_args["privileged_critic"])

    def validate_supported(self) -> None:
        """ Raise if request is outside the standalone task contract """
        TopdownTaskRuntime(task=self.task, env=self.env).validate_supported()

    def action_layout_options(self) -> ActionLayoutOptions:
        """ Return action layout options from parsed CLI values """
        return ActionLayoutOptions(
            include_wrist_roll=bool(self.known_args["include_wrist_roll"]),
            include_waist_yaw=bool(self.known_args["include_waist_yaw"]),
            waist_yaw_action_scale=float(self.known_args["waist_yaw_action_scale"]),
            arm_action_scale_profile=str(self.known_args["arm_action_scale_profile"]),
            arm_controller=self.arm_controller,
        )

    def action_layout(self) -> ActionLayout:
        """ Resolve import-safe action layout for this request """
        return resolve_action_layout(self.action_layout_options())

    def monolith_argv(self) -> tuple[str, ...]:
        """Return exact argv that should be forwarded to the compatibility path"""
        return self.argv


def _path_default(project_root: str, relative_path: str) -> str:
    return os.path.join(project_root or ".", relative_path)


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def build_training_cli_parser(
    *,
    project_root: str | None               = None,  # Param: root directory for project
    env         : Mapping[str, str] | None = None,  # Param: environment or backend object used for runtime calls
) -> argparse.ArgumentParser:
    """Build the import-safe core parser used before compatibility launch

    Steps:
    - Resolve inputs for `build_training_cli_parser` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    root = project_root_from_training_package() if project_root is None else project_root
    source_env = os.environ if env is None else env
    parser = argparse.ArgumentParser(
        description="Train the topdown red-block curriculum with FastTD3",
        add_help=False,
    )
    parser.add_argument("--task", type=str, default=source_env.get("TASK", SUPPORTED_TOPDOWN_TASK))
    parser.add_argument("--num-envs", type=int, default=_env_int(source_env, "NUM_ENVS", 1))
    parser.add_argument("--seed", type=int, default=_env_int(source_env, "SEED", 7))
    parser.add_argument("--total-steps", type=int, default=_env_int(source_env, "TOTAL_STEPS", 100000))
    parser.add_argument("--start-steps", type=int, default=_env_int(source_env, "START_STEPS", 400))
    parser.add_argument("--batch-size", type=int, default=_env_int(source_env, "BATCH_SIZE", 128))
    parser.add_argument("--updates-per-step", type=int, default=_env_int(source_env, "UPDATES_PER_STEP", 4))
    parser.add_argument("--n-step", type=int, default=_env_int(source_env, "N_STEP", 1))
    parser.add_argument("--gamma", type=float, default=_env_float(source_env, "GAMMA", 0.995))
    parser.add_argument("--tau", type=float, default=_env_float(source_env, "TAU", 0.005))
    parser.add_argument("--policy-delay", type=int, default=_env_int(source_env, "POLICY_DELAY", 2))
    parser.add_argument("--actor-lr", type=float, default=_env_float(source_env, "ACTOR_LR", 3e-4))
    parser.add_argument("--critic-lr", type=float, default=_env_float(source_env, "CRITIC_LR", 3e-4))
    parser.add_argument("--device", type=str, default=source_env.get("DEVICE", "cuda:0"))
    parser.add_argument("--headless", action="store_true", default=env_flag("HEADLESS", False, source_env))
    parser.add_argument(
        "--enable-cameras",
        "--enable_cameras",
        dest="enable_cameras",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ENABLE_CAMERAS", False, source_env),
    )
    parser.add_argument("--checkpoint-path", type=str, default=source_env.get("CHECKPOINT_PATH", _path_default(root, "runs/native_training_latest.pt")))
    parser.add_argument("--log-jsonl", type=str, default=source_env.get("LOG_JSONL", _path_default(root, "runs/native_training_log.jsonl")))
    parser.add_argument("--tensorboard-dir", type=str, default=source_env.get("TENSORBOARD_DIR", ""))
    parser.add_argument(
        "--td3-backend",
        choices=("custom", "upstream_fasttd3"),
        default=source_env.get("TD3_BACKEND", "custom"),
    )
    parser.add_argument("--arm-controller", type=str, default=source_env.get("ARM_CONTROLLER", "policy"), choices=("policy", "ik"))
    parser.add_argument(
        "--finger-action-mode",
        type=str,
        default=source_env.get("FINGER_ACTION_MODE", "absolute"),
        choices=("absolute", "delta"),
    )
    parser.add_argument("--finger-delta-scale", type=float, default=_env_float(source_env, "FINGER_DELTA_SCALE", 0.05))
    parser.add_argument("--include-wrist-roll", action="store_true")
    parser.add_argument("--include-waist-yaw", action="store_true")
    parser.add_argument("--waist-yaw-action-scale", type=float, default=_env_float(source_env, "WAIST_YAW_ACTION_SCALE", 1.0))
    parser.add_argument(
        "--arm-action-scale-profile",
        type=str,
        default="side",
        choices=("side", "topdown"),
    )
    parser.add_argument(
        "--privileged-critic",
        action=argparse.BooleanOptionalAction,
        default=env_flag("PRIVILEGED_CRITIC", True, source_env),
    )
    parser.add_argument("--rl-phase-start-steps", type=int, default=_env_int(source_env, "RL_PHASE_START_STEPS", -1))
    register_legacy_training_args(parser, project_root=root, env=source_env)
    return parser


def parse_training_cli(
    argv: Sequence[str] = (),                       # Param: raw command-line arguments forwarded to the parser or subprocess
    *,                                              # Param: Allows for clear separation of argv from other parameters and emphasizes that this function is primarily for parsing argv
    project_root: str | None               = None,  # Param: root directory for project
    env         : Mapping[str, str] | None = None,  # Param: environment or backend object used for runtime calls
) -> TrainingCliRequest:
    """Parse core CLI values while preserving the exact forwarded argv

    Steps:
    - Resolve inputs for `parse_training_cli` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    root = project_root_from_training_package() if project_root is None else project_root
    source_env = dict(os.environ if env is None else env)
    parser = build_training_cli_parser(project_root=root, env=source_env)
    namespace, unknown = parser.parse_known_args(tuple(argv))
    if unknown:
        parser.error(f"unrecognized trainer arguments: {' '.join(unknown)}")
    return TrainingCliRequest(
        argv=tuple(argv),
        known_args=dict(vars(namespace)),
        unknown_args=tuple(unknown),
        project_root=root,
        env=source_env,
    )


def build_static_runtime_context(
    request: TrainingCliRequest,         # Param: normalized request object passed into this helper
    *,
    full_action_dim: int | None = None,  # Param: integer input for full action dim
) -> TrainerRuntimeContext:
    """Build an import-safe context from CLI values known before env creation

    Steps:
    - Resolve inputs for `build_static_runtime_context` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    request.validate_supported()
    layout = request.action_layout()
    action_dim = len(layout.policy_action_spec.joint_names)
    env_action_dim = len(layout.env_action_spec.joint_names)
    priv_obs_dim = TOPDOWN_PRIVILEGED_OBS_DIM if request.privileged_critic else 0
    return TrainerRuntimeContext(
        task=request.task,
        td3_backend=request.td3_backend,
        seed=request.seed,
        device=request.device,
        paths=TrainerPaths(
            checkpoint_path=request.checkpoint_path,
            log_jsonl=request.log_jsonl,
            tensorboard_dir=request.tensorboard_dir,
        ),
        dims=TrainerDimensions(
            obs_dim=TOPDOWN_POLICY_OBS_DIM,
            action_dim=action_dim,
            full_action_dim=env_action_dim if full_action_dim is None else int(full_action_dim),
            priv_obs_dim=priv_obs_dim,
        ),
        action=build_action_context(
            arm_controller=request.arm_controller,
            finger_action_mode=request.finger_action_mode,
            finger_delta_scale=request.finger_delta_scale,
            policy_action_spec=layout.policy_action_spec,
            env_action_spec=layout.env_action_spec,
        ),
        obs_schema_version=OBS_SCHEMA_VERSION,
        obs_keys=TOPDOWN_POLICY_OBS_KEYS,
        env=request.env,
        args=request.known_args,
    )
