"""Training, optimizer, and runtime configuration groups.

These groups describe the learning schedule around a fixed task/teacher.  They
do not know about hand geometry; they only decide how much data to collect,
when BC/DAgger gives way to RL, how aggressive TD3 updates are, and where run
artifacts are written.  Separating this from ``config.teacher`` is important:
it lets us warm-start from the same v32 teacher while trying different
Teacher -> BC -> RL schedules without changing the demonstrated behavior.
"""

from __future__ import annotations  # keeps annotations lazy for forward references

from dataclasses import dataclass  # imports dataclass helpers used by config groups

from .base import add_arg, add_flag, bool01, clean_dict  # imports shared env and CLI conversion helpers


@dataclass(frozen=True)  # makes the following config group immutable
class RunIOConfig:  # defines the run IO config group
    """Run identity and checkpoint inputs.

    This is the source of truth for run directory and warm-start/resume
    semantics.  A profile should choose exactly one story: fresh run, actor
    init, full resume, phase1 seed, or handoff replay.  The launcher validates
    required input checkpoint paths before starting Isaac Sim.
    """

    run_dir                                : str  # Sets the directory where logs and checkpoints are written
    checkpoint_path                        : str | None = None  # Sets the output checkpoint path when it differs from the run latest checkpoint
    actor_init_checkpoint                  : str | None = None  # Loads actor weights from a seed checkpoint without resuming optimizer state
    resume_checkpoint                      : str | None = None  # Loads a full trainer checkpoint for resume runs
    phase1_checkpoint                      : str | None = None  # Loads the phase one policy checkpoint used to seed later stages
    phase1_teacher_only                    : bool       = False  # Controls whether phase one runs teacher actions without RL updates
    reset_optimizers_on_resume             : bool       = False  # Controls whether resume runs discard optimizer state
    reset_obs_stats_on_resume              : bool       = False  # Controls whether resume runs discard observation statistics
    allow_warmstart                        : bool       = True  # Controls whether compatible checkpoints may initialize a fresh run
    save_replay_in_checkpoint              : bool       = False  # Controls whether replay buffer state is stored in checkpoints
    resume_replay                          : bool       = False  # Controls whether replay buffer state is restored during resume
    resume_global_step                     : bool       = False  # Controls whether global step count is restored during resume
    force_dagger_after_resume              : bool       = False  # Controls whether DAgger supervision resumes after loading a checkpoint
    dagger_resume_policy_assist_mix        : float      = -1.0  # Sets the resumed policy assist mix override
    dagger_resume_policy_assist_mix_floor  : float      = -1.0  # Sets the lower bound for resumed policy assist mix decay
    dagger_resume_policy_assist_decay_steps: int        = -1  # Sets decay length for resumed policy assist mixing
    allow_handoff_source_hash_mismatch     : bool       = False  # Allows handoff checkpoints even when source hashes differ
    handoff_checkpoint_path                : str | None = None  # Sets the checkpoint copied into the handoff path
    final_handoff_checkpoint_path          : str | None = None  # Sets the final checkpoint emitted for downstream handoff
    stop_after_handoff_checkpoint          : bool       = False  # Controls whether the run exits after writing the handoff checkpoint

    def resolved_checkpoint_path(self) -> str:  # chooses explicit checkpoint path or the run latest checkpoint
        """Return the explicit checkpoint path or the default latest checkpoint for the run."""
        return self.checkpoint_path or f"{self.run_dir}/latest.pt"  # returns the computed value

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "RUN_DIR": self.run_dir,  # Exports RUN_DIR from the run dir setting
                "CHECKPOINT_PATH": self.resolved_checkpoint_path(),  # Exports CHECKPOINT_PATH from the resolved checkpoint path helper result
                "ACTOR_INIT_CKPT": self.actor_init_checkpoint,  # Exports ACTOR_INIT_CKPT from the actor init checkpoint setting
                "RESUME_CKPT": self.resume_checkpoint,  # Exports RESUME_CKPT from the resume checkpoint setting
                "PHASE1_CKPT": self.phase1_checkpoint,  # Exports PHASE1_CKPT from the phase1 checkpoint setting
                "PHASE1_TEACHER_ONLY": bool01(self.phase1_teacher_only),  # Exports PHASE1_TEACHER_ONLY as legacy 0 or 1 from the phase1 teacher only setting
                "RESET_OPTIMIZERS_ON_RESUME": bool01(self.reset_optimizers_on_resume),  # Exports RESET_OPTIMIZERS_ON_RESUME as legacy 0 or 1 from the reset optimizers on resume setting
                "RESET_OBS_STATS_ON_RESUME": bool01(self.reset_obs_stats_on_resume),  # Exports RESET_OBS_STATS_ON_RESUME as legacy 0 or 1 from the reset obs stats on resume setting
                "ALLOW_WARMSTART": bool01(self.allow_warmstart),  # Exports ALLOW_WARMSTART as legacy 0 or 1 from the allow warmstart setting
                "SAVE_REPLAY_IN_CHECKPOINT": bool01(self.save_replay_in_checkpoint),  # Exports SAVE_REPLAY_IN_CHECKPOINT as legacy 0 or 1 from the save replay in checkpoint setting
                "RESUME_REPLAY": bool01(self.resume_replay),  # Exports RESUME_REPLAY as legacy 0 or 1 from the resume replay setting
                "RESUME_GLOBAL_STEP": bool01(self.resume_global_step),  # Exports RESUME_GLOBAL_STEP as legacy 0 or 1 from the resume global step setting
                "FORCE_DAGGER_AFTER_RESUME": bool01(self.force_dagger_after_resume),  # Exports FORCE_DAGGER_AFTER_RESUME as legacy 0 or 1 from the force dagger after resume setting
                "DAGGER_RESUME_POLICY_ASSIST_MIX": self.dagger_resume_policy_assist_mix,  # Exports DAGGER_RESUME_POLICY_ASSIST_MIX from the dagger resume policy assist mix setting
                "DAGGER_RESUME_POLICY_ASSIST_MIX_FLOOR": self.dagger_resume_policy_assist_mix_floor,  # Exports DAGGER_RESUME_POLICY_ASSIST_MIX_FLOOR from the dagger resume policy assist mix floor setting
                "DAGGER_RESUME_POLICY_ASSIST_DECAY_STEPS": self.dagger_resume_policy_assist_decay_steps,  # Exports DAGGER_RESUME_POLICY_ASSIST_DECAY_STEPS from the dagger resume policy assist decay steps setting
                "ALLOW_HANDOFF_SOURCE_HASH_MISMATCH": bool01(self.allow_handoff_source_hash_mismatch),  # Exports ALLOW_HANDOFF_SOURCE_HASH_MISMATCH as legacy 0 or 1 from the allow handoff source hash mismatch setting
                "HANDOFF_CHECKPOINT_PATH": self.handoff_checkpoint_path,  # Exports HANDOFF_CHECKPOINT_PATH from the handoff checkpoint path setting
                "FINAL_HANDOFF_CHECKPOINT_PATH": self.final_handoff_checkpoint_path,  # Exports FINAL_HANDOFF_CHECKPOINT_PATH from the final handoff checkpoint path setting
                "STOP_AFTER_HANDOFF_CHECKPOINT": bool01(self.stop_after_handoff_checkpoint),  # Exports STOP_AFTER_HANDOFF_CHECKPOINT as legacy 0 or 1 from the stop after handoff checkpoint setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args = ["--log-jsonl", f"{self.run_dir}/log.jsonl", "--checkpoint-path", self.resolved_checkpoint_path()]  # Collects trainer CLI arguments before return
        if self.actor_init_checkpoint:  # Checks whether actor init checkpoint
            args.extend(["--actor-init-checkpoint", self.actor_init_checkpoint])  # appends these trainer CLI tokens
        if self.resume_checkpoint:  # Checks whether resume checkpoint
            args.extend(["--resume-checkpoint", self.resume_checkpoint])  # appends these trainer CLI tokens
        if self.phase1_checkpoint:  # Checks whether phase1 checkpoint
            args.extend(["--phase1-checkpoint", self.phase1_checkpoint])  # appends these trainer CLI tokens
        add_flag(args, self.phase1_teacher_only, "--phase1-teacher-only")  # adds the trainer CLI flag when enabled
        add_flag(args, self.reset_optimizers_on_resume, "--reset-optimizers-on-resume")  # adds the trainer CLI flag when enabled
        add_flag(args, self.save_replay_in_checkpoint, "--save-replay-in-checkpoint")  # adds the trainer CLI flag when enabled
        add_flag(args, self.resume_replay, "--resume-replay")  # adds the trainer CLI flag when enabled
        add_flag(args, self.resume_global_step, "--resume-global-step")  # adds the trainer CLI flag when enabled
        add_flag(args, self.force_dagger_after_resume, "--force-dagger-after-resume")  # adds the trainer CLI flag when enabled
        if self.dagger_resume_policy_assist_mix >= 0.0:  # Checks whether dagger resume policy assist mix >= 0 point 0
            args.extend(["--dagger-resume-policy-assist-mix", str(self.dagger_resume_policy_assist_mix)])  # appends these trainer CLI tokens
        if self.dagger_resume_policy_assist_mix_floor >= 0.0:  # Checks whether dagger resume policy assist mix floor >= 0 point 0
            args.extend(["--dagger-resume-policy-assist-mix-floor", str(self.dagger_resume_policy_assist_mix_floor)])  # appends these trainer CLI tokens
        if self.dagger_resume_policy_assist_decay_steps >= 0:  # Checks whether dagger resume policy assist decay steps >= 0
            args.extend([  # appends these trainer CLI tokens
                "--dagger-resume-policy-assist-decay-steps",  # Adds trainer option --dagger-resume-policy-assist-decay-steps
                str(self.dagger_resume_policy_assist_decay_steps),  # Converts the dagger resume policy assist decay steps setting to CLI text
            ])  # closes the current expression
        add_flag(  # adds the trainer CLI flag when enabled
            args,  # continues this config expression
            self.allow_handoff_source_hash_mismatch,  # Passes the allow handoff source hash mismatch setting into the surrounding call
            "--allow-handoff-source-hash-mismatch",  # Adds trainer option --allow-handoff-source-hash-mismatch
        )  # closes the current expression
        if self.handoff_checkpoint_path:  # Checks whether handoff checkpoint path
            args.extend(["--handoff-checkpoint-path", self.handoff_checkpoint_path])  # appends these trainer CLI tokens
        if self.final_handoff_checkpoint_path:  # Checks whether final handoff checkpoint path
            args.extend(["--final-handoff-checkpoint-path", self.final_handoff_checkpoint_path])  # appends these trainer CLI tokens
        add_flag(args, self.stop_after_handoff_checkpoint, "--stop-after-handoff-checkpoint")  # adds the trainer CLI flag when enabled
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class CoreTrainingConfig:  # defines the core training config group
    """Seed, rollout size, replay horizon, and TD3 core cadence.

    ``start_steps`` and ``bc_only_steps`` control how much teacher-shaped data
    reaches the replay buffer before the actor is trusted.  ``rl_phase_start``
    is the handoff point where RL-specific optimizer/noise overrides may take
    effect.
    """

    seed                : int   = 7  # Sets the random seed for trainer and environment initialization
    num_envs            : int   = 100  # Sets the number of parallel simulated environments
    total_steps         : int   = 1_000_000  # Sets the total number of environment steps to train
    start_steps         : int   = 10_000  # Sets how many initial steps collect data before policy updates
    bc_only_steps       : int   = 100_000  # Sets how long behavior cloning dominates before RL handoff
    rl_phase_start_steps: int   = 100_000  # Sets the global step where RL-phase overrides begin
    replay_size         : int   = 1_000_000  # Sets replay buffer capacity in transitions
    batch_size          : int   = 256  # Sets learner minibatch size
    n_step              : int   = 3  # Sets the multi-step return horizon
    gamma               : float = 0.995  # Sets the reward discount factor
    tau                 : float = 0.005  # Sets the target network soft-update rate
    updates_per_step    : int   = 8  # Sets learner update count per environment step
    policy_delay        : int   = 2  # Sets how often delayed policy updates run

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "SEED": self.seed,  # Exports SEED from the seed setting
                "NUM_ENVS": self.num_envs,  # Exports NUM_ENVS from the num envs setting
                "TOTAL_STEPS": self.total_steps,  # Exports TOTAL_STEPS from the total steps setting
                "START_STEPS": self.start_steps,  # Exports START_STEPS from the start steps setting
                "BC_ONLY_STEPS": self.bc_only_steps,  # Exports BC_ONLY_STEPS from the BC only steps setting
                "RL_PHASE_START_STEPS": self.rl_phase_start_steps,  # Exports RL_PHASE_START_STEPS from the RL phase start steps setting
                "REPLAY_SIZE": self.replay_size,  # Exports REPLAY_SIZE from the replay size setting
                "BATCH_SIZE": self.batch_size,  # Exports BATCH_SIZE from the batch size setting
                "N_STEP": self.n_step,  # Exports N_STEP from the n step setting
                "GAMMA": self.gamma,  # Exports GAMMA from the gamma setting
                "TAU": self.tau,  # Exports TAU from the tau setting
                "UPDATES_PER_STEP": self.updates_per_step,  # Exports UPDATES_PER_STEP from the updates per step setting
                "POLICY_DELAY": self.policy_delay,  # Exports POLICY_DELAY from the policy delay setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for name, raw in (  # iterates over configured values
            ("--seed", self.seed),  # Pairs trainer option --seed with the seed setting
            ("--num-envs", self.num_envs),  # Pairs trainer option --num-envs with the num envs setting
            ("--total-steps", self.total_steps),  # Pairs trainer option --total-steps with the total steps setting
            ("--start-steps", self.start_steps),  # Pairs trainer option --start-steps with the start steps setting
            ("--bc-only-steps", self.bc_only_steps),  # Pairs trainer option --bc-only-steps with the BC only steps setting
            ("--rl-phase-start-steps", self.rl_phase_start_steps),  # Pairs trainer option --rl-phase-start-steps with the RL phase start steps setting
            ("--replay-size", self.replay_size),  # Pairs trainer option --replay-size with the replay size setting
            ("--batch-size", self.batch_size),  # Pairs trainer option --batch-size with the batch size setting
            ("--n-step", self.n_step),  # Pairs trainer option --n-step with the n step setting
            ("--gamma", self.gamma),  # Pairs trainer option --gamma with the gamma setting
            ("--tau", self.tau),  # Pairs trainer option --tau with the tau setting
            ("--updates-per-step", self.updates_per_step),  # Pairs trainer option --updates-per-step with the updates per step setting
            ("--policy-delay", self.policy_delay),  # Pairs trainer option --policy-delay with the policy delay setting
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class OptimizationConfig:  # defines the optimization config group
    """Base-phase optimizer and exploration settings."""

    actor_lr                : float = 3e-5  # Sets actor optimizer learning rate
    critic_lr               : float = 5e-5  # Sets critic optimizer learning rate
    target_q_clip           : float = 50  # Clips TD target Q values to bound critic targets
    critic_grad_clip        : float = 5.0  # Clips critic gradients before optimizer updates
    exploration_noise       : float = 0.01  # Sets arm action exploration noise during rollout
    exploration_noise_finger: float = 0.03  # Sets finger action exploration noise during rollout
    policy_noise            : float = 0.02  # Sets TD3 target policy smoothing noise for arm actions
    policy_noise_finger     : float = 0.04  # Sets TD3 target policy smoothing noise for finger actions
    noise_clip              : float = 0.12  # Clips target policy smoothing noise magnitude
    actor_pre_tanh_l2       : float = 0.08  # Penalizes actor pre-tanh activation magnitude
    actor_freeze_steps      : int   = 0  # Keeps actor updates frozen for the initial step window
    critic_burn_in_steps    : int   = 0  # Runs critic-only updates before actor learning starts

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "ACTOR_LR": self.actor_lr,  # Exports ACTOR_LR from the actor learning rate setting
                "CRITIC_LR": self.critic_lr,  # Exports CRITIC_LR from the critic learning rate setting
                "TARGET_Q_CLIP": self.target_q_clip,  # Exports TARGET_Q_CLIP from the target Q clip setting
                "CRITIC_GRAD_CLIP": self.critic_grad_clip,  # Exports CRITIC_GRAD_CLIP from the critic grad clip setting
                "EXPLORATION_NOISE": self.exploration_noise,  # Exports EXPLORATION_NOISE from the exploration noise setting
                "EXPLORATION_NOISE_FINGER": self.exploration_noise_finger,  # Exports EXPLORATION_NOISE_FINGER from the exploration noise finger setting
                "POLICY_NOISE": self.policy_noise,  # Exports POLICY_NOISE from the policy noise setting
                "POLICY_NOISE_FINGER": self.policy_noise_finger,  # Exports POLICY_NOISE_FINGER from the policy noise finger setting
                "NOISE_CLIP": self.noise_clip,  # Exports NOISE_CLIP from the noise clip setting
                "ACTOR_PRE_TANH_L2": self.actor_pre_tanh_l2,  # Exports ACTOR_PRE_TANH_L2 from the actor pre tanh l2 setting
                "ACTOR_FREEZE_STEPS": self.actor_freeze_steps,  # Exports ACTOR_FREEZE_STEPS from the actor freeze steps setting
                "CRITIC_BURN_IN_STEPS": self.critic_burn_in_steps,  # Exports CRITIC_BURN_IN_STEPS from the critic burn in steps setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for name, raw in (  # iterates over configured values
            ("--actor-lr", self.actor_lr),  # Pairs trainer option --actor-lr with the actor learning rate setting
            ("--critic-lr", self.critic_lr),  # Pairs trainer option --critic-lr with the critic learning rate setting
            ("--target-q-clip", self.target_q_clip),  # Pairs trainer option --target-q-clip with the target Q clip setting
            ("--critic-grad-clip", self.critic_grad_clip),  # Pairs trainer option --critic-grad-clip with the critic grad clip setting
            ("--exploration-noise", self.exploration_noise),  # Pairs trainer option --exploration-noise with the exploration noise setting
            ("--exploration-noise-finger", self.exploration_noise_finger),  # Pairs trainer option --exploration-noise-finger with the exploration noise finger setting
            ("--policy-noise", self.policy_noise),  # Pairs trainer option --policy-noise with the policy noise setting
            ("--policy-noise-finger", self.policy_noise_finger),  # Pairs trainer option --policy-noise-finger with the policy noise finger setting
            ("--noise-clip", self.noise_clip),  # Pairs trainer option --noise-clip with the noise clip setting
            ("--actor-pre-tanh-l2", self.actor_pre_tanh_l2),  # Pairs trainer option --actor-pre-tanh-l2 with the actor pre tanh l2 setting
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        if self.actor_freeze_steps > 0:  # Checks whether actor freeze steps > 0
            add_arg(args, "--actor-freeze-steps", self.actor_freeze_steps)  # adds a scalar trainer CLI option
        if self.critic_burn_in_steps > 0:  # Checks whether critic burn in steps > 0
            add_arg(args, "--critic-burn-in-steps", self.critic_burn_in_steps)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class DaggerConfig:  # defines the dagger config group
    """Behavior cloning and DAgger-label settings before the RL switch.

    ``policy_assist_mix`` controls action execution, while BC weights control
    actor supervised loss.  Keeping those separate lets a run execute mostly
    teacher-assisted behavior while still training the actor on clean teacher
    labels, which was essential for the rough IK teacher.
    """

    policy_bc_relabel              : bool  = True  # Controls whether policy BC relabel is enabled
    policy_assist_mix              : float = 0.25  # Sets the policy assist mix config value
    policy_assist_mix_floor        : float = 0.0  # Sets the policy assist mix floor config value
    policy_assist_decay_steps      : int   = 100_000  # Sets the number of steps for policy assist decay
    policy_assist_decay_start_steps: int   = -1  # Sets the number of steps for policy assist decay start
    bc_only_weight                 : float = 0  # Sets optimization weight for BC only
    bc_only_arm_weight             : float = 8  # Sets optimization weight for BC only arm
    bc_only_finger_weight          : float = 2  # Sets optimization weight for BC only finger
    teacher_bc_weight              : float = 0  # Sets optimization weight for teacher BC
    teacher_bc_arm_weight          : float = 8  # Sets optimization weight for teacher BC arm
    teacher_bc_finger_weight       : float = 2  # Sets optimization weight for teacher BC finger
    teacher_bc_decay_steps         : int   = 600_000  # Sets the number of steps for teacher BC decay
    assist_noise_arm               : float = 0.0  # Sets the assist noise arm config value
    assist_noise_finger            : float = 0.0  # Sets the assist noise finger config value

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "POLICY_BC_RELABEL": bool01(self.policy_bc_relabel),  # Exports POLICY_BC_RELABEL as legacy 0 or 1 from the policy BC relabel setting
                "POLICY_ASSIST_MIX": self.policy_assist_mix,  # Exports POLICY_ASSIST_MIX from the policy assist mix setting
                "POLICY_ASSIST_MIX_FLOOR": self.policy_assist_mix_floor,  # Exports POLICY_ASSIST_MIX_FLOOR from the policy assist mix floor setting
                "POLICY_ASSIST_DECAY_STEPS": self.policy_assist_decay_steps,  # Exports POLICY_ASSIST_DECAY_STEPS from the policy assist decay steps setting
                "POLICY_ASSIST_DECAY_START_STEPS": self.policy_assist_decay_start_steps,  # Exports POLICY_ASSIST_DECAY_START_STEPS from the policy assist decay start steps setting
                "BC_ONLY_WEIGHT": self.bc_only_weight,  # Exports BC_ONLY_WEIGHT from the BC only weight setting
                "BC_ONLY_ARM_WEIGHT": self.bc_only_arm_weight,  # Exports BC_ONLY_ARM_WEIGHT from the BC only arm weight setting
                "BC_ONLY_FINGER_WEIGHT": self.bc_only_finger_weight,  # Exports BC_ONLY_FINGER_WEIGHT from the BC only finger weight setting
                "TEACHER_BC_WEIGHT": self.teacher_bc_weight,  # Exports TEACHER_BC_WEIGHT from the teacher BC weight setting
                "TEACHER_BC_ARM_WEIGHT": self.teacher_bc_arm_weight,  # Exports TEACHER_BC_ARM_WEIGHT from the teacher BC arm weight setting
                "TEACHER_BC_FINGER_WEIGHT": self.teacher_bc_finger_weight,  # Exports TEACHER_BC_FINGER_WEIGHT from the teacher BC finger weight setting
                "TEACHER_BC_DECAY_STEPS": self.teacher_bc_decay_steps,  # Exports TEACHER_BC_DECAY_STEPS from the teacher BC decay steps setting
                "ASSIST_NOISE_ARM": self.assist_noise_arm,  # Exports ASSIST_NOISE_ARM from the assist noise arm setting
                "ASSIST_NOISE_FINGER": self.assist_noise_finger,  # Exports ASSIST_NOISE_FINGER from the assist noise finger setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for name, raw in (  # iterates over configured values
            ("--policy-bc-relabel", int(self.policy_bc_relabel)),  # continues this config expression
            ("--policy-assist-mix", self.policy_assist_mix),  # Pairs trainer option --policy-assist-mix with the policy assist mix setting
            ("--policy-assist-mix-floor", self.policy_assist_mix_floor),  # Pairs trainer option --policy-assist-mix-floor with the policy assist mix floor setting
            ("--policy-assist-decay-steps", self.policy_assist_decay_steps),  # Pairs trainer option --policy-assist-decay-steps with the policy assist decay steps setting
            ("--policy-assist-decay-start-steps", self.policy_assist_decay_start_steps),  # Pairs trainer option --policy-assist-decay-start-steps with the policy assist decay start steps setting
            ("--bc-only-weight", self.bc_only_weight),  # Pairs trainer option --bc-only-weight with the BC only weight setting
            ("--bc-only-arm-weight", self.bc_only_arm_weight),  # Pairs trainer option --bc-only-arm-weight with the BC only arm weight setting
            ("--bc-only-finger-weight", self.bc_only_finger_weight),  # Pairs trainer option --bc-only-finger-weight with the BC only finger weight setting
            ("--teacher-bc-weight", self.teacher_bc_weight),  # Pairs trainer option --teacher-bc-weight with the teacher BC weight setting
            ("--teacher-bc-arm-weight", self.teacher_bc_arm_weight),  # Pairs trainer option --teacher-bc-arm-weight with the teacher BC arm weight setting
            ("--teacher-bc-finger-weight", self.teacher_bc_finger_weight),  # Pairs trainer option --teacher-bc-finger-weight with the teacher BC finger weight setting
            ("--teacher-bc-decay-steps", self.teacher_bc_decay_steps),  # Pairs trainer option --teacher-bc-decay-steps with the teacher BC decay steps setting
            ("--assist-noise-arm", self.assist_noise_arm),  # Pairs trainer option --assist-noise-arm with the assist noise arm setting
            ("--assist-noise-finger", self.assist_noise_finger),  # Pairs trainer option --assist-noise-finger with the assist noise finger setting
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class RlSwitchConfig:  # defines the RL switch config group
    """One in-process switch from DAgger/BC replay fill to TD3 refinement."""

    updates_per_step                 : int   = 50  # Sets learner update count per environment step
    n_step                           : int   = 3  # Sets the multi-step return horizon
    policy_delay                     : int   = 4  # Sets how often delayed policy updates run
    gamma                            : float = -1.0  # Sets the reward discount factor
    tau                              : float = 0.0005  # Sets the target network soft-update rate
    actor_lr                         : float = 2e-5  # Sets actor optimizer learning rate
    critic_lr                        : float = 5e-5  # Sets critic optimizer learning rate
    target_q_clip                    : float = 50  # Clips TD target Q values to bound critic targets
    critic_grad_clip                 : float = -1.0  # Clips critic gradients before optimizer updates
    actor_pre_tanh_l2                : float = -1.0  # Penalizes actor pre-tanh activation magnitude
    exploration_noise                : float = 0.02  # Sets arm action exploration noise during rollout
    exploration_noise_finger         : float = 0.05  # Sets finger action exploration noise during rollout
    policy_noise                     : float = 0.02  # Sets TD3 target policy smoothing noise for arm actions
    policy_noise_finger              : float = 0.05  # Sets TD3 target policy smoothing noise for finger actions
    noise_clip                       : float = 0.10  # Clips target policy smoothing noise magnitude
    policy_bc_relabel                : bool  = False  # Controls whether policy BC relabel is enabled
    teacher_bc_weight                : float = 0  # Sets optimization weight for teacher BC
    teacher_bc_arm_weight            : float = 2  # Sets optimization weight for teacher BC arm
    teacher_bc_finger_weight         : float = 0.3  # Sets optimization weight for teacher BC finger
    teacher_bc_decay_steps           : int   = 700_000  # Sets the number of steps for teacher BC decay
    actor_freeze_steps               : int   = 25_000  # Keeps actor updates frozen for the initial step window
    sync_targets_on_switch           : bool  = True  # Controls whether sync targets on switch is enabled
    reset_critic_optimizers_on_switch: bool  = True  # Controls whether reset critic optimizers on switch is enabled

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "RL_UPDATES_PER_STEP": self.updates_per_step,  # Exports RL_UPDATES_PER_STEP from the updates per step setting
                "RL_N_STEP": self.n_step,  # Exports RL_N_STEP from the n step setting
                "RL_POLICY_DELAY": self.policy_delay,  # Exports RL_POLICY_DELAY from the policy delay setting
                "RL_GAMMA": self.gamma,  # Exports RL_GAMMA from the gamma setting
                "RL_TAU": self.tau,  # Exports RL_TAU from the tau setting
                "RL_ACTOR_LR": self.actor_lr,  # Exports RL_ACTOR_LR from the actor learning rate setting
                "RL_CRITIC_LR": self.critic_lr,  # Exports RL_CRITIC_LR from the critic learning rate setting
                "RL_TARGET_Q_CLIP": self.target_q_clip,  # Exports RL_TARGET_Q_CLIP from the target Q clip setting
                "RL_CRITIC_GRAD_CLIP": self.critic_grad_clip,  # Exports RL_CRITIC_GRAD_CLIP from the critic grad clip setting
                "RL_ACTOR_PRE_TANH_L2": self.actor_pre_tanh_l2,  # Exports RL_ACTOR_PRE_TANH_L2 from the actor pre tanh l2 setting
                "RL_EXPLORATION_NOISE": self.exploration_noise,  # Exports RL_EXPLORATION_NOISE from the exploration noise setting
                "RL_EXPLORATION_NOISE_FINGER": self.exploration_noise_finger,  # Exports RL_EXPLORATION_NOISE_FINGER from the exploration noise finger setting
                "RL_POLICY_NOISE": self.policy_noise,  # Exports RL_POLICY_NOISE from the policy noise setting
                "RL_POLICY_NOISE_FINGER": self.policy_noise_finger,  # Exports RL_POLICY_NOISE_FINGER from the policy noise finger setting
                "RL_NOISE_CLIP": self.noise_clip,  # Exports RL_NOISE_CLIP from the noise clip setting
                "RL_POLICY_BC_RELABEL": bool01(self.policy_bc_relabel),  # Exports RL_POLICY_BC_RELABEL as legacy 0 or 1 from the policy BC relabel setting
                "RL_TEACHER_BC_WEIGHT": self.teacher_bc_weight,  # Exports RL_TEACHER_BC_WEIGHT from the teacher BC weight setting
                "RL_TEACHER_BC_ARM_WEIGHT": self.teacher_bc_arm_weight,  # Exports RL_TEACHER_BC_ARM_WEIGHT from the teacher BC arm weight setting
                "RL_TEACHER_BC_FINGER_WEIGHT": self.teacher_bc_finger_weight,  # Exports RL_TEACHER_BC_FINGER_WEIGHT from the teacher BC finger weight setting
                "RL_TEACHER_BC_DECAY_STEPS": self.teacher_bc_decay_steps,  # Exports RL_TEACHER_BC_DECAY_STEPS from the teacher BC decay steps setting
                "RL_ACTOR_FREEZE_STEPS": self.actor_freeze_steps,  # Exports RL_ACTOR_FREEZE_STEPS from the actor freeze steps setting
                "RL_SYNC_TARGETS_ON_SWITCH": bool01(self.sync_targets_on_switch),  # Exports RL_SYNC_TARGETS_ON_SWITCH as legacy 0 or 1 from the sync targets on switch setting
                "RL_RESET_CRITIC_OPTIMIZERS_ON_SWITCH": bool01(self.reset_critic_optimizers_on_switch),  # Exports RL_RESET_CRITIC_OPTIMIZERS_ON_SWITCH as legacy 0 or 1 from the reset critic optimizers on switch setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for name, raw in (  # iterates over configured values
            ("--rl-updates-per-step", self.updates_per_step),  # Pairs trainer option --rl-updates-per-step with the updates per step setting
            ("--rl-n-step", self.n_step),  # Pairs trainer option --rl-n-step with the n step setting
            ("--rl-policy-delay", self.policy_delay),  # Pairs trainer option --rl-policy-delay with the policy delay setting
            ("--rl-gamma", self.gamma),  # Pairs trainer option --rl-gamma with the gamma setting
            ("--rl-tau", self.tau),  # Pairs trainer option --rl-tau with the tau setting
            ("--rl-actor-lr", self.actor_lr),  # Pairs trainer option --rl-actor-lr with the actor learning rate setting
            ("--rl-critic-lr", self.critic_lr),  # Pairs trainer option --rl-critic-lr with the critic learning rate setting
            ("--rl-target-q-clip", self.target_q_clip),  # Pairs trainer option --rl-target-q-clip with the target Q clip setting
            ("--rl-critic-grad-clip", self.critic_grad_clip),  # Pairs trainer option --rl-critic-grad-clip with the critic grad clip setting
            ("--rl-actor-pre-tanh-l2", self.actor_pre_tanh_l2),  # Pairs trainer option --rl-actor-pre-tanh-l2 with the actor pre tanh l2 setting
            ("--rl-exploration-noise", self.exploration_noise),  # Pairs trainer option --rl-exploration-noise with the exploration noise setting
            ("--rl-exploration-noise-finger", self.exploration_noise_finger),  # Pairs trainer option --rl-exploration-noise-finger with the exploration noise finger setting
            ("--rl-policy-noise", self.policy_noise),  # Pairs trainer option --rl-policy-noise with the policy noise setting
            ("--rl-policy-noise-finger", self.policy_noise_finger),  # Pairs trainer option --rl-policy-noise-finger with the policy noise finger setting
            ("--rl-noise-clip", self.noise_clip),  # Pairs trainer option --rl-noise-clip with the noise clip setting
            ("--rl-policy-bc-relabel", int(self.policy_bc_relabel)),  # continues this config expression
            ("--rl-teacher-bc-weight", self.teacher_bc_weight),  # Pairs trainer option --rl-teacher-bc-weight with the teacher BC weight setting
            ("--rl-teacher-bc-arm-weight", self.teacher_bc_arm_weight),  # Pairs trainer option --rl-teacher-bc-arm-weight with the teacher BC arm weight setting
            ("--rl-teacher-bc-finger-weight", self.teacher_bc_finger_weight),  # Pairs trainer option --rl-teacher-bc-finger-weight with the teacher BC finger weight setting
            ("--rl-teacher-bc-decay-steps", self.teacher_bc_decay_steps),  # Pairs trainer option --rl-teacher-bc-decay-steps with the teacher BC decay steps setting
            ("--rl-actor-freeze-steps", self.actor_freeze_steps),  # Pairs trainer option --rl-actor-freeze-steps with the actor freeze steps setting
            ("--rl-sync-targets-on-switch", int(self.sync_targets_on_switch)),  # continues this config expression
            ("--rl-reset-critic-optimizers-on-switch", int(self.reset_critic_optimizers_on_switch)),  # continues this config expression
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class RlAssistHandoffConfig:  # defines the RL assist handoff config group
    """Teacher-assist schedule to use after the DAgger-to-RL switch."""

    policy_assist_mix              : float = -1.0  # Sets the policy assist mix config value
    policy_assist_mix_floor        : float = -1.0  # Sets the policy assist mix floor config value
    policy_assist_decay_steps      : int   = -1  # Sets the number of steps for policy assist decay
    policy_assist_decay_start_steps: int   = -1  # Sets the number of steps for policy assist decay start

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "RL_POLICY_ASSIST_MIX": self.policy_assist_mix,  # Exports RL_POLICY_ASSIST_MIX from the policy assist mix setting
                "RL_POLICY_ASSIST_MIX_FLOOR": self.policy_assist_mix_floor,  # Exports RL_POLICY_ASSIST_MIX_FLOOR from the policy assist mix floor setting
                "RL_POLICY_ASSIST_DECAY_STEPS": self.policy_assist_decay_steps,  # Exports RL_POLICY_ASSIST_DECAY_STEPS from the policy assist decay steps setting
                "RL_POLICY_ASSIST_DECAY_START_STEPS": self.policy_assist_decay_start_steps,  # Exports RL_POLICY_ASSIST_DECAY_START_STEPS from the policy assist decay start steps setting
            }  # closes the current expression
        )  # closes the current expression

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for name, raw in (  # iterates over configured values
            ("--rl-policy-assist-mix", self.policy_assist_mix),  # Pairs trainer option --rl-policy-assist-mix with the policy assist mix setting
            ("--rl-policy-assist-mix-floor", self.policy_assist_mix_floor),  # Pairs trainer option --rl-policy-assist-mix-floor with the policy assist mix floor setting
            ("--rl-policy-assist-decay-steps", self.policy_assist_decay_steps),  # Pairs trainer option --rl-policy-assist-decay-steps with the policy assist decay steps setting
            ("--rl-policy-assist-decay-start-steps", self.policy_assist_decay_start_steps),  # Pairs trainer option --rl-policy-assist-decay-start-steps with the policy assist decay start steps setting
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments


@dataclass(frozen=True)  # makes the following config group immutable
class DeterminismConfig:  # defines the determinism config group
    """Seed-adjacent switches needed to reproduce training runs."""

    torch_deterministic    : bool = True  # Controls whether torch deterministic is enabled
    torch_warn_only        : bool = True  # Controls whether torch warn only is enabled
    cudnn_benchmark        : bool = False  # Controls whether cudnn benchmark is enabled
    cudnn_deterministic    : bool = True  # Controls whether cudnn deterministic is enabled
    cublas_workspace_config: str  = ":4096:8"  # Sets the cublas workspace config value
    python_hash_seed       : int  = 7  # Sets the python hash seed config value
    numpy_seed             : bool = True  # Controls whether numpy seed is enabled

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "ENPM690_TORCH_DETERMINISTIC": bool01(self.torch_deterministic),  # Exports ENPM690_TORCH_DETERMINISTIC as legacy 0 or 1 from the torch deterministic setting
                "ENPM690_TORCH_DETERMINISTIC_WARN_ONLY": bool01(self.torch_warn_only),  # Exports ENPM690_TORCH_DETERMINISTIC_WARN_ONLY as legacy 0 or 1 from the torch warn only setting
                "ENPM690_CUDNN_BENCHMARK": bool01(self.cudnn_benchmark),  # Exports ENPM690_CUDNN_BENCHMARK as legacy 0 or 1 from the cudnn benchmark setting
                "ENPM690_CUDNN_DETERMINISTIC": bool01(self.cudnn_deterministic),  # Exports ENPM690_CUDNN_DETERMINISTIC as legacy 0 or 1 from the cudnn deterministic setting
                "CUBLAS_WORKSPACE_CONFIG": self.cublas_workspace_config,  # Exports CUBLAS_WORKSPACE_CONFIG from the cublas workspace config setting
                "PYTHONHASHSEED": self.python_hash_seed,  # Exports PYTHONHASHSEED from the python hash seed setting
                "ENPM690_NUMPY_SEED": bool01(self.numpy_seed),  # Exports ENPM690_NUMPY_SEED as legacy 0 or 1 from the numpy seed setting
            }  # closes the current expression
        )  # closes the current expression


@dataclass(frozen=True)  # makes the following config group immutable
class RuntimeConfig:  # defines the runtime config group
    """Process runtime and checkpoint cadence."""

    python                   : str | None = None  # Sets the python config value
    device                   : str        = "cuda:0"  # Sets the device config value
    headless                 : bool       = True  # Controls whether headless is enabled
    enable_cameras           : bool       = False  # Controls whether enable cameras is enabled
    disable_camera_perception: bool       = True  # Controls whether disable camera perception is enabled
    eval_every               : int        = 0  # Sets transition interval between inline eval runs; 0 uses trainer automatic cadence
    eval_steps               : int        = 600  # Sets the number of steps for eval
    eval_episodes            : int        = 1  # Sets the eval episodes config value
    checkpoint_every         : int        = 50_000  # Sets the checkpoint every filesystem path
    rolling_checkpoint_every : int        = 25_000  # Sets the rolling checkpoint every filesystem path
    rolling_checkpoint_keep  : int        = 20  # Sets the rolling checkpoint keep filesystem path
    log_every                : int        = 250  # Sets the log every config value
    sleep                    : float      = 0.0  # Sets the sleep config value

    def env(self) -> dict[str, str]:  # exports this config group as trainer environment variables
        """Return environment variables consumed by the standalone trainer for this config group."""
        return clean_dict(  # returns env vars after dropping unset values
            {  # opens a nested expression
                "ENPM690_PYTHON": self.python,  # Exports ENPM690_PYTHON from the python setting
                "DEVICE": self.device,  # Exports DEVICE from the device setting
                "HEADLESS": bool01(self.headless),  # Exports HEADLESS as legacy 0 or 1 from the headless setting
                "ENABLE_CAMERAS": bool01(self.enable_cameras),  # Exports ENABLE_CAMERAS as legacy 0 or 1 from the enable cameras setting
                "DISABLE_CAMERA_PERCEPTION": bool01(self.disable_camera_perception),  # Exports DISABLE_CAMERA_PERCEPTION as legacy 0 or 1 from the disable camera perception setting
                "EVAL_EVERY": self.eval_every,  # Exports EVAL_EVERY from the eval every setting
                "EVAL_STEPS": self.eval_steps,  # Exports EVAL_STEPS from the eval steps setting
                "EVAL_EPISODES": self.eval_episodes,  # Exports EVAL_EPISODES from the eval episodes setting
                "CHECKPOINT_EVERY": self.checkpoint_every,  # Exports CHECKPOINT_EVERY from the checkpoint every setting
                "ROLLING_CHECKPOINT_EVERY": self.rolling_checkpoint_every,  # Exports ROLLING_CHECKPOINT_EVERY from the rolling checkpoint every setting
                "ROLLING_CHECKPOINT_KEEP": self.rolling_checkpoint_keep,  # Exports ROLLING_CHECKPOINT_KEEP from the rolling checkpoint keep setting
                "LOG_EVERY": self.log_every,  # Exports LOG_EVERY from the log every setting
                "SLEEP": self.sleep,  # Exports SLEEP from the sleep setting
            }  # closes the current expression
        )  # closes the current expression

    def app_args(self) -> list[str]:  # exports this config group as Isaac app launcher arguments
        """Return Isaac application launcher arguments for this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        add_flag(args, self.headless, "--headless")  # adds the trainer CLI flag when enabled
        add_flag(args, self.enable_cameras, "--enable-cameras")  # adds the trainer CLI flag when enabled
        if self.device:  # Checks whether device
            add_arg(args, "--device", self.device)  # adds a scalar trainer CLI option
        return args  # returns assembled trainer CLI arguments

    def trainer_args(self) -> list[str]:  # exports this config group as trainer CLI arguments
        """Return command-line arguments that mirror this config group."""
        args: list[str] = []  # Collects trainer CLI arguments before return
        for name, raw in (  # iterates over configured values
            ("--eval-every", self.eval_every),  # Pairs trainer option --eval-every with the eval every setting
            ("--eval-steps", self.eval_steps),  # Pairs trainer option --eval-steps with the eval steps setting
            ("--eval-episodes", self.eval_episodes),  # Pairs trainer option --eval-episodes with the eval episodes setting
            ("--checkpoint-every", self.checkpoint_every),  # Pairs trainer option --checkpoint-every with the checkpoint every setting
            ("--rolling-checkpoint-every", self.rolling_checkpoint_every),  # Pairs trainer option --rolling-checkpoint-every with the rolling checkpoint every setting
            ("--rolling-checkpoint-keep", self.rolling_checkpoint_keep),  # Pairs trainer option --rolling-checkpoint-keep with the rolling checkpoint keep setting
            ("--log-every", self.log_every),  # Pairs trainer option --log-every with the log every setting
            ("--sleep", self.sleep),  # Pairs trainer option --sleep with the sleep setting
        ):  # closes the current expression
            add_arg(args, name, raw)  # adds a scalar trainer CLI option
        add_flag(args, self.disable_camera_perception, "--disable-camera-perception")  # adds the trainer CLI flag when enabled
        return args  # returns assembled trainer CLI arguments
