"""

Scalar training schedules with explicit inputs


clamp01:                       Clamp a scalar into the unit interval
linear_decay_scale:            Return a one-to-zero linear decay scale
policy_assist_schedule:        Return one scheduled teacher-assist weight
PolicyAssistScheduleConfig:    Teacher-assist schedule knobs for global arm and finger mixes
global_policy_assist_mix:      Return scheduled global teacher-assist mix
component_policy_assist_mix:   Return arm or finger teacher-assist mix with global fallback
teacher_bc_requested:          Return whether teacher BC is configured
scheduled_teacher_bc_weights:  Return decayed base arm and finger BC weights
"""

from __future__ import annotations

from dataclasses import dataclass


def clamp01(value: float) -> float:
    """Clamp a scalar into the unit interval"""
    return max(0.0, min(1.0, float(value)))


def linear_decay_scale(progress_step: int, decay_steps: int) -> float:
    """Return a one-to-zero linear decay scale"""
    if int(decay_steps) <= 0:
        return 1.0
    progress = max(0.0, float(progress_step)) / float(decay_steps)
    return max(0.0, 1.0 - progress)


def policy_assist_schedule(
    global_step: int,             # Param: current absolute training step
    *,
    start_steps      : int,  # Param: step count used for start steps
    peak             : float,  # Param: floating-point input for peak
    floor            : float,  # Param: floating-point input for floor
    decay_steps      : int,  # Param: step count used for decay steps
    decay_start_steps: int = -1,  # Param: step count used for decay start steps
) -> float:
    """Return one scheduled teacher-assist weight

    Steps:
    - Resolve inputs for `policy_assist_schedule` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    floor_value = clamp01(floor)
    peak_value = clamp01(peak)
    if peak_value <= floor_value:
        return floor_value
    if int(global_step) < int(start_steps):
        return 0.0
    if int(decay_steps) <= 0:
        return peak_value
    decay_start = int(start_steps) if int(decay_start_steps) < 0 else int(decay_start_steps)
    if int(global_step) < decay_start:
        return peak_value
    progress = max(0.0, min(1.0, (int(global_step) - decay_start) / float(decay_steps)))
    weight = peak_value + (floor_value - peak_value) * progress
    return max(floor_value, min(peak_value, weight))


@dataclass(frozen=True)
class PolicyAssistScheduleConfig:
    """Teacher-assist schedule knobs for global arm and finger mixes"""

    start_steps       : int  # step count used for start steps scheduling or reporting
    peak              : float  # floating-point peak value used by policy assist schedule config
    floor             : float  # floating-point floor value used by policy assist schedule config
    decay_steps       : int  # step count used for decay steps scheduling or reporting
    decay_start_steps : int   = -1  # step count used for decay start steps scheduling or reporting
    arm_peak          : float = -1.0  # floating-point arm peak value used by policy assist schedule config
    arm_floor         : float = -1.0  # floating-point arm floor value used by policy assist schedule config
    arm_decay_steps   : int   = -1  # step count used for arm decay steps scheduling or reporting
    finger_peak       : float = -1.0  # floating-point finger peak value used by policy assist schedule config
    finger_floor      : float = -1.0  # floating-point finger floor value used by policy assist schedule config
    finger_decay_steps: int   = -1  # step count used for finger decay steps scheduling or reporting


def global_policy_assist_mix(global_step: int, config: PolicyAssistScheduleConfig) -> float:
    """Return scheduled global teacher-assist mix"""
    return policy_assist_schedule(
        global_step,
        start_steps=config.start_steps,
        peak=config.peak,
        floor=config.floor,
        decay_steps=config.decay_steps,
        decay_start_steps=config.decay_start_steps,
    )


def component_policy_assist_mix(
    global_step: int,  # Param: current absolute training step
    component  : str,  # Param: string input for component
    config     : PolicyAssistScheduleConfig,  # Param: configuration object used by this helper
) -> float:
    """Return arm or finger teacher-assist mix with global fallback

    Steps:
    - Resolve inputs for `component_policy_assist_mix` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if component not in ("arm", "finger"):
        raise ValueError(f"unknown policy assist component: {component}")
    peak = getattr(config, f"{component}_peak")
    floor = getattr(config, f"{component}_floor")
    decay_steps = getattr(config, f"{component}_decay_steps")
    if peak < 0.0 and floor < 0.0 and decay_steps < 0:
        return global_policy_assist_mix(global_step, config)
    return policy_assist_schedule(
        global_step,
        start_steps=config.start_steps,
        peak=config.peak if peak < 0.0 else peak,
        floor=config.floor if floor < 0.0 else floor,
        decay_steps=config.decay_steps if decay_steps < 0 else decay_steps,
        decay_start_steps=config.decay_start_steps,
    )


def teacher_bc_requested(base_weight: float, arm_weight: float, finger_weight: float) -> bool:
    """Return whether teacher BC is configured"""
    return float(base_weight) > 0.0 or float(arm_weight) >= 0.0 or float(finger_weight) >= 0.0


def scheduled_teacher_bc_weights(
    *,
    progress_step: int,  # Param: step count used for progress step
    base_weight  : float,  # Param: weight applied to base
    arm_weight   : float,  # Param: weight applied to arm
    finger_weight: float,  # Param: weight applied to finger
    decay_steps  : int,  # Param: step count used for decay steps
) -> tuple[float, float, float]:
    """Return decayed base arm and finger BC weights

    Steps:
    - Resolve inputs for `scheduled_teacher_bc_weights` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    scale = linear_decay_scale(progress_step, decay_steps)
    base = max(0.0, float(base_weight)) * scale
    arm = float(arm_weight) * scale if float(arm_weight) >= 0.0 else -1.0
    finger = float(finger_weight) * scale if float(finger_weight) >= 0.0 else -1.0
    return base, arm, finger
