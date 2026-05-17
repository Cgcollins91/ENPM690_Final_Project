"""Reward terms for the topdown reach-align-contact curriculum.

Stage-routed via ``ensure_curriculum_stage_updated``. Each term reads the
current stage and returns the raw value scaled by an in-band indicator.
The RewardsCfg supplies the per-stage weight; the term itself only knows
which stage(s) it is active in.

Edge-triggered shell bonuses (reach_shell_bonus, align_shell_bonus,
light_contact_success_bonus) fire exactly once per episode on the rising
edge so V(stage_shell) stays bounded.

Zero default weights in ``cgc_topdown_curriculum_env_cfg.py`` are not a
dead-code marker. Named profiles and ad hoc search scripts override several
of those weights, especially lift shaping and centered-contact terms. Treat
deleting a reward function as a behavior change unless all config/profile/script
references have been checked.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from .state_machine import (
    ensure_curriculum_stage_updated,  # Idempotent per-env curriculum update once per sim step
    palm_distance,  # Grip-center 3D distance to reach/align hover target
    palm_height_error,  # Grip-center vertical error from reach/align hover target
    palm_distance_contact,  # Grip-center 3D distance to lower contact-pose target
    palm_height_error_contact,  # Grip-center vertical error from contact-pose target
    palm_drop_axis_error_rad,  # Palm-to-grip drop-axis angle from world -Z
    palm_yaw_axis_error_rad,  # Thumb/back-finger yaw error against grip face axis
    palm_spread_axis_error_rad,  # Index/middle spread error against block spread axis
    open_hand_alignment_error,  # Active fingertip distance sum to face targets
    fingertip_line_angle_rad,  # Thumb-to-back-finger pinch-line elevation angle
    opposite_face_gate,  # Smooth [0, 1] score for opposed-face fingertip placement
    any_hand_contact_strength,  # Max normalized contact across fingertips and palm
    any_fingertip_contact_strength,  # Max normalized contact across finger chains
    thumb_contact_strength,  # Normalized contact for thumb chain
    index_contact_strength,  # Normalized contact for index-finger chain
    palm_contact_strength,  # Normalized contact for palm sensor
    middle_contact_strength,  # Normalized contact for middle-finger chain
    opposed_contact_strength,  # Thumb-plus-opposed-back-finger pinch contact
    block_displacement,  # Full 3D block displacement from episode spawn position
    block_xy_displacement,  # Horizontal block drift from episode spawn position
    block_lift_height,  # Positive block height gain from episode spawn height
    block_z_velocity,  # Active block vertical velocity
    block_xy_velocity_norm,  # Active block planar speed magnitude
    block_angular_velocity_norm,  # Active block angular speed magnitude
    block_tilt_angle_rad,  # Active block local-Z tilt from world up
    lift_drop_from_max_bad,  # Failure gate for dropping too far from best lift height
    light_contact_success_now,  # Instantaneous Stage-2 light-contact success predicate
    light_contact_success_held,  # Debounced light-contact success after hold steps
    lift_success_now,  # Instantaneous lift success with contact, drift, and tilt gates
    centered_contact_errors,  # Max XY and Z residuals to centered face targets
    finger_unlock_center_errors,  # Pre-curl pocket center/max-XY/Z residuals
    stage_is,  # Float mask for exact requested curriculum stage
    stage_at_least,  # Float mask for at least requested curriculum stage
    stage2_warmup_factor,  # [0, 1] ramp after Stage-2 contact-pose latch
    _face_targets,  # Thumb/index opposed face targets near block top
    _three_finger_face_targets,  # Thumb/index/middle targets for three-finger centering
    _link_pos,  # Named robot link world-position lookup
    _THUMB_LINK,  # Rigid-body name for thumb link in reward geometry
    _INDEX_LINK,  # Rigid-body name for index link in reward geometry
    _MIDDLE_LINK,  # Rigid-body name for middle link in reward geometry
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --- Stage 0 (reach) — primary -----------------------------------------------


def reach_palm_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach palm distance."""
    return palm_distance(env) * stage_is(env, 0)


def reach_palm_height(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach palm height."""
    return palm_height_error(env) * stage_is(env, 0)


def reach_palm_orientation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach palm orientation."""
    return palm_drop_axis_error_rad(env) * stage_is(env, 0)


def reach_palm_yaw_axis(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach palm yaw axis."""
    return palm_yaw_axis_error_rad(env) * stage_is(env, 0)


def reach_palm_spread_axis(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach palm spread axis."""
    return palm_spread_axis_error_rad(env) * stage_is(env, 0)


def reach_alignment_error_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach alignment error quadratic."""
    err = torch.clamp(open_hand_alignment_error(env), min=0.0, max=1.0)
    return err * err * stage_is(env, 0)


def reach_fingertip_line_angle_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for reach fingertip line angle quadratic."""
    angle = torch.clamp(fingertip_line_angle_rad(env), min=0.0, max=math.pi)
    return angle * angle * stage_is(env, 0)


def reach_shell_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Edge-triggered: fires once per episode when stage advances 0 -> 1."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_reach_bonus_fired"):
        env._topdown_reach_bonus_fired = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_reach_bonus_fired[just_reset] = False
    just_advanced = (env._topdown_stage >= 1) & (~env._topdown_reach_bonus_fired)
    env._topdown_reach_bonus_fired = env._topdown_reach_bonus_fired | (env._topdown_stage >= 1)
    return just_advanced.float()


# --- Stage 1 (alignment) — primary + maintenance ------------------------------


def align_palm_distance_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align palm distance maintenance."""
    return palm_distance(env) * stage_is(env, 1)


def align_palm_height_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align palm height maintenance."""
    return palm_height_error(env) * stage_is(env, 1)


def align_palm_orientation_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align palm orientation maintenance."""
    return palm_drop_axis_error_rad(env) * stage_is(env, 1)


def align_palm_yaw_axis_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align palm yaw axis maintenance."""
    return palm_yaw_axis_error_rad(env) * stage_is(env, 1)


def align_alignment_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align alignment error."""
    return open_hand_alignment_error(env) * stage_is(env, 1)


def align_fingertip_line_angle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align fingertip line angle."""
    return fingertip_line_angle_rad(env) * stage_is(env, 1)


def align_alignment_error_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align alignment error quadratic."""
    err = torch.clamp(open_hand_alignment_error(env), min=0.0, max=1.0)
    return err * err * stage_is(env, 1)


def align_fingertip_line_angle_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align fingertip line angle quadratic."""
    angle = torch.clamp(fingertip_line_angle_rad(env), min=0.0, max=math.pi)
    return angle * angle * stage_is(env, 1)


def align_opposite_face(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the reward term for align opposite face."""
    return opposite_face_gate(env) * stage_is(env, 1)


def align_shell_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Edge-triggered: fires once per episode when stage advances 1 -> 2."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_align_bonus_fired"):
        env._topdown_align_bonus_fired = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_align_bonus_fired[just_reset] = False
    just_advanced = (env._topdown_stage >= 2) & (~env._topdown_align_bonus_fired)
    env._topdown_align_bonus_fired = env._topdown_align_bonus_fired | (env._topdown_stage >= 2)
    return just_advanced.float()


# --- Stage 2 (light contact) — primary + maintenance --------------------------


def contact_palm_distance_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 palm-distance maintenance reward term."""
    return palm_distance_contact(env) * stage_is(env, 2)


def contact_palm_height_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 palm-height maintenance reward term."""
    return palm_height_error(env) * stage_is(env, 2)


def contact_palm_orientation_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 palm-orientation maintenance reward term."""
    return palm_drop_axis_error_rad(env) * stage_is(env, 2)


def contact_palm_yaw_axis_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 palm-yaw maintenance reward term."""
    return palm_yaw_axis_error_rad(env) * stage_is(env, 2)


def contact_alignment_error_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 open-hand alignment maintenance reward term."""
    return open_hand_alignment_error(env) * stage_is(env, 2)


def contact_alignment_error_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return a stage-2 quadratic penalty for losing the opposed-face pocket."""
    err = torch.clamp(open_hand_alignment_error(env), min=0.0, max=1.0)
    return err * err * stage_is(env, 2)


def alignment_degradation_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize step-to-step degradation of the opposed-face alignment pocket.

    Absolute alignment penalties tell the critic which states are bad, but they
    do not explicitly punish the failure mode we keep seeing: the actor reaches
    a usable pocket, then quickly walks away from it. This stateful term charges
    only positive increases in open-hand alignment error after reach stage.
    """
    ensure_curriculum_stage_updated(env)
    err = torch.clamp(open_hand_alignment_error(env), min=0.0, max=1.0)
    prev = getattr(env, "_topdown_prev_align_error_reward", None)
    if not torch.is_tensor(prev) or prev.shape != err.shape:
        prev = err.detach().clone()

    just_reset = env.episode_length_buf <= 1
    active = stage_at_least(env, 1)
    degraded = torch.clamp(err - prev, min=0.0, max=0.20) * active
    degraded = torch.where(just_reset, torch.zeros_like(degraded), degraded)
    env._topdown_prev_align_error_reward = err.detach().clone()
    return degraded


def contact_fingertip_line_angle_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 fingertip-line angle maintenance reward term."""
    return fingertip_line_angle_rad(env) * stage_is(env, 2)


def contact_opposite_face_maintenance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the stage-2 opposed-face maintenance reward term."""
    return opposite_face_gate(env) * stage_is(env, 2)


def contact_target_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Sum of fingertip distances to opposed face/edge targets."""
    thumb_pos = _link_pos(env, _THUMB_LINK)
    index_pos = _link_pos(env, _INDEX_LINK)
    use_three = _env_bool("CURRICULUM_THREE_FINGER_CENTERING", False)
    if use_three:
        middle_pos = _link_pos(env, _MIDDLE_LINK)
        thumb_target, index_target, middle_target = _three_finger_face_targets(env)
    else:
        middle_pos = None
        thumb_target, index_target = _face_targets(env)
        middle_target = None
    thumb_d = torch.linalg.norm(thumb_pos - thumb_target, dim=-1)
    index_d = torch.linalg.norm(index_pos - index_target, dim=-1)
    total = thumb_d + index_d
    if middle_pos is not None and middle_target is not None:
        total = total + torch.linalg.norm(middle_pos - middle_target, dim=-1)
    return torch.clamp(total, max=(0.36 if use_three else 0.24)) * stage_is(env, 2)


def contact_vertical_gap(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize finger height above selected opposed face/edge targets."""
    thumb_pos = _link_pos(env, _THUMB_LINK)
    index_pos = _link_pos(env, _INDEX_LINK)
    if _env_bool("CURRICULUM_THREE_FINGER_CENTERING", False):
        middle_pos = _link_pos(env, _MIDDLE_LINK)
        thumb_target, index_target, middle_target = _three_finger_face_targets(env)
    else:
        middle_pos = None
        thumb_target, index_target = _face_targets(env)
        middle_target = None
    thumb_gap = torch.clamp(thumb_pos[:, 2] - thumb_target[:, 2], min=0.0)
    index_gap = torch.clamp(index_pos[:, 2] - index_target[:, 2], min=0.0)
    total = thumb_gap + index_gap
    if middle_pos is not None and middle_target is not None:
        total = total + torch.clamp(middle_pos[:, 2] - middle_target[:, 2], min=0.0)
    return total * stage_is(env, 2)


def _smoothstep_in_band(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth indicator: 1 when x <= lo, 0 when x >= hi, smoothstep between.

    Used to gate Stage 2 contact bonuses on shell membership so the policy can't
    park outside the shell collecting unconditional contact reward (the
    "raptor-claw" / "tap from afar" local minima that broke iter 01 and iter 02).
    """
    span = max(hi - lo, 1.0e-6)
    t = torch.clamp((hi - x) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _env_float(name: str, default: float) -> float:
    """Read a float reward override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean reward override from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _finger_unlock_ready_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Gate contact/no-contact shaping until fingers are actually allowed to act."""
    ensure_curriculum_stage_updated(env)
    progress = getattr(env, "_topdown_finger_unlock_progress", None)
    if progress is None:
        return torch.zeros(env.num_envs, device=env.device)
    unlock_min = _env_float("CURRICULUM_NO_CONTACT_PENALTY_UNLOCK_MIN", 0.95)
    return (progress >= unlock_min).to(dtype=progress.dtype)


def contact_thumb_contact_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Stage-2 bridge reward for thumb contact, gated on contact-pose palm shell.

    Palm shell only (not drop) so the bridge still rewards unilateral
    exploration during the early Stage 2 window when the policy is learning
    bilateral closure, but no longer rewards taps from a retreated wrist pose.
    """
    palm_gate = _smoothstep_in_band(palm_distance_contact(env), lo=0.08, hi=0.14)
    return thumb_contact_strength(env) * palm_gate * stage_is(env, 2)


def contact_index_contact_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Stage-2 bridge reward for index contact, gated on contact-pose palm shell."""
    palm_gate = _smoothstep_in_band(palm_distance_contact(env), lo=0.08, hi=0.14)
    return index_contact_strength(env) * palm_gate * stage_is(env, 2)


def contact_opposed_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Opposed-contact bonus, gated on the strict-shell pose (palm AND drop).

    Only pays inside the success-shell pose envelope: palm distance close (≤ 0.14
    with full credit ≤ 0.08) AND drop axis near vertical (≤ 35° with full credit
    ≤ 25°). Smoothstep ramps avoid a hard cliff at the boundary that could
    cause hesitation. Without this gate the unconditional +6.0/step at any
    opposed contact created a raptor-claw local min where the wrist tilts to
    44° while fingers contact opposite faces — which is exactly the failure
    mode that motivated this curriculum's overnight refinement loop.
    """
    drop_rad = palm_drop_axis_error_rad(env)
    drop_deg = drop_rad * (180.0 / 3.141592653589793)
    drop_gate = _smoothstep_in_band(drop_deg, lo=25.0, hi=35.0)
    palm_gate = _smoothstep_in_band(palm_distance_contact(env), lo=0.08, hi=0.14)
    return opposed_contact_strength(env) * drop_gate * palm_gate * stage_is(env, 2)


def _active_back_contact_strength(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the selected back-finger contact strength for the active profile."""
    back = index_contact_strength(env)
    if _env_bool("TOPDOWN_OPPOSED_CONTACT_USE_MIDDLE_BACK", False):
        back = torch.maximum(back, middle_contact_strength(env))
    return back


def _preunlock_pocket_geometry_quality(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return raw open-hand pocket quality before contact/unlock gating."""
    center_xy, max_xy, center_z = finger_unlock_center_errors(env)
    angle_deg = torch.rad2deg(fingertip_line_angle_rad(env))
    align_err = open_hand_alignment_error(env)

    center_full = _env_float("CURRICULUM_FINGER_CENTER_TIP_XY_MAX", 0.025)
    max_xy_full = _env_float("CURRICULUM_FINGER_CENTER_MAX_TIP_XY_MAX", 0.080)
    z_full = _env_float("CURRICULUM_FINGER_CENTER_TIP_Z_MAX", 0.075)
    angle_full = _env_float("CURRICULUM_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG", 15.0)
    align_full = _env_float("CURRICULUM_FINGER_CENTER_ALIGN_ERR_MAX", 0.20)

    center_zero = max(
        _env_float("CURRICULUM_PREUNLOCK_POCKET_CENTER_XY_ZERO", center_full * 2.5),
        center_full + 1.0e-6,
    )
    max_xy_zero = max(
        _env_float("CURRICULUM_PREUNLOCK_POCKET_MAX_XY_ZERO", max_xy_full * 2.0),
        max_xy_full + 1.0e-6,
    )
    z_zero = max(
        _env_float("CURRICULUM_PREUNLOCK_POCKET_Z_ZERO", z_full * 2.0),
        z_full + 1.0e-6,
    )
    angle_zero = max(
        _env_float("CURRICULUM_PREUNLOCK_POCKET_ANGLE_ZERO_DEG", angle_full * 2.0),
        angle_full + 1.0e-6,
    )

    quality = (
        _smoothstep_in_band(center_xy, lo=center_full, hi=center_zero)
        * _smoothstep_in_band(max_xy, lo=max_xy_full, hi=max_xy_zero)
        * _smoothstep_in_band(center_z, lo=z_full, hi=z_zero)
        * _smoothstep_in_band(angle_deg, lo=angle_full, hi=angle_zero)
    )
    if align_full > 0.0:
        align_zero = max(
            _env_float("CURRICULUM_PREUNLOCK_POCKET_ALIGN_ZERO", align_full * 2.0),
            align_full + 1.0e-6,
        )
        quality = quality * _smoothstep_in_band(align_err, lo=align_full, hi=align_zero)
    return quality


def contact_preunlock_pocket_quality(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward the pre-curl pocket that makes the finger-unlock gate reachable.

    The v32 teacher can enter Stage 2 and then dwell with ``unlock=0`` while
    alternating one-sided contacts. This term gives RL a dense gradient toward
    the exact open-hand centering contract that unlocks closure, then fades out
    once fingertips touch or the unlock progress starts paying.
    """
    quality = _preunlock_pocket_geometry_quality(env)
    progress = getattr(env, "_topdown_finger_unlock_progress", None)
    if torch.is_tensor(progress) and progress.shape == quality.shape:
        preunlock_gate = 1.0 - progress.clamp(0.0, 1.0)
    else:
        preunlock_gate = torch.ones_like(quality)
    no_fingertip_contact = 1.0 - any_fingertip_contact_strength(env).clamp(0.0, 1.0)
    return quality * preunlock_gate * no_fingertip_contact * stage_is(env, 2)


def contact_preunlock_no_contact_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize Stage-2 no-contact dwell that is not in the unlock pocket."""
    progress = getattr(env, "_topdown_finger_unlock_progress", None)
    if torch.is_tensor(progress):
        preunlock_gate = 1.0 - progress.clamp(0.0, 1.0)
    else:
        preunlock_gate = torch.ones(env.num_envs, device=env.device)
    no_fingertip_contact = 1.0 - any_fingertip_contact_strength(env).clamp(0.0, 1.0)
    pocket_quality = _preunlock_pocket_geometry_quality(env).clamp(0.0, 1.0)
    return preunlock_gate * no_fingertip_contact * (1.0 - pocket_quality) * stage_is(env, 2)


def contact_bilateral_contact_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward actual opposed bilateral contact without the strict palm/drop shell."""
    return opposed_contact_strength(env) * stage_is(env, 2)


def contact_bilateral_imbalance_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize thumb/back-finger imbalance using the active back-finger choice."""
    thumb = thumb_contact_strength(env)
    back = _active_back_contact_strength(env)
    imbalance = torch.abs(thumb - back)
    return imbalance * imbalance * stage_is(env, 2)


def contact_one_sided_flip_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Charge a one-step cost when one-sided contact switches sides in Stage 2."""
    ensure_curriculum_stage_updated(env)
    thumb = thumb_contact_strength(env)
    back = _active_back_contact_strength(env)
    margin = max(_env_float("CURRICULUM_ONE_SIDED_FLIP_MARGIN", 0.10), 0.0)
    diff = thumb - back
    sign = torch.zeros_like(diff)
    sign = torch.where(diff > margin, torch.ones_like(sign), sign)
    sign = torch.where(diff < -margin, -torch.ones_like(sign), sign)

    prev = getattr(env, "_topdown_prev_one_sided_contact_sign", None)
    if not torch.is_tensor(prev) or prev.shape != sign.shape:
        prev = torch.zeros_like(sign)

    stage_gate = stage_is(env, 2)
    active = stage_gate > 0.0
    just_reset = env.episode_length_buf <= 1
    flip = (prev * sign < 0.0) & (sign != 0.0) & active & (~just_reset)
    next_prev = torch.where(
        just_reset | (~active),
        torch.zeros_like(sign),
        torch.where(sign != 0.0, sign, prev),
    )
    env._topdown_prev_one_sided_contact_sign = next_prev.detach().clone()
    contact_mag = torch.maximum(thumb, back).clamp(0.0, 1.0)
    return flip.float() * contact_mag * stage_gate


def contact_deep_shell_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Smooth multiplicative bonus inside the strict-shell core, gated on contact.

    Iter 03 (shell-gated opposed bonus) prevented the off-shell retreat but
    left the policy parked at the gate boundary with no gradient pulling
    it deeper into the strict success shell. This term installs an explicit
    descent direction in (palm, drop, align) jointly: full credit at the
    success shell core (palm ≤ 0.05, drop ≤ 15°, align ≤ 0.08), zero outside.

    Iter 04 update: also multiply by `opposed_contact_strength` to require
    actual finger-on-face contact for the bonus to fire. Without this, the
    policy was rewarded for being in shell pose without contacting (Q-landscape
    distortion via the phantom carrot, even though the bonus rarely fired
    explicitly). With contact gating the bonus pays only on the actual
    success-track behavior: in-shell + pressing.
    """
    palm = palm_distance_contact(env)
    drop_deg = palm_drop_axis_error_rad(env) * (180.0 / 3.141592653589793)
    align = open_hand_alignment_error(env)
    palm_term = _smoothstep_in_band(palm, lo=0.05, hi=0.08)
    drop_term = _smoothstep_in_band(drop_deg, lo=15.0, hi=25.0)
    align_term = _smoothstep_in_band(align, lo=0.08, hi=0.14)
    contact_term = opposed_contact_strength(env)
    return palm_term * drop_term * align_term * contact_term * stage_is(env, 2)


def stage2_floor_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Small always-on positive signal for staying in the contact/lift stage."""
    return stage_is(env, 2)


def contact_one_sided_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize single-finger contacts whenever Stage 2 is active.

    Earlier iters multiplied this by stage2_warmup_factor so the early
    Stage-2 window paid no cost; the empirical result was that the policy
    locked into an index-only contact mode before the ramp completed and
    then never escaped (its hover-with-one-finger return was net positive).
    Removing the warmup gate makes one-sided contact net-negative from the
    moment the policy enters Stage 2.
    """
    thumb = thumb_contact_strength(env)
    index = index_contact_strength(env)
    one_sided = torch.clamp(torch.maximum(thumb, index) - torch.minimum(thumb, index), min=0.0)
    return one_sided * one_sided * stage_is(env, 2)


def contact_overforce_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize dirty block motion.

    During lift training vertical block motion is the task, so only horizontal
    drift is overforce. Non-lift contact training keeps the historical total
    displacement penalty.
    """
    disp = block_xy_displacement(env) if os.environ.get("TOPDOWN_LIFT_TASK", "0") == "1" else block_displacement(env)
    return disp * stage_is(env, 2)


def contact_lift_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward clean vertical lift after opposed thumb/index contact is established."""
    target = max(_env_float("CURRICULUM_LIFT_PROGRESS_HEIGHT", 0.05), 1.0e-6)
    lift = (block_lift_height(env) / target).clamp(0.0, 1.0)
    drift_default = _env_float("TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX", 0.045)
    drift_max = max(_env_float("CURRICULUM_LIFT_PROGRESS_XY_DRIFT_MAX", drift_default), 1.0e-6)
    drift = (block_xy_displacement(env) / drift_max).clamp(0.0, 1.0)
    drift_factor = (1.0 - drift) * (1.0 - drift)
    return lift * drift_factor * opposed_contact_strength(env) * stage_is(env, 2)


def _lift_task_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return a mask that disables lift-only terms outside lift tasks."""
    if os.environ.get("TOPDOWN_LIFT_TASK", "0") != "1":
        return torch.zeros(env.num_envs, device=env.device)
    return stage_is(env, 2)


def _lift_height_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return positive incremental block-height progress for lift shaping."""
    target = max(_env_float("CURRICULUM_LIFT_HEIGHT_PROGRESS_TARGET", 0.10), 1.0e-6)
    return (block_lift_height(env) / target).clamp(0.0, 1.0)


def _lift_penalty_height_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Delay lift-quality penalties until the block has a small exploration band."""
    start = _env_float("CURRICULUM_LIFT_PENALTY_HEIGHT_START", 0.0)
    if start <= 0.0:
        return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    ramp = max(_env_float("CURRICULUM_LIFT_PENALTY_HEIGHT_RAMP", 0.02), 1.0e-6)
    return ((block_lift_height(env) - start) / ramp).clamp(0.0, 1.0)


def lift_height_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vertical progress for lift mode, optionally gated by opposed grip."""
    progress = _lift_height_progress(env) * _lift_task_gate(env)
    if _env_bool("CURRICULUM_LIFT_HEIGHT_PROGRESS_REQUIRES_GRIP", True):
        contact_min = max(_env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30), 1.0e-6)
        grip_gate = (opposed_contact_strength(env) / contact_min).clamp(0.0, 1.0)
        progress = progress * grip_gate
    return progress


def lift_with_grip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vertical lift progress that only pays while an opposed pinch is present."""
    return (
        _lift_height_progress(env)
        * opposed_contact_strength(env)
        * _lift_task_gate(env)
    )


def vertical_lift_velocity_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward active upward motion during a clean opposed lift."""
    contact_min = _env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
    contact_gate = (opposed_contact_strength(env) >= contact_min).to(dtype=torch.float32)
    z_vel = torch.clamp(block_z_velocity(env), min=0.0, max=0.5)
    return z_vel * contact_gate * _lift_task_gate(env)


def block_xy_velocity_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the act of shoving the block sideways during lift mode."""
    soft_cap = max(_env_float("CURRICULUM_BLOCK_XY_VEL_SOFT_CAP", 0.5), 1.0e-6)
    xy_vel = soft_cap * torch.tanh(block_xy_velocity_norm(env) / soft_cap)
    return xy_vel * _lift_task_gate(env)


def block_angular_velocity_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize rotating the block while developing or holding a lift."""
    height_min = _env_float("CURRICULUM_BLOCK_ANG_VEL_HEIGHT_MIN", 0.005)
    soft_cap = max(_env_float("CURRICULUM_BLOCK_ANG_VEL_SOFT_CAP", 5.0), 1.0e-6)
    lift_gate = (block_lift_height(env) >= height_min).to(dtype=torch.float32)
    ang_vel = soft_cap * torch.tanh(block_angular_velocity_norm(env) / soft_cap)
    return ang_vel * lift_gate * _lift_task_gate(env)


def block_upright_lift_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward lifted opposed grasps that keep the cube upright instead of tilted."""
    tilt_deg = torch.rad2deg(block_tilt_angle_rad(env))
    full = _env_float("CURRICULUM_BLOCK_UPRIGHT_FULL_DEG", 5.0)
    zero = _env_float("CURRICULUM_BLOCK_UPRIGHT_ZERO_DEG", 25.0)
    upright_quality = _smoothstep_in_band(tilt_deg, lo=full, hi=max(zero, full + 1.0e-6))
    contact_min = _env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
    contact_gate = (opposed_contact_strength(env) >= contact_min).to(dtype=torch.float32)
    return _lift_height_progress(env) * upright_quality * contact_gate * _lift_task_gate(env)


def centered_upright_lift_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward the pyramid-ready condition: centered opposed contact plus upright lift."""
    tilt_deg = torch.rad2deg(block_tilt_angle_rad(env))
    full = _env_float("CURRICULUM_BLOCK_UPRIGHT_FULL_DEG", 5.0)
    zero = _env_float("CURRICULUM_BLOCK_UPRIGHT_ZERO_DEG", 25.0)
    upright_quality = _smoothstep_in_band(tilt_deg, lo=full, hi=max(zero, full + 1.0e-6))
    return (
        _lift_height_progress(env)
        * contact_centered_contact_continuous(env)
        * upright_quality
        * _lift_task_gate(env)
    )


def centered_lift_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vertical lift progress that only pays through centered opposed contact."""
    return _lift_height_progress(env) * contact_centered_contact_continuous(env) * _lift_task_gate(env)


def lift_xy_drift_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize cumulative sideways block displacement once the block is lifted."""
    drift_default = _env_float("TOPDOWN_LIFT_SUCCESS_XY_DRIFT_MAX", 0.04)
    drift_free = max(_env_float("CURRICULUM_LIFT_XY_DRIFT_PENALTY_FREE", drift_default), 1.0e-6)
    drift_zero = max(
        _env_float("CURRICULUM_LIFT_XY_DRIFT_PENALTY_ZERO", 0.12),
        drift_free + 1.0e-6,
    )
    drift = block_xy_displacement(env)
    drift_badness = torch.clamp((drift - drift_free) / (drift_zero - drift_free), 0.0, 1.0)
    return _lift_height_progress(env) * _lift_penalty_height_gate(env) * drift_badness * _lift_task_gate(env)


def block_tilt_lift_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize tilted carried-block states, not only angular velocity."""
    tilt_deg = torch.rad2deg(block_tilt_angle_rad(env))
    full = _env_float("CURRICULUM_BLOCK_UPRIGHT_FULL_DEG", 5.0)
    zero = _env_float("CURRICULUM_BLOCK_UPRIGHT_ZERO_DEG", 25.0)
    tilt_badness = torch.clamp((tilt_deg - full) / max(zero - full, 1.0e-6), 0.0, 1.0)
    return _lift_height_progress(env) * _lift_penalty_height_gate(env) * tilt_badness * _lift_task_gate(env)


def uncentered_lift_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize lifting while the thumb/index contact geometry is off-center."""
    center_quality = contact_centered_contact_continuous(env).clamp(0.0, 1.0)
    return _lift_height_progress(env) * _lift_penalty_height_gate(env) * (1.0 - center_quality) * _lift_task_gate(env)


def block_off_table_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot milestone when the block is lifted off the table."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_block_off_table_bonus_fired"):
        env._topdown_block_off_table_bonus_fired = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_block_off_table_bonus_fired[just_reset] = False
    contact_min = _env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
    height_min = _env_float("CURRICULUM_BLOCK_OFF_TABLE_HEIGHT", 0.05)
    clean_now = (
        (block_lift_height(env) >= height_min)
        & (opposed_contact_strength(env) >= contact_min)
        & (_lift_task_gate(env) > 0.0)
    )
    just_fired = clean_now & (~env._topdown_block_off_table_bonus_fired)
    env._topdown_block_off_table_bonus_fired = env._topdown_block_off_table_bonus_fired | clean_now
    return just_fired.float()


def sustained_lift_grip_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Continuous hold reward for maintaining opposed grip while lifted."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_sustained_lift_grip_hold"):
        env._topdown_sustained_lift_grip_hold = torch.zeros(n, dtype=torch.long, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_sustained_lift_grip_hold[just_reset] = 0
    contact_min = _env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
    height_min = _env_float("CURRICULUM_SUSTAINED_LIFT_HEIGHT", 0.02)
    holding_now = (
        (block_lift_height(env) >= height_min)
        & (opposed_contact_strength(env) >= contact_min)
        & (_lift_task_gate(env) > 0.0)
    )
    env._topdown_sustained_lift_grip_hold = torch.where(
        holding_now,
        env._topdown_sustained_lift_grip_hold + 1,
        torch.zeros_like(env._topdown_sustained_lift_grip_hold),
    )
    hold = env._topdown_sustained_lift_grip_hold.to(dtype=torch.float32)
    return torch.tanh(hold / 20.0) * _lift_task_gate(env)


def block_drop_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot penalty after a lifted block is dropped or loses peak height."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_lift_reward_been_high"):
        env._topdown_lift_reward_been_high = torch.zeros(n, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_topdown_block_drop_penalty_fired"):
        env._topdown_block_drop_penalty_fired = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_lift_reward_been_high[just_reset] = False
        env._topdown_block_drop_penalty_fired[just_reset] = False

    contact_min = _env_float("TOPDOWN_LIFT_SUCCESS_CONTACT_MIN", 0.30)
    high_min = _env_float("CURRICULUM_BLOCK_OFF_TABLE_HEIGHT", 0.05)
    drop_height = _env_float("CURRICULUM_BLOCK_DROP_HEIGHT", 0.01)
    drop_contact = _env_float("CURRICULUM_BLOCK_DROP_CONTACT_MAX", 0.10)
    clean_high = (
        (block_lift_height(env) >= high_min)
        & (opposed_contact_strength(env) >= contact_min)
        & (_lift_task_gate(env) > 0.0)
    )
    env._topdown_lift_reward_been_high = env._topdown_lift_reward_been_high | clean_high
    dropped = (
        env._topdown_lift_reward_been_high
        & (block_lift_height(env) < drop_height)
        & (opposed_contact_strength(env) < drop_contact)
        & (_lift_task_gate(env) > 0.0)
    )
    dropped = dropped | lift_drop_from_max_bad(env)
    just_fired = dropped & (~env._topdown_block_drop_penalty_fired)
    env._topdown_block_drop_penalty_fired = env._topdown_block_drop_penalty_fired | just_fired
    return just_fired.float()


def contact_pose_ready_no_contact_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cost the policy for sitting in the contact-pose shell without contacting.

    Reward diagnostics identified a hover plateau: pose_ready frames without
    contact pay a near-zero per-step return (~-0.6) versus +39 for both-finger
    contact, but the gradient between them is weak because Stage-2 maintenance
    terms peak in the hover state. This term adds an explicit per-step cost
    proportional to ``(1 - any_contact_strength)`` once the contact-pose latch
    has been held long enough (>= 50 steps, after fingers are fully unlocked)
    to reasonably expect contact. Active only when ``_topdown_contact_pose_ready``
    is set, so it does not punish envs that are still descending into the shell.
    """
    ensure_curriculum_stage_updated(env)
    if not hasattr(env, "_topdown_contact_pose_ready") or not hasattr(env, "_topdown_contact_pose_age"):
        return torch.zeros(env.num_envs, device=env.device)
    ready = env._topdown_contact_pose_ready.float()
    age_factor = (env._topdown_contact_pose_age.float() / 50.0).clamp(0.0, 1.0)
    contact = any_hand_contact_strength(env)
    unlock_gate = _finger_unlock_ready_gate(env)
    return ready * unlock_gate * age_factor * (1.0 - contact) * stage_is(env, 2)


def contact_smooth_pose_no_contact_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty in the SOFT pose neighborhood when there is no contact.

    Reward diagnostics identified the binding failure: actor parks
    at palm_c=0.085 -- exactly 5mm outside the strict success shell at palm_c<=0.08.
    The existing ``contact_pose_ready_no_contact_penalty`` is gated on the strict
    latch and therefore CANNOT fire at this hover location. This term replaces
    that gate with the same soft pose smoothstep used by ``contact_smooth_success
    _pose_continuous`` so the penalty fires anywhere in the soft pose
    neighborhood that lacks contact. Net effect at palm_c=0.085, no contact:
    pose_factor ~= 0.4, so reward = -0.4 * weight = strong negative; with
    contact, the (1-contact) factor zeros it. Breaks the boundary equilibrium
    by making "hover near the shell without contact" strictly costly.
    """
    pose = _smooth_success_pose_factor(env)
    no_contact = 1.0 - any_hand_contact_strength(env)
    return pose * _finger_unlock_ready_gate(env) * no_contact * stage_is(env, 2)


def _finger_center_pair_delta_xy(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return fingertip center error components in the tabletop plane."""
    thumb_pos = _link_pos(env, _THUMB_LINK)
    index_pos = _link_pos(env, _INDEX_LINK)
    middle_pos = _link_pos(env, _MIDDLE_LINK)
    if _env_bool("CURRICULUM_THREE_FINGER_CENTERING", False):
        thumb_target, index_target, middle_target = _three_finger_face_targets(env)
    else:
        middle_pos = None
        thumb_target, index_target = _face_targets(env)
        middle_target = None
    thumb_delta = thumb_target - thumb_pos
    index_delta = index_target - index_pos
    if middle_pos is not None and middle_target is not None:
        middle_delta = middle_target - middle_pos
        return (thumb_delta[:, :2] + index_delta[:, :2] + middle_delta[:, :2]) / 3.0
    return 0.5 * (thumb_delta[:, :2] + index_delta[:, :2])


def _finger_center_error_scale() -> float:
    """Return the scale used to normalize finger-center errors."""
    return max(
        _env_float(
            "CURRICULUM_CONTACT_FINGER_CENTER_ERR_SCALE",
            _env_float("CURRICULUM_FINGER_CENTER_TIP_XY_MAX", 0.025),
        ),
        1.0e-6,
    )


def _finger_center_error_norm_cap() -> float:
    """Return the cap applied to normalized finger-center penalties."""
    return max(_env_float("CURRICULUM_CONTACT_FINGER_CENTER_ERR_NORM_CAP", 3.0), 0.0)


def contact_finger_center_x_error_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the contact-stage x-centering penalty for opposing fingertips."""
    delta_xy = _finger_center_pair_delta_xy(env)
    norm_err = (torch.abs(delta_xy[:, 0]) / _finger_center_error_scale()).clamp(
        max=_finger_center_error_norm_cap()
    )
    return norm_err * norm_err * stage_is(env, 2)


def contact_finger_center_y_error_quadratic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the contact-stage y-centering penalty for opposing fingertips."""
    delta_xy = _finger_center_pair_delta_xy(env)
    norm_err = (torch.abs(delta_xy[:, 1]) / _finger_center_error_scale()).clamp(
        max=_finger_center_error_norm_cap()
    )
    return norm_err * norm_err * stage_is(env, 2)


def contact_centered_contact_continuous(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Dense reward for the final centered-contact shell.

    Success requires opposed contact plus centered fingertip geometry. The older
    smooth-success terms reward palm pose and contact, but not the final
    per-fingertip centered shell directly. This term fills that gap without
    paying for hover states: it is zero until thumb and index have opposed
    contact, then ramps up as max fingertip XY/Z error and line angle enter the
    same bands used by the unlock/success gates.
    """
    xy_err, z_err = centered_contact_errors(env)
    angle_deg = torch.rad2deg(fingertip_line_angle_rad(env))

    xy_full = _env_float(
        "CURRICULUM_CENTERED_CONTACT_XY_FULL",
        _env_float("CURRICULUM_SUCCESS_CENTER_TIP_XY_MAX", 0.015),
    )
    xy_zero = _env_float(
        "CURRICULUM_CENTERED_CONTACT_XY_ZERO",
        _env_float("CURRICULUM_FINGER_CENTER_MAX_TIP_XY_MAX", 0.055),
    )
    z_full = _env_float(
        "CURRICULUM_CENTERED_CONTACT_Z_FULL",
        _env_float("TOPDOWN_CONTACT_TEACHER_FINGER_GEOM_Z_DONE", 0.010),
    )
    z_zero = _env_float(
        "CURRICULUM_CENTERED_CONTACT_Z_ZERO",
        _env_float("CURRICULUM_FINGER_CENTER_TIP_Z_MAX", 0.075),
    )
    angle_full = _env_float(
        "CURRICULUM_CENTERED_CONTACT_ANGLE_FULL_DEG",
        _env_float("CURRICULUM_SUCCESS_CENTER_ALIGN_ANGLE_MAX_DEG", 8.0),
    )
    angle_zero = _env_float(
        "CURRICULUM_CENTERED_CONTACT_ANGLE_ZERO_DEG",
        _env_float("CURRICULUM_FINGER_CENTER_ALIGN_ANGLE_MAX_DEG", 15.0),
    )

    xy_zero = max(xy_zero, xy_full + 1.0e-6)
    z_zero = max(z_zero, z_full + 1.0e-6)
    angle_zero = max(angle_zero, angle_full + 1.0e-6)

    center = (
        _smoothstep_in_band(xy_err, lo=xy_full, hi=xy_zero)
        * _smoothstep_in_band(z_err, lo=z_full, hi=z_zero)
        * _smoothstep_in_band(angle_deg, lo=angle_full, hi=angle_zero)
    )
    return center * opposed_contact_strength(env) * stage_is(env, 2)


def contact_success_now_continuous(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Continuous +1 reward (per-step) when ALL 9 success conditions hold simultaneously.

    Iter 03-07 ceiling at success_rate ≤ 0.10 was caused by the policy hitting
    each success metric AT DIFFERENT TIMESTEPS. The eval `median_best_*` columns
    looked great (palm 0.07, align 0.10, drop 5°, contact 1.0) but came from
    non-overlapping mode A (hover, no contact) and mode B (contact, bad pose).

    This term reward the JOINT shell — the actual `light_contact_success_now`
    predicate which requires all 9 conditions in the same frame. Provides a
    sustained gradient pulling the policy to maintain ALL conditions
    simultaneously, complementing the once-per-episode +50/+200 success bonus
    (which is too sparse to coordinate 9 conditions).
    """
    return light_contact_success_now(env).float()


def contact_smooth_success_continuous(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Smooth approximation of success_now via per-gate smoothsteps.

    Iter 08 result: the continuous-success reward (binary on success_now) only
    fires when ALL 9 conditions are simultaneously true — but the policy never
    reaches that state, so the reward never fires. Same chicken-and-egg as
    contact_deep_shell_bonus.

    This term is the SMOOTH version: a product of per-gate smoothsteps with
    bands that extend OUTSIDE the strict shell, so the reward is positive
    everywhere inside a soft neighborhood of the shell. Gradient is non-zero
    even when individual gates are missed by 1-2x their strict threshold,
    pulling the policy in toward all 9 simultaneously.

    Gate band design:
    - palm_dist: 0.04→0.12 (full credit ≤ 0.04, 0 outside 0.12 = 1.5x strict 0.08)
    - palm_h: 0.02→0.08 (full ≤ 0.02, 0 outside 0.08 = 2x strict 0.04)
    - drop: 15°→45° (full ≤ 15°, 0 outside 45° = ~1.3x strict 35°)
    - yaw: 10°→45°
    - align: 0.08→0.30 (full ≤ 0.08, 0 outside 0.30 = 1.5x strict 0.20)
    - block_disp: 0→0.04 (full = 0, 0 outside 0.04 = 2x strict 0.02)
    - lift: 0→0.02
    - opposed_face: linear 0-1 (already a gate)
    - opposed_strength: linear 0-1 (already a gate)
    """
    palm_d = palm_distance_contact(env)
    palm_h = palm_height_error_contact(env)
    drop_deg = palm_drop_axis_error_rad(env) * (180.0 / 3.141592653589793)
    yaw_deg = palm_yaw_axis_error_rad(env) * (180.0 / 3.141592653589793)
    align_e = open_hand_alignment_error(env)
    opp_face = opposite_face_gate(env)
    opp_strength = opposed_contact_strength(env)
    blk_disp = block_xy_displacement(env) if os.environ.get("TOPDOWN_LIFT_TASK", "0") == "1" else block_displacement(env)
    lift = block_lift_height(env)

    palm_d_term = _smoothstep_in_band(palm_d, lo=0.04, hi=0.12)
    palm_h_term = _smoothstep_in_band(palm_h, lo=0.02, hi=0.08)
    drop_term = _smoothstep_in_band(drop_deg, lo=15.0, hi=45.0)
    yaw_term = _smoothstep_in_band(yaw_deg, lo=10.0, hi=45.0)
    align_term = _smoothstep_in_band(align_e, lo=0.08, hi=0.30)
    blk_disp_term = _smoothstep_in_band(blk_disp, lo=0.0, hi=0.04)
    lift_term = _smooth_success_lift_term(lift)
    return (
        palm_d_term
        * palm_h_term
        * drop_term
        * yaw_term
        * align_term
        * opp_face
        * opp_strength
        * blk_disp_term
        * lift_term
        * stage_is(env, 2)
    )


def _smooth_success_pose_factor(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Shared pose-only smoothstep product used by the split smooth-success terms.

    Excludes the linear contact gates (``opp_face``, ``opp_strength``) so it has
    a non-zero gradient pre-contact. Mirrors the bands used by
    ``contact_smooth_success_continuous``; bands intentionally extend outside
    the strict shell so the soft neighborhood pays.
    """
    palm_d = palm_distance_contact(env)
    palm_h = palm_height_error_contact(env)
    drop_deg = palm_drop_axis_error_rad(env) * (180.0 / 3.141592653589793)
    yaw_deg = palm_yaw_axis_error_rad(env) * (180.0 / 3.141592653589793)
    align_e = open_hand_alignment_error(env)
    blk_disp = block_xy_displacement(env) if os.environ.get("TOPDOWN_LIFT_TASK", "0") == "1" else block_displacement(env)
    lift = block_lift_height(env)

    palm_d_term = _smoothstep_in_band(palm_d, lo=0.04, hi=0.12)
    palm_h_term = _smoothstep_in_band(palm_h, lo=0.02, hi=0.08)
    drop_term = _smoothstep_in_band(drop_deg, lo=15.0, hi=45.0)
    yaw_term = _smoothstep_in_band(yaw_deg, lo=10.0, hi=45.0)
    align_term = _smoothstep_in_band(align_e, lo=0.08, hi=0.30)
    blk_disp_term = _smoothstep_in_band(blk_disp, lo=0.0, hi=0.04)
    lift_term = _smooth_success_lift_term(lift)
    return (
        palm_d_term
        * palm_h_term
        * drop_term
        * yaw_term
        * align_term
        * blk_disp_term
        * lift_term
    )


def _smooth_success_lift_term(lift: torch.Tensor) -> torch.Tensor:
    """Optional anti-lift gate for old light-contact ablations.

    Default is no lift gate: once the hand reaches the contact pose, smooth pose
    reward should not disappear just because the block starts moving upward.
    """
    if os.environ.get("CURRICULUM_SMOOTH_SUCCESS_LIFT_GATE", "0") == "1":
        return _smoothstep_in_band(lift, lo=0.0, hi=0.02)
    return torch.ones_like(lift)


def contact_smooth_success_pose_continuous(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Pose-track smooth success throughout Stage 2.

    The original ``contact_smooth_success_continuous`` multiplied the pose
    smoothstep product by linear ``opp_face`` AND ``opp_strength`` gates, which
    zeroed the gradient until contact was already established (chicken-and-egg
    that prevented the policy from learning to descend into the contact pose).
    This term keeps paying for pose through and after the contact-pose latch so
    finger closure is not forced to trade away the pose gradient exactly when
    the policy is trying to make contact.
    """
    stage_gate = stage_is(env, 2)
    if os.environ.get("CURRICULUM_SMOOTH_POSE_POST_LATCH", "1") == "1":
        return _smooth_success_pose_factor(env) * stage_gate

    ready = getattr(env, "_topdown_contact_pose_ready", None)
    if ready is None:
        pre_latch_gate = torch.ones_like(stage_gate)
    else:
        pre_latch_gate = 1.0 - ready.float()
    return _smooth_success_pose_factor(env) * stage_gate * pre_latch_gate


def contact_smooth_success_with_contact_continuous(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Contact-track smooth success: pose × opposed-face geometry × opposed strength.

    Mirrors the original joint smooth-success definition. Kept so that contact
    itself still pays, but with a smaller weight than the pose track so the
    gradient is dominated by getting into the pose first, contact second.
    """
    contact_floor = _env_float("CURRICULUM_CONTACT_SMOOTH_PARTIAL_FLOOR", 0.0)
    contact_floor = max(0.0, min(1.0, contact_floor))
    contact_gate = contact_floor + (1.0 - contact_floor) * opposed_contact_strength(env)
    return (
        _smooth_success_pose_factor(env)
        * opposite_face_gate(env)
        * contact_gate
        * stage_is(env, 2)
    )


def light_contact_success_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Edge-triggered: fires once per episode on rising edge of success_now."""
    ensure_curriculum_stage_updated(env)
    n = env.num_envs
    if not hasattr(env, "_topdown_success_bonus_fired"):
        env._topdown_success_bonus_fired = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._topdown_success_bonus_fired[just_reset] = False
    if os.environ.get("TOPDOWN_LIFT_TASK", "0") == "1":
        now = lift_success_now(env)
    else:
        now = light_contact_success_now(env)
    just_fired = now & (~env._topdown_success_bonus_fired)
    env._topdown_success_bonus_fired = env._topdown_success_bonus_fired | now
    return just_fired.float()


# --- Always-on shaping --------------------------------------------------------


def step_cost(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the per-step time penalty for active episodes."""
    return torch.ones(env.num_envs, device=env.device)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Mean-square per-step action delta (matches the standard action-smoothness penalty)."""
    am = env.action_manager
    cur = am.action
    prev = am.prev_action
    if prev is None or prev.shape != cur.shape:
        return torch.zeros(env.num_envs, device=cur.device)
    delta = cur - prev
    return torch.mean(delta * delta, dim=-1)
