"""

Episode metric tensor selection helpers

File map:

EpisodeMetricTensors:                   Metric tensors used to update episode bests
TopdownProgressMetricTensors:           Topdown tensors used to build progress summaries
select_done_or_live:                    Select done-row values from pre-reset tensors and live otherwise
episode_lift_with_strict_contact:       Return lift only where strict contact meets threshold
lift_with_strict_contact:               Compatibility alias for strict-contact lift selection
build_episode_metric_tensors:           Build selected metric tensors for one vectorized training step
build_topdown_progress_metric_tensors:  Build selected topdown tensors for progress summary metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpisodeMetricTensors:
    """Metric tensors used to update episode bests"""

    tip                     : torch.Tensor  # tensor containing tip values for batched env rows
    phase1_palm             : torch.Tensor  # tensor containing phase1 palm values for batched env rows
    phase1_orient           : torch.Tensor  # tensor containing phase1 orient values for batched env rows
    contact                 : torch.Tensor  # tensor containing contact values for batched env rows
    strict_contact          : torch.Tensor  # tensor containing strict contact values for batched env rows
    lift                    : torch.Tensor  # tensor containing lift values for batched env rows
    lift_with_strict_contact: torch.Tensor  # tensor containing lift with strict contact values for batched env rows
    curl                    : torch.Tensor  # tensor containing curl values for batched env rows
    topdown_stage           : torch.Tensor  # current topdown curriculum stage per environment
    topdown_unlock          : torch.Tensor  # tensor containing topdown unlock values for batched env rows


@dataclass(frozen=True)
class TopdownProgressMetricTensors:
    """Topdown tensors used to build progress summaries"""

    stage              : torch.Tensor  # tensor containing stage values for batched env rows
    reach_hold         : torch.Tensor  # tensor containing reach hold values for batched env rows
    align_hold         : torch.Tensor  # tensor containing align hold values for batched env rows
    stage2_age         : torch.Tensor  # tensor containing stage2 age values for batched env rows
    contact_pose_hold  : torch.Tensor  # tensor containing contact pose hold values for batched env rows
    contact_pose_ready : torch.Tensor  # boolean/tensor readiness state for contact pose
    contact_pose_age   : torch.Tensor  # tensor containing contact pose age values for batched env rows
    unlock             : torch.Tensor  # tensor containing unlock values for batched env rows
    contact_pose_shell : torch.Tensor  # tensor containing contact pose shell values for batched env rows
    contact_palm_dist  : torch.Tensor  # tensor containing contact palm dist values for batched env rows
    contact_palm_height: torch.Tensor  # tensor containing contact palm height values for batched env rows


def select_done_or_live(
    done_flags : torch.Tensor,  # Param: per-env done flags returned by the latest env step
    done_values: torch.Tensor,  # Param: tensor input carrying done values values
    live_values: torch.Tensor,  # Param: tensor input carrying live values values
) -> torch.Tensor:
    """Select done-row values from pre-reset tensors and live otherwise"""
    mask = done_flags.to(device=live_values.device, dtype=torch.bool)
    return torch.where(mask, done_values.to(device=live_values.device), live_values)


def episode_lift_with_strict_contact(
    lift          : torch.Tensor,  # Param: tensor input carrying lift values
    strict_contact: torch.Tensor,  # Param: tensor input carrying strict contact values
    *,
    contact_min: float,            # Param: floating-point input for contact min
) -> torch.Tensor:
    """Return lift only where strict contact meets threshold"""
    strict = strict_contact.to(device=lift.device, dtype=torch.float32)
    return torch.where(
        strict >= float(contact_min),
        lift.to(dtype=torch.float32),
        torch.zeros_like(lift, dtype=torch.float32),
    )


def lift_with_strict_contact(
    lift          : torch.Tensor,  # Param: tensor input carrying lift values
    strict_contact: torch.Tensor,  # Param: tensor input carrying strict contact values
    *,
    contact_min: float,            # Param: floating-point input for contact min
) -> torch.Tensor:
    """Compatibility alias for strict-contact lift selection"""
    return episode_lift_with_strict_contact(
        lift,
        strict_contact,
        contact_min=contact_min,
    )


def build_episode_metric_tensors(
    *,
    done_flags        : torch.Tensor,  # Param: per-env done flags returned by the latest env step
    pre_tip           : torch.Tensor,  # Param: tensor input carrying pre tip values
    tip               : torch.Tensor,  # Param: tensor input carrying tip values
    pre_phase1_palm   : torch.Tensor,  # Param: tensor input carrying pre phase1 palm values
    phase1_palm       : torch.Tensor,  # Param: tensor input carrying phase1 palm values
    pre_phase1_orient : torch.Tensor,  # Param: tensor input carrying pre phase1 orient values
    phase1_orient     : torch.Tensor,  # Param: tensor input carrying phase1 orient values
    pre_contact       : torch.Tensor,  # Param: tensor input carrying pre contact values
    contact           : torch.Tensor,  # Param: tensor input carrying contact values
    pre_strict_contact: torch.Tensor,  # Param: tensor input carrying pre strict contact values
    strict_contact    : torch.Tensor,  # Param: tensor input carrying strict contact values
    pre_lift          : torch.Tensor,  # Param: tensor input carrying pre lift values
    lift              : torch.Tensor,  # Param: tensor input carrying lift values
    pre_curl          : torch.Tensor,  # Param: tensor input carrying pre curl values
    curl              : torch.Tensor,  # Param: tensor input carrying curl values
    pre_topdown_stage : torch.Tensor,  # Param: tensor input carrying pre topdown stage values
    topdown_stage     : torch.Tensor,  # Param: tensor input carrying topdown stage values
    pre_topdown_unlock: torch.Tensor,  # Param: tensor input carrying pre topdown unlock values
    topdown_unlock    : torch.Tensor,  # Param: tensor input carrying topdown unlock values
    strict_contact_min: float,  # Param: floating-point input for strict contact min
) -> EpisodeMetricTensors:
    """Build selected metric tensors for one vectorized training step"""
    selected_strict = select_done_or_live(done_flags, pre_strict_contact, strict_contact)
    selected_lift = select_done_or_live(done_flags, pre_lift, lift)
    return EpisodeMetricTensors(
        tip=select_done_or_live(done_flags, pre_tip, tip),
        phase1_palm=select_done_or_live(done_flags, pre_phase1_palm, phase1_palm),
        phase1_orient=select_done_or_live(done_flags, pre_phase1_orient, phase1_orient),
        contact=select_done_or_live(done_flags, pre_contact, contact),
        strict_contact=selected_strict,
        lift=selected_lift,
        lift_with_strict_contact=episode_lift_with_strict_contact(
            selected_lift,
            selected_strict,
            contact_min=strict_contact_min,
        ),
        curl=select_done_or_live(done_flags, pre_curl, curl),
        topdown_stage=select_done_or_live(done_flags, pre_topdown_stage, topdown_stage),
        topdown_unlock=select_done_or_live(done_flags, pre_topdown_unlock, topdown_unlock),
    )


def build_topdown_progress_metric_tensors(
    *,
    done_flags             : torch.Tensor,  # Param: per-env done flags returned by the latest env step
    pre_stage              : torch.Tensor,  # Param: tensor input carrying pre stage values
    stage                  : torch.Tensor,  # Param: tensor input carrying stage values
    pre_reach_hold         : torch.Tensor,  # Param: tensor input carrying pre reach hold values
    reach_hold             : torch.Tensor,  # Param: tensor input carrying reach hold values
    pre_align_hold         : torch.Tensor,  # Param: tensor input carrying pre align hold values
    align_hold             : torch.Tensor,  # Param: tensor input carrying align hold values
    pre_stage2_age         : torch.Tensor,  # Param: tensor input carrying pre stage2 age values
    stage2_age             : torch.Tensor,  # Param: tensor input carrying stage2 age values
    pre_contact_pose_hold  : torch.Tensor,  # Param: tensor input carrying pre contact pose hold values
    contact_pose_hold      : torch.Tensor,  # Param: tensor input carrying contact pose hold values
    pre_contact_pose_ready : torch.Tensor,  # Param: mask or boolean input marking pre contact pose as ready
    contact_pose_ready     : torch.Tensor,  # Param: mask or boolean input marking contact pose as ready
    pre_contact_pose_age   : torch.Tensor,  # Param: tensor input carrying pre contact pose age values
    contact_pose_age       : torch.Tensor,  # Param: tensor input carrying contact pose age values
    pre_unlock             : torch.Tensor,  # Param: tensor input carrying pre unlock values
    unlock                 : torch.Tensor,  # Param: tensor input carrying unlock values
    pre_contact_pose_shell : torch.Tensor,  # Param: tensor input carrying pre contact pose shell values
    contact_pose_shell     : torch.Tensor,  # Param: tensor input carrying contact pose shell values
    pre_contact_palm_dist  : torch.Tensor,  # Param: tensor input carrying pre contact palm dist values
    contact_palm_dist      : torch.Tensor,  # Param: tensor input carrying contact palm dist values
    pre_contact_palm_height: torch.Tensor,  # Param: tensor input carrying pre contact palm height values
    contact_palm_height    : torch.Tensor,  # Param: tensor input carrying contact palm height values
) -> TopdownProgressMetricTensors:
    """Build selected topdown tensors for progress summary metrics"""
    return TopdownProgressMetricTensors(
        stage=select_done_or_live(done_flags, pre_stage, stage),
        reach_hold=select_done_or_live(done_flags, pre_reach_hold, reach_hold),
        align_hold=select_done_or_live(done_flags, pre_align_hold, align_hold),
        stage2_age=select_done_or_live(done_flags, pre_stage2_age, stage2_age),
        contact_pose_hold=select_done_or_live(
            done_flags,
            pre_contact_pose_hold,
            contact_pose_hold,
        ),
        contact_pose_ready=select_done_or_live(
            done_flags,
            pre_contact_pose_ready,
            contact_pose_ready,
        ),
        contact_pose_age=select_done_or_live(
            done_flags,
            pre_contact_pose_age,
            contact_pose_age,
        ),
        unlock=select_done_or_live(done_flags, pre_unlock, unlock),
        contact_pose_shell=select_done_or_live(
            done_flags,
            pre_contact_pose_shell,
            contact_pose_shell,
        ),
        contact_palm_dist=select_done_or_live(
            done_flags,
            pre_contact_palm_dist,
            contact_palm_dist,
        ),
        contact_palm_height=select_done_or_live(
            done_flags,
            pre_contact_palm_height,
            contact_palm_height,
        ),
    )
