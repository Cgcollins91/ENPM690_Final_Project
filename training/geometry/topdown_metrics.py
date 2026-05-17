"""

Topdown curriculum metric reductions for rollout and eval summaries

File map:

SOURCE_BLOCK_NAMES:                       Define source block names constant
TopdownAxisThresholds:                    Stage-0 topdown reach shell thresholds
topdown_source_block_name:                Return a display name for a topdown source block index
_masked_tensor:                           Handle masked tensor logic
_masked_stage:                            Handle masked stage logic
topdown_stage_metrics:                    Aggregate topdown stage and contact-pose tensors
topdown_progress_metric_mask:             Use active rows when present and all rows otherwise
topdown_progress_metrics:                 Build the topdown progress metric dictionary for one training log step
topdown_axis_metrics:                     Aggregate topdown axis reach diagnostics
topdown_axis_state:                       Return one-env topdown axis state
topdown_source_conditioned_metrics:       Aggregate rollout outcomes by sampled source block
_expanded_row_values:                     Handle expanded row values logic
topdown_eval_source_conditioned_metrics:  Aggregate eval outcomes by source block
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics

import torch


SOURCE_BLOCK_NAMES = ("red", "yellow", "blue")


@dataclass(frozen=True)
class TopdownAxisThresholds:
    """Stage-0 topdown reach shell thresholds"""

    palm_dist_max    : float  # floating-point palm dist max value used by topdown axis thresholds
    palm_height_max  : float  # floating-point palm height max value used by topdown axis thresholds
    drop_axis_max_deg: float  # floating-point drop axis max deg value used by topdown axis thresholds
    yaw_axis_max_deg : float  # floating-point yaw axis max deg value used by topdown axis thresholds


def topdown_source_block_name(source_idx: int, names: Sequence[str] = SOURCE_BLOCK_NAMES) -> str:
    """Return a display name for a topdown source block index"""
    if 0 <= int(source_idx) < len(names):
        return str(names[int(source_idx)])
    return "unknown"


def _masked_tensor(
    tensor   : torch.Tensor | None,  # Param: tensor input carrying tensor values
    reference: torch.Tensor,  # Param: tensor input carrying reference values
    mask     : torch.Tensor | None,  # Param: boolean mask selecting mask rows
    *,
    dtype  : torch.dtype,  # Param: torch dtype used when converting or allocating tensors
    default: float = 0.0,  # Param: fallback value used when the input omits or rejects a setting
) -> torch.Tensor:
    if tensor is None or not torch.is_tensor(tensor) or tensor.shape != reference.shape:
        selected = torch.full_like(reference, float(default), dtype=dtype)
    else:
        selected = tensor.to(device=reference.device, dtype=dtype)
    if mask is not None and mask.shape == reference.shape and bool(mask.any().item()):
        selected = selected[mask.to(device=reference.device, dtype=torch.bool)]
    return selected


def _masked_stage(stage: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if not torch.is_tensor(stage) or stage.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    selected = stage.to(dtype=torch.long)
    if mask is not None and mask.shape == stage.shape and bool(mask.any().item()):
        selected = selected[mask.to(device=stage.device, dtype=torch.bool)]
    return selected


def topdown_stage_metrics(
    *,
    stage                 : torch.Tensor,  # Param: tensor input carrying stage values
    mask                  : torch.Tensor | None = None,  # Param: boolean mask selecting mask rows
    source_idx            : torch.Tensor | None = None,  # Param: index selecting the source entry
    finger_unlock_progress: torch.Tensor | None = None,  # Param: tensor input carrying finger unlock progress values
    reach_hold            : torch.Tensor | None = None,  # Param: tensor input carrying reach hold values
    align_hold            : torch.Tensor | None = None,  # Param: tensor input carrying align hold values
    stage2_age            : torch.Tensor | None = None,  # Param: tensor input carrying stage2 age values
    stage2_fallout_hold   : torch.Tensor | None = None,  # Param: tensor input carrying stage2 fallout hold values
    contact_pose_hold     : torch.Tensor | None = None,  # Param: tensor input carrying contact pose hold values
    contact_pose_ready    : torch.Tensor | None = None,  # Param: mask or boolean input marking contact pose as ready
    contact_pose_age      : torch.Tensor | None = None,  # Param: tensor input carrying contact pose age values
    contact_pose_shell    : torch.Tensor | None = None,  # Param: tensor input carrying contact pose shell values
    contact_palm_dist     : torch.Tensor | None = None,  # Param: tensor input carrying contact palm dist values
    contact_palm_height   : torch.Tensor | None = None,  # Param: tensor input carrying contact palm height values
    prefix                : str                 = "topdown",  # Param: string input for prefix
) -> dict[str, float]:
    """Aggregate topdown stage and contact-pose tensors

    Steps:
    - Resolve inputs for `topdown_stage_metrics` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    selected_stage = _masked_stage(stage, mask)
    if selected_stage.numel() == 0:
        return {}
    selected_mask = mask if mask is not None and mask.shape == stage.shape and bool(mask.any().item()) else None

    def metric_tensor(tensor: torch.Tensor | None, default: float = 0.0) -> torch.Tensor:
        return _masked_tensor(tensor, stage, selected_mask, dtype=torch.float32, default=default)

    metrics: dict[str, float] = {
        f"{prefix}_stage0_rate": float((selected_stage == 0).float().mean().item()),
        f"{prefix}_stage1_rate": float((selected_stage == 1).float().mean().item()),
        f"{prefix}_stage2_rate": float((selected_stage == 2).float().mean().item()),
        f"{prefix}_stage_ge1_rate": float((selected_stage >= 1).float().mean().item()),
        f"{prefix}_stage_ge2_rate": float((selected_stage >= 2).float().mean().item()),
        f"{prefix}_stage_mean": float(selected_stage.to(dtype=torch.float32).mean().item()),
    }

    if source_idx is not None and torch.is_tensor(source_idx) and source_idx.shape == stage.shape:
        selected_source = _masked_tensor(source_idx, stage, selected_mask, dtype=torch.long, default=-1.0)
        for idx, name in enumerate(SOURCE_BLOCK_NAMES):
            metrics[f"{prefix}_source_{name}_rate"] = float((selected_source == idx).float().mean().item())

    unlock = metric_tensor(finger_unlock_progress)
    reach = metric_tensor(reach_hold)
    align = metric_tensor(align_hold)
    age = metric_tensor(stage2_age)
    contact_hold = metric_tensor(contact_pose_hold)
    contact_ready = metric_tensor(contact_pose_ready)
    contact_age = metric_tensor(contact_pose_age)
    metrics.update(
        {
            f"{prefix}_finger_unlock_progress_mean": float(unlock.mean().item()),
            f"{prefix}_finger_unlock_progress_max": float(unlock.max().item()),
            f"{prefix}_reach_hold_mean": float(reach.mean().item()),
            f"{prefix}_reach_hold_max": float(reach.max().item()),
            f"{prefix}_align_hold_mean": float(align.mean().item()),
            f"{prefix}_align_hold_max": float(align.max().item()),
            f"{prefix}_stage2_age_mean": float(age.mean().item()),
            f"{prefix}_stage2_age_max": float(age.max().item()),
            f"{prefix}_contact_pose_ready_rate": float(contact_ready.mean().item()),
            f"{prefix}_contact_pose_hold_mean": float(contact_hold.mean().item()),
            f"{prefix}_contact_pose_hold_max": float(contact_hold.max().item()),
            f"{prefix}_contact_pose_age_mean": float(contact_age.mean().item()),
            f"{prefix}_contact_pose_age_max": float(contact_age.max().item()),
        }
    )

    if stage2_fallout_hold is not None:
        fallout = metric_tensor(stage2_fallout_hold)
        metrics[f"{prefix}_stage2_fallout_hold_mean"] = float(fallout.mean().item())
        metrics[f"{prefix}_stage2_fallout_hold_max"] = float(fallout.max().item())
    if contact_pose_shell is not None:
        shell = metric_tensor(contact_pose_shell)
        metrics[f"{prefix}_contact_pose_shell_rate"] = float(shell.mean().item())
    if contact_palm_dist is not None:
        palm_dist = metric_tensor(contact_palm_dist)
        metrics[f"{prefix}_contact_palm_dist_mean"] = float(palm_dist.mean().item())
        metrics[f"{prefix}_contact_palm_dist_min"] = float(palm_dist.min().item())
    if contact_palm_height is not None:
        palm_height = metric_tensor(contact_palm_height)
        metrics[f"{prefix}_contact_palm_height_mean"] = float(palm_height.mean().item())
        metrics[f"{prefix}_contact_palm_height_min"] = float(palm_height.min().item())
    return metrics


def topdown_progress_metric_mask(active_env_mask: torch.Tensor) -> torch.Tensor:
    """Use active rows when present and all rows otherwise"""
    mask = active_env_mask.to(dtype=torch.bool)
    if bool(mask.any().item()):
        return mask
    return torch.ones_like(mask, dtype=torch.bool)


def topdown_progress_metrics(
    *,
    active_env_mask       : torch.Tensor,  # Param: mask selecting env rows still active for collection or reset handling
    stage                 : torch.Tensor,  # Param: tensor input carrying stage values
    finger_unlock_progress: torch.Tensor,  # Param: tensor input carrying finger unlock progress values
    reach_hold            : torch.Tensor,  # Param: tensor input carrying reach hold values
    align_hold            : torch.Tensor,  # Param: tensor input carrying align hold values
    stage2_age            : torch.Tensor,  # Param: tensor input carrying stage2 age values
    contact_pose_hold     : torch.Tensor,  # Param: tensor input carrying contact pose hold values
    contact_pose_ready    : torch.Tensor,  # Param: mask or boolean input marking contact pose as ready
    contact_pose_age      : torch.Tensor,  # Param: tensor input carrying contact pose age values
    contact_pose_shell    : torch.Tensor,  # Param: tensor input carrying contact pose shell values
    contact_palm_dist     : torch.Tensor,  # Param: tensor input carrying contact palm dist values
    contact_palm_height   : torch.Tensor,  # Param: tensor input carrying contact palm height values
    source_idx            : torch.Tensor | None        = None,  # Param: index selecting the source entry
    success_flags         : torch.Tensor | None        = None,  # Param: flag values describing success
    strict_contact        : torch.Tensor | None        = None,  # Param: tensor input carrying strict contact values
    contact               : torch.Tensor | None        = None,  # Param: tensor input carrying contact values
    lift                  : torch.Tensor | None        = None,  # Param: tensor input carrying lift values
    axis_metrics          : Mapping[str, float] | None = None,  # Param: string input for axis metrics
    contact_threshold     : float                      = 0.30,  # Param: cutoff used when evaluating contact
) -> dict[str, float]:
    """Build the topdown progress metric dictionary for one training log step

    Steps:
    - Resolve inputs for `topdown_progress_metrics` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    metric_mask = topdown_progress_metric_mask(active_env_mask)
    metrics = topdown_stage_metrics(
        stage=stage,
        mask=metric_mask,
        finger_unlock_progress=finger_unlock_progress,
        reach_hold=reach_hold,
        align_hold=align_hold,
        stage2_age=stage2_age,
        contact_pose_hold=contact_pose_hold,
        contact_pose_ready=contact_pose_ready,
        contact_pose_age=contact_pose_age,
        contact_pose_shell=contact_pose_shell,
        contact_palm_dist=contact_palm_dist,
        contact_palm_height=contact_palm_height,
    )
    if not metrics:
        return {}
    if axis_metrics:
        metrics.update({str(key): float(value) for key, value in axis_metrics.items()})
    if success_flags is not None:
        success = _masked_tensor(success_flags, stage, metric_mask, dtype=torch.float32)
        metrics["topdown_success_rate"] = float(success.mean().item())
    if strict_contact is not None:
        strict = _masked_tensor(strict_contact, stage, metric_mask, dtype=torch.float32)
        metrics["topdown_strict_contact_mean"] = float(strict.mean().item())
    if contact is not None:
        contact_t = _masked_tensor(contact, stage, metric_mask, dtype=torch.float32)
        metrics["topdown_contact_mean"] = float(contact_t.mean().item())
    if lift is not None:
        lift_t = _masked_tensor(lift, stage, metric_mask, dtype=torch.float32)
        metrics["topdown_lift_mean"] = float(lift_t.mean().item())
        metrics["topdown_lift_max"] = float(lift_t.max().item())
    if source_idx is not None:
        metrics.update(
            topdown_source_conditioned_metrics(
                source_idx,
                mask=metric_mask,
                success_flags=success_flags,
                strict_contact=strict_contact,
                contact=contact,
                lift=lift,
                stage=stage,
                contact_threshold=contact_threshold,
            )
        )
    return metrics


def topdown_axis_metrics(
    *,
    palm_dist  : torch.Tensor,  # Param: tensor input carrying palm dist values
    palm_height: torch.Tensor,  # Param: tensor input carrying palm height values
    drop_deg   : torch.Tensor,  # Param: tensor input carrying drop deg values
    yaw_deg    : torch.Tensor,  # Param: tensor input carrying yaw deg values
    spread_deg : torch.Tensor,  # Param: tensor input carrying spread deg values
    thresholds : TopdownAxisThresholds,  # Param: input value used as thresholds
    mask       : torch.Tensor | None = None,  # Param: boolean mask selecting mask rows
    prefix     : str                 = "topdown",  # Param: string input for prefix
) -> dict[str, float]:
    """Aggregate topdown axis reach diagnostics

    Steps:
    - Resolve inputs for `topdown_axis_metrics` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if mask is not None:
        mask = mask.to(device=palm_dist.device, dtype=torch.bool)
        if mask.shape == palm_dist.shape and bool(mask.any().item()):
            palm_dist = palm_dist[mask]
            palm_height = palm_height[mask]
            drop_deg = drop_deg[mask]
            yaw_deg = yaw_deg[mask]
            spread_deg = spread_deg[mask]
    if palm_dist.numel() == 0:
        return {}

    dist_pass = (palm_dist <= float(thresholds.palm_dist_max)).float()
    height_pass = (palm_height <= float(thresholds.palm_height_max)).float()
    drop_pass = (drop_deg <= float(thresholds.drop_axis_max_deg)).float()
    yaw_pass = (yaw_deg <= float(thresholds.yaw_axis_max_deg)).float()
    reach_pass = dist_pass * height_pass * drop_pass * yaw_pass
    return {
        f"{prefix}_drop_axis_deg_mean": float(drop_deg.mean().item()),
        f"{prefix}_drop_axis_deg_min": float(drop_deg.min().item()),
        f"{prefix}_yaw_axis_deg_mean": float(yaw_deg.mean().item()),
        f"{prefix}_yaw_axis_deg_min": float(yaw_deg.min().item()),
        f"{prefix}_spread_axis_deg_mean": float(spread_deg.mean().item()),
        f"{prefix}_spread_axis_deg_min": float(spread_deg.min().item()),
        f"{prefix}_dist_pass_rate": float(dist_pass.mean().item()),
        f"{prefix}_height_pass_rate": float(height_pass.mean().item()),
        f"{prefix}_drop_pass_rate": float(drop_pass.mean().item()),
        f"{prefix}_yaw_pass_rate": float(yaw_pass.mean().item()),
        f"{prefix}_reach_pass_rate": float(reach_pass.mean().item()),
    }


def topdown_axis_state(
    *,
    palm_dist  : torch.Tensor,  # Param: tensor input carrying palm dist values
    palm_height: torch.Tensor,  # Param: tensor input carrying palm height values
    drop_deg   : torch.Tensor,  # Param: tensor input carrying drop deg values
    yaw_deg    : torch.Tensor,  # Param: tensor input carrying yaw deg values
    spread_deg : torch.Tensor,  # Param: tensor input carrying spread deg values
    thresholds : TopdownAxisThresholds,  # Param: input value used as thresholds
    env_id     : int = 0,  # Param: integer input for env id
    prefix     : str = "topdown",  # Param: string input for prefix
) -> dict[str, float]:
    """Return one-env topdown axis state

    Steps:
    - Resolve inputs for `topdown_axis_state` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    idx = int(env_id)
    dist = float(palm_dist[idx].item())
    height = float(palm_height[idx].item())
    drop = float(drop_deg[idx].item())
    yaw = float(yaw_deg[idx].item())
    spread = float(spread_deg[idx].item())
    dist_pass = float(dist <= float(thresholds.palm_dist_max))
    height_pass = float(height <= float(thresholds.palm_height_max))
    drop_pass = float(drop <= float(thresholds.drop_axis_max_deg))
    yaw_pass = float(yaw <= float(thresholds.yaw_axis_max_deg))
    return {
        f"{prefix}_palm_dist": dist,
        f"{prefix}_palm_height_err": height,
        f"{prefix}_drop_axis_deg": drop,
        f"{prefix}_yaw_axis_deg": yaw,
        f"{prefix}_spread_axis_deg": spread,
        f"{prefix}_dist_pass": dist_pass,
        f"{prefix}_height_pass": height_pass,
        f"{prefix}_drop_pass": drop_pass,
        f"{prefix}_yaw_pass": yaw_pass,
        f"{prefix}_reach_pass": dist_pass * height_pass * drop_pass * yaw_pass,
    }


def topdown_source_conditioned_metrics(
    source_idx: torch.Tensor | None,                  # Param: index selecting the source entry
    *,
    mask             : torch.Tensor | None = None,  # Param: boolean mask selecting mask rows
    success_flags    : torch.Tensor | None = None,  # Param: flag values describing success
    strict_contact   : torch.Tensor | None = None,  # Param: tensor input carrying strict contact values
    contact          : torch.Tensor | None = None,  # Param: tensor input carrying contact values
    lift             : torch.Tensor | None = None,  # Param: tensor input carrying lift values
    stage            : torch.Tensor | None = None,  # Param: tensor input carrying stage values
    contact_threshold: float               = 0.30,  # Param: cutoff used when evaluating contact
    prefix           : str                 = "topdown_block",  # Param: string input for prefix
    block_names      : Sequence[str]       = SOURCE_BLOCK_NAMES,  # Param: ordered candidate names used to resolve block
) -> dict[str, float]:
    """Aggregate rollout outcomes by sampled source block

    Steps:
    - Resolve inputs for `topdown_source_conditioned_metrics` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if source_idx is None or not torch.is_tensor(source_idx):
        return {}
    source_idx = source_idx.to(dtype=torch.long)
    if source_idx.numel() == 0:
        return {}
    base_mask = torch.ones_like(source_idx, dtype=torch.bool)
    if mask is not None and mask.shape == source_idx.shape:
        base_mask = mask.to(device=source_idx.device, dtype=torch.bool)
    denom = int(base_mask.sum().item())
    if denom <= 0:
        return {}

    def float_metric(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or not torch.is_tensor(tensor) or tensor.shape != source_idx.shape:
            return None
        return tensor.to(device=source_idx.device, dtype=torch.float32)

    success_t = float_metric(success_flags)
    strict_t = float_metric(strict_contact)
    contact_t = float_metric(contact)
    lift_t = float_metric(lift)
    stage_t = (
        stage.to(device=source_idx.device, dtype=torch.long)
        if torch.is_tensor(stage) and stage.shape == source_idx.shape
        else None
    )
    metrics: dict[str, float] = {}
    for idx, name in enumerate(block_names):
        block_mask = base_mask & (source_idx == idx)
        count = int(block_mask.sum().item())
        metrics[f"{prefix}_{name}_count"] = float(count)
        metrics[f"{prefix}_{name}_sample_rate"] = float(count / denom)
        if count <= 0:
            continue
        if success_t is not None:
            metrics[f"{prefix}_{name}_success_rate"] = float(success_t[block_mask].mean().item())
        if strict_t is not None:
            strict_block = strict_t[block_mask]
            metrics[f"{prefix}_{name}_strict_contact_mean"] = float(strict_block.mean().item())
            metrics[f"{prefix}_{name}_strict_contact_rate"] = float(
                (strict_block >= float(contact_threshold)).float().mean().item()
            )
        if contact_t is not None:
            metrics[f"{prefix}_{name}_contact_mean"] = float(contact_t[block_mask].mean().item())
        if lift_t is not None:
            metrics[f"{prefix}_{name}_lift_mean"] = float(lift_t[block_mask].mean().item())
            metrics[f"{prefix}_{name}_lift_max"] = float(lift_t[block_mask].max().item())
        if stage_t is not None:
            stage_block = stage_t[block_mask]
            metrics[f"{prefix}_{name}_stage_ge1_rate"] = float((stage_block >= 1).float().mean().item())
            metrics[f"{prefix}_{name}_stage_ge2_rate"] = float((stage_block >= 2).float().mean().item())
    return metrics


def _expanded_row_values(row: Mapping[str, object], key: str, default: object, env_count: int) -> list[object]:
    env_key = key.replace("eval_", "eval_env_", 1)
    env_values = row.get(env_key)
    if isinstance(env_values, list) and len(env_values) == env_count:
        return list(env_values)
    return [row.get(key, default) for _ in range(env_count)]


def topdown_eval_source_conditioned_metrics(
    summaries: Sequence[Mapping[str, object]],        # Param: string input for summaries
    *,
    contact_threshold: float         = 0.30,  # Param: cutoff used when evaluating contact
    prefix           : str           = "eval_topdown_block",  # Param: string input for prefix
    block_names      : Sequence[str] = SOURCE_BLOCK_NAMES,  # Param: ordered candidate names used to resolve block
) -> dict[str, float]:
    """Aggregate eval outcomes by source block

    Steps:
    - Resolve inputs for `topdown_eval_source_conditioned_metrics` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    flat_rows: list[dict[str, object]] = []
    for row in summaries:
        env_source_idxs = row.get("eval_env_topdown_source_pose_idx")
        if isinstance(env_source_idxs, list):
            env_count = len(env_source_idxs)
            successes = _expanded_row_values(row, "eval_success", False, env_count)
            strict_contacts = _expanded_row_values(row, "eval_best_strict_light_contact", 0.0, env_count)
            contacts = _expanded_row_values(row, "eval_best_contact", 0.0, env_count)
            lifts = _expanded_row_values(row, "eval_best_lift", 0.0, env_count)
            stages = _expanded_row_values(row, "eval_best_topdown_stage", -1, env_count)
            for env_id, source_idx in enumerate(env_source_idxs):
                flat_rows.append(
                    {
                        "source_idx"    : int(source_idx),
                        "success"       : bool(successes[env_id]),
                        "strict_contact": float(strict_contacts[env_id]),
                        "contact"       : float(contacts[env_id]),
                        "lift"          : float(lifts[env_id]),
                        "stage"         : int(stages[env_id]),
                    }
                )
        else:
            flat_rows.append(
                {
                    "source_idx"    : int(row.get("eval_topdown_source_pose_idx", -1)),
                    "success"       : bool(row.get("eval_success", False)),
                    "strict_contact": float(row.get("eval_best_strict_light_contact", 0.0)),
                    "contact"       : float(row.get("eval_best_contact", 0.0)),
                    "lift"          : float(row.get("eval_best_lift", 0.0)),
                    "stage"         : int(row.get("eval_best_topdown_stage", -1)),
                }
            )
    if not flat_rows:
        return {}
    metrics: dict[str, float] = {}
    denom = len(flat_rows)
    for idx, name in enumerate(block_names):
        rows = [row for row in flat_rows if int(row.get("source_idx", -1)) == idx]
        count = len(rows)
        metrics[f"{prefix}_{name}_count"] = float(count)
        metrics[f"{prefix}_{name}_sample_rate"] = float(count / max(denom, 1))
        if count <= 0:
            continue
        successes = [1.0 if row.get("success", False) else 0.0 for row in rows]
        strict_contacts = [float(row.get("strict_contact", 0.0)) for row in rows]
        contacts = [float(row.get("contact", 0.0)) for row in rows]
        lifts = [float(row.get("lift", 0.0)) for row in rows]
        stages = [int(row.get("stage", -1)) for row in rows]
        metrics[f"{prefix}_{name}_success_rate"] = float(statistics.mean(successes))
        metrics[f"{prefix}_{name}_success_count"] = float(sum(successes))
        metrics[f"{prefix}_{name}_strict_contact_rate"] = float(
            statistics.mean([1.0 if value >= float(contact_threshold) else 0.0 for value in strict_contacts])
        )
        metrics[f"{prefix}_{name}_strict_contact_mean"] = float(statistics.mean(strict_contacts))
        metrics[f"{prefix}_{name}_contact_mean"] = float(statistics.mean(contacts))
        metrics[f"{prefix}_{name}_median_best_lift"] = float(statistics.median(lifts))
        metrics[f"{prefix}_{name}_stage_ge1_rate"] = float(
            statistics.mean([1.0 if stage >= 1 else 0.0 for stage in stages])
        )
        metrics[f"{prefix}_{name}_stage_ge2_rate"] = float(
            statistics.mean([1.0 if stage >= 2 else 0.0 for stage in stages])
        )
    return metrics
