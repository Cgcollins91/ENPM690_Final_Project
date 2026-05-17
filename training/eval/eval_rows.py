"""

Evaluation JSON row filtering helpers

File map:

NON_LIFT_EVAL_NOISE_KEYS:            Define non lift eval noise keys constant
NON_LIFT_EVAL_STEP_NOISE_KEYS:       Define non lift eval step noise keys constant
eval_summary_needs_lift_noise_drop:  Return whether lift-only eval fields should be dropped
filtered_eval_summary:               Return eval summary with task-specific noisy fields removed
filtered_eval_step_row:              Return eval step row with task-specific noisy fields removed
write_eval_step_row:                 Write one eval step JSON row when enabled
write_eval_summary_row:              Write one eval summary JSON row
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .eval_metrics import EvalTaskKind
from ..logging.jsonl import write_jsonl_row


NON_LIFT_EVAL_NOISE_KEYS = (
    "eval_best_lift",
    "eval_best_lift_with_strict_contact",
    "eval_final_lift",
    "eval_off_table",
    "eval_block_drift",
)

NON_LIFT_EVAL_STEP_NOISE_KEYS = (
    "lift_height",
    "off_table",
    "terminal_off_table_inferred",
    "block_drift",
    "terminal_block_drift_inferred",
    "last_live_lift_height",
    "post_reset_lift_height",
)


def eval_summary_needs_lift_noise_drop(task_kind: EvalTaskKind) -> bool:
    """Return whether lift-only eval fields should be dropped"""
    return task_kind in ("grasp_align", "topdown")


def filtered_eval_summary(
    summary: Mapping[str, Any],  # Param: string input for summary
    *,
    task_kind: EvalTaskKind,     # Param: input value used as task kind
) -> dict[str, Any]:
    """Return eval summary with task-specific noisy fields removed"""
    row = dict(summary)
    if eval_summary_needs_lift_noise_drop(task_kind):
        for key in NON_LIFT_EVAL_NOISE_KEYS:
            row.pop(key, None)
    return row


def filtered_eval_step_row(
    row: Mapping[str, Any],   # Param: string input for row
    *,
    task_kind: EvalTaskKind,  # Param: input value used as task kind
) -> dict[str, Any]:
    """Return eval step row with task-specific noisy fields removed"""
    filtered = dict(row)
    if eval_summary_needs_lift_noise_drop(task_kind):
        for key in NON_LIFT_EVAL_STEP_NOISE_KEYS:
            filtered.pop(key, None)
    return filtered


def write_eval_step_row(
    log_file: Any,  # Param: input value used as log file
    row     : Mapping[str, Any],  # Param: string input for row
    *,
    enabled: bool,  # Param: boolean input controlling enabled
    flush  : bool = True,  # Param: boolean input controlling flush
) -> bool:
    """Write one eval step JSON row when enabled"""
    if not enabled:
        return False
    write_jsonl_row(log_file, row, flush=flush)
    return True


def write_eval_summary_row(log_file: Any, summary: Mapping[str, Any], *, flush: bool = True) -> None:
    """Write one eval summary JSON row"""
    write_jsonl_row(log_file, summary, flush=flush)
