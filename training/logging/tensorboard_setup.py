"""

TensorBoard startup planning helpers

File map:

TensorBoardPlan:            Resolved TensorBoard writer setup plan
resolve_tensorboard_plan:   Resolve TensorBoard directory and availability message
create_tensorboard_writer:  Create a TensorBoard writer from a resolved plan
close_tensorboard_writer:   Flush and close a writer when present
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class TensorBoardPlan:
    """Resolved TensorBoard writer setup plan"""

    enabled    : bool  # Field: whether this optional feature path is enabled
    unavailable: bool  # Field: boolean value indicating the unavailable state for tensor board plan
    log_dir    : str | None  # Field: filesystem location for log dir
    message    : str | None  # Field: human-readable status or error detail


def resolve_tensorboard_plan(
    *,
    tensorboard_dir: str,  # Param: directory where TensorBoard events are written
    log_jsonl      : str,  # Param: JSONL file that receives structured training or eval rows
    tb_available   : bool,  # Param: boolean input controlling tb available
) -> TensorBoardPlan:
    """Resolve TensorBoard directory and availability message"""
    tb_dir_arg = (tensorboard_dir or "").strip()
    if tb_available and tb_dir_arg.lower() != "off":
        tb_dir = tb_dir_arg or os.path.join(os.path.dirname(log_jsonl) or ".", "tb")
        return TensorBoardPlan(
            enabled=True,
            unavailable=False,
            log_dir=tb_dir,
            message=f"tensorboard_dir={tb_dir}",
        )
    if not tb_available:
        return TensorBoardPlan(
            enabled=False,
            unavailable=True,
            log_dir=None,
            message="tensorboard: skipped (torch.utils.tensorboard unavailable)",
        )
    return TensorBoardPlan(enabled=False, unavailable=False, log_dir=None, message=None)


def create_tensorboard_writer(plan: TensorBoardPlan, summary_writer_cls: type | None) -> Any:
    """Create a TensorBoard writer from a resolved plan

    Steps:
    - Resolve inputs for `create_tensorboard_writer` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not plan.enabled:
        return None
    if summary_writer_cls is None:
        raise RuntimeError("TensorBoard plan is enabled but SummaryWriter is unavailable")
    assert plan.log_dir is not None
    os.makedirs(plan.log_dir, exist_ok=True)
    return summary_writer_cls(log_dir=plan.log_dir)


def close_tensorboard_writer(writer: Any) -> bool:
    """Flush and close a writer when present"""
    if writer is None:
        return False
    if hasattr(writer, "flush"):
        writer.flush()
    if hasattr(writer, "close"):
        writer.close()
    return True
