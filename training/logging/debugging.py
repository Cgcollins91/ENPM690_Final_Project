"""

Debug trace and nonfinite update helpers

This module provides helper functions and data structures for emitting debug traces on signals and nonfinite updates,
used by the training loop and signal handler installation

File map:

EVAL_TRACE_PREFIX:              Define eval trace prefix constant
DEBUG_SIGNAL_NAMES:             Define debug signal names constant
eval_trace_line:                Format one eval trace line
emit_eval_trace:                Emit one eval trace line when enabled
signal_name:                    Return a display name for a process signal
signal_received_trace_line:     Format one signal trace line
signal_exit_code:               Return conventional process exit code for a signal
handle_debug_signal:            Emit signal diagnostics and exit with conventional signal code
available_debug_signal_names:   Return configured signal names that exist on this platform
install_debug_signal_handlers:  Install debug signal handlers and return installed signal names
actor_loss_is_finite:           Return whether an actor loss is absent or finite
_float_metric:                  Handle float metric logic
nonfinite_actor_update_line:    Format compact nonfinite actor update diagnostics
nonfinite_actor_error_message:  Format an actor nonfinite exception message
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Iterable
from collections.abc import Mapping
import math
import signal
from types import FrameType

import torch


EVAL_TRACE_PREFIX = "[eval-trace]"
DEBUG_SIGNAL_NAMES = ("SIGTERM", "SIGINT", "SIGHUP", "SIGUSR1", "SIGUSR2")


def eval_trace_line(message: str) -> str:
    """Format one eval trace line"""
    return f"{EVAL_TRACE_PREFIX} {message}"


def emit_eval_trace(
    message: str,                           # Param: string input for message
    *,
    enabled: bool,  # Param: boolean input controlling enabled
    emit   : Callable[[str], object] = print,  # Param: callback used to compute or fetch emit
) -> bool:
    """Emit one eval trace line when enabled"""
    if not enabled:
        return False
    emit(eval_trace_line(message))
    return True


def signal_name(signum: int) -> str:
    """Return a display name for a process signal"""
    try:
        return signal.Signals(int(signum)).name
    except Exception:
        return str(signum)


def signal_received_trace_line(signum: int) -> str:
    """Format one signal trace line"""
    return eval_trace_line(f"signal_received signum={int(signum)} name={signal_name(signum)}")


def signal_exit_code(signum: int) -> int:
    """Return conventional process exit code for a signal"""
    return 128 + int(signum)


def handle_debug_signal(
    signum: int,  # Param: integer input for signum
    _frame: FrameType | None = None,  # Param: input value used as frame
    *,
    emit: Callable[[str], object] = print,  # Param: callback used to compute or fetch emit
) -> None:
    """Emit signal diagnostics and exit with conventional signal code"""
    emit(signal_received_trace_line(signum))
    raise SystemExit(signal_exit_code(signum))


def available_debug_signal_names() -> tuple[str, ...]:
    """Return configured signal names that exist on this platform"""
    return tuple(name for name in DEBUG_SIGNAL_NAMES if getattr(signal, name, None) is not None)


def install_debug_signal_handlers(
    *,
    enabled      : bool,  # Param: boolean input controlling enabled
    signal_names : Iterable[str]                                                    = DEBUG_SIGNAL_NAMES,  # Param: ordered candidate names used to resolve signal
    emit         : Callable[[str], object]                                          = print,  # Param: callback used to compute or fetch emit
    register_exit: Callable[[Callable[[], object]], object] | None                  = atexit.register,  # Param: callback used to compute or fetch register exit
    signal_setter: Callable[[int, Callable[[int, FrameType | None], None]], object] = signal.signal,  # Param: callback used to compute or fetch signal setter
) -> tuple[str, ...]:
    """Install debug signal handlers and return installed signal names

    Steps:
    - Resolve inputs for `install_debug_signal_handlers` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    if not enabled:
        return ()
    if register_exit is not None:
        register_exit(lambda: emit(eval_trace_line("atexit normal_exit")))

    installed: list[str] = []

    def _handle_signal(signum: int, frame: FrameType | None) -> None:
        handle_debug_signal(signum, frame, emit=emit)

    for sig_name in signal_names:
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal_setter(sig, _handle_signal)
        except Exception:
            continue
        installed.append(sig_name)
    return tuple(installed)


def actor_loss_is_finite(actor_loss: torch.Tensor | None) -> bool:
    """Return whether an actor loss is absent or finite"""
    if actor_loss is None:
        return True
    return bool(torch.isfinite(actor_loss.detach()).all().item())


def _float_metric(metrics: Mapping[str, object], key: str, default: float = math.nan) -> float:
    value = metrics.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().item())
    return float(default)


def nonfinite_actor_update_line(
    *,
    train_step   : int,  # Param: step count used for train step
    progress_step: int,  # Param: step count used for progress step
    actor_loss   : torch.Tensor | float,  # Param: tensor input carrying actor loss values
    metrics      : Mapping[str, object],  # Param: metric mapping emitted with the result or log row
) -> str:
    """Format compact nonfinite actor update diagnostics"""
    actor_loss_value = float(actor_loss.detach().item()) if torch.is_tensor(actor_loss) else float(actor_loss)
    return (
        "nonfinite_actor_update "
        f"train_step={int(train_step)} progress_step={int(progress_step)} "
        f"actor_loss={actor_loss_value} "
        f"actor_q_mean={_float_metric(metrics, 'actor_q_mean')} "
        f"actor_q_finite_frac={_float_metric(metrics, 'actor_q_finite_frac'):.6f} "
        f"actor_action_finite_frac={_float_metric(metrics, 'actor_action_finite_frac'):.6f} "
        f"actor_raw_finite_frac={_float_metric(metrics, 'actor_raw_finite_frac'):.6f} "
        f"obs_finite_frac={_float_metric(metrics, 'obs_finite_frac'):.6f} "
        f"reward_finite_frac={_float_metric(metrics, 'reward_finite_frac'):.6f} "
        f"target_finite_frac={_float_metric(metrics, 'target_finite_frac'):.6f} "
        f"critic1_param_finite_frac={_float_metric(metrics, 'critic1_param_finite_frac'):.6f} "
        f"actor_param_finite_frac={_float_metric(metrics, 'actor_param_finite_frac'):.6f}"
    )


def nonfinite_actor_error_message(*, train_step: int, progress_step: int, backend: str = "custom") -> str:
    """Format an actor nonfinite exception message"""
    prefix = "nonfinite actor_loss"
    if backend == "upstream_fasttd3":
        prefix = "nonfinite upstream FastTD3 actor_loss"
    if backend == "upstream_fasttd3":
        return f"{prefix} at train_step={int(train_step)}"
    return f"{prefix} at train_step={int(train_step)} progress_step={int(progress_step)}"
