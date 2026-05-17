"""

Human-readable training progress line formatting

File map:

ProgressLineSummary:    Scalar fields printed on periodic training progress lines
_parse_key_value_bits:  Parse progress bit strings into ordered key/value text fields
_pop_fields:            Pop selected keys from a field map and format them as key=value tokens
_pop_prefixed_fields:   Pop remaining fields whose names match a prefix group
format_progress_line:   Format one periodic trainer progress line
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressLineSummary:
    """Scalar fields printed on periodic training progress lines"""

    global_step       : int  # training step associated with this record or action
    episode_idx       : int  # training episode index associated with this record
    episode_step      : int  # per-env step count inside the current episode
    phase_name        : str  # string phase name value used by progress line summary
    action_source     : str  # string action source value used by progress line summary
    reward            : float  # reward tensor or scalar produced by the environment step
    tip               : float  # floating-point tip value used by progress line summary
    palm              : float  # floating-point palm value used by progress line summary
    palm_height_error : float  # floating-point palm height error value used by progress line summary
    orient_deg        : float  # floating-point orient deg value used by progress line summary
    thumb_err         : float  # floating-point thumb err value used by progress line summary
    idx_err           : float  # floating-point idx err value used by progress line summary
    thumb_target_delta: tuple[float, float, float]  # floating-point thumb target delta value used by progress line summary
    index_target_delta: tuple[float, float, float]  # floating-point index target delta value used by progress line summary
    done_envs         : int  # integer done envs value tracked by progress line summary
    geometry_frame    : str  # string geometry frame value used by progress line summary
    replay_size       : int  # configured or observed replay-buffer size
    assist_mix        : float  # floating-point assist mix value used by progress line summary
    assist_arm_mix    : float  # floating-point assist arm mix value used by progress line summary
    assist_finger_mix : float  # floating-point assist finger mix value used by progress line summary
    progress_bits     : str = ""  # string progress bits value used by progress line summary
    stage_bits        : str = ""  # string stage bits value used by progress line summary
    done_bits         : str = ""  # string done bits value used by progress line summary
    update_bits       : str = ""  # string update bits value used by progress line summary
    reward_term_bits  : str = ""  # string reward term bits value used by progress line summary


def _parse_key_value_bits(*bit_strings: str) -> dict[str, str]:
    """Parse progress bit strings into ordered key/value text fields."""
    fields: dict[str, str] = {}
    for bit_string in bit_strings:
        for token in str(bit_string or "").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key and key not in fields:
                fields[key] = value
    return fields


def _pop_fields(fields: dict[str, str], keys: tuple[str, ...]) -> list[str]:
    """Pop selected keys from a field map and format them as key=value tokens."""
    tokens: list[str] = []
    for key in keys:
        value = fields.pop(key, None)
        if value is not None:
            tokens.append(f"{key}={value}")
    return tokens


def _pop_prefixed_fields(fields: dict[str, str], prefixes: tuple[str, ...]) -> list[str]:
    """Pop remaining fields whose names match a prefix group."""
    keys = tuple(key for key in fields if key.startswith(prefixes))
    return _pop_fields(fields, keys)


def format_progress_line(summary: ProgressLineSummary) -> str:
    """Format one periodic trainer progress line

    Steps:
    - Resolve inputs for `format_progress_line` and capture local config or state
    - Run guard branches and early exits before heavier work
    - Build intermediate tensors, records, commands, or helper objects in dependency order
    - Apply side effects such as state mutation, file IO, env calls, or optimizer updates when present
    - Return computed output or leave updated state for caller use
    """
    thumb_dx, thumb_dy, thumb_dz = summary.thumb_target_delta
    index_dx, index_dy, index_dz = summary.index_target_delta
    fields = {
        "step"     : f"{int(summary.global_step):05d}",
        "ep"       : f"{int(summary.episode_idx):03d}",
        "t"        : f"{int(summary.episode_step):03d}",
        "phase"    : str(summary.phase_name),
        "src"      : str(summary.action_source),
        "reward"   : f"{float(summary.reward):+.3f}",
        "tip"      : f"{float(summary.tip):.3f}",
        "palm"     : f"{float(summary.palm):.3f}",
        "palmzerr" : f"{float(summary.palm_height_error):.3f}",
        "orient"   : f"{float(summary.orient_deg):.1f}",
        "thumb_err": f"{float(summary.thumb_err):.3f}",
        "idx_err"  : f"{float(summary.idx_err):.3f}",
        "thumb_tdx": f"{float(thumb_dx):+.3f}",
        "thumb_tdy": f"{float(thumb_dy):+.3f}",
        "thumb_tdz": f"{float(thumb_dz):+.3f}",
        "idx_tdx"  : f"{float(index_dx):+.3f}",
        "idx_tdy"  : f"{float(index_dy):+.3f}",
        "idx_tdz"  : f"{float(index_dz):+.3f}",
        "done_envs": str(int(summary.done_envs)),
        "frame"    : str(summary.geometry_frame),
        "replay"   : str(int(summary.replay_size)),
        "assist": (
            f"{float(summary.assist_mix):.2f}/"
            f"{float(summary.assist_arm_mix):.2f}/"
            f"{float(summary.assist_finger_mix):.2f}"
        ),
    }
    fields.update(
        _parse_key_value_bits(
            summary.progress_bits,
            summary.stage_bits,
            summary.done_bits,
            summary.update_bits,
            summary.reward_term_bits,
        )
    )
    tokens: list[str] = []
    tokens += _pop_fields(
        fields,
        (
            "step",
            "ep",
            "t",
            "stage",
            "assist",
            "reward",
            "align",
            "align_angle",
            "lift_latch",
            "thumb_c",
            "idx_c",
            "lift",
            "blk_disp",
            "contact",
            "strict",
            "unlock_prog",
            "orient",
            "phase",
            "s1_rate",
            "s2_rate",
            "q1",
            "q2",
            "tgt",
            "c1",
            "c2",
            "actor",
            "bc",
        ),
    )
    tokens += _pop_fields(fields, ("src",))
    tokens += _pop_fields(
        fields,
        (
            "tip",
            "palm",
            "palmzerr",
            "thumb_err",
            "idx_err",
            "thumb_tdx",
            "thumb_tdy",
            "thumb_tdz",
            "idx_tdx",
            "idx_tdy",
            "idx_tdz",
        ),
    )
    tokens += _pop_fields(
        fields,
        (
            "best_stage",
            "reach_hold",
            "align_hold",
            "pose_ready",
            "pose_hold",
            "pose_shell",
            "palm_c",
            "palm_ch",
            "s2_age",
            "eff_unlock",
            "raw_unlock",
        ),
    )
    tokens += _pop_fields(fields, ("opp", "lift_cnt", "lift_prog", "zblend", "xyfix", "hold_rel", "ik_posonly"))
    tokens += _pop_fields(fields, ("finger_hold_gate", "prehold_servo", "align_line_z", "align_servo_q", "align_servo_on"))
    tokens += _pop_fields(fields, ("pocket_sweep_q", "pocket_score"))
    tokens += _pop_fields(
        fields,
        (
            "finger_cent_live",
            "finger_cent",
            "finger_cent_xy",
            "finger_cent_max_xy",
            "finger_cent_z",
            "finger_cent_ang",
            "finger_cent_hold",
        ),
    )
    tokens += _pop_prefixed_fields(fields, ("tct_",))
    tokens += _pop_fields(fields, ("success", "block_drift", "done", "shell_drift", "off_table", "done_envs", "frame", "replay"))
    tokens += _pop_fields(fields, ("a2t_arm", "a2t_f", "rterm"))
    tokens += _pop_fields(fields, tuple(fields))
    return " ".join(tokens)
