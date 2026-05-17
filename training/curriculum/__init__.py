"""Run-level curriculum phase definitions for topdown training."""

from training.curriculum.phases import (
    build_default_phases,
    build_pure_rl_v35_phases,
    build_reanchor_recovery_phases,
)
from training.curriculum.spec import PhaseSpec

__all__ = [
    "PhaseSpec",
    "build_default_phases",
    "build_pure_rl_v35_phases",
    "build_reanchor_recovery_phases",
]
