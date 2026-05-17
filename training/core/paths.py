"""

Import-safe filesystem path helpers for native training

File map:

project_root_from_training_package:  Resolve the project root from the installed training package path
"""

from __future__ import annotations

import os


def project_root_from_training_package() -> str:
    """Resolve the project root from the installed training package path"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
