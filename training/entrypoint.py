"""Compatibility wrapper for training.core.entrypoint"""

from __future__ import annotations

from .core.entrypoint import *  # noqa: F401,F403

if __name__ == "__main__":
    from .core.entrypoint import main

    raise SystemExit(main())
