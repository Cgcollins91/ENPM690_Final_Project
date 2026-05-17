"""Compatibility wrapper for training.native.native_entrypoint"""

from __future__ import annotations

from .native.native_entrypoint import *  # noqa: F401,F403

if __name__ == "__main__":
    from .core.entrypoint import main

    raise SystemExit(main())
