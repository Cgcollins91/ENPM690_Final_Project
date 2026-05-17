"""Package entrypoint for the compatibility training launcher"""

from __future__ import annotations

from .core.entrypoint import main


if __name__ == "__main__":
    raise SystemExit(main())
