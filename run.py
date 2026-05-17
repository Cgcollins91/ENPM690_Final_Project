"""Small Python entrypoint for the ENPM690 final-project launcher.

This keeps the runnable surface out of shell scripts.  It simply places the
local ``src`` directory on ``sys.path`` and delegates to the package CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from enpm690_final_project.launch import main


if __name__ == "__main__":
    raise SystemExit(main())
