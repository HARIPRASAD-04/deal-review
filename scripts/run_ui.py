"""Launcher script for the Streamlit UI — Module 8.

Usage:
    python scripts/run_ui.py
"""

from __future__ import annotations

import pathlib
import sys
from streamlit.web import cli as stcli

APP_PATH = pathlib.Path(__file__).parent.parent / "app" / "ui" / "streamlit_app.py"


def main() -> None:
    sys.argv = ["streamlit", "run", str(APP_PATH)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
