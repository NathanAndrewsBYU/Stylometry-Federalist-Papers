"""Launcher: puts src/ on the import path and runs stylo.cli.

Run from anywhere:  python scripts/run_screen.py --help
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from stylo.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
