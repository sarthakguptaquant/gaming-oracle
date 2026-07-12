"""Make the package importable in tests without an editable install.

The repo was never `pip install -e`'d; the suite relies on the project root
being on sys.path. Adding it here keeps `uv run pytest` working from a clean
checkout (and matches how Cycle 1 was run).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
