#!/usr/bin/env python3
"""Backward-compatible path. The entry point is run_identification.py."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "run_identification.py"
    runpy.run_path(str(target), run_name="__main__")
