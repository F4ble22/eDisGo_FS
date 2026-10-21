from __future__ import annotations

import os

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

from IPython.display import display

from edisgo import EDisGo
from edisgo.io.db import engine as egon_engine

# ---- central parameters -------------------------------------------------
# Directory the example grid is downloaded into. Override via the EDISGO_DATA_DIR
# environment variable to point at your own data location.
DATA_DIR = Path(os.environ.get("EDISGO_DATA_DIR", Path.home() / ".edisgo"))
GRID_ID = "ding0_test_network_3"  # small eDisGo example grid (egon-data format)
SCENARIO = "eGon2035"

grid_path = DATA_DIR / GRID_ID


def download_example_grid(target: Path) -> None:
    """Download the eDisGo example ding0 grid (egon-data format) from GitHub."""
    target.mkdir(parents=True, exist_ok=True)
    base = (
        "https://raw.githubusercontent.com/openego/eDisGo/dev/"
        "tests/data/ding0_test_network_3"
    )
    filenames = [
        "buses",
        "generators",
        "lines",
        "loads",
        "network",
        "switches",
        "transformers",
        "transformers_hvmv",
    ]
    for name in filenames:
        resp = requests.get(f"{base}/{name}.csv")
        resp.raise_for_status()
        (target / f"{name}.csv").write_bytes(resp.content)


# Fetch the example grid on first run. Replace ``grid_path`` with your own ding0
# grid directory (egon-data format) to analyse a different grid.
if not grid_path.is_dir():
    download_example_grid(grid_path)

assert grid_path.is_dir(), f"Grid folder missing: {grid_path}"
print("Grid path:", grid_path)

# Build the OEDB engine (reads the token automatically from the package config).
# ssh=False -> public OEDB via token. Only actually needed from Stage 3 on.
engine = egon_engine(path=None, ssh=False, token=None)
# Print only the host, NOT engine.url (which would contain the token in clear text):
print("OEDB engine connected to:", engine.url.host)
