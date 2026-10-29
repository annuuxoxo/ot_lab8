"""Shared helpers: seeding, timing, and saving results/plots."""
import os
import random
import time
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def set_seed(seed: int = 0) -> None:
    """Fix every random number generator we use, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer():
    """Usage: with timer() as t: ...; then t['s'] holds elapsed seconds."""
    out = {}
    start = time.perf_counter()
    yield out
    out["s"] = time.perf_counter() - start


def _track_dir(track: str) -> str:
    path = os.path.join(RESULTS, track)
    os.makedirs(path, exist_ok=True)
    return path


def save_results(df, track: str, name: str) -> str:
    """Write a pandas DataFrame to results/<track>/<name>.csv and return the path."""
    path = os.path.join(_track_dir(track), f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def save_plot(fig, track: str, name: str) -> str:
    """Save a matplotlib figure to results/<track>/<name>.png and return the path."""
    path = os.path.join(_track_dir(track), f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
