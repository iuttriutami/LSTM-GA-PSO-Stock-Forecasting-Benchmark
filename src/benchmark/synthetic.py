"""Synthetic battery with known structure (Section III-E, Table VII).

"Random walk + drift" has a constant, stable drift (0.0008 per day on the log level), so its expected price change is
predictable in principle. It is not a process without predictable mean structure; the random walk with drift
(baselines.random_walk_drift, Eqs. 13-14) is the additional baseline for it (scripts/synthetic_drift_baseline.py).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PROCESSES = {
    "Random walk": "none", "Random walk + drift": "constant drift", "GARCH random walk": "none",
    "AR(1) returns, phi=0.3": "weak", "Mean-reverting level, phi=0.6": "moderate", "Seasonal + AR": "strong",
}


def synth(kind: str, n: int = 1449, seed: int = 7, start: str = "2019-07-01") -> pd.DataFrame:
    """Positive price series 100*exp(x). Realizations in the paper use seeds 7, 8, 9."""
    r = np.random.RandomState(seed)
    t = np.arange(n)
    if kind == "Random walk":
        x = np.cumsum(r.normal(0, .015, n))
    elif kind == "Random walk + drift":
        x = np.cumsum(r.normal(.0008, .015, n))
    elif kind == "GARCH random walk":
        e, s2 = np.zeros(n), np.full(n, 2e-4)
        for i in range(1, n):
            s2[i] = 1e-5 + .10 * e[i - 1] ** 2 + .85 * s2[i - 1]
            e[i] = r.normal(0, math.sqrt(s2[i]))
        x = np.cumsum(e)
    elif kind == "AR(1) returns, phi=0.3":
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = .3 * e[i - 1] + r.normal(0, .01)
        x = np.cumsum(e)
    elif kind == "Mean-reverting level, phi=0.6":
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = .6 * x[i - 1] + r.normal(0, .02)
    elif kind == "Seasonal + AR":
        a = np.zeros(n)
        for i in range(1, n):
            a[i] = .3 * a[i - 1] + r.normal(0, .005)
        x = .08 * np.sin(2 * np.pi * t / 5) + .05 * np.sin(2 * np.pi * t / 21) + a
    else:
        raise ValueError(kind)
    return pd.DataFrame({"Date": pd.date_range(start, periods=n, freq="B"), "Close": 100 * np.exp(x)})
