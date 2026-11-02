"""Accuracy measures and significance tests."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def mase(y, p, y_train) -> float:
    """MAE scaled by the in-sample mean absolute one-step change (naive forecast)."""
    return mae(y, p) / float(np.mean(np.abs(np.diff(np.asarray(y_train, float)))))


def dm_test(e_model, e_base, h: int = 1):
    """Diebold-Mariano test with absolute-error loss, two-sided, Harvey-Leybourne-Newbold correction.

    Returns (DM, p). DM > 0 means the first forecast (the model) has the larger loss."""
    d = np.abs(np.asarray(e_model, float)) - np.abs(np.asarray(e_base, float))
    n = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    dm = dbar / math.sqrt(var) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def ci95(values):
    a = np.asarray(values, float)
    half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))
    return a.mean() - half, a.mean() + half
