"""Persistence and statistical baselines, evaluated on the same test window as the learned models."""
from __future__ import annotations

import numpy as np

TRAIN_FRAC = 0.8


def split_point(y) -> int:
    return int(len(y) * TRAIN_FRAC)


def random_walk(y):
    """One-step random walk: the forecast for day t is the price of day t-1."""
    y = np.asarray(y, float)
    n_tr = split_point(y)
    return y[n_tr - 1:-1]


def random_walk_drift(y, h: int = 1):
    """Random walk plus h times the mean daily price change of the training period (Eqs. 13-14)."""
    y = np.asarray(y, float)
    n_tr = split_point(y)
    drift = np.mean(np.diff(y[:n_tr]))
    return y[n_tr - 1:-1] + h * drift


def arima_rolling(y, order=(1, 1, 1)):
    """ARIMA refitted at every origin of the test window (rolling one-step forecasts)."""
    from statsmodels.tsa.arima.model import ARIMA
    y = np.asarray(y, float)
    n_tr = split_point(y)
    return np.array([float(ARIMA(y[:t], order=order).fit().forecast(1)[0]) for t in range(n_tr, len(y))])


def ar_on_differences(y, p: int):
    """Linear AR(p) on first differences, fitted by OLS on the training part (used for non-financial series)."""
    y = np.asarray(y, float)
    n_tr = split_point(y)
    d = np.diff(y)
    X = np.array([d[i - p:i][::-1] for i in range(p, len(d))])
    Y = d[p:]
    idx = np.arange(p, len(d)) + 1
    A = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(A[idx < n_tr], Y[idx < n_tr], rcond=None)[0]
    te = idx >= n_tr
    return y[idx[te] - 1] + A[te] @ w
