"""Random walk with drift as an additional baseline on the synthetic battery (Table VII, Appendix).

A constant, stable drift is predictable in principle, so the "Random walk + drift" process is not a process
without predictable mean structure. This script scores the drift forecast of Eqs. (13)-(14), estimated on the
training part, against the plain random walk on all 18 realizations (seeds 7, 8, 9). It needs only numpy, pandas
and scipy (no TensorFlow):

    PYTHONPATH=src python scripts/synthetic_drift_baseline.py

Output: results/TableVII_drift_baseline.csv

To compare the LSTM with the drift baseline, pass the seed-averaged LSTM forecast of a realization to
``evaluation.dm_test(y - lstm, y - drift)``; the arrays are produced in notebooks/03_E6_...ipynb (Table VII cell).
"""
import pathlib

import numpy as np
import pandas as pd

from benchmark import baselines, evaluation, synthetic

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
rows = []
for kind in synthetic.PROCESSES:
    for rz in range(3):
        y = synthetic.synth(kind, seed=7 + rz).Close.values
        n_tr = baselines.split_point(y)
        yt, rw, dr = y[n_tr:], baselines.random_walk(y), baselines.random_walk_drift(y)
        dm, p = evaluation.dm_test(yt - dr, yt - rw)          # DM > 0: drift forecast is less accurate than RW
        rows.append(dict(series=kind, realization=rz, drift_per_day=float(np.mean(np.diff(y[:n_tr]))),
                         MAPE_RW=evaluation.mape(yt, rw), MAPE_drift=evaluation.mape(yt, dr),
                         MASE_drift=evaluation.mase(yt, dr, y[:n_tr]), DM_drift_vs_RW=dm, p=p))
out = pd.DataFrame(rows).round(4)
out.to_csv(RES / "TableVII_drift_baseline.csv", index=False)
print(out.to_string(index=False))
