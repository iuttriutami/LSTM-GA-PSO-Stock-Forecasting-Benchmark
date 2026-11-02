"""Export the per-evaluation search history of the main (paper) protocol to results/SearchHistory_E7a.csv.

Notebooks 04a/04b keep, for every GA/PSO run, the list ``hist`` of tuples
(calls, distinct, best_validation_mse, best_config_key) in ``cache_e7/*.pkl``. These caches are not committed
(``*.pkl`` is git-ignored), so this script turns them into a small CSV that is. Point it at the cache folder
(for example the notebook output on Kaggle or ./out_*/cache_e7):

    python scripts/export_search_history.py path/to/cache_e7
    python scripts/make_fig_convergence_300.py        # now draws the full per-evaluation curve

The optimizer and run index are read from the file name when it contains 'GA'/'PSO' and 'r<k>' (the notebooks
print names such as 'search GA r0'); otherwise the file stem is used as run id. Check the printed summary.
"""
import glob
import pathlib
import pickle
import re
import sys

import pandas as pd

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cache_e7")
rows = []
for f in sorted(glob.glob(str(src / "*.pkl"))):
    try:
        res = pickle.load(open(f, "rb"))
    except Exception:
        continue
    if not isinstance(res, dict) or "hist" not in res:
        continue
    stem = pathlib.Path(f).stem
    opt = "PSO" if re.search("pso", stem, re.I) else "GA" if re.search(r"(^|[^a-z])ga", stem, re.I) else "unknown"
    m = re.search(r"r(\d+)", stem)
    for calls, distinct, best, _ in res["hist"]:
        rows.append(dict(optimizer=opt, run=int(m.group(1)) if m else stem, calls=calls, distinct=distinct, best_val_mse=best))
out = pd.DataFrame(rows)
if out.empty:
    sys.exit("no search histories found in " + str(src))
out.to_csv(pathlib.Path(__file__).resolve().parents[1] / "results" / "SearchHistory_E7a.csv", index=False)
print(out.groupby(["optimizer", "run"]).agg(calls=("calls", "max"), distinct=("distinct", "max")))
