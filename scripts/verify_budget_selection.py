"""Check that the configuration reported at budget 100 was selected after exactly 100 unique configurations.

Notebook 03 stores, per search, ``hist``: one entry (best validation MSE so far, best key) for every NEW unique
configuration. A search finishes its current generation or iteration, so it ends with 100-119 unique
configurations, but the reported configuration is ``hist[99]`` (the incumbent after the 100th). This script
compares ``hist[b-1]`` with ``best_val_mse`` in results/TableV_E6a_repeated_search_runs.csv. No training is run.

    python scripts/verify_budget_selection.py path/to/cache_e6     # folder with srch_<space>_<opt>_<mode>_r<k>.pkl
"""
import glob, pathlib, pickle, re, sys
import pandas as pd

runs = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / "results" / "TableV_E6a_repeated_search_runs.csv")
ok = bad = 0
for f in sorted(glob.glob(str(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cache_e6") / "srch_*.pkl"))):
    m = re.match(r"srch_(\w+?)_(GA|PSO)_(shared|nested)_r(\d+)", pathlib.Path(f).stem)
    if not m:
        continue
    sp, opt, mode, r = m.group(1), m.group(2), m.group(3), int(m.group(4))
    h = pickle.load(open(f, "rb"))["hist"]
    for b in (25, 50, 100):
        row = runs[(runs.space == sp) & (runs.optimizer == opt) & (runs.selection == mode) & (runs.run == r) & (runs.budget == b)]
        if row.empty:
            continue
        match = abs(h[min(b, len(h)) - 1][0] - float(row.best_val_mse.iloc[0])) < 1e-5
        ok, bad = ok + match, bad + (not match)
    print(f"{pathlib.Path(f).stem}: {len(h)} unique configurations in total")
print(f"checkpoints consistent: {ok}, inconsistent: {bad}")
