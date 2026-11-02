"""Download and cache the data used in the paper as CSV files in data/ (requires Internet).

    python scripts/download_data.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from benchmark import data  # noqa: E402

out = pathlib.Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)
for tk in data.STOCKS:
    d = data.load_stock(tk)
    d.to_csv(out / f"{tk.replace('.JK', '')}_features.csv", index=False)
    print(f"{tk}: {len(d)} rows, {d.Date.min().date()} to {d.Date.max().date()}")
