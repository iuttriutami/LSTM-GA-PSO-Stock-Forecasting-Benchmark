# Data

Raw prices are not redistributed in this repository. They are downloaded from Yahoo Finance on demand:

* SIDO.JK, BBCA.JK, BMRI.JK, TLKM.JK, ASII.JK, UNVR.JK: daily OHLCV, 2019-07-01 to 2025-07-01, adjusted for
  dividends and splits
* ^JKSE (Jakarta Composite Index) and USDIDR=X (USD/IDR): daily close, same period
* ^GSPC, ^N225, BTC-USD, EURUSD=X (2019-2025) and ^GSPC, ^JKSE (2012-2018): daily close (Table VIII)

Run `PYTHONPATH=src python scripts/download_data.py` to write `<STOCK>_features.csv` with the 16 features used in
the paper (OHLCV, 9 technical indicators, JCI, USD/IDR) after warm-up rows are removed. The non-financial series
(weekly Mauna Loa CO2, monthly El Nino SST) ship with `statsmodels` (`sm.datasets.co2`, `sm.datasets.elnino`).

Check value: SIDO has 1,430 rows after warm-up, and the one-step random-walk MAPE on the test window (last 20%) is
1.368%.
