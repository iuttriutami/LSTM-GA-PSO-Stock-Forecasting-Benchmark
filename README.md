# LSTM-GA-PSO Stock Forecasting Benchmark

Code, notebooks and results for the paper

> I. T. Utami, P. Kartikasari, W. Pujiati, and R. Arisandi, "Visual Benchmarking of Metaheuristic Optimized Deep Learning Against Random Walk Stock Forecasts," submitted to *JOIV: International Journal on Informatics Visualization*.

The repository implements a baseline-centered benchmarking protocol. Learned forecasters (LSTM tuned by a Genetic
Algorithm or Particle Swarm Optimization, plus GRU, BiLSTM, TCN, Transformer and MLP) are compared with a random
walk, a random walk with drift and a rolling ARIMA(1,1,1). The comparison uses repeated runs, Diebold–Mariano tests
with the Harvey–Leybourne–Newbold correction and Holm–Bonferroni adjustment, multi-horizon forecasts, cross-stock
transfer, synthetic control series and series from other markets and domains.

**Main result.** On SIDO (Indonesia Stock Exchange, 2019–2025), no learned model beats the random walk (one-step
MAPE 1.37%). This holds for every configuration tested, for repeated GA/PSO searches under the paper's own
300-evaluation protocol, and for all six architectures tuned with an equal budget and nested validation. It also
holds for five other IDX stocks and six series from other markets and periods. On series with real structure
(synthetic seasonal series, monthly El Niño temperature), learned models do beat persistence, so the evaluated cases
suggest that the protocol can distinguish improvements over persistence under several controlled and real-world
conditions.

## Repository structure

```
src/benchmark/        reusable modules
  data.py             download (Yahoo Finance with fallback) and 9 causal technical indicators
  models.py           LSTM, GRU, BiLSTM, TCN, Transformer, MLP; windowing; final training protocol
  baselines.py        random walk, random walk with drift, rolling ARIMA(1,1,1), AR on differences
  search.py           GA and PSO (paper protocol and distinct-budget variant); shared or nested fitness
  evaluation.py       MAPE, RMSE, MAE, MASE, Diebold-Mariano (HLN), Holm-Bonferroni, 95% CI
  synthetic.py        synthetic processes with known structure
notebooks/            one self-contained Kaggle notebook per experiment (see table below)
results/              every table of the paper as CSV, named by table number (see results/README.md)
figures/              every figure of the paper (600 dpi PNG)
scripts/              data download; Fig. 1 and Fig. 5 drawn from results/
tests/                fast smoke test on a synthetic series
```

## Reproducing the paper

Each notebook is self-contained: it downloads the data, builds the features, trains the models, and writes the
CSV and PNG files listed below. The notebooks were written for Kaggle. Use **Accelerator: None** (CPU) except where noted,
turn **Internet on**, and run with *Save Version → Save & Run All (Commit)*. Every expensive step is cached, and the
long notebooks stop cleanly before Kaggle's 12-hour limit. To continue, add the notebook's previous output as
input (*Add Input → Your Work*) and commit again.

| Paper item | Notebook | Output in `results/` / `figures/` | Approx. time |
|---|---|---|---|
| Table I, Table II, Table III, feature ablation, Figs. 2–4 | `01_main_benchmark_Tables_I-III_Figs_2-4` | `TableI_one_step.csv`, `TableII_multistep_all_horizons.csv`, `TableIII_cross_stock.csv`, `Ablation_feature_sets.csv`, `manuscript_numbers.json`, `Fig2–Fig4` | 6–9 h (GPU) |
| Table IV, Fig. 5 | `02_sensitivity_E4_E5_Table_IV_Fig_5` | `TableIV_E4_fixed_configurations.csv`, `TableIV_E5_tightened_space.csv`; Fig. 5 via `scripts/make_fig1_fig5.py` | 3–6 h |
| Table V (re-implemented search), Fig. 6, nested selection, DM for all models, Table VII, Table VIII (financial) | `03_E6_repeated_search_nested_DM_synthetic_markets_…` | `TableV_E6a_*`, `TableE6b_*`, `TableE6c_DM_all_models.csv`, `TableVII_E6d_synthetic_with_AR_benchmark.csv`, `TableVIII_E6e_cross_market.csv`, `Fig6_repeated_search.png` | 15–20 h (2 commits) |
| Table V (paper protocol, GA / PSO) | `04a_E7a_paper_protocol_GA`, `04b_E7a_paper_protocol_PSO` | `TableV_E7a_paper_protocol_runs.csv`, `TableV_E7a_paper_protocol_summary.csv`, `TableV_E7a_budget.csv`, `FigS1_E7a_paper_protocol_PSO.png` | 5–9 h each |
| Table VI (equal-budget tuning of six architectures) | `05a_E7b_…LSTM_GRU_BiLSTM`, `05b_E7b_…TCN_Transformer_MLP` | `TableVI_E7b_equal_budget_runs.csv`, `TableVI_E7b_equal_budget_summary.csv` | 4–9 h each |
| Table VIII (non-financial series) | `06_E7c_nonfinancial_series_Table_VIII` | `TableVIII_E7c_nonfinancial.csv` | ~30 min |
| Main protocol repeated with disjoint selection and early-stopping blocks (SIDO) | `07_main_protocol_nested_repeat_Table_I` (`MODE = 'QUICK'` or `'FULL'`) | `TableI_nested_repeat_runs.csv`, `TableI_nested_repeat_summary.csv`, `SearchHistory_nested.csv`, `Fig_nested_repeat.png` (written to the notebook output; copy to `results/` and `figures/`) | QUICK 4–7 h; FULL 12–18 h (2–3 commits); estimates |
| Fig. 1 | `scripts/make_fig1_fig5.py` | `Fig1_workflow.png` | seconds |
| Convergence of the main protocol to 300 candidate positions | `scripts/make_fig_convergence_300.py` (full curve: `scripts/export_search_history.py` first) | `FigS2_paper_protocol_convergence_300.png` | seconds |
| Random walk with drift baseline on the synthetic battery | `scripts/synthetic_drift_baseline.py` | `TableVII_drift_baseline.csv` | seconds |

### Evaluation budget

The main protocol uses 300 **candidate positions** per search (20 × 15), counted whether or not a candidate repeats an
earlier one. The number of **unique configurations** and of **fitness trainings** is lower because fitness is memoised
(mean 226.4 for GA and 289.8 for PSO per run; `results/TableV_budget_accounting.csv`). The distinct-budget searches
(Fig. 6, rows "Original, 100" and "Tight, 100" of Table V) finish the generation or iteration that reaches 100 unique
configurations (100–119 trained in total), but the reported configuration is the incumbent after exactly 100; check
with `scripts/verify_budget_selection.py`.

### Synthetic battery

The "Random walk + drift" process has a constant, stable drift, which is predictable in principle. It is therefore
scored against the random walk with drift (Eqs. 13–14) as well as the plain random walk; only the random walk and the
GARCH random walk are processes without predictable mean structure.

### Validation protocol

Table I and the main experiments use one validation block for both hyperparameter selection and early stopping.
Nested selection (disjoint blocks) is a robustness check of limited scope (six LSTM searches and the equal-budget
architecture comparison). It does not remove possible selection bias from the main experiments.

The CSV files in `results/` are the exact outputs used in the paper. Results from separate notebooks (for example
the two parts of Table VI) were merged, with Holm adjustment and Kruskal–Wallis tests recomputed over all six
architectures.

### Running locally

```bash
git clone https://github.com/iuttriutami/LSTM-GA-PSO-Stock-Forecasting-Benchmark.git
cd LSTM-GA-PSO-Stock-Forecasting-Benchmark
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests                # smoke test, about one minute on CPU (also run by CI)
PYTHONPATH=src python scripts/download_data.py          # writes data/*_features.csv
python scripts/make_fig1_fig5.py                        # redraws Fig. 1 and Fig. 5 from results/
```

The notebooks also run in Jupyter: open them from `notebooks/` and run all cells. Outputs are written to
`./out_*` instead of `/kaggle/working`.

## Data

Daily prices for SIDO.JK, BBCA.JK, BMRI.JK, TLKM.JK, ASII.JK and UNVR.JK, the Jakarta Composite Index (^JKSE) and
USD/IDR (USDIDR=X) are downloaded from Yahoo Finance for 1 July 2019 to 1 July 2025. Prices are adjusted for
dividends and splits. The external series (^GSPC, ^N225, BTC-USD, EURUSD=X) come from the same source. Weekly
Mauna Loa CO₂ and monthly El Niño sea surface temperature are distributed with `statsmodels`. Raw price data are
not redistributed here; see `data/README.md`.

## Notes on reproducibility

* Seeds are fixed for Python, NumPy and TensorFlow (42 + k). Deep learning training on different hardware (GPU or
  CPU) is not bit-for-bit deterministic, so individual runs can differ slightly from the CSV files. The
  conclusions rest on repeated runs and do not depend on single values.
* The Kaggle runs used Python 3.12–3.13, TensorFlow 2.18–2.20 and Keras 3. The main benchmark ran on an NVIDIA T4
  GPU. The robustness experiments ran on Kaggle GPU or CPU sessions.
* yfinance versions on hosted notebooks change often. Every notebook includes a fallback to the Yahoo chart
  endpoint, so the downloaded data are the same either way (SIDO: 1,430 rows after warm-up, random walk MAPE
  1.368%).
* Exact software versions of the main runs are listed in `ENVIRONMENT.md`.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT License, see `LICENSE`.
