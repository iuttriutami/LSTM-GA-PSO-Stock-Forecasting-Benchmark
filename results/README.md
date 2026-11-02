# Results

Every CSV is the exact output used in the paper, named by table number. `E4`–`E7` are internal experiment labels.

| File | Paper item |
|---|---|
| `TableI_one_step.csv`, `TableII_multistep_all_horizons.csv`, `TableIII_cross_stock.csv`, `Ablation_feature_sets.csv`, `manuscript_numbers.json` | Tables I–III, feature ablation |
| `TableIV_E4_fixed_configurations.csv`, `TableIV_E5_tightened_space.csv` | Table IV |
| `TableV_E7a_paper_protocol_*.csv`, `TableV_E7a_budget.csv` | Table V, rows "Main, 300" (checkpoints at 50, 100, 300 candidate positions) |
| `TableV_E6a_*.csv` | Table V, rows "Original, 100" and "Tight, 100"; Fig. 6 |
| `TableV_budget_accounting.csv` | Evaluation budget: candidate positions, unique configurations, fitness trainings |
| `TableE6b_nested_*.csv`, `TableE6c_DM_all_models.csv` | Nested vs shared validation (robustness check); DM for all models |
| `TableVI_E7b_equal_budget_*.csv` | Table VI |
| `TableVII_E6d_synthetic_with_AR_benchmark.csv` | Table VII (LSTM vs plain random walk) |
| `TableVII_drift_baseline.csv` | Random walk with drift vs random walk on the 18 synthetic realizations |
| `TableVIII_E6e_cross_market.csv`, `TableVIII_E7c_nonfinancial.csv` | Table VIII |

## Evaluation budget

* **Candidate positions**: 300 per search under the main protocol (population or swarm of 20 × 15 generations or
  iterations), counted whether or not a candidate repeats an earlier one.
* **Unique configurations**: distinct decoded hyperparameter sets among those positions (`search.config_key`).
* **Fitness trainings**: networks actually trained during the search. Fitness is memoised (`search.Evaluator`), so one
  training is run per unique configuration, and repeated candidates are not retrained.

Under the main protocol the mean number of unique configurations (= fitness trainings) per run is 226.4 for GA
(range 222–229) and 289.8 for PSO (range 279–298). A distinct-budget search finishes the generation or iteration in which the budget is reached, so it trains 100–119 unique
configurations in total (`TableV_budget_accounting.csv`). The reported configuration is the incumbent after exactly
100 (or 25, 50) unique configurations (`hist[b-1]` in notebook 03), so the 0–19 extra trainings do not affect selection.
