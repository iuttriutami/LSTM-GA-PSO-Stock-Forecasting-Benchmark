"""Baseline-centered visual benchmarking of metaheuristic-tuned deep learning forecasters against a random walk.

Modules
-------
data        download (Yahoo Finance) and causal feature engineering
models      LSTM/GRU/BiLSTM/TCN/Transformer/MLP builder, windowing, final training protocol
baselines   random walk, random walk with drift, rolling ARIMA(1,1,1), AR on differences
search      GA and PSO (paper protocol and distinct-budget re-implementation), shared/nested fitness
evaluation  MAPE, RMSE, MAE, MASE, Diebold-Mariano (HLN), Holm-Bonferroni, 95% CI
synthetic   synthetic processes with known structure
"""
__version__ = "1.0.0"
