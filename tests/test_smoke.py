"""Fast end-to-end check on a synthetic series (about one minute on CPU): python -m pytest -q tests"""
import numpy as np

from benchmark import baselines, evaluation, models, search, synthetic


def test_pipeline_runs():
    df = synthetic.synth("Seasonal + AR", n=400)
    hp = dict(models.INTERIOR_HP, units=8, timestep=10)
    out = models.train_predict(hp, seed=42, df=df, feats=["Close"], epochs=2)
    assert len(out["pred"]) == len(out["y"]) == len(out["rw"])
    dm, p = evaluation.dm_test(out["y"] - out["pred"], out["y"] - out["rw"])
    assert np.isfinite(dm) and 0 <= p <= 1
    assert np.allclose(baselines.random_walk(df.Close.values), out["rw"])


def test_search_runs():
    df = synthetic.synth("Random walk", n=400)
    lb, ub = search.bounds("tight")
    search.FIT_EPOCHS = 1
    F = search.Evaluator(lambda hp: search.fitness(dict(hp, units=8), df, ["Close"], mode="nested"))
    search.ga_paper(F, seed=42, lb=lb, ub=ub, gens=2, pop=4)
    hp, val = F.best_at(8)
    assert F.calls == 8 and val > 0 and set(hp) >= {"units", "lr", "timestep"}


def test_holm_and_mase():
    adj = evaluation.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluation.mase(y, y + 1, y) == 1.0
