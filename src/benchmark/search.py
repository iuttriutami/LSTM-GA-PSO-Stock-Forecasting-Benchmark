"""GA and PSO hyperparameter search.

Two implementations are provided:

* ``ga_paper`` / ``pso_paper`` - the protocol of Section II-C and Algorithms 1-2: elitism (best quarter), blend
  crossover with alpha = 0.25, uniform mutation with probability 0.3, constriction PSO (w = 0.729,
  c1 = c2 = 1.49445), population or swarm of 20 over 15 generations or iterations (300 evaluations, repeated
  candidates counted).
* ``ga_distinct`` / ``pso_distinct`` - the re-implementation used in the sensitivity analyses (uniform crossover,
  Gaussian mutation with rate 0.2, PSO re-initialisation of the worst quarter after 15 stalled iterations), with a
  budget of distinct candidates.

Fitness is the validation MSE after 20 epochs. ``mode='shared'`` scores candidates on the last 20% of the training
part (the block also used for early stopping, as in the paper); ``mode='nested'`` scores them on the 60-80% block so
that selection and early stopping use disjoint data.
"""
from __future__ import annotations

import math

import numpy as np

from .models import OPTIMIZERS, build_model, sequences, set_seed

POP = 20
FIT_EPOCHS = 20
SPACES = {
    "original": dict(units=(16, 160), n_layers=(1, 3), dropout=(0.10, 0.50), lr=(1e-4, 1e-3), batch_size=(16, 128), timestep=(10, 30)),
    "tight": dict(units=(32, 96), n_layers=(1, 2), dropout=(0.20, 0.45), lr=(1e-4, 5e-4), batch_size=(32, 96), timestep=(15, 25)),
}


def bounds(space: str = "original"):
    s = SPACES[space]
    lb = [s["units"][0], s["n_layers"][0], s["dropout"][0], math.log10(s["lr"][0]), s["batch_size"][0], s["timestep"][0], 0]
    ub = [s["units"][1], s["n_layers"][1], s["dropout"][1], math.log10(s["lr"][1]), s["batch_size"][1], s["timestep"][1],
          len(OPTIMIZERS) - 1e-9]
    return np.array(lb, float), np.array(ub, float)


def decode(v) -> dict:
    """Map a continuous position to valid integer and categorical hyperparameters."""
    return dict(units=int(round(v[0])), n_layers=int(round(v[1])), dropout=round(float(v[2]), 3), lr=float(10 ** v[3]),
                batch_size=int(round(v[4])), timestep=int(round(v[5])),
                optimizer=OPTIMIZERS[int(min(int(v[6]), len(OPTIMIZERS) - 1))])


def config_key(v) -> tuple:
    h = decode(v)
    return (h["units"], h["n_layers"], round(h["dropout"], 2), round(math.log10(h["lr"]), 2), h["batch_size"],
            h["timestep"], h["optimizer"])


def key_to_hp(k) -> dict:
    return dict(units=int(k[0]), n_layers=int(k[1]), dropout=float(k[2]), lr=float(10 ** k[3]), batch_size=int(k[4]),
                timestep=int(k[5]), optimizer=k[6])


def on_bounds(hp: dict, space: str = "original") -> str:
    s, out = SPACES[space], []
    for k in ("units", "dropout", "lr", "batch_size", "timestep"):
        lo, hi, v = s[k][0], s[k][1], float(hp[k])
        if k == "lr":
            lo, hi, v = math.log10(lo), math.log10(hi), math.log10(v)
        if min(abs(v - lo), abs(v - hi)) <= 0.01 * (hi - lo):
            out.append(k)
    return ",".join(out) or "none"


def fitness(hp: dict, df, feats, mode: str = "shared", arch: str = "lstm", seed: int = 42) -> float:
    from tensorflow.keras import backend as K
    set_seed(seed)
    Xtr, Ytr, _, _, _ = sequences(df, feats, hp["timestep"])
    n = len(Xtr)
    b = int(0.2 * n)
    if mode == "shared":
        Xa, ya, Xv, yv = Xtr[:n - b], Ytr[:n - b], Xtr[n - b:], Ytr[n - b:]
    else:
        Xa, ya, Xv, yv = Xtr[:n - 2 * b], Ytr[:n - 2 * b], Xtr[n - 2 * b:n - b], Ytr[n - 2 * b:n - b]
    m = build_model(Xa.shape[1:], hp, arch)
    m.fit(Xa, ya, epochs=FIT_EPOCHS, shuffle=False, batch_size=hp["batch_size"], verbose=0)
    v = float(np.mean((m.predict(Xv, verbose=0).ravel() - yv.ravel()) ** 2))
    del m
    K.clear_session()
    return v


class Evaluator:
    """Memoised fitness; records (calls, distinct, best fitness, best key) after every call."""

    def __init__(self, fn):
        self.fn, self.memo, self.history, self.calls = fn, {}, [], 0

    def __call__(self, v):
        k = config_key(v)
        self.calls += 1
        if k not in self.memo:
            self.memo[k] = self.fn(key_to_hp(k))
        best = min(self.memo, key=self.memo.get)
        self.history.append((self.calls, len(self.memo), self.memo[best], best))
        return self.memo[k]

    def best_at(self, budget: int, by: str = "calls"):
        col = 0 if by == "calls" else 1
        rows = [h for h in self.history if h[col] <= budget] or self.history
        return key_to_hp(rows[-1][3]), rows[-1][2]


def ga_paper(F: Evaluator, seed: int, lb, ub, gens: int = 15, pop: int = POP):
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, size=(pop, len(lb)))
    f = np.array([F(x) for x in X])
    for _ in range(gens - 1):
        order = np.argsort(f)
        new = [X[i].copy() for i in order[:pop // 4]]
        while len(new) < pop:
            p1, p2 = [X[rng.choice(order[:pop // 2])] for _ in range(2)]
            child = p1 + rng.uniform(-0.25, 1.25, len(lb)) * (p2 - p1)          # blend crossover, alpha = 0.25
            child = np.where(rng.rand(len(lb)) < 0.3, rng.uniform(lb, ub), child)  # uniform mutation, p = 0.3
            new.append(np.clip(child, lb, ub))
        X = np.array(new)
        f = np.array([F(x) for x in X])
    return F


def pso_paper(F: Evaluator, seed: int, lb, ub, iters: int = 15, swarm: int = POP, w=0.729, c1=1.49445, c2=1.49445):
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, size=(swarm, len(lb)))
    V = np.zeros_like(X)
    pbf = np.array([F(x) for x in X])
    pb, gb, gbf = X.copy(), X[pbf.argmin()].copy(), pbf.min()
    for _ in range(iters - 1):
        r1, r2 = rng.rand(*X.shape), rng.rand(*X.shape)
        V = w * V + c1 * r1 * (pb - X) + c2 * r2 * (gb - X)
        X = np.clip(X + V, lb, ub)
        for i in range(swarm):
            f = F(X[i])
            if f < pbf[i]:
                pbf[i], pb[i] = f, X[i].copy()
            if f < gbf:
                gbf, gb = f, X[i].copy()
    return F


def ga_distinct(F: Evaluator, seed: int, lb, ub, budget: int = 100, pop: int = POP):
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, size=(pop, len(lb)))
    f = np.array([F(x) for x in X])
    stall = 0
    while len(F.memo) < budget and stall < 50:
        before = len(F.memo)
        order = np.argsort(f)
        new = [X[i].copy() for i in order[:pop // 4]]
        while len(new) < pop:
            a, b = [X[rng.choice(order[:pop // 2])] for _ in range(2)]
            child = np.where(rng.rand(len(lb)) < 0.5, a, b)
            mut = rng.rand(len(lb)) < 0.2
            new.append(np.clip(np.where(mut, child + rng.normal(0, 0.1, len(lb)) * (ub - lb), child), lb, ub))
        X = np.array(new)
        f = np.array([F(x) for x in X])
        stall = stall + 1 if len(F.memo) == before else 0
    return F


def pso_distinct(F: Evaluator, seed: int, lb, ub, budget: int = 100, swarm: int = POP, w=0.729, c1=1.49445, c2=1.49445):
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, size=(swarm, len(lb)))
    V = np.zeros_like(X)
    pbf = np.array([F(x) for x in X])
    pb, gb, gbf = X.copy(), X[pbf.argmin()].copy(), pbf.min()
    stall = it = 0
    while len(F.memo) < budget and it < 500:
        it += 1
        before = len(F.memo)
        r1, r2 = rng.rand(*X.shape), rng.rand(*X.shape)
        V = w * V + c1 * r1 * (pb - X) + c2 * r2 * (gb - X)
        X = np.clip(X + V, lb, ub)
        for i in range(swarm):
            f = F(X[i])
            if f < pbf[i]:
                pbf[i], pb[i] = f, X[i].copy()
            if f < gbf:
                gbf, gb = f, X[i].copy()
        if len(F.memo) == before:
            stall += 1
            if stall >= 15:
                k = max(1, swarm // 4)
                worst = np.argsort(pbf)[-k:]
                X[worst] = rng.uniform(lb, ub, size=(k, len(lb)))
                V[worst] = 0
                pbf[worst] = np.inf
                stall = 0
        else:
            stall = 0
    return F
