"""Model builder (LSTM, GRU, BiLSTM, TCN, Transformer, MLP), supervised windowing and the final training protocol.

Protocol (Section II of the paper): chronological 80/20 split, min-max scaling fitted on the training part only,
sliding windows of length ``timestep``, MSE loss, final training for up to 100 epochs with early stopping on the
last 20% of the training part (patience 10, best weights restored).
"""
from __future__ import annotations

import gc
import random

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET

TRAIN_FRAC = 0.8
FINAL_EPOCHS = 100
OPTIMIZERS = ["adam", "rmsprop", "nadam"]

# Configurations reported in the paper
DEFAULT_HP = dict(units=64, n_layers=1, dropout=0.2, lr=1e-3, batch_size=32, timestep=20, optimizer="adam")
GA_HP = dict(units=126, n_layers=1, dropout=0.140, lr=5.261e-4, batch_size=61, timestep=22, optimizer="rmsprop")
PSO_HP = dict(units=160, n_layers=1, dropout=0.1, lr=1e-3, batch_size=16, timestep=21, optimizer="nadam")
INTERIOR_HP = dict(units=88, n_layers=1, dropout=0.3, lr=3.2e-4, batch_size=64, timestep=20, optimizer="adam")


def set_seed(seed: int) -> None:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _keras():
    import tensorflow as tf
    from tensorflow.keras import layers as KL
    return tf, KL


def build_model(input_shape, hp: dict, arch: str = "lstm", out: int = 1):
    """``n_layers``: recurrent layers (LSTM/GRU/BiLSTM), dilated conv blocks (TCN: n_layers+1 layers with dilation
    1, 2, 4, ...), encoder blocks (Transformer) or hidden layers (MLP)."""
    tf, KL = _keras()
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

    class LastStep(KL.Layer):
        def call(self, x):
            return x[:, -1, :]

    class PosEnc(KL.Layer):
        def build(self, shape):
            T, D = int(shape[1]), int(shape[2])
            pos, i = np.arange(T)[:, None], np.arange(D)[None, :]
            ang = pos / np.power(10000, (2 * (i // 2)) / D)
            self.pe = tf.constant(np.where(i % 2 == 0, np.sin(ang), np.cos(ang))[None], dtype="float32")

        def call(self, x):
            return x + self.pe

    K.clear_session()
    inp = KL.Input(shape=input_shape)
    u, dr, L = int(hp["units"]), float(hp.get("dropout", 0.2)), int(hp.get("n_layers", 1))
    if arch in ("lstm", "gru", "bilstm"):
        x = inp
        for i in range(L):
            cell = (KL.GRU if arch == "gru" else KL.LSTM)(u, return_sequences=(i < L - 1))
            x = KL.Bidirectional(cell)(x) if arch == "bilstm" else cell(x)
            if dr > 0:
                x = KL.Dropout(dr)(x)
    elif arch == "tcn":
        x = inp
        for j in range(L + 1):
            x = KL.Conv1D(u, 3, padding="causal", dilation_rate=2 ** j, activation="relu")(x)
            if dr > 0:
                x = KL.Dropout(dr)(x)
        x = LastStep()(x)
    elif arch == "mlp":
        x = KL.Flatten()(inp)
        for _ in range(L):
            x = KL.Dense(u, activation="relu")(x)
            if dr > 0:
                x = KL.Dropout(dr)(x)
    elif arch == "transformer":
        x = PosEnc()(KL.Dense(u)(inp))
        for _ in range(L):
            a = KL.MultiHeadAttention(num_heads=4, key_dim=max(1, u // 4), dropout=dr)(x, x)
            x = KL.LayerNormalization()(x + a)
            f = KL.Dense(u)(KL.Dense(u, activation="relu")(x))
            x = KL.LayerNormalization()(x + f)
        x = KL.GlobalAveragePooling1D()(x)
        if dr > 0:
            x = KL.Dropout(dr)(x)
    else:
        raise ValueError(f"unknown architecture {arch}")
    opt = {"adam": Adam, "rmsprop": RMSprop, "nadam": Nadam}[hp.get("optimizer", "adam")](hp.get("lr", 1e-3))
    m = Model(inp, KL.Dense(out, activation="linear")(x))
    m.compile(loss="mse", optimizer=opt)
    return m


def sequences(df: pd.DataFrame, feats, timestep: int, horizon: int = 1):
    """Inputs end at i-1, targets are y[i..i+horizon-1]; scalers are fitted on the training part only.
    Returns X_train, Y_train, X_test, test indices, target scaler."""
    from sklearn.preprocessing import MinMaxScaler
    dX = df[list(feats)].values.astype("float32")
    dy = df[[TARGET]].values.astype("float32")
    n_tr = int(len(df) * TRAIN_FRAC)
    sx, sy = MinMaxScaler().fit(dX[:n_tr]), MinMaxScaler().fit(dy[:n_tr])
    Xs, ys = sx.transform(dX), sy.transform(dy).ravel()
    idx = np.arange(timestep, len(df) - horizon + 1)
    X = np.stack([Xs[i - timestep:i] for i in idx])
    Y = np.stack([ys[i:i + horizon] for i in idx])
    train = idx + horizon - 1 < n_tr
    test = idx >= n_tr
    return X[train], Y[train], X[test], idx[test], sy


def train_predict(hp: dict, seed: int, df: pd.DataFrame, feats=FEATURES, arch: str = "lstm", horizon: int = 1,
                  epochs: int = FINAL_EPOCHS, keep_history: bool = False) -> dict:
    """Final training protocol; returns test predictions, actual values and the aligned random-walk forecast."""
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    set_seed(seed)
    Xtr, Ytr, Xte, ite, sy = sequences(df, feats, int(hp["timestep"]), horizon)
    m = build_model(Xtr.shape[1:], hp, arch, out=horizon)
    h = m.fit(Xtr, Ytr, validation_split=0.2, shuffle=False, epochs=epochs, batch_size=int(hp["batch_size"]),
              callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)], verbose=0)
    p = m.predict(Xte, verbose=0).reshape(len(Xte), horizon)
    pred = sy.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)
    y = df[TARGET].values.astype("float64")
    if horizon == 1:
        out = dict(pred=pred[:, 0], y=y[ite], rw=y[ite - 1])
    else:
        out = dict(pred=pred, y=np.stack([y[ite + k] for k in range(horizon)], 1), last=y[ite - 1])
    if keep_history:
        out["history"] = {k: list(map(float, v)) for k, v in h.history.items()}
    del m
    K.clear_session()
    gc.collect()
    return out
