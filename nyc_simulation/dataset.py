import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES = ["total_flow", "avg_speed", "avg_occupancy"]


def build_windows_from_csv(
    csv_path: str,
    out_dir: str,
    seq_len: int = 6,
    horizon: int = 3,
    stride: int = 1,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if not all(f in df.columns for f in FEATURES):
        raise ValueError(f"CSV must contain columns {FEATURES}")

    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES].to_numpy(dtype=np.float32))

    values = df[FEATURES].to_numpy(dtype=np.float32)
    timestamps = df["time"].to_numpy()

    # Temporal splits by timestamp quantiles
    ts_series = pd.to_datetime(timestamps, unit="s")
    train_cut = pd.Series(ts_series).quantile(0.7)
    val_cut = pd.Series(ts_series).quantile(0.85)

    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    for i in range(0, len(values) - seq_len - horizon + 1, stride):
        j = i + seq_len
        k = j + horizon
        end_time = ts_series[k - 1]
        X_window = values[i:j]
        Y_window = values[j:k]

        if end_time <= train_cut:
            X_train.append(X_window)
            Y_train.append(Y_window)
        elif end_time <= val_cut:
            X_val.append(X_window)
            Y_val.append(Y_window)
        else:
            X_test.append(X_window)
            Y_test.append(Y_window)

    X_clean_train = np.stack(X_train, axis=0) if X_train else np.zeros((0, seq_len, len(FEATURES)), dtype=np.float32)
    Y_train = np.stack(Y_train, axis=0) if Y_train else np.zeros((0, horizon, len(FEATURES)), dtype=np.float32)
    X_clean_val = np.stack(X_val, axis=0) if X_val else np.zeros((0, seq_len, len(FEATURES)), dtype=np.float32)
    Y_val = np.stack(Y_val, axis=0) if Y_val else np.zeros((0, horizon, len(FEATURES)), dtype=np.float32)
    X_clean_test = np.stack(X_test, axis=0) if X_test else np.zeros((0, seq_len, len(FEATURES)), dtype=np.float32)
    Y_test = np.stack(Y_test, axis=0) if Y_test else np.zeros((0, horizon, len(FEATURES)), dtype=np.float32)

    np.save(os.path.join(out_dir, "X_clean_train.npy"), X_clean_train)
    np.save(os.path.join(out_dir, "X_clean_val.npy"), X_clean_val)
    np.save(os.path.join(out_dir, "X_clean_test.npy"), X_clean_test)
    np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(out_dir, "Y_val.npy"), Y_val)
    np.save(os.path.join(out_dir, "Y_test.npy"), Y_test)

    meta = {
        "features": FEATURES,
        "target_features": FEATURES,
        "seq_len": seq_len,
        "horizon": horizon,
        "scaler": {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()},
    }
    with open(os.path.join(out_dir, "timeseries_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir, os.path.join(out_dir, "timeseries_meta.json")
