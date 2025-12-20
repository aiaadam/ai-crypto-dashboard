import os
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# CONFIG
# ----------------------------

# timeframe name -> (csv_path, interval_string)
TIMEFRAMES = {
    "1m": ("data/btcusdt_1m.csv", "1m"),
    "5m": ("data/btcusdt_5m.csv", "5m"),
    "15m": ("data/btcusdt_15m.csv", "15m"),
}

TP_PCT = 0.01   # +1% take profit
SL_PCT = 0.005  # -0.5% stop loss
LOOKAHEAD = 48  # next 48 bars

N_SPLITS = 3
N_ESTIMATORS = 300
RANDOM_STATE = 42

MODEL_DIR = "models"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    if "time" in df.columns:
        t = df["time"].astype(float)
        if t.max() > 1e11:
            df["time"] = (t // 1000).astype(int)
        else:
            df["time"] = t.astype(int)
    else:
        df["time"] = np.arange(len(df))

    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # EMAs + distances
    ema20 = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_20"] = ema20
    df["ema_50"] = ema50
    df["dist_ema_20"] = df["close"] / ema20 - 1
    df["dist_ema_50"] = df["close"] / ema50 - 1

    # EMA slopes
    df["ema_20_slope"] = ema20.pct_change(5)
    df["ema_50_slope"] = ema50.pct_change(5)

    # MACD
    macd_ind = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # Bollinger
    bb_ind = ta.volatility.BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2,
    )
    df["bb_high"] = bb_ind.bollinger_hband()
    df["bb_low"] = bb_ind.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

    # Volume features
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_rel"] = df["volume"] / df["vol_ma_20"]

    df = df.dropna().reset_index(drop=True)
    return df


def create_tp_sl_labels(df: pd.DataFrame,
                        tp_pct: float,
                        sl_pct: float,
                        lookahead: int) -> pd.Series:
    close = df["close"].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        entry = close[i]
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)

        hit_tp = False
        hit_sl = False
        end = min(n, i + lookahead + 1)
        for j in range(i + 1, end):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]
            if low <= sl:
                hit_sl = True
                break
            if high >= tp:
                hit_tp = True
                break

        if hit_tp and not hit_sl:
            labels[i] = 1
        else:
            labels[i] = 0

    return pd.Series(labels, index=df.index)


def train_for_timeframe(tf_name: str, csv_path: str, interval_str: str):
    print(f"\n==============================")
    print(f"[TRAIN] Timeframe: {tf_name} | CSV: {csv_path}")
    print(f"==============================")

    if not os.path.exists(csv_path):
        print(f"[TRAIN] CSV not found for {tf_name}: {csv_path} (skipping)")
        return

    df_raw = load_data(csv_path)
    print("[TRAIN] Raw rows:", len(df_raw))

    df_feat = build_features(df_raw)
    print("[TRAIN] Rows after features/dropna:", len(df_feat))

    if len(df_feat) == 0:
        print(f"[TRAIN] No usable rows for {tf_name} – need real candle data in {csv_path}")
        return

    labels = create_tp_sl_labels(df_raw.loc[df_feat.index], TP_PCT, SL_PCT, LOOKAHEAD)
    df_feat["label"] = labels

    feature_cols = [c for c in df_feat.columns if c not in ("time", "label")]
    X = df_feat[feature_cols].values
    y = df_feat["label"].values

    print("[TRAIN] Feature shape:", X.shape, "Labels shape:", y.shape)

    if len(df_feat) > 1000:
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=None,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print(f"\n[TRAIN] {tf_name} Fold {fold} classification report:")
            print(classification_report(y_test, y_pred, digits=3))
    else:
        print(f"[TRAIN] Not enough data for CV on {tf_name}; training on all data.")

    print(f"[TRAIN] Fitting final model on all data for {tf_name}...")
    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "interval": interval_str,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"crypto_model_{tf_name}.pkl")
    joblib.dump(bundle, model_path)
    print(f"[TRAIN] Saved {tf_name} model to {model_path}")


def main():
    for tf_name, (csv_path, interval_str) in TIMEFRAMES.items():
        train_for_timeframe(tf_name, csv_path, interval_str)


if __name__ == "__main__":
    main()
