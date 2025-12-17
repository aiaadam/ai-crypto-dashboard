import requests
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_klines(symbol: str, interval: str = "15m", limit: int = 1500):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(BINANCE_URL, params=params)
    response.raise_for_status()
    return response.json()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time"] = df["time"] // 1000
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # Basic returns
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # EMAs
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    # Distance from EMAs
    df["dist_ema_20"] = df["close"] / df["ema_20"] - 1
    df["dist_ema_50"] = df["close"] / df["ema_50"] - 1

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

    # Bollinger Bands
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

    df = df.dropna()
    return df


def add_labels(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.0025):
    """
    horizon: how many candles ahead to look (e.g. 3 x 15m = 45m)
    threshold: min move to count as up or down (0.25% here)
    """
    df = df.copy()
    future_price = df["close"].shift(-horizon)
    future_ret = (future_price - df["close"]) / df["close"]

    conditions = [
        future_ret > threshold,
        future_ret < -threshold,
    ]
    choices = [1, -1]  # 1 = up, -1 = down
    df["label"] = np.select(conditions, choices, default=0)  # 0 = flat

    df = df.dropna()
    return df


def main():
    symbols = ["BTCUSDT", "ETHUSDT"]
    interval = "15m"

    all_rows = []

    for sym in symbols:
        print(f"Downloading data for {sym} {interval}...")
        raw = get_klines(sym, interval=interval, limit=1500)
        df = pd.DataFrame(
            raw,
            columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "_1",
                "_2",
                "_3",
                "_4",
                "_5",
                "_6",
            ],
        )
        df = build_features(df)
        df = add_labels(df, horizon=3, threshold=0.0025)
        df["symbol"] = sym
        all_rows.append(df)

    data = pd.concat(all_rows, ignore_index=True)
    print("Total samples:", len(data))

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_6",
        "rsi",
        "dist_ema_20",
        "dist_ema_50",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "vol_rel",
    ]

    X = data[feature_cols].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("Validation report:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs("models", exist_ok=True)
    model_path = "models/crypto_model.pkl"
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "interval": interval,
            "horizon": 3,
            "threshold": 0.0025,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
