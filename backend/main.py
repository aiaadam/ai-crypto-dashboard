from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
import pandas as pd
import ta
import joblib
import os
import time
from datetime import datetime, timedelta


app = FastAPI(
    title="AI Crypto Dashboard",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# CORS
origins = [
    "https://aiautotrades.onrender.com",  # your frontend
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # not ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FRONTEND (optional static mount, safe to leave)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
print("FRONTEND_DIR:", FRONTEND_DIR)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    return {"status": "ok", "message": "Aadam AutoTrades backend is running"}


@app.get("/__test")
def test_route():
    return {"msg": "this is the correct main.py"}


# -------------------------------
# CoinGecko data fetch (replaces Binance)
# -------------------------------

COINGECKO_OHLC_URL = "https://api.coingecko.com/api/v3/coins/{id}/ohlc"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

# Map your trading symbols to CoinGecko IDs
SYMBOL_TO_COINGECKO_ID = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    # add more here if needed
}

# Rough mapping of your requested interval to "days" window
INTERVAL_TO_DAYS = {
    "1h": 7,    # 7 days of hourly candles
    "4h": 30,
    "1d": 90,
}


def get_klines(symbol: str, interval: str = "1h", limit: int = 500):
    symbol = symbol.upper()
    cg_id = SYMBOL_TO_COINGECKO_ID.get(symbol)
    if cg_id is None:
        raise ValueError(f"Unsupported symbol for CoinGecko: {symbol}")

    days = INTERVAL_TO_DAYS.get(interval, 7)

    params = {
        "vs_currency": "usd",
        "days": days,
    }

    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY

    url = COINGECKO_OHLC_URL.format(id=cg_id)
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 429:
        # rate limited – return empty list instead of crashing
        print("[Data] CoinGecko rate limit hit (429)")
        return []
    resp.raise_for_status()
    raw = resp.json()
    # raw: [[timestamp_ms, open, high, low, close], ...]

    klines = []
    for item in raw[-limit:]:
        t, o, h, l, c = item
        volume = 0.0  # CoinGecko OHLC doesn't include volume
        klines.append([
            t,
            str(o),
            str(h),
            str(l),
            str(c),
            str(volume),
            "0", "0", "0", "0", "0", "0",
        ])

    return klines


# -------------------------------
# Rule-based signal
# -------------------------------

def generate_signal(row):
    close = row["close"]
    ema_50 = row["ema_50"]
    ema_200 = row["ema_200"]
    rsi = row["rsi"]
    macd = row["macd"]
    macd_signal = row["macd_signal"]
    bb_high = row["bb_high"]
    bb_low = row["bb_low"]
    vol = row["volume"]
    vol_ma = row["vol_ma_20"]

    uptrend = close > ema_200
    downtrend = close < ema_200

    # Aggressive BUY logic
    if (
        (uptrend or close >= ema_50)
        and rsi < 60
        and (macd > macd_signal or close <= bb_low * 1.02)
        and (vol >= 0.7 * vol_ma)
    ):
        return "BUY"

    if (
        close > ema_50
        and macd > macd_signal
        and vol >= 1.2 * vol_ma
    ):
        return "BUY"

    # SELL logic
    if (
        downtrend
        and rsi > 40
        and (macd < macd_signal or close >= bb_high * 0.98)
    ):
        return "SELL"

    if (
        rsi > 70
        and close >= bb_high
        and vol >= 1.5 * vol_ma
    ):
        return "SELL"

    return "HOLD"


@app.get("/crypto/{symbol}")
def crypto(symbol: str, interval: str = Query("1h")):
    data = get_klines(symbol.upper(), interval=interval)

    if not data:
        # make it explicit so frontend can show a message
        return {
            "ok": False,
            "time": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "signal": [],
            "interval": interval,
            "symbol": symbol.upper(),
            "info": "No data (possibly rate limited by data provider)",
        }

    df = pd.DataFrame(
        data,
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

    df["time"] = df["time"] // 1000
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # Indicators
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    macd_ind = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()

    bb_ind = ta.volatility.BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2,
    )
    df["bb_high"] = bb_ind.bollinger_hband()
    df["bb_low"] = bb_ind.bollinger_lband()

    df["vol_ma_20"] = df["volume"].rolling(20).mean()

    # NEW: keep rows, fill indicator NaNs instead of dropping everything
    df[["rsi", "ema_50", "ema_200", "macd", "macd_signal",
        "bb_high", "bb_low", "vol_ma_20"]] = df[[
            "rsi", "ema_50", "ema_200", "macd", "macd_signal",
            "bb_high", "bb_low", "vol_ma_20"
        ]].fillna(method="bfill").fillna(method="ffill")

    df["signal"] = df.apply(generate_signal, axis=1)

    return {
        "ok": True,
        "time": df["time"].tolist(),
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "signal": df["signal"].tolist(),
        "interval": interval,
        "symbol": symbol.upper(),
    }


# -------------------------------
# ML model: load + predict
# -------------------------------

MODEL_PATH = "models/crypto_model.pkl"

ml_model = None
ml_feature_cols = None
ml_interval = None


def load_model():
    global ml_model, ml_feature_cols, ml_interval
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[ML] Model file not found at {MODEL_PATH} - skipping ML load")
            return
        bundle = joblib.load(MODEL_PATH)
        ml_model = bundle["model"]
        ml_feature_cols = bundle["feature_cols"]
        ml_interval = bundle["interval"]
        print(f"[ML] Loaded model from {MODEL_PATH} (interval={ml_interval})")
    except Exception as e:
        print(f"[ML] Failed to load model: {e}")
        ml_model = None
        ml_feature_cols = None
        ml_interval = None


load_model()


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time"] = df["time"] // 1000
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    ema20 = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_20"] = ema20
    df["ema_50"] = ema50
    df["dist_ema_20"] = df["close"] / ema20 - 1
    df["dist_ema_50"] = df["close"] / ema50 - 1

    macd_ind = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    bb_ind = ta.volatility.BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2,
    )
    df["bb_high"] = bb_ind.bollinger_hband()
    df["bb_low"] = bb_ind.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_rel"] = df["volume"] / df["vol_ma_20"]

    df = df.dropna()
    return df


def map_label_to_signal(label: int) -> str:
    if label == -1:
        return "SELL"
    return "BUY"


@app.get("/predict/{symbol}")
def predict(symbol: str):
    # if model not loaded, be explicit
    if ml_model is None:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": ml_interval or "unknown",
            "prediction": "HOLD",
            "info": "ML model not loaded",
        }

    interval = ml_interval or "15m"

    raw = get_klines(symbol.upper(), interval=interval, limit=300)
    if not raw:
        # explicitly tell frontend it’s a data issue (often rate limit)
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": interval,
            "prediction": "HOLD",
            "info": "No data from provider (rate limited or unavailable)",
        }

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
    df = build_ml_features(df)

    if len(df) == 0:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": interval,
            "prediction": "HOLD",
            "info": "Not enough data for features",
        }

    latest = df.iloc[-1]
    X = latest[ml_feature_cols].values.reshape(1, -1)
    label = int(ml_model.predict(X)[0])
    pred_signal = map_label_to_signal(label)

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": interval,
        "prediction": pred_signal,
        "raw_label": label,
        "time": int(latest["time"]),
    }
