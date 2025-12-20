from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import pandas as pd
import ta
import joblib
import os

app = FastAPI(
    title="AI Crypto Dashboard",
    version="0.4.1",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# CORS
origins = [
    "https://aiautotrades.onrender.com",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FRONTEND
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
print("FRONTEND_DIR:", FRONTEND_DIR)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "message": "Aadam AutoTrades backend is running"}


@app.get("/__test")
def test_route():
    return {"msg": "this is the correct main.py"}


# -------------------------------
# CryptoCompare OHLC data
# -------------------------------

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2"

SYMBOL_MAP = {
    "BTCUSDT": ("BTC", "USDT"),
    "ETHUSDT": ("ETH", "USDT"),
}

INTERVAL_CONFIG = {
    "1m":  ("histominute", 1),
    "5m":  ("histominute", 5),
    "15m": ("histominute", 15),
    "30m": ("histominute", 30),
    "1h":  ("histohour",   1),
    "4h":  ("histohour",   4),
    "1d":  ("histoday",    1),
}


def get_klines(symbol: str, interval: str = "1m", limit: int = 200):
    symbol = symbol.upper()
    if symbol not in SYMBOL_MAP:
        raise ValueError(f"Unsupported symbol for CryptoCompare: {symbol}")

    fsym, tsym = SYMBOL_MAP[symbol]
    cfg = INTERVAL_CONFIG.get(interval)
    if cfg is None:
        cfg = INTERVAL_CONFIG["1m"]
    endpoint, aggregate = cfg

    url = f"{CRYPTOCOMPARE_BASE}/{endpoint}"
    params = {
        "fsym": fsym,
        "tsym": tsym,
        "limit": limit,
        "aggregate": aggregate,
    }

    resp = requests.get(url, params=params)
    if resp.status_code in (400, 401, 403, 404, 429, 500, 503):
        print(f"[Data] CryptoCompare error {resp.status_code} for {url} params={params}")
        return []

    resp.raise_for_status()
    data = resp.json()
    if data.get("Response") != "Success":
        print(f"[Data] CryptoCompare non-success response: {data.get('Message')}")
        return []

    bars = data["Data"]["Data"]

    klines = []
    for b in bars:
        t_sec = b["time"]
        o = b["open"]
        h = b["high"]
        l = b["low"]
        c = b["close"]
        v = b["volumefrom"]
        t_ms = t_sec * 1000
        klines.append([
            t_ms,
            str(o),
            str(h),
            str(l),
            str(c),
            str(v),
            "0", "0", "0", "0", "0", "0",
        ])

    return klines


# -------------------------------
# Rule-based signal (chart signals)
# -------------------------------

def generate_signal(row, interval: str = "1h"):
    close = row["close"]
    ema_50 = row["ema_50"]
    ema_200 = row["ema_200"]
    rsi = row["rsi"]
    macd = row["macd"]
    macd_signal = row["macd_signal"]
    bb_high = row["bb_high"]
    bb_low = row["bb_low"]
    atr_perc = row.get("atr_perc", 0.0)

    ema_uptrend = ema_50 > ema_200
    ema_downtrend = ema_50 < ema_200

    rsi_bullish_zone = rsi >= 40
    rsi_strong_bull = rsi >= 55
    rsi_bearish_zone = rsi <= 60
    rsi_strong_bear = rsi <= 45

    macd_bull = macd > macd_signal
    macd_bear = macd < macd_signal
    macd_diff = macd - macd_signal

    if interval == "1m":
        macd_strong = 0.0005
        min_vol = 0.001
    elif interval in ["5m", "15m"]:
        macd_strong = 0.0008
        min_vol = 0.0008
    else:
        macd_strong = 0.0010
        min_vol = 0.0005

    strong_macd_up = macd_diff > macd_strong
    strong_macd_down = macd_diff < -macd_strong

    near_lower_band = bb_low is not None and bb_low > 0 and close <= bb_low
    near_upper_band = bb_high is not None and bb_high > 0 and close >= bb_high

    if atr_perc is not None and atr_perc < min_vol:
        return "HOLD"

    # BUY
    if (
        rsi_strong_bull
        and macd_bull
        and strong_macd_up
        and (ema_uptrend or not ema_downtrend)
    ):
        return "BUY"

    if (
        rsi_bullish_zone
        and macd_bull
        and macd_diff > 0
    ):
        if near_upper_band and rsi > 70:
            return "HOLD"
        return "BUY"

    # SELL
    if (
        rsi_strong_bear
        and macd_bear
        and strong_macd_down
        and (ema_downtrend or not ema_uptrend)
    ):
        return "SELL"

    if (
        rsi_bearish_zone
        and macd_bear
        and macd_diff < 0
    ):
        if near_lower_band and rsi < 30:
            return "HOLD"
        return "SELL"

    return "HOLD"


@app.get("/crypto/{symbol}")
def crypto(symbol: str, interval: str = Query("1h")):
    print(f"[CRYPTO] symbol={symbol.upper()} interval={interval}")

    data = get_klines(symbol.upper(), interval=interval)

    if not data:
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
            "info": "No data from provider",
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

    atr_ind = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr_ind.average_true_range()
    df["atr_perc"] = df["atr"] / df["close"]

    df["vol_ma_20"] = df["volume"].rolling(20).mean()

    cols = ["rsi", "ema_50", "ema_200", "macd", "macd_signal",
            "bb_high", "bb_low", "vol_ma_20", "atr", "atr_perc"]
    df[cols] = df[cols].bfill().ffill()

    df["signal"] = df.apply(lambda row: generate_signal(row, interval), axis=1)

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
# ML models: load + predict
# -------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATHS = {
    "1m": os.path.join(MODEL_DIR, "crypto_model_1m.pkl"),
    "5m": os.path.join(MODEL_DIR, "crypto_model_5m.pkl"),
    "15m": os.path.join(MODEL_DIR, "crypto_model_15m.pkl"),
}

ml_bundles = {}  # timeframe -> {"model": ..., "feature_cols": [...], "interval": "1m/5m/15m"}


def load_models():
    global ml_bundles
    for tf, path in MODEL_PATHS.items():
        try:
            if not os.path.exists(path):
                print(f"[ML] Model file for {tf} not found at {path}")
                continue
            bundle = joblib.load(path)
            ml_bundles[tf] = {
                "model": bundle["model"],
                "feature_cols": bundle["feature_cols"],
                "interval": bundle.get("interval", tf),
            }
            print(f"[ML] Loaded {tf} model from {path}")
        except Exception as e:
            print(f"[ML] Failed to load {tf} model from {path}: {e}")


load_models()


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
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    ema20 = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_20"] = ema20
    df["ema_50"] = ema50
    df["dist_ema_20"] = df["close"] / ema20 - 1
    df["dist_ema_50"] = df["close"] / ema50 - 1

    df["ema_20_slope"] = ema20.pct_change(5)
    df["ema_50_slope"] = ema50.pct_change(5)

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

    df = df.dropna().reset_index(drop=True)
    return df


def map_proba_to_signal(p_good: float, risk: str, timeframe: str) -> str:
    """
    p_good = probability this is a good long (TP before SL).
    thresholds are relaxed for 1m so it doesn't spam HOLD.
    """
    risk = risk.lower()
    timeframe = timeframe.lower()

    if timeframe == "1m":
        if risk == "risky":
            buy_th = 0.52
            hold_th = 0.50
        elif risk == "strict":
            buy_th = 0.70
            hold_th = 0.55
        else:  # medium
            buy_th = 0.60
            hold_th = 0.52
    else:
        if risk == "risky":
            buy_th = 0.55
            hold_th = 0.50
        elif risk == "strict":
            buy_th = 0.75
            hold_th = 0.60
        else:  # medium
            buy_th = 0.65
            hold_th = 0.55

    if p_good >= buy_th:
        return "BUY"
    if p_good >= hold_th:
        return "HOLD"
    return "SELL"


@app.get("/predict/{symbol}")
def predict(
    symbol: str,
    risk: str = Query("medium"),
    timeframe: str = Query("15m", regex="^(1m|5m|15m)$"),
):
    """
    AI-based prediction using timeframe-specific TP/SL models.
    risk: "risky" | "medium" | "strict"
    timeframe: "1m" | "5m" | "15m"
    """
    bundle = ml_bundles.get(timeframe)
    if not bundle:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": timeframe,
            "prediction": "HOLD",
            "info": f"ML model for timeframe {timeframe} not loaded",
        }

    ml_model = bundle["model"]
    ml_feature_cols = bundle["feature_cols"]
    ml_interval = bundle["interval"]

    risk = risk.lower()
    if risk not in ("risky", "medium", "strict"):
        risk = "medium"

    interval = ml_interval or timeframe

    raw = get_klines(symbol.upper(), interval=interval, limit=300)
    if not raw:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": interval,
            "prediction": "HOLD",
            "info": "No data from provider",
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

    try:
        proba = ml_model.predict_proba(X)[0]
    except Exception as e:
        print(f"[ML] predict_proba failed: {e}")
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": interval,
            "prediction": "HOLD",
            "info": "ML predict_proba error",
        }

    classes = list(ml_model.classes_)
    class_to_proba = {cls: p for cls, p in zip(classes, proba)}
    p_good = float(class_to_proba.get(1, 0.0))

    final_signal = map_proba_to_signal(p_good, risk, timeframe)

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": interval,
        "prediction": final_signal,
        "p_good": p_good,
        "risk": risk,
        "time": int(latest["time"]),
    }
