from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import pandas as pd
import ta
import os

app = FastAPI(
    title="AI Crypto Dashboard",
    version="0.2.0",
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
def serve_dashboard():
    return {"status": "ok", "message": "Aadam AutoTrades backend is running"}


@app.get("/__test")
def test_route():
    return {"msg": "this is the correct main.py"}


# -------------------------------
# CryptoCompare OHLC data
# -------------------------------

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com"

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

    url = f"{CRYPTOCOMPARE_BASE}/data/v2/{endpoint}"
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
# Rule-based signal (RSI+MACD brain)
# -------------------------------

def generate_signal(row, interval: str, strictness: float):
    """
    strictness in [0,1]:
    - 0.0 = very aggressive (lots of trades)
    - 0.5 = medium
    - 1.0 = very strict (only strong setups)
    """
    close = row["close"]
    ema_50 = row["ema_50"]
    ema_200 = row["ema_200"]
    rsi = row["rsi"]
    macd = row["macd"]
    macd_signal = row["macd_signal"]
    bb_high = row["bb_high"]
    bb_low = row["bb_low"]
    atr_perc = row.get("atr_perc", 0.0)

    # Trend from EMAs
    ema_uptrend = ema_50 > ema_200
    ema_downtrend = ema_50 < ema_200

    # RSI zones
    rsi_bullish_zone = rsi >= 40
    rsi_strong_bull = rsi >= 55
    rsi_bearish_zone = rsi <= 60
    rsi_strong_bear = rsi <= 45

    # MACD
    macd_bull = macd > macd_signal
    macd_bear = macd < macd_signal
    macd_diff = macd - macd_signal

    # Timeframe‑specific base sensitivity
    if interval == "1m":
        base_macd_strong = 0.0005
        base_min_vol = 0.001
    elif interval in ["5m", "15m"]:
        base_macd_strong = 0.0008
        base_min_vol = 0.0008
    else:
        base_macd_strong = 0.0010
        base_min_vol = 0.0005

    # Adjust sensitivity by strictness
    # stricter = require stronger MACD and more volatility
    macd_strong = base_macd_strong * (1.0 + strictness)
    min_vol = base_min_vol * (1.0 + 0.5 * strictness)

    strong_macd_up = macd_diff > macd_strong
    strong_macd_down = macd_diff < -macd_strong

    # Bands
    near_lower_band = bb_low is not None and bb_low > 0 and close <= bb_low
    near_upper_band = bb_high is not None and bb_high > 0 and close >= bb_high

    # Volatility guard: if ultra-low vol and strict, avoid trades
    if strictness > 0.5 and atr_perc is not None and atr_perc < min_vol:
        return "HOLD"

    # ---------------- BUY LOGIC ----------------
    # Strong buy
    if (
        rsi_strong_bull
        and macd_bull
        and strong_macd_up
        and (ema_uptrend or not ema_downtrend)
    ):
        return "BUY"

    # Normal buy: RSI bull zone + MACD bullish
    if (
        rsi_bullish_zone
        and macd_bull
        and macd_diff > 0
    ):
        if near_upper_band and rsi > 70 and strictness > 0.3:
            return "HOLD"
        return "BUY"

    # ---------------- SELL LOGIC ---------------
    # Strong sell
    if (
        rsi_strong_bear
        and macd_bear
        and strong_macd_down
        and (ema_downtrend or not ema_uptrend)
    ):
        return "SELL"

    # Normal sell
    if (
        rsi_bearish_zone
        and macd_bear
        and macd_diff < 0
    ):
        if near_lower_band and rsi < 30 and strictness > 0.3:
            return "HOLD"
        return "SELL"

    # Disagreement / no clear edge
    return "HOLD"


def strictness_from_risk(risk: str) -> float:
    risk = (risk or "medium").lower()
    if risk == "risky":
        return 0.0
    if risk == "strict":
        return 1.0
    return 0.5


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

    # Volatility
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

    # Default risk strictness = medium
    strictness = strictness_from_risk("medium")
    df["signal"] = df.apply(lambda row: generate_signal(row, interval, strictness), axis=1)

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
# Simple "AI" prediction endpoint with risk
# -------------------------------

@app.get("/predict/{symbol}")
def predict(symbol: str, risk: str = Query("medium")):
    """
    Uses the same indicator brain but applies a different strictness
    based on risk: "risky" | "medium" | "strict".
    """
    interval = "15m"
    strictness = strictness_from_risk(risk)

    data = get_klines(symbol.upper(), interval=interval, limit=300)
    if not data:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": interval,
            "prediction": "HOLD",
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

    atr_ind = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr_ind.average_true_range()
    df["atr_perc"] = df["atr"] / df["close"]

    df = df.dropna()
    if len(df) == 0:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "interval": interval,
            "prediction": "HOLD",
            "info": "Not enough data after indicators",
        }

    latest = df.iloc[-1]
    pred_signal = generate_signal(latest, interval, strictness)

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": interval,
        "prediction": pred_signal,
        "risk": risk.lower(),
        "time": int(latest["time"]),
    }
