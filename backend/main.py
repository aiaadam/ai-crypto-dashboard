from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import pandas as pd
import ta
import joblib
import os
import shutil
import base64
import json

from openai import OpenAI

# -------------------------------
# FastAPI app setup
# -------------------------------

app = FastAPI(
    title="AI Crypto Dashboard",
    version="0.7.0",
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
# Grok client + uploads dir
# -------------------------------

GROK_API_KEY = os.getenv("GROK_API_KEY")

client = None
if GROK_API_KEY:
    client = OpenAI(
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1",
    )
else:
    print("WARNING: GROK_API_KEY not set - AI endpoints disabled")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------------------
# REAL chart-image AI analysis
# -------------------------------

@app.post("/analyze_chart_image")
async def analyze_chart_image(
    file: UploadFile = File(...),
):
    """
    Accepts an uploaded chart image and returns AI analysis text
    using Grok vision via OpenAI-compatible client.
    """

    contents = await file.read()
    size_kb = len(contents) / 1024

    if size_kb > 4096:
        return {
            "ok": False,
            "message": "Image too large (max ~4MB)",
        }

    safe_name = file.filename.replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as buffer:
        buffer.write(contents)

    base64_image = encode_image_to_base64(save_path)

    if client is None:
        return {
            "ok": False,
            "filename": file.filename,
            "analysis": "Server missing GROK_API_KEY. Contact admin.",
        }

    prompt = """
You are a professional day trader. Analyze this TRADING CHART image only.
If this is NOT a trading chart, clearly say that and do NOT invent prices.

If it IS a trading chart, extract REAL price levels from the chart and answer
in EXACTLY this format (no extra text, no bullet dashes):

AI Image Analysis
Bias: <Bullish/Bearish/Neutral> (<short reason>)
Action: <BUY/SELL/HOLD> (<confidence %>)
Entry: <exact price or small range>
Stop loss: <exact price> (<why here>)
Take profit 1: <exact price> (<what this level is>)
Take profit 2: <exact price> (<what this level is>)
Risk ratio: <R:R number>
Structure: <key support/resistance + structure notes>
Risks: <3 short risks in one line>

Use only prices you can clearly see on the chart (y-axis / labels / candles).
If you cannot read prices, say: 'Cannot read price scale clearly, no exact levels.'
"""

    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=800,
        )

        analysis_text = response.choices[0].message.content

    except Exception as e:
        print("[VISION] Error:", e)
        return {
            "ok": False,
            "filename": file.filename,
            "analysis": f"Vision API error: {e}",
        }

    return {
        "ok": True,
        "filename": file.filename,
        "analysis": analysis_text,
    }

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
        macd_strong = 0.0002
        min_vol = 0.0003
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

    if interval == "1m":
        if 50 <= rsi <= 60 and ema_50 > ema_200:
            return "BUY"
        if 40 <= rsi <= 50 and ema_50 < ema_200:
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

def map_proba_to_signal(p_good: float, timeframe: str) -> str:
    timeframe = timeframe.lower()

    if timeframe == "1m":
        buy_th = 0.60
        hold_th = 0.50
    else:
        buy_th = 0.65
        hold_th = 0.55

    if p_good >= buy_th:
        return "BUY"
    if p_good >= hold_th:
        return "HOLD"
    return "SELL"

# -------------------------------
# Simple SMC helpers (BoS/CHoCH + FVG)
# -------------------------------

def detect_bos_choch(df: pd.DataFrame):
    if len(df) < 10:
        return {
            "bos_bull": False,
            "bos_bear": False,
            "choch_bull": False,
            "choch_bear": False,
        }

    d = df.tail(50).reset_index(drop=True)

    swing_high = (d["high"] > d["high"].shift(1)) & (d["high"] > d["high"].shift(-1))
    swing_low = (d["low"] < d["low"].shift(1)) & (d["low"] < d["low"].shift(-1))

    d["swing_high"] = swing_high
    d["swing_low"] = swing_low

    highs = d[d["swing_high"]]
    lows = d[d["swing_low"]]

    bos_bull = bos_bear = choch_bull = choch_bear = False

    if len(highs) >= 2:
        last_high = highs.iloc[-1]["high"]
        prev_high = highs.iloc[-2]["high"]
        if last_high > prev_high:
            bos_bull = True

    if len(lows) >= 2:
        last_low = lows.iloc[-1]["low"]
        prev_low = lows.iloc[-2]["low"]
        if last_low < prev_low:
            bos_bear = True

    if bos_bull and len(lows) >= 2:
        last_low = lows.iloc[-1]["low"]
        prev_low = lows.iloc[-2]["low"]
        if last_low < prev_low:
            choch_bear = True

    if bos_bear and len(highs) >= 2:
        last_high = highs.iloc[-1]["high"]
        prev_high = highs.iloc[-2]["high"]
        if last_high > prev_high:
            choch_bull = True

    return {
        "bos_bull": bool(bos_bull),
        "bos_bear": bool(bos_bear),
        "choch_bull": bool(choch_bull),
        "choch_bear": bool(choch_bear),
    }

def detect_fvg(df: pd.DataFrame):
    if len(df) < 3:
        return {"fvg_bull": False, "fvg_bear": False}

    d = df.tail(3)
    h0, h2 = d["high"].iloc[0], d["high"].iloc[2]
    l0, l2 = d["low"].iloc[0], d["low"].iloc[2]

    fvg_bear = h0 < l2
    fvg_bull = l0 > h2

    return {"fvg_bull": bool(fvg_bull), "fvg_bear": bool(fvg_bear)}

def smc_overlay(df: pd.DataFrame, final_signal: str, p_good: float) -> str:
    latest = df.iloc[-1]

    rsi = latest["rsi"]
    ema_20 = latest.get("ema_20", latest.get("ema_50", 0.0))
    ema_50 = latest.get("ema_50", ema_20)

    swing_info = detect_bos_choch(df)
    fvg_info = detect_fvg(df)

    bullish_setup = swing_info["choch_bull"] and fvg_info["fvg_bull"]
    bearish_setup = swing_info["choch_bear"] and fvg_info["fvg_bear"]

    if bullish_setup:
        if rsi < 70 and ema_20 > ema_50 and p_good > 0.50:
            if final_signal == "SELL":
                final_signal = "HOLD"
            elif final_signal == "HOLD":
                final_signal = "BUY"

    if bearish_setup:
        if rsi > 30 and ema_20 < ema_50 and p_good < 0.50:
            if final_signal == "BUY":
                final_signal = "HOLD"
            elif final_signal == "HOLD":
                final_signal = "SELL"

    return final_signal

@app.get("/predict/{symbol}")
def predict(
    symbol: str,
    timeframe: str = Query("15m", regex="^(1m|5m|15m)$"),
):
    """
    AI-based prediction using timeframe-specific TP/SL models.
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

    final_signal = map_proba_to_signal(p_good, timeframe)
    final_signal = smc_overlay(df, final_signal, p_good)

    if timeframe == "1m":
        last = df.tail(3)
        price_trend = last["close"].iloc[-1] - last["close"].iloc[0]
        ema20 = last["ema_20"].iloc[-1] if "ema_20" in last.columns else None
        ema50 = last["ema_50"].iloc[-1] if "ema_50" in last.columns else None

        if ema20 is not None and ema50 is not None:
            strong_down = price_trend < 0 and ema20 < ema50
            strong_up = price_trend > 0 and ema20 > ema50

            if final_signal == "BUY" and strong_down:
                final_signal = "HOLD"

            if final_signal == "HOLD" and strong_down and p_good < 0.45:
                final_signal = "SELL"

            if final_signal == "HOLD" and strong_up and p_good > 0.55:
                final_signal = "BUY"

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": interval,
        "prediction": final_signal,
        "p_good": p_good,
        "time": int(latest["time"]),
    }

# -------------------------------
# Multi-timeframe summary + Grok AI decision
# -------------------------------

MULTI_TF_LIST = ["1m", "5m", "15m", "1h", "4h", "1d"]

def build_tf_summary(symbol: str):
    symbol = symbol.upper()
    lines = []

    for tf in MULTI_TF_LIST:
        raw = get_klines(symbol, interval=tf, limit=200)
        if not raw:
            lines.append(f"{tf}: no data.")
            continue

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

        df["time"] = df["time"] // 1000
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

        macd_ind = ta.trend.MACD(
            close=df["close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()

        df = df.dropna().reset_index(drop=True)
        if len(df) == 0:
            lines.append(f"{tf}: not enough candles.")
            continue

        last = df.iloc[-1]
        price = last["close"]
        rsi = last["rsi"]
        ema20 = last["ema_20"]
        ema50 = last["ema_50"]
        macd_val = last["macd"]
        macd_sig = last["macd_signal"]

        trend = "uptrend" if ema20 > ema50 else "downtrend" if ema20 < ema50 else "flat"
        macd_state = "bullish" if macd_val > macd_sig else "bearish"

        line = (
            f"{tf}: close={price:.2f}, rsi={rsi:.1f}, "
            f"trend={trend}, macd={macd_state}."
        )
        lines.append(line)

    summary_text = "\n".join(lines)
    return summary_text

@app.get("/ai_multi_tf_signal/{symbol}")
def ai_multi_tf_signal(symbol: str):
    """
    Grok-powered multi-timeframe BUY/HOLD/SELL decision
    used by the 'Sync AI with chart' button.
    """
    if client is None:
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "decision": "HOLD",
            "reason": "Server missing GROK_API_KEY.",
        }

    summary = build_tf_summary(symbol)
    if not summary.strip():
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "decision": "HOLD",
            "reason": "No multi-timeframe data.",
        }

    system_prompt = """
You are an advanced crypto trading assistant.
You receive multi-timeframe data (1m, 5m, 15m, 1h, 4h, 1d) for a symbol.
Return ONE clear technical action: BUY, SELL, or HOLD.

You MUST respond in EXACT JSON:

{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": "<number between 0 and 100>",
  "reason": "<one short sentence>"
}
"""

    user_prompt = f"Symbol: {symbol.upper()}\n\nMulti-timeframe summary:\n{summary}"

    try:
        resp = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            temperature=0.4,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print("[AI_MULTI_TF] Grok error:", e)
        return {
            "ok": False,
            "symbol": symbol.upper(),
            "decision": "HOLD",
            "reason": f"Error talking to Grok: {e}",
        }

    try:
        data = json.loads(content)
        decision = str(data.get("decision", "HOLD")).upper()
        confidence = data.get("confidence")
        reason = str(data.get("reason", ""))
    except Exception:
        text = content.upper()
        if "BUY" in text and "SELL" not in text:
            decision = "BUY"
        elif "SELL" in text and "BUY" not in text:
            decision = "SELL"
        else:
            decision = "HOLD"
        confidence = None
        reason = content[:200]

    if decision not in ("BUY", "SELL", "HOLD"):
        decision = "HOLD"

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
    }
