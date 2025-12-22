from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import pandas as pd
import ta
import os
import base64
from openai import OpenAI

# -------------------------------
# FastAPI app setup
# -------------------------------

app = FastAPI(
    title="AI Crypto Dashboard",
    version="0.8.0",
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
# OpenAI client + uploads dir
# -------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("WARNING: OPENAI_API_KEY is not set - AI endpoints will be disabled.")

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
    Accepts an uploaded chart image and returns REAL AI analysis text
    using OpenAI GPT-4o vision. Gives bias, action (BUY/SELL/HOLD),
    entry, SL, TP1, TP2, R:R, structure and risks.
    """

    # read + size check
    contents = await file.read()
    size_kb = len(contents) / 1024

    if size_kb > 4096:
        return {
            "ok": False,
            "message": "Image too large (max ~4MB)",
        }

    # save to disk
    safe_name = file.filename.replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as buffer:
        buffer.write(contents)

    # encode to base64 for OpenAI
    base64_image = encode_image_to_base64(save_path)

    # if no API key, fail nicely
    if not OPENAI_API_KEY or client is None:
        return {
            "ok": False,
            "filename": file.filename,
            "analysis": "Server missing OPENAI_API_KEY. Contact admin.",
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
            model="gpt-4o-mini",
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
# Rule-based signal (chart signals ONLY)
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

    # TUNED: looser thresholds on 1m to reduce HOLD spam
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
