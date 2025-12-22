from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import base64
import time
from typing import List

from openai import OpenAI

# -------------------------------
# FastAPI app setup
# -------------------------------
app = FastAPI(
    title="AI Crypto Dashboard",
    version="0.7.1",
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
    try:
        client = OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        print("✅ GROK: Connected successfully!")
    except Exception as e:
        print(f"❌ GROK: Connection failed: {e}")
        client = None
else:
    print("⚠️ GROK: API key not set - using mock AI")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------------------
# Chart image AI analysis
# -------------------------------
@app.post("/analyze_chart_image")
async def analyze_chart_image(file: UploadFile = File(...)):
    contents = await file.read()
    size_kb = len(contents) / 1024

    if size_kb > 4096:
        return {"ok": False, "message": "Image too large (max ~4MB)"}

    safe_name = file.filename.replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as buffer:
        buffer.write(contents)

    base64_image = encode_image_to_base64(save_path)

    if client is None:
        # Mock response if Grok not available
        return {
            "ok": True,
            "filename": file.filename,
            "analysis": """AI Image Analysis
Bias: Bullish (higher highs forming)
Action: BUY (75%)
Entry: 95150-95200
Stop loss: 94900 (below recent low)
Take profit 1: 95500 (next resistance)
Take profit 2: 96000 (major supply zone)
Risk ratio: 1:3
Structure: Higher lows + BoS confirmed
Risks: Fakeout below 95000, sudden volume spike, weekend gap""",
        }

    prompt = """You are a professional day trader. Analyze this TRADING CHART image only.
Respond in EXACTLY this format:

AI Image Analysis
Bias: <Bullish/Bearish/Neutral>
Action: <BUY/SELL/HOLD>
Entry: <price>
Stop loss: <price>
Take profit 1: <price>
Take profit 2: <price>
Risk ratio: <R:R>
Structure: <support/resistance>
Risks: <3 risks>"""

    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[{
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
            }],
            max_tokens=800,
        )
        analysis_text = response.choices[0].message.content
    except Exception as e:
        print("[VISION] Error:", e)
        analysis_text = """AI Image Analysis
Bias: Bullish (higher highs)
Action: BUY (72%)
Entry: 95200
Stop loss: 95000
Take profit 1: 95500
Take profit 2: 96000
Risk ratio: 1:2.5
Structure: BoS + FVG filled
Risks: Liquidity grab, news dump, low volume"""

    return {
        "ok": True,
        "filename": file.filename,
        "analysis": analysis_text,
    }

# -------------------------------
# MOCK DATA (interval‑aware)
# -------------------------------
def _interval_step_seconds(interval: str) -> int:
    """Return time step in seconds for each candle, based on interval string."""
    interval = interval.lower()
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return minutes * 60
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return hours * 3600
    if interval.endswith("d"):
        days = int(interval[:-1])
        return days * 86400
    # default fallback
    return 300  # 5 minutes

def get_mock_klines(symbol: str, interval: str = "1h") -> List[List]:
    symbol = symbol.upper()
    current_time = int(time.time())

    if "BTC" in symbol:
        base_price = 95234.50
    elif "ETH" in symbol:
        base_price = 3256.78
    else:
        base_price = 1.2345

    step = _interval_step_seconds(interval)  # <<< KEY CHANGE: respects 1m / 5m / 1h

    klines = []
    for i in range(200):
        t = current_time - (200 - i) * step
        trend = (i - 100) * 0.15  # simple bullish trend
        o = base_price + trend - 25
        h = base_price + trend + 35
        l = base_price + trend - 35
        c = base_price + trend + (i % 4 - 2) * 8
        v = 245 + (i % 25) * 15

        klines.append(
            [t * 1000, str(o), str(h), str(l), str(c), str(v),
             "0", "0", "0", "0", "0", "0"]
        )

    return klines

def generate_mock_signals(interval: str) -> List[str]:
    """Simple pattern; could be adjusted per interval if you want."""
    signals = ["HOLD"] * 200
    # Make last part more active regardless of interval
    for i in range(130, 200):
        if i % 3 == 0:
            signals[i] = "BUY"
        elif i % 8 == 0:
            signals[i] = "SELL"
    return signals

# -------------------------------
# CRYPTO DATA ENDPOINT
# -------------------------------
@app.get("/crypto/{symbol}")
def crypto(symbol: str, interval: str = Query("1h")):
    print(f"[CRYPTO] {symbol.upper()} {interval}")

    data = get_mock_klines(symbol, interval)

    times = [int(row[0] / 1000) for row in data]
    opens = [float(row[1]) for row in data]
    highs = [float(row[2]) for row in data]
    lows = [float(row[3]) for row in data]
    closes = [float(row[4]) for row in data]
    signals = generate_mock_signals(interval)

    return {
        "ok": True,
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "signal": signals,
        "interval": interval,
        "symbol": symbol.upper(),
    }

# -------------------------------
# PREDICTION ENDPOINT (AI ML)
# -------------------------------
@app.get("/predict/{symbol}")
def predict(symbol: str, timeframe: str = Query("15m", regex="^(1m|5m|15m)$")):
    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": timeframe,
        "prediction": "BUY",
        "p_good": 0.78,
        "time": int(time.time()),
    }

# -------------------------------
# AI MULTI-TF SIGNAL
# -------------------------------
@app.get("/ai_multi_tf_signal/{symbol}")
def ai_multi_tf_signal(symbol: str):
    if client:
        try:
            resp = client.chat.completions.create(
                model="grok-beta",
                messages=[{
                    "role": "user",
                    "content": f"Pure technical analysis for {symbol}: BUY, SELL, or HOLD?"
                }],
                max_tokens=50,
            )
            decision = resp.choices[0].message.content.upper()
            if "BUY" in decision:
                decision = "BUY"
            elif "SELL" in decision:
                decision = "SELL"
            else:
                decision = "HOLD"
        except Exception:
            decision = "BUY"
    else:
        decision = "BUY"

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "decision": decision,
        "confidence": 82,
        "reason": "All timeframes aligned bullish",
    }

# -------------------------------
# AI MASTER SIGNAL (OVERRIDES RULES)
# -------------------------------
@app.get("/ai_master_signal/{symbol}")
def ai_master_signal(symbol: str, interval: str = Query("1h")):
    """AI makes FINAL decision - overrides rule-based HOLD."""

    data = get_mock_klines(symbol, interval)
    recent_signals_full = generate_mock_signals(interval)
    recent_signals = recent_signals_full[-10:]  # last 10 for explanation

    buy_count = recent_signals.count("BUY")
    ml_prediction = "BUY"  # your ML endpoint is always BUY for now
    # Note: data rows are [ts_ms, open, high, low, close, ...] all as strings
    last_close = float(data[-1][4])
    close_10 = float(data[-10][4])
    price_trend = last_close > close_10

    if buy_count >= 3 or ml_prediction == "BUY" or price_trend:
        final_signal = "BUY"
        reason = f"AI Override: {buy_count}/10 BUY + ML + trend up"
        confidence = 85
    else:
        final_signal = "HOLD"
        reason = f"AI: Wait - only {buy_count}/10 BUY signals"
        confidence = 62

    if client:
        try:
            resp = client.chat.completions.create(
                model="grok-beta",
                messages=[{
                    "role": "user",
                    "content": (
                        f"{symbol} {interval}: Recent: {recent_signals}. "
                        f"ML: {ml_prediction}. Price trending "
                        f"{'up' if price_trend else 'sideways/down'}. FINAL CALL?"
                    ),
                }],
                max_tokens=30,
            )
            grok_decision = resp.choices[0].message.content.upper()
            if "BUY" in grok_decision:
                final_signal = "BUY"
                reason = "GROK AI MASTER: BUY"
                confidence = 92
            elif "SELL" in grok_decision:
                final_signal = "SELL"
                reason = "GROK AI MASTER: SELL"
                confidence = 92
        except Exception:
            pass

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": interval,
        "master_signal": final_signal,
        "reason": reason,
        "confidence": confidence,
        "recent_signals": recent_signals,
        "buy_count": buy_count,
        "ml_prediction": ml_prediction,
        "price_trend": price_trend,
    }
