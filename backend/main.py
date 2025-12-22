from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import os
import base64
import json
import time
from typing import List

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
    try:
        client = OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1",
            http_client=None  # FIXES proxies error
        )
        print("GROK: Connected successfully!")
    except Exception as e:
        print(f"GROK: Connection failed: {e}")
        client = None
else:
    print("GROK: API key not set")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------------------
# REAL chart-image AI analysis (KEEP WORKING)
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
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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
# MOCK DATA FUNCTIONS (NO pandas needed)
# -------------------------------
def get_mock_klines(symbol: str, interval: str = "1h") -> List[List]:
    """Returns realistic mock OHLCV data for any symbol/interval"""
    symbol = symbol.upper()
    current_time = int(time.time())
    
    # Realistic price levels based on symbol
    if "BTC" in symbol:
        base_price = 95200
    elif "ETH" in symbol:
        base_price = 3250
    else:
        base_price = 1.25
    
    klines = []
    for i in range(200):
        t = current_time - (200 - i) * 60 * 5  # 5min spacing
        price_change = (i - 100) * 0.1  # Trending up
        o = base_price + price_change - 20
        h = base_price + price_change + 30
        l = base_price + price_change - 30
        c = base_price + price_change + (i % 3 - 1) * 5
        v = 150 + (i % 20) * 10
        
        klines.append([
            t * 1000,  # time (ms)
            str(o),    # open
            str(h),    # high  
            str(l),    # low
            str(c),    # close
            str(v),    # volume
            "0", "0", "0", "0", "0", "0",
        ])
    
    return klines

def generate_mock_signals(interval: str) -> List[str]:
    """Generates realistic BUY/HOLD/SELL signals"""
    signals = ["HOLD"] * 200
    # Recent BUY trend
    for i in range(150, 200):
        if i % 3 == 0:
            signals[i] = "BUY"
        elif i % 5 == 0:
            signals[i] = "SELL"
    return signals

# -------------------------------
# MOCK /crypto/{symbol} endpoint
# -------------------------------
@app.get("/crypto/{symbol}")
def crypto(symbol: str, interval: str = Query("1h")):
    print(f"[CRYPTO] symbol={symbol.upper()} interval={interval}")
    
    # MOCK DATA - deploys instantly
    data = get_mock_klines(symbol, interval)
    
    # Extract arrays for frontend
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
# MOCK /predict/{symbol} endpoint
# -------------------------------
@app.get("/predict/{symbol}")
def predict(symbol: str, timeframe: str = Query("15m", regex="^(1m|5m|15m)$")):
    """Mock ML prediction - always trending bullish"""
    current_time = int(time.time())
    
    # Realistic confidence based on timeframe
    if timeframe == "1m":
        p_good = 0.68
        prediction = "BUY"
    else:
        p_good = 0.72
        prediction = "BUY"
    
    return {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": timeframe,
        "prediction": prediction,
        "p_good": p_good,
        "time": current_time,
    }

# -------------------------------
# Mock multi-timeframe AI (uses real Grok if available)
# -------------------------------
MULTI_TF_MOCK_SUMMARY = """
1m: close=95234.50, rsi=58.2, trend=uptrend, macd=bullish.
5m: close=95241.20, rsi=62.1, trend=uptrend, macd=bullish.
15m: close=95248.90, rsi=65.4, trend=uptrend, macd=bullish.
1h: close=95255.30, rsi=67.8, trend=uptrend, macd=bullish.
4h: close=95262.10, rsi=69.2, trend=uptrend, macd=bullish.
1d: close=95270.00, rsi=71.5, trend=uptrend, macd=bullish.
"""

@app.get("/ai_multi_tf_signal/{symbol}")
def ai_multi_tf_signal(symbol: str):
    """Grok-powered OR mock multi-timeframe decision"""
    if client:
        # Try real Grok first
        try:
            system_prompt = """
You are an advanced crypto trading assistant.
Return ONE clear technical action: BUY, SELL, or HOLD in EXACT JSON:
{"decision": "BUY" | "SELL" | "HOLD", "confidence": "<0-100>", "reason": "<short>"}"""
            
            user_prompt = f"Symbol: {symbol.upper()}\n\n{MULTI_TF_MOCK_SUMMARY}"
            
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
            
            try:
                data = json.loads(content)
                decision = str(data.get("decision", "BUY")).upper()
                confidence = data.get("confidence", 75)
                reason = str(data.get("reason", "Multi-TF bullish alignment"))
            except:
                decision = "BUY"
                confidence = 75
                reason = content[:100]
        except Exception as e:
            print("[AI_MULTI_TF] Grok error:", e)
            decision, confidence, reason = "BUY", 75, "Fallback mock signal"
    else:
        # Pure mock if no API key
        decision, confidence, reason = "BUY", 78, "All timeframes bullish"
    
    return {
        "ok": True,
        "symbol": symbol.upper(),
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
    }

