import os
import requests
import pandas as pd

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2/histominute"

# BTCUSDT on CryptoCompare = BTC vs USDT
FSYM = "BTC"
TSYM = "USDT"

def fetch_cryptocompare_minutes(limit: int = 2000, aggregate: int = 15):
    """
    Fetch 15m candles (aggregate=15) from CryptoCompare.
    Max 2000 candles per request.
    """
    params = {
        "fsym": FSYM,
        "tsym": TSYM,
        "limit": limit,
        "aggregate": aggregate,
    }
    resp = requests.get(CRYPTOCOMPARE_BASE, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare error: {data.get('Message')}")
    rows = []
    for b in data["Data"]["Data"]:
        t = int(b["time"])       # seconds since epoch
        o = float(b["open"])
        h = float(b["high"])
        l = float(b["low"])
        c = float(b["close"])
        v = float(b["volumefrom"])
        rows.append([t, o, h, l, c, v])
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    return df

def main():
    os.makedirs("data", exist_ok=True)
    print("[DATA] Downloading BTC/USDT 15m candles from CryptoCompare...")
    df = fetch_cryptocompare_minutes(limit=2000, aggregate=15)
    out_path = "data/btcusdt_15m.csv"
    df.to_csv(out_path, index=False)
    print(f"[DATA] Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
