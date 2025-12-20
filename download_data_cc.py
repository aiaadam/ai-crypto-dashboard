import os
import requests
import pandas as pd

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2/histominute"

FSYM = "BTC"
TSYM = "USDT"

def fetch_cc(fsym: str, tsym: str, aggregate: int, limit: int = 2000) -> pd.DataFrame:
    params = {
        "fsym": fsym,
        "tsym": tsym,
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
        rows.append([
            int(b["time"]),
            float(b["open"]),
            float(b["high"]),
            float(b["low"]),
            float(b["close"]),
            float(b["volumefrom"]),
        ])
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])

def main():
    os.makedirs("data", exist_ok=True)

    timeframes = [
        (1,   "1m"),
        (5,   "5m"),
        (15,  "15m"),
        # 1h via histohour would be a different endpoint; skip for now
    ]

    for agg, name in timeframes:
        print(f"[DATA] Downloading BTC/USDT {name} (aggregate={agg}) from CryptoCompare...")
        df = fetch_cc(FSYM, TSYM, aggregate=agg, limit=2000)
        out_path = f"data/btcusdt_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"[DATA] Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
