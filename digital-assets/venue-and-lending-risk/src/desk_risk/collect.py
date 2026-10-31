"""Capture public responses, including request timing and checksums."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
MORPHO = "https://api.morpho.org/graphql"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Capture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records = []

    def fetch(self, name, url, params=None, body=None, text=False):
        """Save the response before parsing it. Retry only transient failures."""
        method = "POST" if body else "GET"
        attempts = []
        for attempt in range(3):
            started = utc_now()
            clock = time.monotonic()
            try:
                r = requests.request(method, url, params=params, json=body, timeout=30,
                                     headers={"User-Agent": "public-desk-risk-research/1.0"})
                finished = utc_now()
                attempts.append({"started": started, "finished": finished,
                                 "status": r.status_code,
                                 "elapsed_ms": (time.monotonic() - clock) * 1000})
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(min(float(r.headers.get("Retry-After", 2 ** attempt)), 10))
                    continue
                payload = r.content
                path = self.directory / (name + (".txt" if text else ".json"))
                path.write_bytes(payload)
                self.records.append({"file": path.name, "url": r.url, "method": method,
                    "body": body, "attempts": attempts,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload), "http_date": r.headers.get("Date"),
                    "cache_age": r.headers.get("Age")})
                r.raise_for_status()
                result = r.text if text else r.json()
                if isinstance(result, dict) and (result.get("errors") or result.get("error")):
                    raise ValueError(f"API error in {name}: {result}")
                return result
            except requests.ConnectionError:
                attempts.append({"started": started, "error": "connection failure"})
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Request failed: {name}")

    def save_manifest(self, settings):
        (self.directory / "manifest.json").write_text(json.dumps(
            {"completed_at": utc_now(), "settings": settings,
             "requests": sorted(self.records, key=lambda r: r["file"])}, indent=2))


def collect(rounds, interval):
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    capture = Capture(ROOT / "data/raw" / run)
    settings = {"rounds": rounds, "interval_seconds": interval,
                "pair": "ETH/USD", "position_page_cap": 20,
                "selection": "30 largest listed non-idle Ethereum markets by supply USD; top 5 listed USDC V1 vaults"}
    try:
        capture.fetch("coinbase_product", "https://api.exchange.coinbase.com/products/ETH-USD")
        capture.fetch("kraken_product", "https://api.kraken.com/0/public/AssetPairs", {"pair": "ETHUSD"})
        capture.fetch("kraken_hourly", "https://api.kraken.com/0/public/OHLC", {"pair": "ETHUSD", "interval": 60})
        capture.fetch("coinbase_hourly", "https://api.exchange.coinbase.com/products/ETH-USD/candles", {"granularity": 3600})
        markets = capture.fetch("morpho_markets", MORPHO,
            body={"query": (ROOT / "queries/markets.graphql").read_text()})["data"]["markets"]["items"]
        capture.fetch("morpho_vaults", MORPHO,
            body={"query": (ROOT / "queries/vaults.graphql").read_text()})
        chosen = [m for m in markets if m["loanAsset"]["symbol"] == "USDC"][:3]
        settings["position_market_ids"] = [m["marketId"] for m in chosen]
        for i, market in enumerate(chosen):
            for page in range(settings["position_page_cap"]):
                result = capture.fetch(f"morpho_positions_{i}_{page:02d}", MORPHO,
                    body={"query": (ROOT / "queries/positions.graphql").read_text(),
                          "variables": {"ids": [market["marketId"]], "skip": page * 100}})
                if len(result["data"]["marketPositions"]["items"]) < 100:
                    break
                time.sleep(0.2)
        revision = capture.fetch("contract_revision", "https://api.github.com/repos/morpho-org/morpho-blue/commits/main")["sha"]
        settings["contract_revision"] = revision
        for name, path in [("Morpho.sol", "src/Morpho.sol"),
                           ("SharesMathLib.sol", "src/libraries/SharesMathLib.sol"),
                           ("ConstantsLib.sol", "src/libraries/ConstantsLib.sol")]:
            capture.fetch(name, f"https://raw.githubusercontent.com/morpho-org/morpho-blue/{revision}/{path}", text=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i in range(rounds):
                a = pool.submit(capture.fetch, f"coinbase_book_{i:03d}",
                    "https://api.exchange.coinbase.com/products/ETH-USD/book", {"level": 2})
                b = pool.submit(capture.fetch, f"kraken_book_{i:03d}",
                    "https://api.kraken.com/0/public/Depth", {"pair": "ETHUSD", "count": 100})
                a.result(); b.result()
                print(f"Captured book pair {i + 1}/{rounds}", flush=True)
                if i < rounds - 1:
                    time.sleep(interval)
    finally:
        capture.save_manifest(settings)
    (ROOT / "data/LATEST").write_text(run + "\n")
    print(capture.directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--interval", type=float, default=8)
    args = parser.parse_args()
    if args.rounds < 1 or args.interval < 0:
        parser.error("rounds must be positive and interval non-negative")
    collect(args.rounds, args.interval)
