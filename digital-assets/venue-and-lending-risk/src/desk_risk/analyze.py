"""Rebuild tables and figures from an archived collection, without network access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/desk-risk-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .metrics import (book_metrics, round_trip, debt_from_shares, position_health,
                      shocked_price, withdrawal_capacity)

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def elapsed(a, b):
    return (pd.Timestamp(a) - pd.Timestamp(b)).total_seconds()


def verify(directory):
    manifest = read(directory / "manifest.json")
    for request in manifest["requests"]:
        actual = hashlib.sha256((directory / request["file"]).read_bytes()).hexdigest()
        if actual != request["sha256"]:
            raise ValueError(f"Checksum mismatch: {request['file']}")
    return manifest


def parse_book(payload, venue):
    if venue == "kraken":
        if payload["error"]:
            raise ValueError(payload["error"])
        payload = next(iter(payload["result"].values()))
    return tuple([(float(x[0]), float(x[1])) for x in payload[side]] for side in ("bids", "asks"))


def exchange_tables(directory, manifest):
    records = {r["file"]: r for r in manifest["requests"]}
    rows, opportunities = [], []
    for path in sorted(directory.glob("coinbase_book_*.json")):
        suffix = path.stem.rsplit("_", 1)[-1]
        paired = {}
        for venue in ("coinbase", "kraken"):
            file = f"{venue}_book_{suffix}.json"
            record = records[file]
            timing = record["attempts"][-1]
            payload = read(directory / file)
            bids, asks = parse_book(payload, venue)
            stats = book_metrics(bids, asks)
            age = float(record["cache_age"] or 0)
            server_delay = elapsed(timing["finished"], record["http_date"]) if record["http_date"] else math.nan
            # An HTTP date is not a quote timestamp. Flag it separately.
            row = {"round": int(suffix), "venue": venue, "received_at": timing["finished"],
                   "elapsed_ms": timing["elapsed_ms"], "cache_age_seconds": age,
                   "http_date_age_seconds": server_delay,
                   "source_quote_timestamp_available": bool(payload.get("time")),
                   "source_quote_age_seconds": elapsed(timing["finished"], payload["time"]) if payload.get("time") else math.nan,
                   "auction_mode": bool(payload.get("auction_mode", False)),
                   "bid_levels": len(bids), "ask_levels": len(asks), **stats}
            rows.append(row)
            paired[venue] = (bids, asks, row)
        cb, kr = paired["coinbase"], paired["kraken"]
        skew = abs(elapsed(cb[2]["received_at"], kr[2]["received_at"]))
        for buyer, seller in (("coinbase", "kraken"), ("kraken", "coinbase")):
            buy, sell = paired[buyer], paired[seller]
            for notional in (1000, 10000, 50000, 100000):
                quantity = notional / buy[2]["mid"]
                for fee in (0, 5, 10, 25, 40, 60):
                    result = round_trip(buy[1], sell[0], quantity, fee, fee)
                    opportunities.append({"round": int(suffix), "buy_venue": buyer,
                        "sell_venue": seller, "target_notional_usd": notional,
                        "quantity_eth": quantity, "fee_bps_per_leg": fee,
                        "receipt_skew_seconds": skew, "depth_sufficient": result is not None,
                        "timing_gate": skew <= 2 and all(paired[v][2]["elapsed_ms"] <= 2000
                            and paired[v][2]["cache_age_seconds"] <= 2
                            and not paired[v][2]["auction_mode"]
                            and (not paired[v][2]["source_quote_timestamp_available"] or
                                 -1 <= paired[v][2]["source_quote_age_seconds"] <= 2) for v in paired),
                        **(result or {"buy_vwap": None, "sell_vwap": None,
                                     "gross_bps": None, "net_bps": None, "net_usd": None})})
    return pd.DataFrame(rows), pd.DataFrame(opportunities)


def candle_table(directory, manifest):
    cutoffs = {r["file"]: pd.Timestamp(r["attempts"][-1]["started"]).floor("h").timestamp()
               for r in manifest["requests"] if "hourly" in r["file"]}
    cb = pd.DataFrame(read(directory / "coinbase_hourly.json"),
                      columns=["timestamp", "low", "high", "open", "close", "volume"])
    result = read(directory / "kraken_hourly.json")["result"]
    key = next(k for k in result if k != "last")
    kr = pd.DataFrame(result[key], columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    # Kraken's final candle is always incomplete. Exclude it even on delayed captures.
    kr = kr.iloc[:-1].copy()
    frames = []
    for venue, frame in [("coinbase", cb), ("kraken", kr)]:
        frame = frame.apply(pd.to_numeric).sort_values("timestamp")
        frame = frame[frame.timestamp < cutoffs[f"{venue}_hourly.json"]].copy()
        if frame.timestamp.duplicated().any():
            raise ValueError(f"Duplicate candles: {venue}")
        if not ((frame.low <= frame[["open", "close"]].min(axis=1)) &
                (frame.high >= frame[["open", "close"]].max(axis=1)) &
                (frame.low > 0) & (frame.volume >= 0)).all():
            raise ValueError(f"Invalid OHLC bars: {venue}")
        frame["venue"] = venue
        frame["utc"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
        frame["log_return"] = np.log(frame.close).diff()
        frame.loc[frame.timestamp.diff() != 3600, "log_return"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def market_table(directory, manifest):
    markets = read(directory / "morpho_markets.json")["data"]["markets"]["items"]
    rows = []
    for m in markets:
        s = m["state"]
        supply, borrow = int(s["supplyAssets"]), int(s["borrowAssets"])
        u = borrow / supply if supply else math.nan
        rows.append({"market_id": m["marketId"], "loan": m["loanAsset"]["symbol"],
            "collateral": (m["collateralAsset"] or {}).get("symbol", "none"),
            "loan_address": m["loanAsset"]["address"],
            "loan_decimals": m["loanAsset"]["decimals"],
            "lltv": int(m["lltv"]) / 1e18, "supply_usd": s["supplyAssetsUsd"],
            "borrow_usd": s["borrowAssetsUsd"], "liquidity_usd": s["liquidityAssetsUsd"],
            "utilization": u, "api_utilization": s["utilization"],
            "supply_apy": s["supplyApy"], "borrow_apy": s["borrowApy"],
            "state_timestamp": int(s["timestamp"]), "block": int(s["blockNumber"]),
            "state_age_seconds": pd.Timestamp(manifest["completed_at"]).timestamp() - int(s["timestamp"]),
            "oracle": (m["oracle"] or {}).get("address"),
            "oracle_type": (m["oracle"] or {}).get("type"),
            "oracle_price_raw": s["price"],
            "warnings": ";".join(str(w["type"]) for w in m["warnings"])})
    return pd.DataFrame(rows), {m["marketId"]: m for m in markets}


def position_tables(directory, markets):
    rows, seen = [], set()
    for path in sorted(directory.glob("morpho_positions_*.json")):
        for p in read(path)["data"]["marketPositions"]["items"]:
            if p["id"] in seen:
                raise ValueError("Duplicate position across pages; recollect before analysis")
            seen.add(p["id"])
            m = markets[p["market"]["marketId"]]
            s, ps = m["state"], p["state"]
            if ps is None or s["price"] is None:
                raise ValueError("Missing state or oracle price")
            debt = debt_from_shares(ps["borrowShares"], s["borrowAssets"], s["borrowShares"])
            health = position_health(ps["collateral"], s["price"], debt, m["lltv"])
            # Hash public position identifiers in the presentation table, not in the raw archive.
            label = hashlib.sha256(p["id"].encode()).hexdigest()[:12]
            for shock in (0, .05, .10, .20, .30, .40):
                stressed = position_health(ps["collateral"], shocked_price(s["price"], shock), debt, m["lltv"])
                rows.append({"position": label, "market_id": m["marketId"],
                    "collateral": m["collateralAsset"]["symbol"], "loan": m["loanAsset"]["symbol"],
                    "shock": shock, "debt_loan_units": debt / 10 ** m["loanAsset"]["decimals"],
                    "reported_borrow_usd": ps["borrowAssetsUsd"],
                    "health_factor": health["health_factor"],
                    "api_health_factor": p["healthFactor"],
                    "stressed_health_factor": stressed["health_factor"],
                    "liquidatable": stressed["liquidatable"],
                    "position_state_timestamp": int(ps["timestamp"]),
                    "market_state_timestamp": int(s["timestamp"]),
                    "state_time_gap_seconds": abs(int(ps["timestamp"]) - int(s["timestamp"]))})
    positions = pd.DataFrame(rows)
    stress = positions.assign(exposed_debt=lambda d: d.debt_loan_units * d.liquidatable).groupby(
        ["market_id", "collateral", "loan", "shock"], as_index=False).agg(
        sampled_debt_loan_units=("debt_loan_units", "sum"),
        debt_beyond_threshold=("exposed_debt", "sum"), positions=("position", "count"),
        breached_positions=("liquidatable", "sum"))
    stress["breached_debt_fraction"] = stress.debt_beyond_threshold / stress.sampled_debt_loan_units
    coverage = positions[positions.shock == 0].groupby("market_id", as_index=False).agg(
        sampled_debt=("debt_loan_units", "sum"), positions=("position", "count"))
    coverage["market_borrow"] = coverage.market_id.map(lambda key:
        int(markets[key]["state"]["borrowAssets"]) / 10 ** markets[key]["loanAsset"]["decimals"])
    coverage["debt_coverage"] = coverage.sampled_debt / coverage.market_borrow
    return positions, stress, coverage


def vault_tables(directory):
    vaults = read(directory / "morpho_vaults.json")["data"]["vaults"]["items"]
    allocations, summaries = [], []
    for v in vaults:
        s = v["state"]
        total = int(s["totalAssets"])
        vault_rows = []
        for a in s["allocation"]:
            amount = int(a["supplyAssets"])
            if amount <= 0:
                continue
            m, ms = a["market"], a["market"]["state"]
            capacity = withdrawal_capacity(amount, ms["supplyAssets"], ms["borrowAssets"])
            # Only the withdrawal queue is immediately accessible via normal vault withdrawals.
            eligible = a["withdrawQueueIndex"] is not None
            row = {"vault": v["name"], "vault_address": v["address"], "market_id": m["marketId"],
                "collateral": (m["collateralAsset"] or {}).get("symbol", "idle"),
                "collateral_address": (m["collateralAsset"] or {}).get("address", "idle"),
                "allocation_loan_units": amount / 10 ** v["asset"]["decimals"],
                "weight": amount / total if total else math.nan,
                "utilization": ms["utilization"],
                "withdrawable_loan_units": capacity / 10 ** v["asset"]["decimals"] if eligible else 0,
                "withdraw_queue": a["withdrawQueueIndex"],
                "market_state_timestamp": int(ms["timestamp"])}
            allocations.append(row); vault_rows.append(row)
        frame = pd.DataFrame(vault_rows)
        denominator = total / 10 ** v["asset"]["decimals"]
        groups = frame.groupby("collateral_address").weight.sum()
        summaries.append({"vault": v["name"], "address": v["address"],
            "total_assets_usd": s["totalAssetsUsd"], "market_count": len(frame),
            "allocation_coverage": frame.weight.sum(),
            "largest_collateral_weight": groups.max(), "collateral_hhi": (groups ** 2).sum(),
            "high_utilization_weight": frame.loc[frame.utilization >= .95, "weight"].sum(),
            "immediate_withdrawal_fraction": frame.withdrawable_loan_units.sum() / denominator,
            "timelock_hours": int(s["timelock"]) / 3600,
            "state_timestamp": int(s["timestamp"]),
            "warnings": ";".join(str(w["type"]) for w in v["warnings"])})
    return pd.DataFrame(allocations), pd.DataFrame(summaries)


def oracle_table(directory, markets, manifest):
    path = directory / "morpho_oracles_end.json"
    if not path.exists():
        return pd.DataFrame()
    later = read(path)["data"]["markets"]["items"]
    rows = []
    for m in later:
        if m["marketId"] not in markets:
            continue
        first = markets[m["marketId"]]
        old, new = first["state"], m["state"]
        oracle = m["oracle"] or {}
        detail = oracle.get("data") or {}
        feeds = [detail.get(k) for k in ("baseFeedOne", "baseFeedTwo", "quoteFeedOne", "quoteFeedTwo")]
        feeds = [f["address"] for f in feeds if f]
        conversion_vaults = [detail[k]["address"] for k in ("baseOracleVault", "quoteOracleVault") if detail.get(k)]
        old_price, new_price = int(old["price"]), int(new["price"])
        scale = 10 ** (first["collateralAsset"]["decimals"] - first["loanAsset"]["decimals"])
        rows.append({"market_id": m["marketId"], "collateral": first["collateralAsset"]["symbol"],
            "loan": first["loanAsset"]["symbol"], "oracle": oracle.get("address"),
            "oracle_type": oracle.get("type"), "feed_count": len(feeds),
            "feeds": ";".join(feeds), "conversion_vaults": ";".join(conversion_vaults),
            "initial_price_raw": str(old_price), "later_price_raw": str(new_price),
            "later_loan_units_per_collateral": new_price / 10**36 * scale,
            "price_change_bps": (new_price / old_price - 1) * 10000,
            "initial_state_timestamp": int(old["timestamp"]), "later_state_timestamp": int(new["timestamp"]),
            "state_advanced": int(new["timestamp"]) > int(old["timestamp"]),
            "underlying_feed_freshness": "unknown: latestRoundData not retrieved"})
    return pd.DataFrame(rows)


def alerts_table(books, markets, vaults, positions):
    alerts = []
    def add(entity, rule, severity, value, threshold, action):
        alerts.append(dict(entity=entity, rule=rule, severity=severity, value=value,
                           threshold=threshold, action=action))
    for _, m in markets.iterrows():
        if m.utilization >= .95:
            add(m.market_id, "high_utilization", "review", m.utilization, .95, "Review exit liquidity before new allocation")
        if m.state_age_seconds > 3600:
            add(m.market_id, "old_market_state", "data_review", m.state_age_seconds, 3600, "Reconcile state with RPC; not proof of a stale oracle")
        if m.warnings:
            add(m.market_id, "provider_warning", "review", m.warnings, "none", "Inspect provider warning and contract dependencies")
    for _, v in vaults.iterrows():
        if v.largest_collateral_weight > .5:
            add(v.address, "collateral_concentration", "review", v.largest_collateral_weight, .5, "Review correlated collateral and issuer exposure")
        if v.immediate_withdrawal_fraction < .1:
            add(v.address, "limited_exit_liquidity", "review", v.immediate_withdrawal_fraction, .1, "Review withdrawal queues and other lender competition")
    for _, b in books.iterrows():
        if (b.elapsed_ms > 2000 or b.cache_age_seconds > 2 or b.auction_mode or
            (b.source_quote_timestamp_available and not -1 <= b.source_quote_age_seconds <= 2)):
            add(f"{b.venue}:{b['round']}", "snapshot_timing", "reject", b.elapsed_ms, 2000, "Do not use snapshot for execution decisions")
    for _, p in positions[(positions.shock == 0) & (positions.health_factor < 1.05)].iterrows():
        add(p.position, "low_health_factor", "review", p.health_factor, 1.05, "Verify current debt and oracle on a common block")
    return pd.DataFrame(alerts, columns=["entity", "rule", "severity", "value", "threshold", "action"])


def plot_all(books, opportunities, candles, markets, stress, vaults, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140, "font.size": 10})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for venue, frame in books.groupby("venue"):
        x = pd.to_datetime(frame.received_at)
        axes[0].plot(x, frame.spread_bps, marker=".", label=venue.title())
        axes[1].plot(x, frame.ask_depth_10bps_usd / 1000, label=venue.title())
    axes[0].set(title="ETH/USD sampled spreads and displayed liquidity", ylabel="Spread (bps)")
    axes[1].set(ylabel="Ask depth within 10 bps ($000)", xlabel="Receipt time (UTC)")
    axes[0].legend(); axes[1].legend(); fig.autofmt_xdate()
    save(fig, "exchange_liquidity.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    subset = opportunities[opportunities.target_notional_usd == 10000]
    for direction, frame in subset.groupby(["buy_venue", "sell_venue"]):
        values = frame.groupby("fee_bps_per_leg").net_bps.agg(["median", "min", "max"])
        ax.plot(values.index, values["median"], marker="o", label=f"Buy {direction[0]}, sell {direction[1]}")
        ax.fill_between(values.index, values["min"], values["max"], alpha=.15)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="$10,000 paired-book opportunity screen", xlabel="Assumed taker fee per leg (bps)", ylabel="Net price difference (bps)")
    ax.legend(); save(fig, "fee_sensitivity.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for venue, frame in candles.groupby("venue"):
        axes[0].plot(frame.utc, frame.close, label=venue.title())
    axes[0].set(title="ETH/USD public hourly history", ylabel="Close (USD)"); axes[0].legend()
    kr = candles[candles.venue == "kraken"]
    axes[1].plot(kr.utc, kr.log_return.rolling(24, min_periods=24).std() * np.sqrt(24 * 365) * 100)
    axes[1].set(ylabel="24h realised volatility (% annualised)", xlabel="UTC; completed candles only")
    for axis in axes:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    save(fig, "price_and_volatility.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    m = markets.sort_values("supply_usd", ascending=False).head(15).iloc[::-1]
    labels = m.collateral + "/" + m.loan + " " + m.market_id.str[2:8]
    ax.barh(labels, m.utilization * 100, color=np.where(m.utilization >= .95, "#bd5143", "#227c9d"))
    ax.axvline(95, color="#bd5143", linestyle="--", label="Research review threshold")
    ax.set(title="Largest sampled Morpho markets", xlabel="Utilisation (%)"); ax.legend()
    save(fig, "market_utilization.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for (key, collateral), frame in stress.groupby(["market_id", "collateral"]):
        ax.plot(frame.shock * 100, frame.breached_debt_fraction * 100, marker="o", label=f"{collateral}/USDC {key[2:8]}")
    ax.set(title="Static collateral-price stress on sampled borrowers", xlabel="Collateral oracle-price decline (%)",
           ylabel="Sampled debt beyond liquidation threshold (%)")
    ax.legend(); save(fig, "borrower_stress.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y = np.arange(len(vaults))
    axes[0].barh(y, vaults.largest_collateral_weight * 100, color="#227c9d")
    axes[0].set_yticks(y, vaults.vault); axes[0].set(xlabel="Largest collateral weight (%)", title="Collateral concentration")
    axes[1].barh(y, vaults.immediate_withdrawal_fraction * 100, color="#6a994e")
    axes[1].set_yticks(y, [""] * len(y)); axes[1].set(xlabel="Current assets accessible (%)", title="Static withdrawal capacity estimate")
    save(fig, "vault_risk.png")


def main(run=None):
    directory = ROOT / "data/raw" / (run or (ROOT / "data/LATEST").read_text().strip())
    manifest = verify(directory)
    out, figs = ROOT / "outputs" / directory.name, ROOT / "figures" / directory.name
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    books, opportunities = exchange_tables(directory, manifest)
    candles = candle_table(directory, manifest)
    markets, market_map = market_table(directory, manifest)
    positions, stress, coverage = position_tables(directory, market_map)
    allocations, vaults = vault_tables(directory)
    alerts = alerts_table(books, markets, vaults, positions)
    oracles = oracle_table(directory, market_map, manifest)
    tables = dict(books=books, opportunities=opportunities, candles=candles, markets=markets,
                  positions=positions, borrower_stress=stress, position_coverage=coverage,
                  vault_allocations=allocations, vaults=vaults, alerts=alerts, oracles=oracles)
    for name, table in tables.items():
        table.to_csv(out / (name + ".csv"), index=False)
    summary = {"run": directory.name, "retrieved_at": manifest["completed_at"],
        "requests": len(manifest["requests"]), "books": len(books), "markets": len(markets),
        "positions": int((positions.shock == 0).sum()), "vaults": len(vaults),
        "snapshot_seconds": elapsed(books.received_at.max(), books.received_at.min()),
        "candles_by_venue": candles.groupby("venue").size().to_dict(),
        "spread_median_bps": books.groupby("venue").spread_bps.median().to_dict(),
        "max_gross_bps": opportunities.gross_bps.max(),
        "max_net_bps_at_10bps_per_leg": opportunities.loc[opportunities.fee_bps_per_leg == 10, "net_bps"].max(),
        "high_utilization_markets": int((markets.utilization >= .95).sum()),
        "max_utilization_reconciliation_error": abs(markets.utilization - markets.api_utilization).max(),
        "max_receipt_skew_seconds": opportunities.receipt_skew_seconds.max(),
        "oracle_observations_paired": len(oracles),
        "alerts": len(alerts), "contract_revision": manifest["settings"]["contract_revision"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    plot_all(books, opportunities, candles, markets, stress, vaults, figs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    main(parser.parse_args().run)
