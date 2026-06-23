"""
Data fetching: ENTSO-E load, generation, capacity, day-ahead prices + fuel/carbon loaders.
All raw pulls are cached to data/raw/ so re-runs are free.
"""

import os
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BIDDING_ZONE = "10YNL----------L"   # Netherlands
START = pd.Timestamp("2023-01-01", tz="Europe/Amsterdam")
END   = pd.Timestamp("2024-01-01", tz="Europe/Amsterdam")


def _get_client():
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "ENTSOE_API_TOKEN not set. Copy .env.example to .env and add your token."
        )
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=token)


def _cache(name: str, fetch_fn):
    path = RAW / f"{name}.pkl"
    if path.exists():
        print(f"  [cache] {name}")
        return pickle.loads(path.read_bytes())
    print(f"  [fetch] {name} ...")
    data = fetch_fn()
    path.write_bytes(pickle.dumps(data))
    return data


def fetch_load(zone=BIDDING_ZONE, start=START, end=END) -> pd.Series:
    client = _get_client()
    return _cache(
        f"load_{zone}",
        lambda: client.query_load(zone, start=start, end=end)
    )


def fetch_generation(zone=BIDDING_ZONE, start=START, end=END) -> pd.DataFrame:
    client = _get_client()
    return _cache(
        f"generation_{zone}",
        lambda: client.query_generation(zone, start=start, end=end)
    )


def fetch_installed_capacity(zone=BIDDING_ZONE, start=START, end=END) -> pd.DataFrame:
    client = _get_client()
    return _cache(
        f"capacity_{zone}",
        lambda: client.query_installed_generation_capacity(zone, start=start, end=end)
    )


def fetch_day_ahead_prices(zone=BIDDING_ZONE, start=START, end=END) -> pd.Series:
    client = _get_client()
    return _cache(
        f"da_prices_{zone}",
        lambda: client.query_day_ahead_prices(zone, start=start, end=end)
    )


# ---------------------------------------------------------------------------
# Fuel and carbon loaders
# Inputs: daily CSVs dropped into data/raw/ by the user.
# Expected columns:  date (YYYY-MM-DD), price
# ---------------------------------------------------------------------------

def load_fuel_csv(name: str) -> pd.Series:
    """Load a daily fuel/carbon price CSV from data/raw/<name>.csv."""
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path}. Download the daily series and save it there.\n"
            "  TTF gas (EUR/MWh): e.g. from EEX, ICE, or a data vendor.\n"
            "  API2 coal (USD/t): convert to EUR/MWh_th in build_panel().\n"
            "  EUA carbon (EUR/t CO2): EU ETS front contract."
        )
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["price"].rename(name)


def load_ttf_gas() -> pd.Series:
    return load_fuel_csv("ttf_gas_eur_mwh")


def load_api2_coal_usd_t() -> pd.Series:
    return load_fuel_csv("api2_coal_usd_t")


def load_eua_carbon() -> pd.Series:
    return load_fuel_csv("eua_carbon_eur_t")


# ---------------------------------------------------------------------------
# Build merged hourly panel
# ---------------------------------------------------------------------------

def build_panel(zone=BIDDING_ZONE, start=START, end=END) -> pd.DataFrame:
    """
    Merge all inputs into a single hourly DataFrame saved to data/processed/panel.parquet.
    Returns the DataFrame.

    Fuel/carbon prices are daily series forward-filled to hourly.
    Coal USD/t is converted to EUR/MWh_th using:
        calorific value 6.978 MWh_th/t  (API2 spec)
        FX EUR/USD: read from data/raw/eurusd.csv or default to 1.08 (document this).
    """
    print("Building hourly panel ...")

    # --- grid data ---
    load = fetch_load(zone, start, end)
    gen  = fetch_generation(zone, start, end)
    cap  = fetch_installed_capacity(zone, start, end)
    da   = fetch_day_ahead_prices(zone, start, end)

    # Resample everything to hourly UTC-naive index
    idx = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")

    def to_hourly(s):
        s = s.copy()
        if s.index.tz is not None:
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        return s.resample("h").mean().reindex(idx)

    load_h = to_hourly(load if isinstance(load, pd.Series) else load.iloc[:, 0])
    da_h   = to_hourly(da   if isinstance(da,   pd.Series) else da.iloc[:, 0])

    # Generation by type (MW, actual)
    gen_h = gen.copy()
    if gen_h.index.tz is not None:
        gen_h.index = gen_h.index.tz_convert("UTC").tz_localize(None)
    gen_h = gen_h.resample("h").mean().reindex(idx)

    # Aggregate must-run renewables
    re_cols = {
        "wind_onshore":  ["Wind Onshore"],
        "wind_offshore": ["Wind Offshore"],
        "solar":         ["Solar"],
        "hydro_ror":     ["Hydro Run-of-river and poundage"],
    }
    re = {}
    for key, cols in re_cols.items():
        available = [c for c in cols if c in gen_h.columns]
        re[key] = gen_h[available].sum(axis=1) if available else pd.Series(0.0, index=idx)

    nuclear = gen_h.get("Nuclear", pd.Series(0.0, index=idx))

    # --- fuel/carbon ---
    ttf   = load_ttf_gas()
    coal  = load_api2_coal_usd_t()
    eua   = load_eua_carbon()

    # FX: try file, else constant
    fx_path = RAW / "eurusd.csv"
    if fx_path.exists():
        fx = pd.read_csv(fx_path, parse_dates=["date"], index_col="date")["price"]
        fx = fx.reindex(pd.date_range(start, end, freq="D")).ffill().bfill()
    else:
        print("  [warn] eurusd.csv not found; using EUR/USD = 1.08 (document this)")
        fx = pd.Series(1.08, index=pd.date_range(start, end, freq="D"))

    coal_eur_mwh_th = (coal / fx) / 6.978   # USD/t -> EUR/t -> EUR/MWh_th

    def daily_to_hourly(s: pd.Series) -> pd.Series:
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        hourly_idx = idx.tz_localize(None) if idx.tz is not None else idx
        return s.reindex(hourly_idx, method="ffill").ffill().bfill()

    ttf_h  = daily_to_hourly(ttf)
    coal_h = daily_to_hourly(coal_eur_mwh_th)
    eua_h  = daily_to_hourly(eua)

    panel = pd.DataFrame({
        "load_mw":          load_h.values,
        "da_price_eur_mwh": da_h.values,
        "wind_onshore_mw":  re["wind_onshore"].values,
        "wind_offshore_mw": re["wind_offshore"].values,
        "solar_mw":         re["solar"].values,
        "hydro_ror_mw":     re["hydro_ror"].values,
        "nuclear_mw":       nuclear.values,
        "ttf_eur_mwh":      ttf_h.values,
        "coal_eur_mwh_th":  coal_h.values,
        "eua_eur_t":        eua_h.values,
    }, index=idx)

    panel["residual_demand_mw"] = (
        panel["load_mw"]
        - panel["wind_onshore_mw"]
        - panel["wind_offshore_mw"]
        - panel["solar_mw"]
        - panel["hydro_ror_mw"]
        - panel["nuclear_mw"]
    ).clip(lower=0)

    out = ROOT / "data" / "processed" / "panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"  Panel saved to {out}  shape={panel.shape}")
    return panel


if __name__ == "__main__":
    panel = build_panel()
    print(panel.describe())
