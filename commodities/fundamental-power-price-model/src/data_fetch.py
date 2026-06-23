"""
Data fetching: ENTSO-E load, generation, capacity, day-ahead prices + fuel/carbon loaders.
All raw pulls are cached to data/raw/ so re-runs are free.

Nuclear treatment:
  Nuclear is placed in the thermal stack at a low fixed SRMC and is NOT subtracted
  from residual demand before dispatch. The LP decides how much nuclear to commit.
  This avoids double-counting (subtracting nuclear from demand AND dispatching it).
  Consequence: in practice the LP always runs nuclear at full capacity because its
  SRMC is the lowest in the stack, which is the correct economic outcome.

Hydro treatment:
  Run-of-river hydro is subtracted from demand (must-run, zero SRMC, no storage).
  Reservoir hydro is NOT subtracted — it is dispatchable and should ideally sit in
  the stack at near-zero cost. For v1 we exclude it from both sides and flag it as
  a limitation: the residual demand is slightly overstated in hydro-rich hours.

Biomass/waste:
  Not modelled. Actual generation from these sources is ignored on both the supply
  and demand side. Effect is small for NL; flagged in docs/limitations.md.
"""

import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BIDDING_ZONE = "10YNL----------L"   # Netherlands, EIC code
START = pd.Timestamp("2023-01-01", tz="Europe/Amsterdam")
END   = pd.Timestamp("2024-01-01", tz="Europe/Amsterdam")


# ---------------------------------------------------------------------------
# ENTSO-E client
# ---------------------------------------------------------------------------

def _get_client():
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "ENTSOE_API_TOKEN not set. Copy .env.example to .env and fill in your token.\n"
            "Register at https://transparency.entsoe.eu/ and email transparency@entsoe.eu "
            "with 'RESTful API access' in the subject."
        )
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=token)


def _cache(name: str, fetch_fn):
    """Return cached pickle if it exists; otherwise call fetch_fn, cache, and return."""
    path = RAW / f"{name}.pkl"
    if path.exists():
        print(f"  [cache] {name}")
        return pickle.loads(path.read_bytes())
    print(f"  [fetch] {name} ...")
    data = fetch_fn()
    path.write_bytes(pickle.dumps(data))
    return data


def fetch_load(zone=BIDDING_ZONE, start=START, end=END) -> pd.Series:
    """ENTSO-E A65: actual total load (MW)."""
    client = _get_client()
    return _cache(f"load_{zone}", lambda: client.query_load(zone, start=start, end=end))


def fetch_generation(zone=BIDDING_ZONE, start=START, end=END) -> pd.DataFrame:
    """ENTSO-E A75: actual generation per production type (MW)."""
    client = _get_client()
    return _cache(f"generation_{zone}", lambda: client.query_generation(zone, start=start, end=end))


def fetch_installed_capacity(zone=BIDDING_ZONE, start=START, end=END) -> pd.DataFrame:
    """ENTSO-E A68: installed generation capacity per type (MW)."""
    client = _get_client()
    return _cache(
        f"capacity_{zone}",
        lambda: client.query_installed_generation_capacity(zone, start=start, end=end),
    )


def fetch_day_ahead_prices(zone=BIDDING_ZONE, start=START, end=END) -> pd.Series:
    """ENTSO-E A44: day-ahead market clearing price (EUR/MWh)."""
    client = _get_client()
    return _cache(f"da_prices_{zone}", lambda: client.query_day_ahead_prices(zone, start=start, end=end))


# ---------------------------------------------------------------------------
# Fuel and carbon price loaders
# ---------------------------------------------------------------------------

def load_fuel_csv(name: str) -> pd.Series:
    """
    Load a daily fuel or carbon price series from data/raw/<name>.csv.
    Expected columns: date (YYYY-MM-DD), price.
    """
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Place a CSV with columns [date, price] there.\n"
            "  ttf_gas_eur_mwh.csv   : Dutch TTF front-month (EUR/MWh thermal)\n"
            "  api2_coal_usd_t.csv   : API2 CIF ARA coal (USD/tonne)\n"
            "  eua_carbon_eur_t.csv  : EU ETS EUA December front contract (EUR/tonne CO2)\n"
            "  eurusd.csv            : EUR/USD FX rate (optional; defaults to 1.08)"
        )
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["price"].rename(name)


def load_ttf_gas()        -> pd.Series: return load_fuel_csv("ttf_gas_eur_mwh")
def load_api2_coal_usd_t()-> pd.Series: return load_fuel_csv("api2_coal_usd_t")
def load_eua_carbon()     -> pd.Series: return load_fuel_csv("eua_carbon_eur_t")


def _coal_usd_t_to_eur_mwh_th(coal_usd_t: pd.Series, fx_eur_usd: pd.Series) -> pd.Series:
    """
    Convert API2 coal from USD/tonne to EUR/MWh_thermal.

    API2 specification: gross calorific value 6,000 kcal/kg = 6.978 MWh_th/tonne.
    Source: ICE API2 contract spec.

    coal [USD/t] / FX [USD/EUR] / 6.978 [MWh_th/t] = EUR/MWh_th
    """
    coal_aligned = coal_usd_t.reindex(fx_eur_usd.index, method="ffill")
    return (coal_aligned / fx_eur_usd) / 6.978


# ---------------------------------------------------------------------------
# Installed capacity extraction
# ---------------------------------------------------------------------------

# Map from ENTSO-E production type strings to our stack technology names.
CAPACITY_MAP = {
    "Nuclear":                               "nuclear",
    "Fossil Hard coal":                      "hard_coal",
    "Fossil Brown coal/Lignite":             "lignite",
    "Fossil Gas":                            "ccgt",      # CCGT + OCGT aggregated here;
    "Fossil Oil":                            "oil",       # split via OCGT_FRACTION below
}

OCGT_FRACTION = 0.15  # fraction of installed gas capacity assumed to be OCGT


def extract_capacity_from_entsoe(cap_df: pd.DataFrame) -> dict[str, float]:
    """
    Parse the ENTSO-E A68 DataFrame into a {tech: MW} dict.

    ENTSO-E returns annual or monthly snapshots. We take the most recent row.
    Gas capacity is split into CCGT and OCGT using OCGT_FRACTION.
    """
    if isinstance(cap_df.columns, pd.MultiIndex):
        cap_df = cap_df.xs("Actual Aggregated", axis=1, level=1, drop_level=True)

    latest = cap_df.iloc[-1]
    result = {}
    for entsoe_name, tech in CAPACITY_MAP.items():
        val = latest.get(entsoe_name, 0.0)
        val = float(val) if pd.notna(val) else 0.0
        if tech == "ccgt":
            result["ccgt"]  = val * (1 - OCGT_FRACTION)
            result["ocgt"]  = val * OCGT_FRACTION
        else:
            result[tech] = result.get(tech, 0.0) + val

    # Fill any missing tech with zero
    for t in ["nuclear", "lignite", "hard_coal", "ccgt", "ocgt", "oil"]:
        result.setdefault(t, 0.0)

    return result


# ---------------------------------------------------------------------------
# Build merged hourly panel
# ---------------------------------------------------------------------------

def build_panel(zone=BIDDING_ZONE, start=START, end=END) -> tuple[pd.DataFrame, dict]:
    """
    Merge all inputs into a single hourly DataFrame and extract installed capacity.

    Returns
    -------
    panel    : pd.DataFrame — hourly panel (shape ~8760 x 11)
    capacity : dict         — {tech: installed_MW} from ENTSO-E A68

    Columns in panel:
      load_mw              Total actual load
      da_price_eur_mwh     Actual day-ahead clearing price (target variable)
      wind_onshore_mw      Actual wind onshore generation
      wind_offshore_mw     Actual wind offshore generation
      solar_mw             Actual solar PV generation
      hydro_ror_mw         Actual run-of-river hydro (treated as must-run)
      ttf_eur_mwh          TTF gas price (EUR/MWh thermal) — daily, forward-filled
      coal_eur_mwh_th      API2 coal price (EUR/MWh thermal) — converted from USD/t
      eua_eur_t            EU ETS EUA price (EUR/tonne CO2) — daily, forward-filled
      residual_demand_mw   load − (wind_on + wind_off + solar + hydro_ror); floored at 0

    Note: nuclear is NOT subtracted from residual demand. It sits in the thermal
    stack and the dispatch LP commits it at full capacity (lowest SRMC). See module
    docstring for the reasoning.
    """
    print("Building hourly panel ...")

    # --- ENTSO-E grid data ---
    load_raw = fetch_load(zone, start, end)
    gen_raw  = fetch_generation(zone, start, end)
    cap_raw  = fetch_installed_capacity(zone, start, end)
    da_raw   = fetch_day_ahead_prices(zone, start, end)

    # Extract installed capacity dict from A68
    capacity = extract_capacity_from_entsoe(cap_raw)
    print(f"  Installed capacity (MW): {capacity}")

    # UTC-naive hourly index for the full year
    idx = pd.date_range(
        start.tz_convert("UTC").tz_localize(None),
        end.tz_convert("UTC").tz_localize(None),
        freq="h",
        inclusive="left",
    )

    def to_hourly_series(s) -> pd.Series:
        s = s.copy()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if s.index.tz is not None:
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        return s.resample("h").mean().reindex(idx)

    load_h = to_hourly_series(load_raw)
    da_h   = to_hourly_series(da_raw)

    # Generation by type
    gen_h = gen_raw.copy()
    if isinstance(gen_h.columns, pd.MultiIndex):
        # entsoe-py sometimes returns MultiIndex columns; take "Actual Aggregated"
        gen_h = gen_h.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
    if gen_h.index.tz is not None:
        gen_h.index = gen_h.index.tz_convert("UTC").tz_localize(None)
    gen_h = gen_h.resample("h").mean().reindex(idx)

    def gen_col(name: str) -> pd.Series:
        return gen_h[name].fillna(0.0) if name in gen_h.columns else pd.Series(0.0, index=idx)

    wind_on  = gen_col("Wind Onshore")
    wind_off = gen_col("Wind Offshore")
    solar    = gen_col("Solar")
    ror      = gen_col("Hydro Run-of-river and poundage")

    # --- Fuel and carbon prices ---
    ttf  = load_ttf_gas()
    coal = load_api2_coal_usd_t()
    eua  = load_eua_carbon()

    # EUR/USD FX for coal conversion
    fx_path = RAW / "eurusd.csv"
    if fx_path.exists():
        fx = pd.read_csv(fx_path, parse_dates=["date"], index_col="date")["price"]
    else:
        print("  [warn] eurusd.csv not found; using EUR/USD = 1.08 throughout.")
        print("         Coal SRMC sensitivity to FX is ~1% per cent move; acceptable for v1.")
        daily_idx = pd.date_range(start.date(), end.date(), freq="D")
        fx = pd.Series(1.08, index=daily_idx)

    coal_eur = _coal_usd_t_to_eur_mwh_th(coal, fx)

    def daily_to_hourly(s: pd.Series) -> pd.Series:
        """Forward-fill a daily price series onto the hourly UTC index."""
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return (
            s.reindex(idx, method="ffill")
             .ffill()   # fill any remaining gaps at the start
             .bfill()
        )

    ttf_h  = daily_to_hourly(ttf)
    coal_h = daily_to_hourly(coal_eur)
    eua_h  = daily_to_hourly(eua)

    # --- Assemble panel ---
    panel = pd.DataFrame(
        {
            "load_mw":          load_h.values,
            "da_price_eur_mwh": da_h.values,
            "wind_onshore_mw":  wind_on.values,
            "wind_offshore_mw": wind_off.values,
            "solar_mw":         solar.values,
            "hydro_ror_mw":     ror.values,
            "ttf_eur_mwh":      ttf_h.values,
            "coal_eur_mwh_th":  coal_h.values,
            "eua_eur_t":        eua_h.values,
        },
        index=idx,
    )

    # Residual demand = load minus must-run renewables (VRE + run-of-river hydro).
    # Nuclear is excluded here — it is dispatched by the LP at the bottom of the stack.
    panel["residual_demand_mw"] = (
        panel["load_mw"]
        - panel["wind_onshore_mw"]
        - panel["wind_offshore_mw"]
        - panel["solar_mw"]
        - panel["hydro_ror_mw"]
    ).clip(lower=0)

    out = ROOT / "data" / "processed" / "panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"  Panel saved → {out}  shape={panel.shape}")

    cap_path = ROOT / "data" / "processed" / "capacity.csv"
    pd.Series(capacity).to_csv(cap_path, header=["installed_mw"])
    print(f"  Capacity saved → {cap_path}")

    return panel, capacity


if __name__ == "__main__":
    panel, capacity = build_panel()
    print("\nCapacity (MW):")
    for k, v in capacity.items():
        print(f"  {k:12s}: {v:,.0f}")
    print("\nPanel summary:")
    print(panel.describe().round(1))
