"""
Merit-order construction: SRMC per technology per hour.

SRMC_i = (P_fuel / eff) + (P_CO2 * EF / eff) + VOM

Technology parameters from:
  Schröder et al. (2013) "Current and Prospective Costs of Electricity Generation until 2050",
  DIW Data Documentation 68, Berlin.
  ENTSO-E TYNDP 2022 technology assumptions.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Technology parameter table
# Each entry: (efficiency, emission_factor_t_co2_per_mwh_th, vom_eur_mwh_e, fuel_key)
# fuel_key must match a column in the panel: 'ttf_eur_mwh' or 'coal_eur_mwh_th'
# Nuclear uses a fixed SRMC (no fuel price column needed).
# ---------------------------------------------------------------------------

TECH_PARAMS = {
    # name:         (eff,   ef_co2,  vom,  fuel_key)
    "nuclear":      (None,  0.0,     4.0,  None),          # fixed SRMC = 4 EUR/MWh_e
    "lignite":      (0.38,  0.36,    3.0,  "coal_eur_mwh_th"),
    "hard_coal":    (0.40,  0.34,    4.0,  "coal_eur_mwh_th"),
    "ccgt":         (0.54,  0.202,   3.0,  "ttf_eur_mwh"),
    "ocgt":         (0.38,  0.202,   8.0,  "ttf_eur_mwh"),
    "oil":          (0.35,  0.266,  15.0,  None),           # oil ~80 EUR/MWh_th assumed constant
}

NUCLEAR_SRMC = 4.0    # EUR/MWh_e  — placeholder for must-run nuclear
OIL_FUEL_EUR = 80.0  # EUR/MWh_th — approximate, flag as limitation


def srmc(
    tech: str,
    fuel_price_eur_mwh_th: float | pd.Series,
    co2_price_eur_t: float | pd.Series,
) -> float | pd.Series:
    """
    Compute short-run marginal cost for a single technology.
    fuel_price and co2_price can be scalars or equal-length Series.
    """
    eff, ef_co2, vom, _ = TECH_PARAMS[tech]
    if tech == "nuclear":
        return NUCLEAR_SRMC + vom * 0   # vom already baked into the constant
    return (fuel_price_eur_mwh_th / eff) + (co2_price_eur_t * ef_co2 / eff) + vom


def build_stack(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each hour, compute the SRMC of every thermal technology and return a
    wide DataFrame with columns: nuclear, lignite, hard_coal, ccgt, ocgt, oil.

    The result has the same index as panel.
    """
    stack = pd.DataFrame(index=panel.index)

    for tech, (eff, ef_co2, vom, fuel_key) in TECH_PARAMS.items():
        if tech == "nuclear":
            stack[tech] = NUCLEAR_SRMC
        elif tech == "oil":
            stack[tech] = (OIL_FUEL_EUR / eff) + (panel["eua_eur_t"] * ef_co2 / eff) + vom
        else:
            fuel = panel[fuel_key]
            stack[tech] = (fuel / eff) + (panel["eua_eur_t"] * ef_co2 / eff) + vom

    return stack


def merit_order(srmc_row: pd.Series) -> pd.Series:
    """Sort a single-hour SRMC series ascending (cheapest first)."""
    return srmc_row.sort_values()


def fuel_switch_price_gas_vs_coal(
    co2_price: float,
    coal_price: float,
) -> float:
    """
    Analytic TTF gas price at which CCGT overtakes hard coal in the merit order.
    Solve: SRMC_ccgt(gas*) = SRMC_coal
    gas* = (SRMC_coal - vom_ccgt - co2*ef_ccgt/eff_ccgt) * eff_ccgt
    """
    eff_ccgt,  ef_ccgt,  vom_ccgt,  _ = TECH_PARAMS["ccgt"]
    eff_coal,  ef_coal,  vom_coal,  _ = TECH_PARAMS["hard_coal"]

    srmc_coal = (coal_price / eff_coal) + (co2_price * ef_coal / eff_coal) + vom_coal
    gas_star = (srmc_coal - vom_ccgt - (co2_price * ef_ccgt / eff_ccgt)) * eff_ccgt
    return gas_star


if __name__ == "__main__":
    # Quick sanity check with representative prices
    print("SRMC table at TTF=40, Coal=15, EUA=65 EUR:")
    test = {
        "ttf_eur_mwh": 40,
        "coal_eur_mwh_th": 15,
        "eua_eur_t": 65,
    }
    row = pd.Series(test)
    for tech in TECH_PARAMS:
        if tech == "nuclear":
            print(f"  {tech:12s}: {NUCLEAR_SRMC:.1f} EUR/MWh_e")
        elif tech == "oil":
            eff, ef, vom, _ = TECH_PARAMS[tech]
            v = (OIL_FUEL_EUR / eff) + (test["eua_eur_t"] * ef / eff) + vom
            print(f"  {tech:12s}: {v:.1f} EUR/MWh_e")
        else:
            eff, ef, vom, fk = TECH_PARAMS[tech]
            v = (test[fk] / eff) + (test["eua_eur_t"] * ef / eff) + vom
            print(f"  {tech:12s}: {v:.1f} EUR/MWh_e")

    switch = fuel_switch_price_gas_vs_coal(co2_price=65, coal_price=15)
    print(f"\nGas-to-coal fuel-switch TTF price: {switch:.1f} EUR/MWh_th")
