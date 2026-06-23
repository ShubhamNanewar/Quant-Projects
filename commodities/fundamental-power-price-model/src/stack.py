"""
Merit-order construction: short-run marginal cost (SRMC) per technology per hour.

Formula
-------
    SRMC_i = (P_fuel_i / η_i)  +  (P_CO2 · EF_i / η_i)  +  VOM_i

where
    P_fuel_i   fuel price          [EUR / MWh_thermal]
    η_i        net electrical efficiency (LHV basis)  [MWh_e / MWh_th]
    P_CO2      EU ETS carbon price [EUR / t CO2]
    EF_i       emission factor     [t CO2 / MWh_thermal]
    VOM_i      variable O&M        [EUR / MWh_e]

The first term is the fuel cost per MWh of electricity produced.
The second term is the carbon cost: the CO2 emitted per MWh of fuel burned,
  divided by efficiency to express it per MWh of electricity.

Technology parameters
---------------------
Source: Schröder et al. (2013), "Current and Prospective Costs of Electricity
Generation until 2050", DIW Data Documentation 68, Berlin, Table A1.
Supplemented by ENTSO-E TYNDP 2022 technology assumptions (Table 2).

Nuclear
  Nuclear's SRMC is dominated by fixed costs (capital, fuel fabrication, waste).
  The variable cost of running an already-built nuclear plant is very low:
  ~3–6 EUR/MWh_e in European literature. We use 4 EUR/MWh_e (DIW Table A1).
  Fuel cost is excluded from the SRMC formula because uranium cost per MWh_e
  is approximately 0.5–1.5 EUR/MWh_e and nuclear does not face EU ETS carbon cost.
  Nuclear always sits at the bottom of the thermal merit order.

Lignite (brown coal)
  Lignite is not traded on international markets (unlike hard coal). Its cost is
  dominated by fixed mining costs. API2 hard coal prices are used here only as a
  proxy because no public daily lignite price series exists. In practice NL has
  zero lignite capacity so this approximation has no effect on NL results.
  The approximation is documented and flagged for the DE-LU robustness run.

CCGT vs OCGT
  Both burn gas (TTF price). CCGT has higher capital cost but higher efficiency,
  making it cheaper per MWh at moderate load factors. OCGT is the peaking unit:
  lower capital cost, lower efficiency, higher SRMC, committed only when CCGT is
  full or when short run-times make CCGT's higher start-up cost unattractive
  (not modelled in v1; OCGT simply has higher SRMC and thus clears after CCGT).

Oil
  Fuel oil (heavy fuel oil, HFO) is not traded daily on a public free series.
  We approximate the HFO price as a fixed spread above crude: 80 EUR/MWh_th,
  which corresponds roughly to ~$500/t HFO at 1.08 EUR/USD and 11 MWh_th/t.
  This is a limitation; oil-fired generation is a small fraction of the NL stack
  and only runs in extreme scarcity. The sensitivity of modelled prices to this
  assumption is low. See docs/limitations.md.

Fuel-switch price
  The gas-to-coal fuel-switch price is the TTF level at which CCGT and hard coal
  have identical SRMC. Above the switch price, hard coal is cheaper and sets the
  price; below it, gas does. This is analytically derived and checked against the
  observed marginal technology in the dispatch output.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Technology parameter table
# (efficiency, emission_factor [t CO2/MWh_th], VOM [EUR/MWh_e], fuel_key)
# fuel_key references a column in the hourly panel; None = fixed cost below.
# ---------------------------------------------------------------------------

TECH_PARAMS: dict[str, tuple] = {
    "nuclear":   (None,  0.0,   4.0,  None),
    "lignite":   (0.38,  0.364, 3.0,  "coal_eur_mwh_th"),
    "hard_coal": (0.40,  0.341, 4.0,  "coal_eur_mwh_th"),
    "ccgt":      (0.54,  0.202, 3.0,  "ttf_eur_mwh"),
    "ocgt":      (0.38,  0.202, 8.0,  "ttf_eur_mwh"),
    "oil":       (0.35,  0.266, 15.0, None),
}

NUCLEAR_SRMC_EUR_MWH = 4.0    # EUR/MWh_e — variable cost only (DIW 2013, Table A1)
OIL_PRICE_EUR_MWH_TH = 80.0  # EUR/MWh_th — approximate HFO price; see module docstring

# Emission factors reference:
# IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Vol. 2, Table 2.2
# Lignite:    ~0.364 t CO2/MWh_th  (101 t CO2/TJ × 3.6 MJ/kWh)
# Hard coal:  ~0.341 t CO2/MWh_th  (94.6 t CO2/TJ)
# Natural gas: ~0.202 t CO2/MWh_th (56.1 t CO2/TJ)
# Oil:         ~0.266 t CO2/MWh_th (73.3 t CO2/TJ)


def srmc_single(
    tech: str,
    fuel_price: float,
    co2_price: float,
) -> float:
    """
    SRMC for one technology at given fuel and carbon prices (scalar inputs).
    Useful for sensitivity analysis and the fuel-switch calculation.
    """
    eff, ef_co2, vom, _ = TECH_PARAMS[tech]
    if tech == "nuclear":
        return NUCLEAR_SRMC_EUR_MWH
    if tech == "oil":
        return (OIL_PRICE_EUR_MWH_TH / eff) + (co2_price * ef_co2 / eff) + vom
    return (fuel_price / eff) + (co2_price * ef_co2 / eff) + vom


def build_stack(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the hourly SRMC for every technology in the stack.

    Parameters
    ----------
    panel : hourly panel from data_fetch.build_panel()
            must contain columns: ttf_eur_mwh, coal_eur_mwh_th, eua_eur_t

    Returns
    -------
    DataFrame with index = panel.index and columns = technology names.
    Each cell is the SRMC in EUR/MWh_e for that hour and technology.
    """
    stack = pd.DataFrame(index=panel.index)

    for tech, (eff, ef_co2, vom, fuel_key) in TECH_PARAMS.items():
        if tech == "nuclear":
            stack[tech] = NUCLEAR_SRMC_EUR_MWH
        elif tech == "oil":
            stack[tech] = (OIL_PRICE_EUR_MWH_TH / eff) + (panel["eua_eur_t"] * ef_co2 / eff) + vom
        else:
            stack[tech] = (panel[fuel_key] / eff) + (panel["eua_eur_t"] * ef_co2 / eff) + vom

    return stack


def fuel_switch_price_gas_vs_coal(co2_price: float, coal_price_eur_mwh_th: float) -> float:
    """
    Analytic TTF gas price (EUR/MWh_th) at which CCGT and hard coal have equal SRMC.

    Derivation:
        SRMC_ccgt(gas*) = SRMC_coal
        gas*/η_ccgt + P_CO2·EF_ccgt/η_ccgt + VOM_ccgt
            = coal/η_coal + P_CO2·EF_coal/η_coal + VOM_coal

        Solving for gas*:
        gas* = η_ccgt · [SRMC_coal − VOM_ccgt − P_CO2·EF_ccgt/η_ccgt]

    When actual TTF > gas*: hard coal is cheaper → coal sets the price.
    When actual TTF < gas*: CCGT is cheaper → gas sets the price.

    Parameters
    ----------
    co2_price            : EUA price in EUR/t CO2
    coal_price_eur_mwh_th: API2 coal price in EUR/MWh_th
    """
    eff_ccgt, ef_ccgt, vom_ccgt, _ = TECH_PARAMS["ccgt"]
    eff_coal, ef_coal, vom_coal, _ = TECH_PARAMS["hard_coal"]

    srmc_coal = (coal_price_eur_mwh_th / eff_coal) + (co2_price * ef_coal / eff_coal) + vom_coal
    gas_star  = eff_ccgt * (srmc_coal - vom_ccgt - co2_price * ef_ccgt / eff_ccgt)
    return gas_star


def srmc_table_at(ttf: float, coal_eur_mwh_th: float, eua: float) -> pd.Series:
    """Return a Series of SRMC values for all technologies at given prices (for inspection)."""
    results = {}
    for tech in TECH_PARAMS:
        fuel = ttf if TECH_PARAMS[tech][3] == "ttf_eur_mwh" else coal_eur_mwh_th
        results[tech] = srmc_single(tech, fuel, eua)
    return pd.Series(results).sort_values()


if __name__ == "__main__":
    print("SRMC merit order at representative 2023 prices:")
    print("  TTF = 40 EUR/MWh_th,  Coal = 15 EUR/MWh_th,  EUA = 90 EUR/t\n")
    table = srmc_table_at(ttf=40, coal_eur_mwh_th=15, eua=90)
    for tech, cost in table.items():
        print(f"  {tech:12s}: {cost:6.1f} EUR/MWh_e")

    switch = fuel_switch_price_gas_vs_coal(co2_price=90, coal_price_eur_mwh_th=15)
    print(f"\nGas-to-coal switch price: {switch:.1f} EUR/MWh_th")
    print(f"  → If TTF < {switch:.1f}, CCGT is cheaper than hard coal.")
    print(f"  → If TTF > {switch:.1f}, hard coal is cheaper.")
