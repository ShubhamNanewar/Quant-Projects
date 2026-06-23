"""
Trader-facing insight layer:
  1. Fundamental residual (richness/cheapness indicator)
  2. Price sensitivities to gas and carbon
  3. Marginal-technology timeline and share analysis
  4. Fuel-switch price recovery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Fundamental residual
# ---------------------------------------------------------------------------

def compute_residual(panel: pd.DataFrame, dispatch: pd.DataFrame) -> pd.Series:
    """actual DA price minus LP modelled price."""
    residual = panel["da_price_eur_mwh"] - dispatch["price_b"]
    residual.name = "fundamental_residual"
    return residual


def plot_residual_vs_vre(df: pd.DataFrame, residual: pd.Series, save: bool = True):
    vre = (df["wind_onshore_mw"] + df["wind_offshore_mw"] + df["solar_mw"]) / df["load_mw"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(vre, residual, alpha=0.15, s=3, c=df["ttf_eur_mwh"], cmap="viridis")
    ax.axhline(0, color="red", lw=0.8, ls="--")
    plt.colorbar(sc, ax=ax, label="TTF gas (EUR/MWh)")
    ax.set_xlabel("VRE share of load")
    ax.set_ylabel("Fundamental residual (EUR/MWh)")
    ax.set_title("Richness indicator: actual − modelled vs VRE penetration")
    plt.tight_layout()
    if save:
        fig.savefig(FIGS / "residual_vs_vre.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 2. Price sensitivities
# ---------------------------------------------------------------------------

def price_sensitivity_gas(
    panel: pd.DataFrame,
    dispatch_fn,
    srmc_stack: pd.DataFrame,
    capacity: dict,
    delta_gas: float = 1.0,
) -> pd.Series:
    """
    Numerical d(price)/d(TTF): bump TTF by delta_gas, re-run dispatch, compute delta price.
    Returns a Series of hourly sensitivities.

    For speed we use the analytic shortcut: if the marginal tech uses gas,
    sensitivity = 1 / eff_ccgt.  Otherwise 0.
    """
    from stack import TECH_PARAMS
    marginal_tech = dispatch_fn.get("marginal_tech_a", pd.Series("unknown", index=panel.index))

    gas_techs = {t for t, (_, _, _, fk) in TECH_PARAMS.items() if fk == "ttf_eur_mwh"}
    eff_map = {t: TECH_PARAMS[t][0] for t in gas_techs}

    def _sensitivity(tech):
        if tech in gas_techs:
            return 1.0 / eff_map[tech]
        return 0.0

    sens = marginal_tech.map(_sensitivity)
    sens.name = "dprice_dgas"
    return sens


def clean_spark_spread(panel: pd.DataFrame, eff_ccgt: float = 0.54) -> pd.Series:
    """Clean spark spread: DA price − gas cost of generating 1 MWh_e via CCGT."""
    spread = panel["da_price_eur_mwh"] - panel["ttf_eur_mwh"] / eff_ccgt
    spread.name = "clean_spark_spread"
    return spread


def clean_dark_spread(panel: pd.DataFrame, eff_coal: float = 0.40) -> pd.Series:
    """Clean dark spread: DA price − (coal + carbon) cost of generating 1 MWh_e via hard coal."""
    from stack import TECH_PARAMS
    _, ef_coal, _, _ = TECH_PARAMS["hard_coal"]
    cost = panel["coal_eur_mwh_th"] / eff_coal + panel["eua_eur_t"] * ef_coal / eff_coal
    spread = panel["da_price_eur_mwh"] - cost
    spread.name = "clean_dark_spread"
    return spread


# ---------------------------------------------------------------------------
# 3. Marginal-technology timeline
# ---------------------------------------------------------------------------

def marginal_tech_monthly_share(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Monthly share (%) of hours each technology sets the price (Method A)."""
    if "marginal_tech_a" not in dispatch.columns:
        raise ValueError("dispatch must contain marginal_tech_a (run Method A)")
    df = dispatch[["marginal_tech_a"]].copy()
    df["month"] = df.index.to_period("M")
    counts = df.groupby(["month", "marginal_tech_a"]).size().unstack(fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    return share


def plot_marginal_tech_stacked(share: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(14, 5))
    share.plot.area(ax=ax, stacked=True, alpha=0.85)
    ax.set_title("Monthly share of price-setting technology")
    ax.set_ylabel("% of hours")
    ax.set_xlabel("")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save:
        fig.savefig(FIGS / "marginal_tech_timeline.png", dpi=150)
    return fig


def plot_spreads(panel: pd.DataFrame, save: bool = True):
    css = clean_spark_spread(panel)
    cds = clean_dark_spread(panel)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(css.index, css.rolling(24).mean(), lw=0.8)
    axes[0].axhline(0, color="red", lw=0.8, ls="--")
    axes[0].set_title("Clean Spark Spread (24h rolling mean)")
    axes[0].set_ylabel("EUR/MWh")

    axes[1].plot(cds.index, cds.rolling(24).mean(), lw=0.8, color="saddlebrown")
    axes[1].axhline(0, color="red", lw=0.8, ls="--")
    axes[1].set_title("Clean Dark Spread (24h rolling mean)")
    axes[1].set_ylabel("EUR/MWh")

    plt.tight_layout()
    if save:
        fig.savefig(FIGS / "spreads.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. Fuel-switch recovery
# ---------------------------------------------------------------------------

def fuel_switch_analysis(panel: pd.DataFrame, dispatch: pd.DataFrame) -> pd.DataFrame:
    """
    For each hour, compute the analytic gas-to-coal switching price and compare to
    actual TTF to label whether gas or coal should theoretically be the cheaper option.
    Compare against what the dispatch model actually dispatched.
    """
    from stack import fuel_switch_price_gas_vs_coal

    switch_prices = panel.apply(
        lambda r: fuel_switch_price_gas_vs_coal(r["eua_eur_t"], r["coal_eur_mwh_th"]),
        axis=1,
    )

    df = pd.DataFrame({
        "ttf_eur_mwh":    panel["ttf_eur_mwh"],
        "switch_price":   switch_prices,
        "gas_cheaper":    panel["ttf_eur_mwh"] < switch_prices,
        "marginal_tech":  dispatch.get("marginal_tech_a", pd.Series("unknown", index=panel.index)),
    })

    return df


def run_insights(panel: pd.DataFrame, dispatch: pd.DataFrame):
    residual = compute_residual(panel, dispatch)

    df = panel.copy()
    df["residual"] = residual

    print("\n--- Insights ---")
    print(f"Mean fundamental residual: {residual.mean():.2f} EUR/MWh")
    print(f"  (positive = market consistently above fundamentals)")

    # Spreads
    css = clean_spark_spread(panel)
    cds = clean_dark_spread(panel)
    print(f"\nClean Spark Spread  mean: {css.mean():.2f}  std: {css.std():.2f}")
    print(f"Clean Dark Spread   mean: {cds.mean():.2f}  std: {cds.std():.2f}")

    # Fuel switch
    sw = fuel_switch_analysis(panel, dispatch)
    frac = sw["gas_cheaper"].mean()
    print(f"\nFraction of hours where TTF < switch price (gas cheaper): {frac:.1%}")

    # Timeline
    if "marginal_tech_a" in dispatch.columns:
        share = marginal_tech_monthly_share(dispatch)
        plot_marginal_tech_stacked(share)

    # Plots
    plot_residual_vs_vre(df, residual)
    plot_spreads(panel)

    # Save residual
    out = ROOT / "reports" / "fundamental_residual.csv"
    residual.to_csv(out)
    print(f"\n  Residual saved to {out}")

    return residual, css, cds, sw
