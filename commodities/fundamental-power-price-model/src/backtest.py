"""
Validation: modelled clearing price against actual ENTSO-E day-ahead price.

Two types of validation:

1. Price fit — how well does the modelled price track the actual price?
   Overall metrics (MAE, RMSE, bias, correlation) and conditional metrics
   broken out by market regime: VRE penetration, demand level, gas-price
   tercile, season, day type.

2. Dispatch-mix comparison — does the LP dispatch the right fuel mix?
   Compare modelled generation per technology against actual ENTSO-E A75 output.
   If the model dispatches coal in hours when the market ran gas, the fuel-cost
   inputs or efficiency parameters are miscalibrated.

Expected honest results:
  - Good fit in hours when a thermal unit is clearly marginal (mid-demand,
    moderate VRE, stable fuel prices).
  - Poor fit in scarcity hours (price spikes the model cannot reproduce without
    VOLL), very high VRE hours (negative prices from must-run), and strongly
    import-coupled hours (NL's interconnection with BE, DE, GB).
  - The bias (mean signed error) quantifies whether the model systematically
    under- or over-prices, and in which regime.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Scalar fit metrics
# ---------------------------------------------------------------------------

def compute_metrics(actual: pd.Series, modelled: pd.Series) -> dict:
    """
    MAE, RMSE, bias, Pearson correlation, and R² between two price series.

    Bias = mean(actual − modelled).
      Positive bias: model systematically underprices.
      Negative bias: model systematically overprices.
    R² here is 1 − Var(error)/Var(actual), which can be negative if the model
    is worse than the unconditional mean.
    """
    e = actual - modelled
    valid = e.dropna()
    return {
        "n":    len(valid),
        "mae":  valid.abs().mean(),
        "rmse": np.sqrt((valid**2).mean()),
        "bias": valid.mean(),
        "corr": actual.corr(modelled),
        "r2":   1 - valid.var() / actual.reindex(valid.index).var(),
    }


def print_metrics(label: str, m: dict):
    line = (f"  n={m['n']:,}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}  "
            f"bias={m['bias']:+.1f}  corr={m['corr']:.3f}  R²={m['r2']:.3f}")
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(line)


# ---------------------------------------------------------------------------
# Conditional bucketing
# ---------------------------------------------------------------------------

def add_regime_buckets(panel: pd.DataFrame, dispatch: pd.DataFrame) -> pd.DataFrame:
    """
    Add market-regime classification columns to a joined panel + dispatch DataFrame.

    Columns added:
      vre_share      VRE (wind + solar) as fraction of total load
      vre_bucket     low / mid / high VRE penetration
      demand_bucket  off_peak / shoulder / peak (by load quantile)
      gas_bucket     gas_low / gas_mid / gas_high (by TTF tercile)
      season         winter / spring / summer / autumn
      day_type       weekday / weekend

    VRE bucket thresholds (15%, 35%):
      These are approximate for NL 2023. Check the distribution after running;
      if NL VRE share is consistently above 35%, adjust the thresholds so that
      the buckets contain enough observations to be statistically meaningful.
    """
    df = panel.join(dispatch[["price_a", "price_b"]], how="left")

    vre = (df["wind_onshore_mw"] + df["wind_offshore_mw"] + df["solar_mw"]) \
        / df["load_mw"].replace(0, np.nan)
    df["vre_share"] = vre

    df["vre_bucket"] = pd.cut(
        vre,
        bins=[-np.inf, 0.15, 0.35, np.inf],
        labels=["low_vre", "mid_vre", "high_vre"],
    )
    df["demand_bucket"] = pd.qcut(
        df["load_mw"], q=3, labels=["off_peak", "shoulder", "peak"],
    )
    df["gas_bucket"] = pd.qcut(
        df["ttf_eur_mwh"], q=3, labels=["gas_low", "gas_mid", "gas_high"],
    )
    df["season"] = df.index.month.map({
        12: "winter", 1: "winter",  2: "winter",
         3: "spring", 4: "spring",  5: "spring",
         6: "summer", 7: "summer",  8: "summer",
         9: "autumn", 10: "autumn", 11: "autumn",
    })
    df["day_type"] = np.where(df.index.dayofweek < 5, "weekday", "weekend")

    return df


def conditional_metrics_table(
    df: pd.DataFrame,
    modelled_col: str = "price_b",
) -> pd.DataFrame:
    """
    Compute fit metrics for every bucket and return a tidy DataFrame.
    This is the key diagnostic table: it shows where the model works and where it fails.
    """
    rows = []
    for bucket_col in ["vre_bucket", "demand_bucket", "gas_bucket", "season", "day_type"]:
        for label, grp in df.groupby(bucket_col, observed=True):
            m = compute_metrics(grp["da_price_eur_mwh"], grp[modelled_col])
            rows.append({"dimension": bucket_col, "regime": str(label), **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dispatch-mix comparison
# ---------------------------------------------------------------------------

def dispatch_mix_comparison(
    panel: pd.DataFrame,
    dispatch: pd.DataFrame,
    gen_actual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare modelled dispatch (MW) against actual ENTSO-E A75 generation by type.

    Parameters
    ----------
    panel      : hourly panel
    dispatch   : dispatch output with dispatched_{tech} columns
    gen_actual : raw A75 DataFrame (from data_fetch.fetch_generation)

    Returns a DataFrame with columns (actual_X, modelled_X) for each technology.

    Interpretation:
      If modelled_coal >> actual_coal in gas-price-high hours, the coal capacity
      assumption is too large or the coal SRMC is too low.
      If modelled_ccgt << actual_ccgt, the CCGT capacity may be understated.
    """
    # Map ENTSO-E generation type names to our stack names
    entsoe_to_tech = {
        "Nuclear":                           "nuclear",
        "Fossil Hard coal":                  "hard_coal",
        "Fossil Brown coal/Lignite":         "lignite",
        "Fossil Gas":                        "ccgt",
        "Fossil Oil":                        "oil",
    }

    result = pd.DataFrame(index=panel.index)

    gen_h = gen_actual.copy()
    if isinstance(gen_h.columns, pd.MultiIndex):
        gen_h = gen_h.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
    if gen_h.index.tz is not None:
        gen_h.index = gen_h.index.tz_convert("UTC").tz_localize(None)
    gen_h = gen_h.resample("h").mean().reindex(panel.index)

    for entsoe_name, tech in entsoe_to_tech.items():
        actual_col = gen_h.get(entsoe_name, pd.Series(np.nan, index=panel.index))
        modelled_col = dispatch.get(f"dispatched_{tech}", pd.Series(np.nan, index=panel.index))
        result[f"actual_{tech}"]   = actual_col.values
        result[f"modelled_{tech}"] = modelled_col.values

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_price_comparison(
    df: pd.DataFrame,
    modelled_col: str = "price_b",
    sample_days: int = 21,
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel figure:
      Top: time series of actual vs modelled price for the first `sample_days`.
      Bottom: scatter of modelled vs actual across the full year.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # --- Time series ---
    sample = df.iloc[: 24 * sample_days]
    ax = axes[0]
    ax.plot(sample.index, sample["da_price_eur_mwh"], lw=0.9, label="Actual DA price", color="black")
    ax.plot(sample.index, sample[modelled_col], lw=0.9, alpha=0.8, label="Modelled (LP dual)", color="steelblue")
    ax.set_title(f"Day-ahead price: actual vs modelled (first {sample_days} days)")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    # --- Scatter ---
    ax2 = axes[1]
    lo = min(df["da_price_eur_mwh"].quantile(0.01), df[modelled_col].quantile(0.01)) - 5
    hi = max(df["da_price_eur_mwh"].quantile(0.99), df[modelled_col].quantile(0.99)) + 5
    ax2.scatter(df[modelled_col], df["da_price_eur_mwh"],
                alpha=0.12, s=4, linewidths=0, color="steelblue")
    ax2.plot([lo, hi], [lo, hi], "r--", lw=1, label="45° line (perfect fit)")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("Modelled price (EUR/MWh)")
    ax2.set_ylabel("Actual DA price (EUR/MWh)")
    ax2.set_title("Modelled vs actual across full year")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "price_comparison.png", dpi=150)
    return fig


def plot_residuals(
    df: pd.DataFrame,
    modelled_col: str = "price_b",
    save: bool = True,
) -> plt.Figure:
    residual = df["da_price_eur_mwh"] - df[modelled_col]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df.index, residual, lw=0.4, alpha=0.6, color="darkred")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title("Fundamental residual over time (actual − modelled)")
    axes[0].set_ylabel("EUR/MWh")

    axes[1].hist(residual.dropna(), bins=80, color="steelblue", edgecolor="white", lw=0.2)
    axes[1].axvline(0, color="black", lw=1, ls="--")
    axes[1].axvline(residual.mean(), color="red", lw=1, label=f"Mean = {residual.mean():.1f}")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("EUR/MWh")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "residuals.png", dpi=150)
    return fig


def plot_conditional_metrics(cond: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Bar chart of MAE per regime, grouped by dimension.
    The most informative diagnostic: shows which market conditions the model handles well.
    """
    dims = cond["dimension"].unique()
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 5), sharey=False)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        sub = cond[cond["dimension"] == dim].set_index("regime")
        sub["mae"].plot.bar(ax=ax, color="steelblue", alpha=0.85)
        ax.set_title(dim.replace("_", " "))
        ax.set_ylabel("MAE (EUR/MWh)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Conditional MAE by market regime", fontsize=11, y=1.01)
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "conditional_mae.png", dpi=150, bbox_inches="tight")
    return fig


def plot_dispatch_mix(mix: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Monthly actual vs modelled generation share for each thermal technology.
    A good dispatch model should closely match the actual mix.
    """
    techs = ["nuclear", "hard_coal", "ccgt", "ocgt"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, tech in enumerate(techs):
        ax = axes[i]
        actual   = mix.get(f"actual_{tech}", pd.Series(np.nan, index=mix.index))
        modelled = mix.get(f"modelled_{tech}", pd.Series(np.nan, index=mix.index))

        act_m = actual.resample("ME").mean()
        mod_m = modelled.resample("ME").mean()

        ax.plot(act_m.index, act_m.values, lw=1.2, label="Actual (ENTSO-E A75)", color="black")
        ax.plot(mod_m.index, mod_m.values, lw=1.2, label="Modelled (LP)", color="steelblue", ls="--")
        ax.set_title(tech.replace("_", " ").title())
        ax.set_ylabel("Monthly mean MW")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle("Dispatch mix: actual vs modelled generation by technology", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "dispatch_mix.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest(
    panel: pd.DataFrame,
    dispatch: pd.DataFrame,
    gen_actual: pd.DataFrame | None = None,
) -> dict:
    """
    Run all validation steps and save outputs.

    Parameters
    ----------
    panel      : hourly panel from data_fetch.build_panel()
    dispatch   : dispatch output from dispatch.run_dispatch()
    gen_actual : optional raw A75 generation DataFrame for dispatch-mix comparison

    Returns
    -------
    dict with keys: overall, overall_a, conditional, df, mix (if gen_actual provided)
    """
    df = add_regime_buckets(panel, dispatch)

    # --- Overall metrics ---
    m_b = compute_metrics(df["da_price_eur_mwh"], df["price_b"])
    print_metrics("Overall fit — Method B (LP dual price)", m_b)

    if "price_a" in df.columns:
        m_a = compute_metrics(df["da_price_eur_mwh"], df["price_a"])
        print_metrics("Overall fit — Method A (analytic stacking)", m_a)
    else:
        m_a = {}

    # --- Conditional metrics ---
    cond = conditional_metrics_table(df, "price_b")
    print("\nConditional fit by market regime:")
    for _, row in cond.iterrows():
        print(f"  {row['dimension']:15s} | {row['regime']:12s} | "
              f"n={row['n']:,}  MAE={row['mae']:.1f}  bias={row['bias']:+.1f}  R²={row['r2']:.3f}")

    out_cond = ROOT / "reports" / "conditional_metrics.csv"
    cond.to_csv(out_cond, index=False)
    print(f"\n  Saved → {out_cond}")

    # --- Plots ---
    plot_price_comparison(df)
    plot_residuals(df)
    plot_conditional_metrics(cond)

    # --- Dispatch mix ---
    mix = None
    if gen_actual is not None:
        mix = dispatch_mix_comparison(panel, dispatch, gen_actual)
        plot_dispatch_mix(mix)
        out_mix = ROOT / "reports" / "dispatch_mix.csv"
        mix.to_csv(out_mix)
        print(f"  Dispatch mix saved → {out_mix}")

    return {
        "overall":   m_b,
        "overall_a": m_a,
        "conditional": cond,
        "df": df,
        "mix": mix,
    }
