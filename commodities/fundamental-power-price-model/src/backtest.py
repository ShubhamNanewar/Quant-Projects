"""
Validation: compare modelled clearing price against actual day-ahead price.

Reports:
  - Overall fit: MAE, RMSE, correlation, bias
  - Conditional fit by: VRE share, demand level, gas regime, season, day type
  - Dispatch-mix comparison vs actual generation
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
# Scalar metrics
# ---------------------------------------------------------------------------

def metrics(actual: pd.Series, modelled: pd.Series) -> dict:
    e = actual - modelled
    return {
        "n":      len(e.dropna()),
        "mae":    e.abs().mean(),
        "rmse":   np.sqrt((e**2).mean()),
        "bias":   e.mean(),
        "corr":   actual.corr(modelled),
        "r2":     1 - e.var() / actual.var(),
    }


def print_metrics(label: str, m: dict):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  n={m['n']}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}"
          f"  bias={m['bias']:.2f}  corr={m['corr']:.3f}  R²={m['r2']:.3f}")


# ---------------------------------------------------------------------------
# Conditional bucketing
# ---------------------------------------------------------------------------

def add_buckets(panel: pd.DataFrame, dispatch: pd.DataFrame) -> pd.DataFrame:
    """Add classification columns used for conditional metrics."""
    df = panel.join(dispatch[["price_a", "price_b"]], how="left")

    # VRE share of load
    vre = (
        df["wind_onshore_mw"] + df["wind_offshore_mw"] + df["solar_mw"]
    ) / df["load_mw"].replace(0, np.nan)
    df["vre_share"] = vre
    df["vre_bucket"] = pd.cut(
        vre, bins=[-np.inf, 0.15, 0.35, np.inf],
        labels=["low_vre", "mid_vre", "high_vre"]
    )

    # Demand level (by quantile)
    df["demand_bucket"] = pd.qcut(
        df["load_mw"], q=3, labels=["off_peak", "shoulder", "peak"]
    )

    # Gas regime (by TTF tercile)
    df["gas_bucket"] = pd.qcut(
        df["ttf_eur_mwh"], q=3, labels=["gas_low", "gas_mid", "gas_high"]
    )

    # Season
    df["season"] = df.index.month.map(
        {12: "winter", 1: "winter", 2: "winter",
         3: "spring", 4: "spring", 5: "spring",
         6: "summer", 7: "summer", 8: "summer",
         9: "autumn", 10: "autumn", 11: "autumn"}
    )

    # Weekday vs weekend
    df["day_type"] = np.where(df.index.dayofweek < 5, "weekday", "weekend")

    return df


def conditional_metrics(df: pd.DataFrame, modelled_col: str = "price_b") -> pd.DataFrame:
    actual = df["da_price_eur_mwh"]
    modelled = df[modelled_col]

    rows = []
    for bucket_col in ["vre_bucket", "demand_bucket", "gas_bucket", "season", "day_type"]:
        for label, grp in df.groupby(bucket_col, observed=True):
            m = metrics(grp["da_price_eur_mwh"], grp[modelled_col])
            rows.append({"bucket_col": bucket_col, "label": str(label), **m})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_price_comparison(df: pd.DataFrame, modelled_col: str = "price_b", save: bool = True):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time series (first two weeks for readability)
    sample = df.iloc[:24*14]
    ax = axes[0]
    ax.plot(sample.index, sample["da_price_eur_mwh"], label="Actual DA", lw=0.8)
    ax.plot(sample.index, sample[modelled_col], label="Modelled (LP)", lw=0.8, alpha=0.8)
    ax.set_title("Day-ahead price: actual vs modelled (first 14 days)")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Scatter
    ax2 = axes[1]
    lim = [df[["da_price_eur_mwh", modelled_col]].min().min() - 5,
           df[["da_price_eur_mwh", modelled_col]].max().max() + 5]
    ax2.scatter(df[modelled_col], df["da_price_eur_mwh"], alpha=0.15, s=3)
    ax2.plot(lim, lim, "r--", lw=1, label="45° line")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("Modelled price (EUR/MWh)")
    ax2.set_ylabel("Actual price (EUR/MWh)")
    ax2.set_title("Scatter: modelled vs actual")
    ax2.legend()

    plt.tight_layout()
    if save:
        fig.savefig(FIGS / "price_comparison.png", dpi=150)
        print(f"  Saved {FIGS / 'price_comparison.png'}")
    return fig


def plot_residuals(df: pd.DataFrame, modelled_col: str = "price_b", save: bool = True):
    df = df.copy()
    df["residual"] = df["da_price_eur_mwh"] - df[modelled_col]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df.index, df["residual"], lw=0.5, alpha=0.7)
    axes[0].axhline(0, color="red", lw=0.8, ls="--")
    axes[0].set_title("Residual (actual − modelled) over time")
    axes[0].set_ylabel("EUR/MWh")

    axes[1].hist(df["residual"].dropna(), bins=80, edgecolor="white", lw=0.3)
    axes[1].axvline(0, color="red", lw=1, ls="--")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("EUR/MWh")

    plt.tight_layout()
    if save:
        fig.savefig(FIGS / "residuals.png", dpi=150)
        print(f"  Saved {FIGS / 'residuals.png'}")
    return fig


def plot_marginal_tech(dispatch: pd.DataFrame, save: bool = True):
    if "marginal_tech_a" not in dispatch.columns:
        return
    counts = dispatch["marginal_tech_a"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax)
    ax.set_title("Hours by marginal technology (Method A)")
    ax.set_ylabel("Hours")
    ax.set_xlabel("")
    plt.tight_layout()
    if save:
        fig.savefig(FIGS / "marginal_tech.png", dpi=150)
        print(f"  Saved {FIGS / 'marginal_tech.png'}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(panel: pd.DataFrame, dispatch: pd.DataFrame) -> dict:
    df = add_buckets(panel, dispatch)

    # Overall
    m_overall = metrics(df["da_price_eur_mwh"], df["price_b"])
    print_metrics("Overall (Method B / LP)", m_overall)

    if "price_a" in df.columns:
        m_a = metrics(df["da_price_eur_mwh"], df["price_a"])
        print_metrics("Overall (Method A / analytic)", m_a)

    # Conditional
    cond = conditional_metrics(df, "price_b")
    print("\nConditional metrics:")
    print(cond.to_string(index=False))

    out = ROOT / "reports" / "conditional_metrics.csv"
    cond.to_csv(out, index=False)
    print(f"\n  Conditional metrics saved to {out}")

    # Plots
    plot_price_comparison(df)
    plot_residuals(df)
    plot_marginal_tech(dispatch)

    return {"overall": m_overall, "conditional": cond, "df": df}
