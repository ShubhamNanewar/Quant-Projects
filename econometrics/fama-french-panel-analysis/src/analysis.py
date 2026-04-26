from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper


SELECTED_SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
]


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(data_dir / "capmff_2010-2025_prices.csv", parse_dates=["Date"])
    prices = prices.set_index("Date").apply(pd.to_numeric, errors="coerce")

    factors = pd.read_csv(data_dir / "capmff_2010-2025_ff.csv", parse_dates=["Date"])
    factors = factors.rename(
        columns={"Mkt-RF": "mkt_excess", "SMB": "smb", "HML": "hml", "RF": "rf"}
    )
    factors[["mkt_excess", "smb", "hml", "rf"]] = (
        factors[["mkt_excess", "smb", "hml", "rf"]] / 100.0
    )

    sectors = pd.read_csv(data_dir / "capmff_2010-2025_sector.csv")
    return prices, factors, sectors


def prepare_panel(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    sectors: pd.DataFrame,
    coverage_threshold: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = sectors.loc[sectors["sector"].isin(SELECTED_SECTORS), ["Ticker", "sector"]]
    selected_tickers = selected["Ticker"].tolist()

    prices = prices[[ticker for ticker in prices.columns if ticker in selected_tickers]]
    prices = prices.loc[:, prices.notna().mean() >= coverage_threshold]
    returns = prices.pct_change(fill_method=None)

    panel = (
        returns.stack()
        .rename("return")
        .reset_index()
        .rename(columns={"level_1": "Ticker"})
        .merge(factors, left_on="Date", right_on="Date", how="inner")
        .merge(selected, on="Ticker", how="left")
        .dropna(subset=["return", "mkt_excess", "smb", "hml", "rf", "sector"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    panel["excess_return"] = panel["return"] - panel["rf"]

    sector_summary = (
        panel.groupby("sector")
        .agg(
            n_stocks=("Ticker", "nunique"),
            observations=("excess_return", "size"),
            mean_excess_return=("excess_return", "mean"),
            volatility=("excess_return", "std"),
        )
        .sort_index()
    )

    return panel, sector_summary


def fit_robust_ols(formula: str, data: pd.DataFrame) -> RegressionResultsWrapper:
    return smf.ols(formula, data=data).fit(cov_type="HC3")


def tidy_model(
    result: RegressionResultsWrapper, model_name: str, keep: list[str] | None = None
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "coefficient": result.params,
            "std_error": result.bse,
            "t_stat": result.tvalues,
            "p_value": result.pvalues,
        }
    )
    if keep is not None:
        table = table.loc[keep]
    table.index.name = "term"
    table = table.reset_index()
    table.insert(0, "model", model_name)
    table["r_squared"] = result.rsquared
    return table


def fit_entity_fixed_effects(panel: pd.DataFrame) -> RegressionResultsWrapper:
    columns = ["excess_return", "mkt_excess", "smb", "hml"]
    demeaned = panel[columns] - panel.groupby("Ticker")[columns].transform("mean")
    y = demeaned["excess_return"]
    x = demeaned[["mkt_excess", "smb", "hml"]]
    return sm.OLS(y, x).fit(cov_type="HC3")


def estimate_firm_loadings(panel: pd.DataFrame, min_observations: int = 252) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for ticker, group in panel.groupby("Ticker"):
        if len(group) < min_observations:
            continue

        result = smf.ols(
            "excess_return ~ mkt_excess + smb + hml",
            data=group,
        ).fit()
        rows.append(
            {
                "Ticker": ticker,
                "sector": group["sector"].iloc[0],
                "alpha": result.params["Intercept"],
                "beta_mkt": result.params["mkt_excess"],
                "beta_smb": result.params["smb"],
                "beta_hml": result.params["hml"],
                "r_squared": result.rsquared,
                "observations": len(group),
            }
        )

    return pd.DataFrame(rows).sort_values(["sector", "Ticker"]).reset_index(drop=True)


def summarize_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    summary = (
        loadings.groupby("sector")[["alpha", "beta_mkt", "beta_smb", "beta_hml", "r_squared"]]
        .mean()
        .rename(columns={"r_squared": "average_r_squared"})
        .sort_index()
    )
    return summary


def plot_sector_cumulative_returns(panel: pd.DataFrame, output_path: Path) -> None:
    sector_returns = (
        panel.groupby(["Date", "sector"])["return"].mean().unstack("sector").sort_index()
    )
    cumulative = (1.0 + sector_returns).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    for sector in cumulative.columns:
        ax.plot(cumulative.index, cumulative[sector], linewidth=2, label=sector)

    ax.set_title("Equal-Weighted Sector Growth")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_factor_loadings(loadings: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    factor_specs = [
        ("beta_mkt", "Market Beta"),
        ("beta_smb", "SMB Loading"),
        ("beta_hml", "HML Loading"),
    ]

    for ax, (column, title) in zip(axes, factor_specs):
        data = [loadings.loc[loadings["sector"] == sector, column].values for sector in SELECTED_SECTORS]
        ax.boxplot(data, tick_labels=SELECTED_SECTORS, patch_artist=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)

    fig.suptitle("Firm-Level Fama-French Loadings by Sector")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(
    project_dir: Path,
    sector_summary: pd.DataFrame,
    model_tables: list[pd.DataFrame],
    loadings: pd.DataFrame,
    loadings_summary: pd.DataFrame,
) -> None:
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    sector_summary.to_csv(output_dir / "sector_summary.csv")
    pd.concat(model_tables, ignore_index=True).to_csv(output_dir / "model_summary.csv", index=False)
    loadings.to_csv(output_dir / "firm_loadings.csv", index=False)
    loadings_summary.to_csv(output_dir / "sector_loadings_summary.csv")


def run_analysis(project_dir: Path) -> dict[str, object]:
    data_dir = project_dir / "data"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    prices, factors, sectors = load_data(data_dir)
    panel, sector_summary = prepare_panel(prices, factors, sectors)

    capm = fit_robust_ols("excess_return ~ mkt_excess", panel)
    ff3 = fit_robust_ols("excess_return ~ mkt_excess + smb + hml", panel)
    sector_effects = fit_robust_ols(
        "excess_return ~ mkt_excess + smb + hml + C(sector)", panel
    )
    ticker_fixed_effects = fit_entity_fixed_effects(panel)

    loadings = estimate_firm_loadings(panel)
    loadings_summary = summarize_loadings(loadings)

    plot_sector_cumulative_returns(panel, figures_dir / "sector_cumulative_returns.png")
    plot_factor_loadings(loadings, figures_dir / "factor_loadings_by_sector.png")

    model_tables = [
        tidy_model(capm, "CAPM", ["Intercept", "mkt_excess"]),
        tidy_model(ff3, "FF3", ["Intercept", "mkt_excess", "smb", "hml"]),
        tidy_model(
            sector_effects,
            "Sector Effects",
            [
                "Intercept",
                "C(sector)[T.Communication Services]",
                "C(sector)[T.Consumer Cyclical]",
                "C(sector)[T.Consumer Defensive]",
                "mkt_excess",
                "smb",
                "hml",
            ],
        ),
        tidy_model(
            ticker_fixed_effects,
            "Ticker Fixed Effects",
            ["mkt_excess", "smb", "hml"],
        ),
    ]
    write_outputs(project_dir, sector_summary, model_tables, loadings, loadings_summary)

    return {
        "panel": panel,
        "sector_summary": sector_summary,
        "capm": capm,
        "ff3": ff3,
        "sector_effects": sector_effects,
        "ticker_fixed_effects": ticker_fixed_effects,
        "loadings": loadings,
        "loadings_summary": loadings_summary,
    }


def print_summary(results: dict[str, object]) -> None:
    panel = results["panel"]
    sector_summary = results["sector_summary"]
    ff3 = results["ff3"]
    ticker_fixed_effects = results["ticker_fixed_effects"]
    loadings_summary = results["loadings_summary"]
    loadings = results["loadings"]

    print("Panel dimensions")
    print(
        {
            "observations": int(len(panel)),
            "tickers": int(panel["Ticker"].nunique()),
            "dates": int(panel["Date"].nunique()),
        }
    )
    print("\nSector summary")
    print(sector_summary.round(4).to_string())

    print("\nFF3 pooled model")
    print(ff3.params.round(4).to_string())
    print(f"R-squared: {ff3.rsquared:.4f}")

    print("\nTicker fixed-effects slopes")
    print(ticker_fixed_effects.params[["mkt_excess", "smb", "hml"]].round(4).to_string())
    print(f"R-squared: {ticker_fixed_effects.rsquared:.4f}")

    print("\nAverage firm-level loadings by sector")
    print(loadings_summary.round(4).to_string())

    top_betas = loadings.nlargest(5, "beta_mkt")[["Ticker", "sector", "beta_mkt", "r_squared"]]
    print("\nHighest market beta names")
    print(top_betas.round(4).to_string(index=False))


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results = run_analysis(project_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
