from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar import vecm
from statsmodels.tsa.vector_ar.vecm import VECM

warnings.simplefilter("ignore", ValueWarning)


SYMBOLS = ["SPY5.P", "SPY5.SIX", "SPY5l.CHIX"]


def load_prices(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "sp_g18.csv.gz", compression="gzip", parse_dates=["DateTime"])
    wide = (
        df.pivot_table(index="DateTime", columns="Symbol", values="Price", aggfunc="last")
        .sort_index()
        .loc[:, SYMBOLS]
    )
    return wide


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def prepare_daily_series(wide_prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    six_close = wide_prices["SPY5.SIX"].groupby(wide_prices.index.normalize()).last().dropna()
    daily_returns = 100.0 * np.log(six_close).diff().dropna()
    daily_returns = winsorize(daily_returns).reset_index(drop=True)

    intraday_returns = 100.0 * np.log(wide_prices["SPY5.SIX"]).diff().dropna()
    realized_variance = intraday_returns.pow(2).groupby(intraday_returns.index.normalize()).sum()
    realized_variance = realized_variance.loc[realized_variance.index.isin(six_close.index[1:])]
    return daily_returns, realized_variance


def prepare_intraday_series(wide_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_15m = wide_prices.resample("15min").last().dropna(how="any")
    returns_15m = 100.0 * np.log(prices_15m).diff().dropna()
    return prices_15m, returns_15m


def run_arma_grid(daily_returns: pd.Series) -> tuple[pd.DataFrame, ARIMA]:
    train = daily_returns.iloc[:252]
    test = daily_returns.iloc[252:]

    rows: list[dict[str, float | int]] = []
    best_result = None
    best_order = None

    for p in range(3):
        for q in range(3):
            model = ARIMA(
                train,
                order=(p, 0, q),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(method_kwargs={"maxiter": 50})
            forecast = result.forecast(steps=len(test))
            mse = float(np.mean((forecast.to_numpy() - test.to_numpy()) ** 2))
            ljung_box = acorr_ljungbox(result.resid, lags=[10], return_df=True)
            rows.append(
                {
                    "p": p,
                    "q": q,
                    "aic": result.aic,
                    "bic": result.bic,
                    "log_likelihood": result.llf,
                    "oos_mse": mse,
                    "ljung_box_pvalue": ljung_box["lb_pvalue"].iloc[0],
                }
            )
            if best_result is None or result.aic < best_result.aic:
                best_result = result
                best_order = (p, q)

    arma_table = pd.DataFrame(rows).sort_values(["aic", "oos_mse"]).reset_index(drop=True)
    assert best_result is not None and best_order is not None
    best_result.order_display = best_order  # type: ignore[attr-defined]
    return arma_table, best_result


def run_var_models(returns_15m: pd.DataFrame) -> tuple[pd.DataFrame, VAR, VAR]:
    train = returns_15m.loc[returns_15m.index < "2025-01-01"]
    test = returns_15m.loc[returns_15m.index >= "2025-01-01"]

    var2 = VAR(train).fit(2)
    var5 = VAR(train).fit(5)

    forecast_2 = pd.DataFrame(
        var2.forecast(train.values[-2:], steps=len(test)),
        index=test.index,
        columns=test.columns,
    )
    forecast_5 = pd.DataFrame(
        var5.forecast(train.values[-5:], steps=len(test)),
        index=test.index,
        columns=test.columns,
    )

    rows = []
    for label, result, forecast in [("VAR(2)", var2, forecast_2), ("VAR(5)", var5, forecast_5)]:
        mse_by_symbol = ((forecast - test) ** 2).mean()
        row = {
            "model": label,
            "aic": result.aic,
            "bic": result.bic,
            "log_likelihood": result.llf,
        }
        row.update({f"mse_{symbol}": mse_by_symbol[symbol] for symbol in mse_by_symbol.index})
        rows.append(row)

    return pd.DataFrame(rows), var2, var5


def run_vecm(prices_15m: pd.DataFrame) -> tuple[pd.DataFrame, VECM]:
    log_prices = np.log(prices_15m.loc[prices_15m.index < "2025-01-01"]).iloc[::6]
    lag_order = 2
    rank_test = vecm.select_coint_rank(
        log_prices,
        det_order=0,
        k_ar_diff=lag_order,
        method="trace",
        signif=0.05,
    )
    rank = min(rank_test.rank, 2)
    fitted = VECM(log_prices, k_ar_diff=lag_order, coint_rank=rank, deterministic="ci").fit()

    adjustment = pd.DataFrame(
        fitted.alpha,
        index=log_prices.columns,
        columns=[f"ec{i + 1}" for i in range(fitted.alpha.shape[1])],
    )
    adjustment["adf_pvalue"] = [adfuller(log_prices[column])[1] for column in log_prices.columns]
    adjustment["cointegration_rank"] = rank
    adjustment["lag_order"] = lag_order
    return adjustment.reset_index(names="symbol"), fitted


def run_volatility_models(
    daily_returns: pd.Series, realized_variance: pd.Series
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | str]] = []
    best_garch = None
    best_egarch = None

    for vol in ["GARCH", "EGARCH"]:
        for p in [1, 2]:
            for q in [1, 2]:
                result = arch_model(
                    daily_returns,
                    mean="ARX",
                    lags=1,
                    vol=vol,
                    p=p,
                    q=q,
                    dist="normal",
                    rescale=False,
                ).fit(disp="off")
                row = {
                    "model": f"{vol}({p},{q})",
                    "aic": result.aic,
                    "bic": result.bic,
                    "log_likelihood": result.loglikelihood,
                }
                rows.append(row)

                if vol == "GARCH" and (best_garch is None or result.aic < best_garch.aic):
                    best_garch = result
                if vol == "EGARCH" and (best_egarch is None or result.aic < best_egarch.aic):
                    best_egarch = result

    assert best_garch is not None and best_egarch is not None
    volatility_table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)

    comparison = pd.DataFrame(
        {
            "realized_variance": realized_variance.reset_index(drop=True),
            "garch_variance": best_garch.conditional_volatility**2,
            "egarch_variance": best_egarch.conditional_volatility**2,
        }
    ).dropna()

    diagnostics = {
        "best_garch_name": volatility_table.query("model.str.startswith('GARCH')", engine="python")
        .iloc[0]["model"],
        "best_egarch_name": volatility_table.query("model.str.startswith('EGARCH')", engine="python")
        .iloc[0]["model"],
        "comparison": comparison,
    }
    return volatility_table, diagnostics


def plot_daily_diagnostics(daily_returns: pd.Series, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(daily_returns, lags=20, zero=False, ax=axes[0])
    axes[0].set_title("ACF of Daily SIX Log Returns")
    plot_pacf(daily_returns, lags=20, zero=False, ax=axes[1], method="ywm")
    axes[1].set_title("PACF of Daily SIX Log Returns")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_var_residual_correlation(var_result, output_path: Path) -> None:
    corr = pd.DataFrame(var_result.resid, columns=SYMBOLS).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(SYMBOLS)), SYMBOLS, rotation=30)
    ax.set_yticks(range(len(SYMBOLS)), SYMBOLS)
    ax.set_title("Residual Correlation Matrix for VAR(2)")
    for i in range(len(SYMBOLS)):
        for j in range(len(SYMBOLS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_volatility_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(comparison.index, comparison["realized_variance"], label="Realized variance", linewidth=2)
    ax.plot(comparison.index, comparison["garch_variance"], label="Best GARCH variance", linewidth=2)
    ax.plot(comparison.index, comparison["egarch_variance"], label="Best EGARCH variance", linewidth=2)
    ax.set_title("Daily Realized Variance vs Conditional Variance Estimates")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Variance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(
    project_dir: Path,
    arma_table: pd.DataFrame,
    var_table: pd.DataFrame,
    vecm_table: pd.DataFrame,
    volatility_table: pd.DataFrame,
) -> None:
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    arma_table.to_csv(output_dir / "arma_grid.csv", index=False)
    var_table.to_csv(output_dir / "var_comparison.csv", index=False)
    vecm_table.to_csv(output_dir / "vecm_adjustment.csv", index=False)
    volatility_table.to_csv(output_dir / "volatility_models.csv", index=False)


def run_analysis(project_dir: Path) -> dict[str, object]:
    data_dir = project_dir / "data"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    wide_prices = load_prices(data_dir)
    daily_returns, realized_variance = prepare_daily_series(wide_prices)
    prices_15m, returns_15m = prepare_intraday_series(wide_prices)

    arma_table, arma_result = run_arma_grid(daily_returns)
    var_table, var2, var5 = run_var_models(returns_15m)
    vecm_table, vecm_result = run_vecm(prices_15m)
    volatility_table, volatility_diagnostics = run_volatility_models(daily_returns, realized_variance)

    plot_daily_diagnostics(daily_returns, figures_dir / "daily_return_acf_pacf.png")
    plot_var_residual_correlation(var2, figures_dir / "var2_residual_correlation.png")
    plot_volatility_comparison(
        volatility_diagnostics["comparison"],
        figures_dir / "realized_vs_model_variance.png",
    )

    write_outputs(project_dir, arma_table, var_table, vecm_table, volatility_table)

    return {
        "wide_prices": wide_prices,
        "daily_returns": daily_returns,
        "returns_15m": returns_15m,
        "arma_table": arma_table,
        "arma_result": arma_result,
        "var_table": var_table,
        "var2": var2,
        "var5": var5,
        "vecm_table": vecm_table,
        "vecm_result": vecm_result,
        "volatility_table": volatility_table,
        "volatility_diagnostics": volatility_diagnostics,
        "arch_lm": het_arch(arma_result.resid, nlags=5)[:2],
    }


def print_summary(results: dict[str, object]) -> None:
    wide_prices = results["wide_prices"]
    arma_table = results["arma_table"]
    var_table = results["var_table"]
    vecm_table = results["vecm_table"]
    volatility_table = results["volatility_table"]
    arch_lm_stat, arch_lm_pvalue = results["arch_lm"]

    print("Sample overview")
    print(
        {
            "venues": list(wide_prices.columns),
            "minute_observations": int(len(wide_prices)),
            "start": str(wide_prices.index.min()),
            "end": str(wide_prices.index.max()),
        }
    )

    print("\nBest ARMA candidates")
    print(arma_table.head(5).round(4).to_string(index=False))
    print(f"\nARCH LM statistic: {arch_lm_stat:.4f}, p-value: {arch_lm_pvalue:.4f}")

    print("\nVAR comparison")
    print(var_table.round(4).to_string(index=False))

    print("\nVECM adjustment coefficients")
    print(vecm_table.round(4).to_string(index=False))

    print("\nVolatility model ranking")
    print(volatility_table.round(4).to_string(index=False))


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results = run_analysis(project_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
