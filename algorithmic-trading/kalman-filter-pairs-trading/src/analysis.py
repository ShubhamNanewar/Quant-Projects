from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


DATA_START = "2020-01-01"
DATA_END = "2025-12-31"
PAIR = ("RF", "SCHW")
TRAIN_FRACTION = 0.75
ENTRY_Z = 1.25
EXIT_Z = 0.0
TRANSACTION_COST_BPS = 5.0
DELTA = 1e-4


@dataclass
class KalmanState:
    alpha: float
    beta: float
    innovation: float
    innovation_std: float
    z_score: float
    state_var_alpha: float
    state_var_beta: float


def ensure_dirs() -> None:
    for name in ["figures", "outputs", "notebooks", ".mplconfig"]:
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


def download_prices(tickers: tuple[str, str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
    prices = prices.dropna()
    prices.index = pd.to_datetime(prices.index)
    return prices.loc[:, list(tickers)]


def kalman_filter_regression(
    x: pd.Series,
    y: pd.Series,
    delta: float = DELTA,
    measurement_var: float | None = None,
) -> pd.DataFrame:
    if measurement_var is None:
        measurement_var = float((y - x).diff().dropna().var())
    transition_var = measurement_var * delta / (1.0 - delta)

    theta = np.zeros(2)
    covariance = np.eye(2)
    transition_cov = transition_var * np.eye(2)

    records: list[KalmanState] = []
    for xt, yt in zip(x.to_numpy(), y.to_numpy()):
        design = np.array([1.0, xt])
        pred_theta = theta.copy()
        pred_cov = covariance + transition_cov

        forecast = float(design @ pred_theta)
        forecast_var = float(design @ pred_cov @ design.T + measurement_var)
        innovation = float(yt - forecast)
        gain = pred_cov @ design / forecast_var

        theta = pred_theta + gain * innovation
        covariance = pred_cov - np.outer(gain, design) @ pred_cov

        innovation_std = np.sqrt(forecast_var)
        records.append(
            KalmanState(
                alpha=float(theta[0]),
                beta=float(theta[1]),
                innovation=innovation,
                innovation_std=float(innovation_std),
                z_score=float(innovation / innovation_std),
                state_var_alpha=float(covariance[0, 0]),
                state_var_beta=float(covariance[1, 1]),
            )
        )

    return pd.DataFrame(records, index=x.index)


def generate_signal(z_scores: pd.Series, entry_z: float = ENTRY_Z, exit_z: float = EXIT_Z) -> pd.Series:
    signal = np.zeros(len(z_scores), dtype=float)
    current = 0.0
    for i, z in enumerate(z_scores.to_numpy()):
        if current == 0.0:
            if z <= -entry_z:
                current = 1.0
            elif z >= entry_z:
                current = -1.0
        elif current == 1.0 and z >= exit_z:
            current = 0.0
        elif current == -1.0 and z <= -exit_z:
            current = 0.0
        signal[i] = current
    return pd.Series(signal, index=z_scores.index, name="signal")


def compute_strategy_returns(log_prices: pd.DataFrame, beta_path: pd.Series, signal: pd.Series) -> pd.DataFrame:
    log_returns = log_prices.diff().fillna(0.0)
    beta_abs = beta_path.abs().clip(lower=1e-8)
    weight_y = 1.0 / (1.0 + beta_abs)
    weight_x = -beta_path / (1.0 + beta_abs)

    lagged_signal = signal.shift(1).fillna(0.0)
    gross = lagged_signal * (weight_y * log_returns[PAIR[0]] + weight_x * log_returns[PAIR[1]])

    turnover = signal.diff().abs().fillna(signal.abs())
    cost_rate = TRANSACTION_COST_BPS / 10_000
    costs = turnover * cost_rate
    net = gross - costs

    result = pd.DataFrame(
        {
            "gross_return": gross,
            "transaction_cost": costs,
            "net_return": net,
            "signal": signal,
            "beta": beta_path,
            "weight_rf": weight_y * lagged_signal,
            "weight_schw": weight_x * lagged_signal,
        }
    )
    return result


def annualized_sharpe(returns: pd.Series) -> float:
    vol = float(returns.std())
    if vol == 0.0:
        return 0.0
    return float(np.sqrt(252.0) * returns.mean() / vol)


def max_drawdown(cumulative: pd.Series) -> float:
    wealth = np.exp(cumulative.cumsum())
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def summarize_performance(returns: pd.Series, split_label: str) -> dict[str, float | str]:
    return {
        "sample": split_label,
        "observations": int(returns.shape[0]),
        "total_return_pct": float(100 * (np.exp(returns.sum()) - 1.0)),
        "annualized_return_pct": float(100 * 252 * returns.mean()),
        "annualized_vol_pct": float(100 * np.sqrt(252) * returns.std()),
        "sharpe_ratio": annualized_sharpe(returns),
        "max_drawdown_pct": float(100 * max_drawdown(returns)),
    }


def build_summary_tables(strategy: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = strategy.index <= train_end
    test_mask = strategy.index > train_end

    performance = pd.DataFrame(
        [
            summarize_performance(strategy.loc[train_mask, "gross_return"], "Train Gross"),
            summarize_performance(strategy.loc[test_mask, "gross_return"], "Test Gross"),
            summarize_performance(strategy.loc[train_mask, "net_return"], "Train Net"),
            summarize_performance(strategy.loc[test_mask, "net_return"], "Test Net"),
        ]
    )

    diagnostics = pd.DataFrame(
        {
            "metric": [
                "entry_z",
                "exit_z",
                "transaction_cost_bps",
                "long_days",
                "short_days",
                "flat_days",
                "signal_changes",
                "avg_abs_beta",
                "avg_alpha",
            ],
            "value": [
                ENTRY_Z,
                EXIT_Z,
                TRANSACTION_COST_BPS,
                int((strategy["signal"] == 1).sum()),
                int((strategy["signal"] == -1).sum()),
                int((strategy["signal"] == 0).sum()),
                int(strategy["signal"].diff().abs().fillna(strategy["signal"].abs()).sum()),
                float(strategy["beta"].abs().mean()),
                float(strategy["alpha"].mean()) if "alpha" in strategy.columns else np.nan,
            ],
        }
    )
    return performance, diagnostics


def save_figures(log_prices: pd.DataFrame, states: pd.DataFrame, strategy: pd.DataFrame, train_end: pd.Timestamp) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    log_prices.plot(ax=ax)
    ax.set_title("Log Prices: RF and SCHW")
    ax.axvline(train_end, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Log Price")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "log_prices.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    states["beta"].plot(ax=axes[0], color="steelblue")
    axes[0].axvline(train_end, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Kalman Filter Hedge Ratio")
    axes[0].set_ylabel("Beta")

    states["alpha"].plot(ax=axes[1], color="darkorange")
    axes[1].axvline(train_end, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Kalman Filter Intercept")
    axes[1].set_ylabel("Alpha")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "state_paths.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    states["z_score"].plot(ax=axes[0], color="purple")
    axes[0].axhline(ENTRY_Z, color="red", linestyle="--")
    axes[0].axhline(-ENTRY_Z, color="green", linestyle="--")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(train_end, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("One-Step-Ahead Mispricing Z-Score")
    axes[0].set_ylabel("Z")

    strategy["signal"].plot(ax=axes[1], color="teal")
    axes[1].axvline(train_end, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Trading Signal")
    axes[1].set_ylabel("Position")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "signals.png", dpi=160)
    plt.close(fig)

    cumulative = strategy[["gross_return", "net_return"]].cumsum().apply(np.exp) - 1.0
    fig, ax = plt.subplots(figsize=(12, 5))
    cumulative.plot(ax=ax)
    ax.axvline(train_end, color="black", linestyle="--", linewidth=1)
    ax.set_title("Cumulative Strategy Returns")
    ax.set_ylabel("Return")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "cumulative_returns.png", dpi=160)
    plt.close(fig)


def run_analysis() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    prices = download_prices(PAIR, DATA_START, DATA_END)
    log_prices = np.log(prices)
    train_size = int(TRAIN_FRACTION * len(log_prices))
    train_end = log_prices.index[train_size - 1]

    states = kalman_filter_regression(log_prices[PAIR[1]], log_prices[PAIR[0]])
    states.index = log_prices.index
    signal = generate_signal(states["z_score"])

    strategy = compute_strategy_returns(log_prices, states["beta"], signal)
    strategy["alpha"] = states["alpha"]
    strategy["innovation"] = states["innovation"]
    strategy["innovation_std"] = states["innovation_std"]
    strategy["z_score"] = states["z_score"]

    performance, diagnostics = build_summary_tables(strategy, train_end)

    save_figures(log_prices, states, strategy, train_end)

    prices.to_csv(PROJECT_ROOT / "outputs" / "prices.csv")
    states.to_csv(PROJECT_ROOT / "outputs" / "kalman_states.csv")
    strategy.to_csv(PROJECT_ROOT / "outputs" / "strategy_returns.csv")
    performance.to_csv(PROJECT_ROOT / "outputs" / "performance_summary.csv", index=False)
    diagnostics.to_csv(PROJECT_ROOT / "outputs" / "strategy_diagnostics.csv", index=False)

    return {
        "prices": prices,
        "states": states,
        "strategy": strategy,
        "performance": performance,
        "diagnostics": diagnostics,
    }


def main() -> None:
    results = run_analysis()
    print("\nPerformance Summary")
    print(results["performance"].round(4).to_string(index=False))
    print("\nStrategy Diagnostics")
    print(results["diagnostics"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
