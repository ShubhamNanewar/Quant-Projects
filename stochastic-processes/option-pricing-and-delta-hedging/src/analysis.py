from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


DATA_PATH = ROOT / "data" / "SP500.csv"
FIGURES_DIR = ROOT / "figures"

RISK_FREE_RATE = 0.03
STRIKE = 6500.0
MATURITY_YEARS = 0.25
TRADING_WEEKS = 13
HEDGE_CONTRACTS = 20
CONTRACT_SIZE = 100
MONTE_CARLO_PATHS = 10_000
SEED = 42


@dataclass(frozen=True)
class MarketInputs:
    spot: float
    annual_volatility: float
    annual_drift_from_data: float


def load_market_inputs(path: Path) -> MarketInputs:
    data = pd.read_csv(path)
    prices = data["SP500"].astype(float).to_numpy()
    log_returns = np.diff(np.log(prices))

    monthly_sigma = float(np.std(log_returns, ddof=1))
    monthly_mu_tilde = float(np.mean(log_returns) + 0.5 * monthly_sigma**2)

    annual_sigma = monthly_sigma * np.sqrt(12)
    annual_mu = monthly_mu_tilde * 12
    spot = float(prices[-1])

    return MarketInputs(
        spot=spot,
        annual_volatility=annual_sigma,
        annual_drift_from_data=annual_mu,
    )


def black_scholes_call_price(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def black_scholes_delta(spot: np.ndarray | float, strike: float, rate: float, sigma: float, tau: float) -> np.ndarray | float:
    if tau <= 0:
        spot_array = np.asarray(spot)
        delta = np.where(spot_array > strike, 1.0, 0.0)
        return float(delta) if np.isscalar(spot) else delta

    spot_array = np.asarray(spot)
    d1 = (np.log(spot_array / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    delta = norm.cdf(d1)
    return float(delta) if np.isscalar(spot) else delta


def black_scholes_gamma(spot: np.ndarray | float, strike: float, rate: float, sigma: float, tau: float) -> np.ndarray | float:
    if tau <= 0:
        spot_array = np.asarray(spot)
        gamma = np.zeros_like(spot_array, dtype=float)
        return float(gamma) if np.isscalar(spot) else gamma

    spot_array = np.asarray(spot)
    d1 = (np.log(spot_array / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    gamma = norm.pdf(d1) / (spot_array * sigma * np.sqrt(tau))
    return float(gamma) if np.isscalar(spot) else gamma


def crr_binomial_call_price(spot: float, strike: float, rate: float, sigma: float, maturity: float, steps: int) -> float:
    dt = maturity / steps
    up = np.exp(sigma * np.sqrt(dt))
    down = np.exp(-sigma * np.sqrt(dt))
    discount = np.exp(-rate * dt)
    q = (np.exp(rate * dt) - down) / (up - down)

    nodes = np.arange(steps, -1, -1, dtype=float)
    terminal_prices = spot * up**nodes * down ** (steps - nodes)
    option_values = np.maximum(terminal_prices - strike, 0.0)

    for _ in range(steps):
        option_values = discount * (q * option_values[:-1] + (1 - q) * option_values[1:])

    return float(option_values[0])


def monte_carlo_call_price(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    steps: int,
    paths: int,
    seed: int,
    scheme: str,
) -> float:
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.standard_normal((paths, steps))

    if scheme == "euler":
        prices = np.full(paths, spot, dtype=float)
        for step in range(steps):
            prices = prices * (1 + rate * dt + sigma * np.sqrt(dt) * shocks[:, step])
    elif scheme == "log_euler":
        log_prices = np.full(paths, np.log(spot), dtype=float)
        drift = (rate - 0.5 * sigma**2) * dt
        for step in range(steps):
            log_prices = log_prices + drift + sigma * np.sqrt(dt) * shocks[:, step]
        prices = np.exp(log_prices)
    else:
        raise ValueError("scheme must be 'euler' or 'log_euler'")

    payoff = np.maximum(prices - strike, 0.0)
    return float(np.exp(-rate * maturity) * payoff.mean())


def crank_nicolson_call_price(spot: float, strike: float, rate: float, sigma: float, maturity: float, grid_points: int, time_steps: int) -> float:
    max_spot = 3 * spot
    d_spot = max_spot / grid_points
    dt = maturity / time_steps
    stock_grid = np.linspace(0, max_spot, grid_points + 1)

    option_values = np.maximum(stock_grid - strike, 0.0)
    alpha = np.zeros(grid_points - 1)
    beta = np.zeros(grid_points - 1)
    gamma = np.zeros(grid_points - 1)

    for j in range(1, grid_points):
        stock = j * d_spot
        alpha[j - 1] = 0.25 * dt * ((sigma**2 * stock**2 / d_spot**2) - (rate * stock / d_spot))
        beta[j - 1] = -0.5 * dt * ((sigma**2 * stock**2 / d_spot**2) + rate)
        gamma[j - 1] = 0.25 * dt * ((sigma**2 * stock**2 / d_spot**2) + (rate * stock / d_spot))

    left_matrix = diags([-alpha[1:], 1 - beta, -gamma[:-1]], [-1, 0, 1], format="csc")
    right_matrix = diags([alpha[1:], 1 + beta, gamma[:-1]], [-1, 0, 1], format="csc")

    interior = option_values[1:-1].copy()

    for n in range(time_steps):
        time = maturity - (n + 1) * dt
        upper_boundary = max_spot - strike * np.exp(-rate * time)
        rhs = right_matrix.dot(interior)
        rhs[0] += alpha[0] * 0.0
        rhs[-1] += gamma[-1] * upper_boundary
        interior = spsolve(left_matrix, rhs)

    option_values[1:-1] = interior
    option_values[0] = 0.0
    option_values[-1] = max_spot - strike * np.exp(-rate * maturity)
    return float(np.interp(spot, stock_grid, option_values))


def simulate_gbm_paths(spot: float, mu: float, sigma: float, maturity: float, steps: int, paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.standard_normal((paths, steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    prices = np.empty((paths, steps + 1), dtype=float)
    prices[:, 0] = spot
    prices[:, 1:] = spot * np.exp(log_paths)
    return prices


def simulate_delta_hedge_pnl(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    hedge_steps: int,
    paths: int,
    physical_drift: float,
    contracts: int,
    contract_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    price_paths = simulate_gbm_paths(spot, physical_drift, sigma, maturity, hedge_steps, paths, seed)
    dt = maturity / hedge_steps
    option_count = contracts * contract_size
    initial_premium = black_scholes_call_price(spot, strike, rate, sigma, maturity)

    stock_position = np.zeros(paths, dtype=float)
    cash_account = np.full(paths, option_count * initial_premium, dtype=float)
    gamma_track = np.zeros((paths, hedge_steps), dtype=float)

    for step in range(1, hedge_steps + 1):
        tau = maturity - step * dt
        current_prices = price_paths[:, step]
        delta_new = np.asarray(black_scholes_delta(current_prices, strike, rate, sigma, tau), dtype=float)
        gamma_new = np.asarray(black_scholes_gamma(current_prices, strike, rate, sigma, tau), dtype=float)

        gamma_track[:, step - 1] = np.abs(gamma_new)
        cash_account -= option_count * (delta_new - stock_position) * current_prices
        stock_position = delta_new
        cash_account *= np.exp(rate * dt)

    terminal_prices = price_paths[:, -1]
    terminal_portfolio = option_count * stock_position * terminal_prices + cash_account
    option_payoff = option_count * np.maximum(terminal_prices - strike, 0.0)
    pnl = terminal_portfolio - option_payoff
    average_abs_gamma = gamma_track.mean(axis=1)

    return pnl, average_abs_gamma


def save_line_plot(x: list[int], y: list[float], reference: float, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", color="#1f77b4")
    ax.axhline(reference, color="#d62728", linestyle="--", label="Black-Scholes benchmark")
    ax.set_title(title)
    ax.set_xlabel("Simulation / Grid Size")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def save_histograms(results: dict[str, np.ndarray], filename: str) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.8), sharey=True)
    for ax, (label, values) in zip(axes, results.items()):
        ax.hist(values, bins=50, color="#1f77b4", edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("Terminal P&L")
    axes[0].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def save_gamma_scatter(gamma: np.ndarray, pnl: np.ndarray, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gamma, pnl, s=8, alpha=0.35, color="#1f77b4")
    ax.set_title("Gamma Exposure and Hedging P&L")
    ax.set_xlabel("Average Absolute Gamma")
    ax.set_ylabel("Terminal P&L")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    market = load_market_inputs(DATA_PATH)
    spot = market.spot
    sigma = market.annual_volatility
    drift = market.annual_drift_from_data

    bs_price = black_scholes_call_price(spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS)

    binomial_steps = [25, 50, 100, 300]
    monte_carlo_paths = [100, 500, 1_000, 5_000, 10_000]
    crank_nicolson_grids = [50, 100, 200, 400]

    binomial_prices = [
        crr_binomial_call_price(spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, steps)
        for steps in binomial_steps
    ]
    euler_prices = [
        monte_carlo_call_price(spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, TRADING_WEEKS, paths, SEED, "euler")
        for paths in monte_carlo_paths
    ]
    log_euler_prices = [
        monte_carlo_call_price(spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, TRADING_WEEKS, paths, SEED, "log_euler")
        for paths in monte_carlo_paths
    ]
    crank_nicolson_prices = [
        crank_nicolson_call_price(spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, grid, grid)
        for grid in crank_nicolson_grids
    ]

    hedging_weekly, gamma_weekly = simulate_delta_hedge_pnl(
        spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, 13, MONTE_CARLO_PATHS, 0.05, HEDGE_CONTRACTS, CONTRACT_SIZE, SEED
    )
    hedging_monthly, _ = simulate_delta_hedge_pnl(
        spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, 3, MONTE_CARLO_PATHS, 0.05, HEDGE_CONTRACTS, CONTRACT_SIZE, SEED
    )
    hedging_daily, _ = simulate_delta_hedge_pnl(
        spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, 63, MONTE_CARLO_PATHS, 0.05, HEDGE_CONTRACTS, CONTRACT_SIZE, SEED
    )
    hedging_daily_low_mu, _ = simulate_delta_hedge_pnl(
        spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, 63, MONTE_CARLO_PATHS, 0.01, HEDGE_CONTRACTS, CONTRACT_SIZE, SEED
    )
    hedging_daily_high_mu, _ = simulate_delta_hedge_pnl(
        spot, STRIKE, RISK_FREE_RATE, sigma, MATURITY_YEARS, 63, MONTE_CARLO_PATHS, 0.09, HEDGE_CONTRACTS, CONTRACT_SIZE, SEED
    )

    save_line_plot(
        binomial_steps,
        binomial_prices,
        bs_price,
        "Binomial Price Convergence",
        "Call Price",
        "binomial_convergence.png",
    )
    save_line_plot(
        monte_carlo_paths,
        euler_prices,
        bs_price,
        "Monte Carlo Euler Convergence",
        "Call Price",
        "euler_convergence.png",
    )
    save_line_plot(
        monte_carlo_paths,
        log_euler_prices,
        bs_price,
        "Monte Carlo Log-Euler Convergence",
        "Call Price",
        "log_euler_convergence.png",
    )
    save_line_plot(
        crank_nicolson_grids,
        crank_nicolson_prices,
        bs_price,
        "Crank-Nicolson Convergence",
        "Call Price",
        "crank_nicolson_convergence.png",
    )
    save_histograms(
        {
            "Monthly hedge": hedging_monthly,
            "Weekly hedge": hedging_weekly,
            "Daily hedge": hedging_daily,
        },
        "hedging_pnl_histograms.png",
    )
    save_gamma_scatter(gamma_weekly, hedging_weekly, "gamma_vs_pnl.png")

    print("Market inputs")
    print(
        pd.Series(
            {
                "spot": spot,
                "annual_volatility": sigma,
                "annual_drift_from_data": drift,
                "black_scholes_call": bs_price,
            }
        )
        .round(6)
        .to_string()
    )

    print("\nPricing comparison")
    pricing_table = pd.DataFrame(
        {
            "method": (
                [f"Binomial ({steps} steps)" for steps in binomial_steps]
                + [f"Euler MC ({paths} paths)" for paths in monte_carlo_paths]
                + [f"Log-Euler MC ({paths} paths)" for paths in monte_carlo_paths]
                + [f"Crank-Nicolson ({grid}x{grid})" for grid in crank_nicolson_grids]
            ),
            "price": binomial_prices + euler_prices + log_euler_prices + crank_nicolson_prices,
        }
    )
    pricing_table["diff_vs_bs"] = pricing_table["price"] - bs_price
    print(pricing_table.round(6).to_string(index=False))

    print("\nDelta hedging summary")
    hedging_table = pd.DataFrame(
        [
            {"strategy": "Monthly hedge", "mean_pnl": hedging_monthly.mean(), "std_pnl": hedging_monthly.std(ddof=1)},
            {"strategy": "Weekly hedge", "mean_pnl": hedging_weekly.mean(), "std_pnl": hedging_weekly.std(ddof=1)},
            {"strategy": "Daily hedge", "mean_pnl": hedging_daily.mean(), "std_pnl": hedging_daily.std(ddof=1)},
            {"strategy": "Daily hedge, mu=1%", "mean_pnl": hedging_daily_low_mu.mean(), "std_pnl": hedging_daily_low_mu.std(ddof=1)},
            {"strategy": "Daily hedge, mu=9%", "mean_pnl": hedging_daily_high_mu.mean(), "std_pnl": hedging_daily_high_mu.std(ddof=1)},
        ]
    )
    print(hedging_table.round(6).to_string(index=False))

    print("\nGamma exposure")
    print(
        pd.Series(
            {
                "average_absolute_gamma": gamma_weekly.mean(),
                "gamma_pnl_correlation": np.corrcoef(gamma_weekly, hedging_weekly)[0, 1],
            }
        )
        .round(6)
        .to_string()
    )


if __name__ == "__main__":
    main()
