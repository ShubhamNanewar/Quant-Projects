from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import binom, jarque_bera, norm, shapiro
from statsmodels.stats.weightstats import ztest


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

DATA_PATH = ROOT / "data" / "SP500.csv"
FIGURES_DIR = ROOT / "figures"

MONTHLY_RISK_FREE_RATE = 0.03 / 12
QUARTERLY_RISK_FREE_RATE = 0.03 / 4
ANNUAL_RISK_FREE_RATE = 0.03
STRIKE = 6500.0
TERMINAL_LEVEL = 6300.0
BASE_UP = 1.037296
BASE_DOWN = 0.96553
SEED = 45


@dataclass(frozen=True)
class ReturnSummary:
    mean: float
    std: float
    z_stat: float
    z_p_value: float
    shapiro_stat: float
    shapiro_p_value: float


def load_price_series(path: Path) -> pd.Series:
    data = pd.read_csv(path, parse_dates=["observation_date"])
    return data["SP500"].astype(float)


def compute_simple_returns(prices: pd.Series) -> np.ndarray:
    return prices.pct_change().dropna().to_numpy()


def compute_log_returns(prices: pd.Series) -> np.ndarray:
    return np.log(prices / prices.shift(1)).dropna().to_numpy()


def summarize_returns(returns: np.ndarray) -> ReturnSummary:
    z_stat, z_p_value = ztest(returns, value=0)
    shapiro_stat, shapiro_p_value = shapiro(returns)
    return ReturnSummary(
        mean=float(np.mean(returns)),
        std=float(np.std(returns, ddof=1)),
        z_stat=float(z_stat),
        z_p_value=float(z_p_value),
        shapiro_stat=float(shapiro_stat),
        shapiro_p_value=float(shapiro_p_value),
    )


def monte_carlo_confidence_interval_width(
    mu: float,
    sigma: float,
    n_months: int,
    n_simulations: int,
    z_value: float = 1.96,
    seed: int = SEED,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mu, sigma, size=(n_simulations, n_months))
    sample_means = simulated_returns.mean(axis=1)
    sample_stds = simulated_returns.std(axis=1, ddof=1)
    margin = z_value * sample_stds / np.sqrt(n_months)
    return pd.Series(
        {
            "avg_lower_bound": float(np.mean(sample_means - margin)),
            "avg_upper_bound": float(np.mean(sample_means + margin)),
            "avg_interval_width": float(np.mean(2 * margin)),
        }
    )


def estimate_gbm_parameters(log_returns: np.ndarray) -> tuple[float, float]:
    sigma_tilde = float(np.std(log_returns, ddof=1))
    mean_log_return = float(np.mean(log_returns))
    mu_tilde = mean_log_return + 0.5 * sigma_tilde**2
    return mu_tilde, sigma_tilde


def expected_future_price(current_price: float, mu_tilde: float, months: int) -> float:
    return float(current_price * np.exp(months * mu_tilde))


def simulate_terminal_prices(
    s0: float,
    mu_tilde: float,
    sigma_tilde: float,
    months: int,
    n_simulations: int,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(n_simulations, months))
    log_growth = (mu_tilde - 0.5 * sigma_tilde**2) + sigma_tilde * shocks
    cumulative_log_growth = log_growth.sum(axis=1)
    return s0 * np.exp(cumulative_log_growth)


def risk_neutral_probability_closed_form(
    s0: float,
    level: float,
    sigma_tilde: float,
    months: int,
    risk_free_rate_monthly: float = 0.0,
) -> float:
    mean = np.log(s0) + (risk_free_rate_monthly - 0.5 * sigma_tilde**2) * months
    std = sigma_tilde * np.sqrt(months)
    return float(norm.cdf((np.log(level) - mean) / std))


def black_scholes_prices(
    s0: float,
    strike: float,
    annual_rate: float,
    annual_sigma: float,
    maturity_years: float,
) -> tuple[float, float]:
    d1 = (
        np.log(s0 / strike)
        + (annual_rate + 0.5 * annual_sigma**2) * maturity_years
    ) / (annual_sigma * np.sqrt(maturity_years))
    d2 = d1 - annual_sigma * np.sqrt(maturity_years)
    call = s0 * norm.cdf(d1) - strike * np.exp(-annual_rate * maturity_years) * norm.cdf(d2)
    put = strike * np.exp(-annual_rate * maturity_years) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    return float(call), float(put)


def construct_stock_tree(s0: float, up: float, down: float, steps: int) -> list[np.ndarray]:
    return [
        np.array([s0 * up ** (period - j) * down**j for j in range(period + 1)], dtype=float)
        for period in range(steps + 1)
    ]


def scaled_up_down(base_up: float, base_down: float, steps: int) -> tuple[float, float]:
    factor = np.sqrt(3 / steps)
    return 1 + (base_up - 1) * factor, 1 + (base_down - 1) * factor


def binomial_option_price(
    s0: float,
    strike: float,
    up: float,
    down: float,
    rate_per_step: float,
    steps: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    q_up = (1 + rate_per_step - down) / (up - down)
    powers = np.arange(steps, -1, -1, dtype=float)
    terminal_prices = s0 * up**powers * down ** (steps - powers)

    if option_type == "call":
        option_values = np.maximum(terminal_prices - strike, 0.0)
    elif option_type == "put":
        option_values = np.maximum(strike - terminal_prices, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    for period in range(steps - 1, -1, -1):
        continuation = (q_up * option_values[:-1] + (1 - q_up) * option_values[1:]) / (1 + rate_per_step)
        if exercise.lower() == "american":
            current_powers = np.arange(period, -1, -1, dtype=float)
            current_prices = s0 * up**current_powers * down ** (period - current_powers)
            if option_type == "call":
                exercise_value = np.maximum(current_prices - strike, 0.0)
            else:
                exercise_value = np.maximum(strike - current_prices, 0.0)
            option_values = np.maximum(continuation, exercise_value)
        else:
            option_values = continuation

    return float(option_values[0])


def empirical_risk_neutral_return(
    q_up: float,
    q_down: float,
    up: float,
    down: float,
    steps: int,
) -> float:
    values = [
        comb(steps, i) * q_up ** (steps - i) * q_down**i * up ** (steps - i) * down**i
        for i in range(steps + 1)
    ]
    return float(np.sum(values) - 1)


def convergence_series(
    s0: float,
    strike: float,
    base_up: float,
    base_down: float,
    quarterly_rate: float,
    steps_list: list[int],
) -> pd.DataFrame:
    rows = []
    for steps in steps_list:
        up, down = scaled_up_down(base_up, base_down, steps)
        rate_per_step = quarterly_rate / steps
        rows.append(
            {
                "steps": steps,
                "call_price": binomial_option_price(
                    s0, strike, up, down, rate_per_step, steps, option_type="call"
                ),
                "put_price": binomial_option_price(
                    s0, strike, up, down, rate_per_step, steps, option_type="put"
                ),
            }
        )
    return pd.DataFrame(rows)


def terminal_log_return_distribution(
    up: float,
    down: float,
    q_down: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_returns = np.array(
        [np.log(up) * (steps - i) + np.log(down) * i for i in range(steps + 1)],
        dtype=float,
    )
    weights = binom.pmf(np.arange(steps + 1), steps, q_down)
    return log_returns, weights


def save_histogram(values: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, color="#1f77b4", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_convergence_plot(
    steps: np.ndarray,
    values: np.ndarray,
    reference: float,
    path: Path,
    ylabel: str,
    reference_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, values, marker="o", color="#1f77b4")
    ax.axhline(reference, color="#d62728", linestyle="--", label=reference_label)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_distribution_plot(log_returns: np.ndarray, weights: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(log_returns, weights, width=0.005, color="#1f77b4")
    ax.set_xlim(-0.4, 0.4)
    ax.set_xlabel("Terminal Log Return")
    ax.set_ylabel("Risk-Neutral Probability")
    ax.set_title("Binomial Terminal Log-Return Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    prices = load_price_series(DATA_PATH)
    simple_returns = compute_simple_returns(prices)
    log_returns = compute_log_returns(prices)
    return_summary = summarize_returns(simple_returns)

    ci_summary = monte_carlo_confidence_interval_width(
        mu=return_summary.mean,
        sigma=return_summary.std,
        n_months=prices.size - 1,
        n_simulations=10_000,
    )

    mu_tilde, sigma_tilde = estimate_gbm_parameters(log_returns)
    current_price = float(prices.iloc[-1])
    forecast_months = 60
    simulated_terminal_prices = simulate_terminal_prices(
        current_price,
        mu_tilde,
        sigma_tilde,
        forecast_months,
        10_000,
    )

    jb_stat, jb_p_value = jarque_bera(np.log(simulated_terminal_prices))

    risk_neutral_terminal_prices = simulate_terminal_prices(
        current_price,
        mu_tilde=0.0,
        sigma_tilde=sigma_tilde,
        months=forecast_months,
        n_simulations=10_000,
    )
    simulated_put_probability = float(np.mean(risk_neutral_terminal_prices < TERMINAL_LEVEL))
    closed_form_put_probability = risk_neutral_probability_closed_form(
        current_price,
        TERMINAL_LEVEL,
        sigma_tilde,
        forecast_months,
    )

    annual_sigma = sigma_tilde * np.sqrt(12)
    bs_call, bs_put = black_scholes_prices(
        current_price,
        STRIKE,
        ANNUAL_RISK_FREE_RATE,
        annual_sigma,
        maturity_years=0.25,
    )

    one_step_returns = np.array([BASE_UP - 1, BASE_DOWN - 1], dtype=float)
    one_step_probabilities = np.array([0.55, 0.45], dtype=float)
    one_step_mean = float(np.dot(one_step_probabilities, one_step_returns))
    one_step_variance = float(
        np.dot(one_step_probabilities, (one_step_returns - one_step_mean) ** 2)
    )

    scaled_steps = 6
    up_scaled, down_scaled = scaled_up_down(BASE_UP, BASE_DOWN, scaled_steps)
    scaled_rate = QUARTERLY_RISK_FREE_RATE / scaled_steps
    q_up_scaled = (1 + scaled_rate - down_scaled) / (up_scaled - down_scaled)
    q_down_scaled = 1 - q_up_scaled
    empirical_return = empirical_risk_neutral_return(
        q_up_scaled, q_down_scaled, up_scaled, down_scaled, scaled_steps
    )
    theoretical_return = (1 + scaled_rate) ** scaled_steps - 1

    steps_list = [3, 6, 10, 100, 1_000, 10_000]
    convergence = convergence_series(
        current_price,
        STRIKE,
        BASE_UP,
        BASE_DOWN,
        QUARTERLY_RISK_FREE_RATE,
        steps_list,
    )

    steps_10k = 10_000
    up_10k, down_10k = scaled_up_down(BASE_UP, BASE_DOWN, steps_10k)
    rate_10k = QUARTERLY_RISK_FREE_RATE / steps_10k
    q_up_10k = (1 + rate_10k - down_10k) / (up_10k - down_10k)
    q_down_10k = 1 - q_up_10k
    binomial_log_returns, binomial_weights = terminal_log_return_distribution(
        up_10k, down_10k, q_down_10k, steps_10k
    )
    terminal_expectation = float(np.sum(binomial_log_returns * binomial_weights))
    theoretical_terminal_expectation = QUARTERLY_RISK_FREE_RATE - 0.5 * 3 * sigma_tilde**2

    american_call = binomial_option_price(
        current_price,
        STRIKE,
        up_10k,
        down_10k,
        rate_10k,
        steps_10k,
        option_type="call",
        exercise="american",
    )
    american_put = binomial_option_price(
        current_price,
        STRIKE,
        up_10k,
        down_10k,
        rate_10k,
        steps_10k,
        option_type="put",
        exercise="american",
    )

    put_call_parity_gap = (
        convergence.loc[convergence["steps"] == steps_10k, "put_price"].iloc[0]
        + current_price
        - (
            convergence.loc[convergence["steps"] == steps_10k, "call_price"].iloc[0]
            + STRIKE / (1 + rate_10k) ** steps_10k
        )
    )

    save_histogram(
        simulated_terminal_prices,
        FIGURES_DIR / "gbm_terminal_prices.png",
        "GBM Terminal Prices After 60 Months",
        "Price",
    )
    save_convergence_plot(
        convergence["steps"].to_numpy(),
        convergence["call_price"].to_numpy(),
        bs_call,
        FIGURES_DIR / "call_convergence.png",
        "Call Price",
        "Black-Scholes Call",
    )
    save_convergence_plot(
        convergence["steps"].to_numpy(),
        convergence["put_price"].to_numpy(),
        bs_put,
        FIGURES_DIR / "put_convergence.png",
        "Put Price",
        "Black-Scholes Put",
    )
    save_distribution_plot(
        binomial_log_returns,
        binomial_weights,
        FIGURES_DIR / "binomial_log_return_distribution.png",
    )

    print("Monthly simple return summary")
    print(pd.Series(return_summary.__dict__).round(6).to_string())
    print("\nMonte Carlo confidence interval summary")
    print(ci_summary.round(6).to_string())
    print("\nGBM parameter estimates")
    print(pd.Series({"mu_tilde": mu_tilde, "sigma_tilde": sigma_tilde}).round(6).to_string())
    print("\n60-month forecast")
    print(
        pd.Series(
            {
                "expected_price_60m": expected_future_price(current_price, mu_tilde, forecast_months),
                "simulated_mean_terminal_price": float(np.mean(simulated_terminal_prices)),
                "lognormal_jb_stat": float(jb_stat),
                "lognormal_jb_p_value": float(jb_p_value),
            }
        )
        .round(6)
        .to_string()
    )
    print("\nRisk-neutral probability check")
    print(
        pd.Series(
            {
                "simulated_put_probability": simulated_put_probability,
                "closed_form_put_probability": closed_form_put_probability,
                "absolute_difference": abs(simulated_put_probability - closed_form_put_probability),
            }
        )
        .round(6)
        .to_string()
    )
    print("\nThree-step tree diagnostics")
    print(
        pd.Series(
            {
                "up_times_down": BASE_UP * BASE_DOWN,
                "one_step_mean": one_step_mean,
                "one_step_variance": one_step_variance,
            }
        )
        .round(6)
        .to_string()
    )
    print("\nBlack-Scholes benchmark")
    print(pd.Series({"call_price": bs_call, "put_price": bs_put}).round(6).to_string())
    print("\nScaled tree check")
    print(
        pd.Series(
            {
                "q_up": q_up_scaled,
                "q_down": q_down_scaled,
                "empirical_return": empirical_return,
                "theoretical_return": theoretical_return,
                "absolute_difference": abs(theoretical_return - empirical_return),
            }
        )
        .round(6)
        .to_string()
    )
    print("\nConvergence table")
    print(convergence.round(6).to_string(index=False))
    print("\nTerminal distribution check")
    print(
        pd.Series(
            {
                "empirical_expectation": terminal_expectation,
                "theoretical_expectation": theoretical_terminal_expectation,
                "absolute_difference": abs(theoretical_terminal_expectation - terminal_expectation),
            }
        )
        .round(6)
        .to_string()
    )
    print("\nAmerican option prices")
    print(pd.Series({"american_call": american_call, "american_put": american_put}).round(6).to_string())
    print("\nPut-call parity gap at 10,000 steps")
    print(round(float(put_call_parity_gap), 6))


if __name__ == "__main__":
    main()
