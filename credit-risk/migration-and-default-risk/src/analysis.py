from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
TRANSITION_MATRIX = np.array(
    [
        [0.91115, 0.08179, 0.00607, 0.00072, 0.00024, 0.00003, 0.00000, 0.00000],
        [0.00844, 0.89626, 0.08954, 0.00437, 0.00064, 0.00036, 0.00018, 0.00021],
        [0.00055, 0.02595, 0.91138, 0.05509, 0.00499, 0.00107, 0.00045, 0.00052],
        [0.00031, 0.00147, 0.04289, 0.90584, 0.03898, 0.00708, 0.00175, 0.00168],
        [0.00007, 0.00044, 0.00446, 0.06741, 0.83274, 0.07667, 0.00895, 0.00926],
        [0.00008, 0.00031, 0.00150, 0.00490, 0.05373, 0.82531, 0.07894, 0.03523],
        [0.00000, 0.00015, 0.00023, 0.00091, 0.00388, 0.07630, 0.83035, 0.08818],
    ]
)
CURRENT_VALUES = {"AAA": 99.40, "AA": 98.39, "A": 97.22, "BBB": 92.79, "BB": 90.11, "B": 86.60, "CCC": 77.16}
FORWARD_VALUES = {"AAA": 99.50, "AA": 98.51, "A": 97.53, "BBB": 92.77, "BB": 90.48, "B": 88.25, "CCC": 77.88, "D": 60.00}
PORTFOLIOS = {
    "Portfolio I": [("AAA", 0.60), ("AA", 0.30), ("BBB", 0.10)],
    "Portfolio II": [("BB", 0.60), ("B", 0.35), ("CCC", 0.05)],
}
RHO_GRID = {"0%": 0.0, "33%": 0.33, "66%": 0.66, "100%": 1.0}
REV_RATINGS = RATINGS[::-1]
PV0 = 1500.0


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def build_thresholds() -> dict[str, np.ndarray]:
    thresholds: dict[str, np.ndarray] = {}
    for i, start_rating in enumerate(RATINGS[:-1]):
        reversed_row = TRANSITION_MATRIX[i][::-1]
        cdf = np.cumsum(reversed_row)
        thresholds[start_rating] = norm.ppf(np.clip(cdf[:-1], 1e-12, 1 - 1e-12))
    return thresholds


THRESHOLDS = build_thresholds()


def risk_metrics_from_values(portfolio_values: np.ndarray, pv0: float = PV0) -> dict[str, float]:
    loss = pv0 - portfolio_values
    var90 = float(np.quantile(loss, 0.90))
    var995 = float(np.quantile(loss, 0.995))
    return {
        "Expected Value": float(portfolio_values.mean()),
        "90% VaR": var90,
        "99.5% VaR": var995,
        "90% ES": float(loss[loss >= var90].mean()),
        "99.5% ES": float(loss[loss >= var995].mean()),
    }


def simulate_concentrated_portfolio(
    portfolio: list[tuple[str, float]],
    rho: float,
    n_scenarios: int,
    seed: int,
    pv0: float = PV0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    systematic = rng.standard_normal(n_scenarios)
    portfolio_value = np.zeros(n_scenarios)

    for start_rating, weight in portfolio:
        epsilon = rng.standard_normal(n_scenarios)
        latent = np.sqrt(rho) * systematic + np.sqrt(1.0 - rho) * epsilon
        migrated_idx = np.searchsorted(THRESHOLDS[start_rating], latent, side="right")
        migrated_ratings = np.array(REV_RATINGS)[migrated_idx]
        value_ratio = np.array([FORWARD_VALUES[r] for r in migrated_ratings]) / CURRENT_VALUES[start_rating]
        portfolio_value += weight * pv0 * value_ratio

    return portfolio_value


def simulate_diversified_portfolio(
    portfolio: list[tuple[str, float]],
    rho: float,
    n_scenarios: int,
    seed: int,
    issuers_per_bucket: int = 100,
    pv0: float = PV0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    systematic = rng.standard_normal(n_scenarios)
    portfolio_value = np.zeros(n_scenarios)

    for start_rating, weight in portfolio:
        epsilon = rng.standard_normal((n_scenarios, issuers_per_bucket))
        latent = np.sqrt(rho) * systematic[:, None] + np.sqrt(1.0 - rho) * epsilon
        migrated_idx = np.searchsorted(THRESHOLDS[start_rating], latent, side="right")
        migrated_ratings = np.array(REV_RATINGS)[migrated_idx]
        values = np.vectorize(FORWARD_VALUES.get)(migrated_ratings)
        bucket_ratio = (values / CURRENT_VALUES[start_rating]).mean(axis=1)
        portfolio_value += weight * pv0 * bucket_ratio

    return portfolio_value


def portfolio_ii_var995(rho: float, n_scenarios: int, seed: int) -> float:
    values = simulate_concentrated_portfolio(PORTFOLIOS["Portfolio II"], rho, n_scenarios, seed)
    loss = PV0 - values
    return float(np.quantile(loss, 0.995))


def convergence_check(rho: float = 0.33, seeds: tuple[int, ...] = (42, 43, 44), target_rel: float = 0.01) -> tuple[pd.DataFrame, int]:
    grid = [50_000, 100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000]
    rows = []
    chosen = grid[-1]
    for n_scenarios in grid:
        vars_ = [portfolio_ii_var995(rho, n_scenarios, seed) for seed in seeds]
        mean_var = float(np.mean(vars_))
        width = float(max(vars_) - min(vars_))
        rel = width / mean_var
        rows.append(
            {
                "N": n_scenarios,
                "VaR99.5_seed_42": vars_[0],
                "VaR99.5_seed_43": vars_[1],
                "VaR99.5_seed_44": vars_[2],
                "Range": width,
                "Relative Range": rel,
            }
        )
        if rel <= target_rel and chosen == grid[-1]:
            chosen = n_scenarios
            break
    return pd.DataFrame(rows), chosen


def static_validation_table() -> pd.DataFrame:
    default_prob = TRANSITION_MATRIX[RATINGS.index("BBB"), RATINGS.index("D")]
    threshold = norm.ppf(default_prob)
    return pd.DataFrame(
        [
            {
                "Item": "BBB default probability",
                "Value": default_prob,
            },
            {
                "Item": "BBB default threshold",
                "Value": threshold,
            },
        ]
    )


def make_results_table(simulator, n_scenarios: int, seed: int) -> pd.DataFrame:
    rows = []
    for portfolio_name, portfolio in PORTFOLIOS.items():
        for rho_label, rho in RHO_GRID.items():
            values = simulator(portfolio, rho, n_scenarios, seed)
            metrics = risk_metrics_from_values(values)
            rows.append({"Portfolio": portfolio_name, "Rho": rho_label, **metrics})
    result = pd.DataFrame(rows)
    result["Rho"] = pd.Categorical(result["Rho"], categories=list(RHO_GRID.keys()), ordered=True)
    return result.sort_values(["Portfolio", "Rho"]).reset_index(drop=True)


def plot_loss_panels(simulator, n_scenarios: int, seed: int, title: str, filename: str, suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (portfolio_name, portfolio) in zip(axes, PORTFOLIOS.items()):
        for rho_label, rho in RHO_GRID.items():
            values = simulator(portfolio, rho, n_scenarios, seed)
            loss = PV0 - values
            ax.hist(loss, bins=60, density=True, alpha=0.35, label=f"$\\rho$={rho_label}")
        ax.set_title(f"{portfolio_name}{suffix}")
        ax.set_xlabel("Loss")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def run_analysis(seed: int = 42) -> dict[str, pd.DataFrame | int]:
    ensure_dirs()
    convergence_df, final_n = convergence_check()
    q1_df = make_results_table(simulate_concentrated_portfolio, final_n, seed)
    q2_df = make_results_table(simulate_diversified_portfolio, final_n, seed)
    validation_df = static_validation_table()

    convergence_df.to_csv(OUTPUTS_DIR / "convergence_check.csv", index=False)
    q1_df.to_csv(OUTPUTS_DIR / "question_1_risk_metrics.csv", index=False)
    q2_df.to_csv(OUTPUTS_DIR / "question_2_risk_metrics.csv", index=False)
    validation_df.to_csv(OUTPUTS_DIR / "validation_checks.csv", index=False)

    plot_loss_panels(
        simulate_concentrated_portfolio,
        final_n,
        seed,
        "Loss Distributions vs Correlation (Single Issuer per Rating)",
        "question_1_loss_distributions.png",
        "",
    )
    plot_loss_panels(
        simulate_diversified_portfolio,
        final_n,
        seed,
        "Loss Distributions vs Correlation (100 Issuers per Rating Bucket)",
        "question_2_loss_distributions.png",
        " (100 issuers per rating)",
    )
    return {"convergence": convergence_df, "q1": q1_df, "q2": q2_df, "validation": validation_df, "final_n": final_n}


def main() -> None:
    results = run_analysis()
    print("Chosen Monte Carlo sample size:", results["final_n"])
    print()
    print("Question 1 risk metrics")
    print(results["q1"].to_string(index=False))
    print()
    print("Question 2 risk metrics")
    print(results["q2"].to_string(index=False))


if __name__ == "__main__":
    main()
