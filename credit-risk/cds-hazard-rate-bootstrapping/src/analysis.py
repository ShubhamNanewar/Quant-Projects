from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MATURITIES = np.array([1, 3, 5, 7, 10], dtype=float)
SPREADS_BPS = np.array([100, 110, 120, 120, 125], dtype=float)
LGD = 0.40
RISK_FREE_RATE = 0.03


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def simple_model_table() -> pd.DataFrame:
    spreads = SPREADS_BPS / 10000.0
    lambda_avg = spreads / LGD
    cumulative_hazard = lambda_avg * MATURITIES
    survival = np.exp(-cumulative_hazard)
    h_prev = np.r_[0.0, cumulative_hazard[:-1]]
    t_prev = np.r_[0.0, MATURITIES[:-1]]
    lambda_forward = (cumulative_hazard - h_prev) / (MATURITIES - t_prev)
    survival_prev = np.r_[1.0, survival[:-1]]
    default_prob = survival_prev - survival

    return pd.DataFrame(
        {
            "Maturity": MATURITIES,
            "CDS Spread (bps)": SPREADS_BPS,
            "Average Hazard Rate": lambda_avg,
            "Forward Hazard Rate": lambda_forward,
            "Default Probability": default_prob,
        }
    )


def schedule_quarters(maturity: float, dt: float = 0.25) -> np.ndarray:
    n_steps = int(round(maturity / dt))
    return np.array([i * dt for i in range(1, n_steps + 1)], dtype=float)


def cumulative_hazard(t: float, nodes: np.ndarray, lambdas: np.ndarray) -> float:
    prev = 0.0
    total = 0.0
    for end, lam in zip(nodes, lambdas):
        if t <= prev:
            break
        seg_end = min(t, end)
        if seg_end > prev:
            total += lam * (seg_end - prev)
        prev = end
    return total


def survival_probability(t: float, nodes: np.ndarray, lambdas: np.ndarray) -> float:
    return float(np.exp(-cumulative_hazard(t, nodes, lambdas)))


def cds_value(maturity: float, spread: float, risk_free_rate: float, lgd: float, nodes: np.ndarray, lambdas: np.ndarray, dt: float = 0.25) -> float:
    pay_times = schedule_quarters(maturity, dt)
    pv_premium_regular = 0.0
    pv_premium_accrued = 0.0
    pv_protection = 0.0
    prev_t = 0.0

    for t_i in pay_times:
        midpoint = 0.5 * (prev_t + t_i)
        q_prev = survival_probability(prev_t, nodes, lambdas)
        q_curr = survival_probability(t_i, nodes, lambdas)
        discount_ti = np.exp(-risk_free_rate * t_i)
        discount_mid = np.exp(-risk_free_rate * midpoint)

        pv_premium_regular += spread * dt * discount_ti * q_curr
        pv_premium_accrued += spread * 0.5 * dt * discount_mid * (q_prev - q_curr)
        pv_protection += lgd * discount_mid * (q_prev - q_curr)
        prev_t = t_i

    return pv_premium_regular + pv_premium_accrued - pv_protection


def strip_hazards_exact(nodes: np.ndarray, spreads_bps: np.ndarray, risk_free_rate: float, lgd: float, dt: float = 0.25) -> np.ndarray:
    solved: list[float] = []
    for k, maturity in enumerate(nodes):
        spread = spreads_bps[k] / 10000.0

        def objective(lam_k: float) -> float:
            lam_vec = np.array(solved + [lam_k], dtype=float)
            if len(lam_vec) < len(nodes):
                lam_vec = np.r_[lam_vec, np.zeros(len(nodes) - len(lam_vec))]
            return cds_value(maturity, spread, risk_free_rate, lgd, nodes, lam_vec, dt)

        lo, hi = 1e-10, 5.0
        f_lo, f_hi = objective(lo), objective(hi)
        while f_lo * f_hi > 0:
            hi *= 2.0
            if hi > 200:
                raise RuntimeError(f"Could not bracket root for maturity {maturity}Y")
            f_hi = objective(hi)

        solved.append(float(brentq(objective, lo, hi)))

    return np.array(solved, dtype=float)


def exact_model_table(nodes: np.ndarray, spreads_bps: np.ndarray, risk_free_rate: float, lgd: float, dt: float = 0.25) -> tuple[pd.DataFrame, np.ndarray]:
    lambdas = strip_hazards_exact(nodes, spreads_bps, risk_free_rate, lgd, dt)
    avg_hazards = []
    default_probs = []
    prev_survival = 1.0
    for maturity in nodes:
        hazard = cumulative_hazard(maturity, nodes, lambdas)
        avg_hazards.append(hazard / maturity)
        survival = survival_probability(maturity, nodes, lambdas)
        default_probs.append(prev_survival - survival)
        prev_survival = survival

    table = pd.DataFrame(
        {
            "Maturity": nodes,
            "CDS Spread (bps)": spreads_bps,
            "Average Hazard Rate": avg_hazards,
            "Bootstrap Hazard Rate": lambdas,
            "Default Probability": default_probs,
        }
    )
    return table, lambdas


def validation_table(lambdas: np.ndarray, dt: float = 0.25) -> pd.DataFrame:
    maturity = 7.0
    spread = SPREADS_BPS[3] / 10000.0
    pay_times = schedule_quarters(maturity, dt)
    pv_regular = 0.0
    pv_accrued = 0.0
    pv_protection = 0.0
    prev_t = 0.0

    for t_i in pay_times:
        midpoint = 0.5 * (prev_t + t_i)
        q_prev = survival_probability(prev_t, MATURITIES, lambdas)
        q_curr = survival_probability(t_i, MATURITIES, lambdas)
        discount_ti = np.exp(-RISK_FREE_RATE * t_i)
        discount_mid = np.exp(-RISK_FREE_RATE * midpoint)
        pv_regular += spread * dt * discount_ti * q_curr
        pv_accrued += spread * 0.5 * dt * discount_mid * (q_prev - q_curr)
        pv_protection += LGD * discount_mid * (q_prev - q_curr)
        prev_t = t_i

    return pd.DataFrame(
        [
            {
                "Contract": "7Y CDS",
                "PV Premium Regular": pv_regular,
                "PV Premium Accrued": pv_accrued,
                "PV Protection": pv_protection,
                "Net PV": pv_regular + pv_accrued - pv_protection,
            }
        ]
    )


def sensitivity_high_rate_table() -> pd.DataFrame:
    exact_table, _ = exact_model_table(MATURITIES, SPREADS_BPS, 0.10, LGD)
    exact_table["Risk-Free Rate"] = 0.10
    return exact_table


def plot_hazard_curves(simple_df: pd.DataFrame, exact_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(simple_df["Maturity"], simple_df["Average Hazard Rate"], "o--", label="Simple average hazard")
    plt.step(exact_df["Maturity"], exact_df["Bootstrap Hazard Rate"], where="post", label="Exact bootstrap hazard")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Hazard rate")
    plt.title("CDS-Implied Hazard Rate Term Structure")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hazard_rate_term_structure.png", dpi=200)
    plt.close()


def run_analysis() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    simple_df = simple_model_table()
    exact_df, lambdas = exact_model_table(MATURITIES, SPREADS_BPS, RISK_FREE_RATE, LGD)
    validation_df = validation_table(lambdas)
    high_rate_df = sensitivity_high_rate_table()

    simple_df.to_csv(OUTPUTS_DIR / "simple_model_table.csv", index=False)
    exact_df.to_csv(OUTPUTS_DIR / "exact_model_table.csv", index=False)
    validation_df.to_csv(OUTPUTS_DIR / "validation_7y_cds.csv", index=False)
    high_rate_df.to_csv(OUTPUTS_DIR / "high_rate_sensitivity.csv", index=False)
    plot_hazard_curves(simple_df, exact_df)
    return {"simple": simple_df, "exact": exact_df, "validation": validation_df, "high_rate": high_rate_df}


def main() -> None:
    results = run_analysis()
    print("Simple model")
    print(results["simple"].to_string(index=False))
    print()
    print("Exact model")
    print(results["exact"].to_string(index=False))
    print()
    print("7Y validation")
    print(results["validation"].to_string(index=False))


if __name__ == "__main__":
    main()
