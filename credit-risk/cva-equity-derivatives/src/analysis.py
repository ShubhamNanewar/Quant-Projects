from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

r = 0.03
q = 0.02
sigma_aex = 0.15
sigma_sx5e = 0.15
rho = 0.8
LGD = 0.4
T = 5.0
steps_per_year = 12
times = np.linspace(0.0, T, int(T * steps_per_year) + 1)

S0_aex = 1000.0
S0_sx5e = 6000.0
Notional_aex = 55_000.0
Notional_sx5e = 10_000.0
K_fwd_aex = 1000.0
K_fwd_sx5e = 6000.0
K_put_aex = 800.0
K_put_sx5e = 4800.0
hazard_buckets = {"0-1": 0.02, "1-3": 0.0215, "3-5": 0.0220}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def simulate_two_assets_gbm(
    S0_1: float,
    S0_2: float,
    r_: float,
    q_: float,
    sigma1: float,
    sigma2: float,
    rho_: float,
    time_grid: np.ndarray,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_times = len(time_grid)
    dt = time_grid[1] - time_grid[0]
    z1 = rng.standard_normal((n_paths, n_times - 1))
    z2_indep = rng.standard_normal((n_paths, n_times - 1))
    z2 = rho_ * z1 + np.sqrt(1.0 - rho_**2) * z2_indep

    drift1 = (r_ - q_ - 0.5 * sigma1**2) * dt
    drift2 = (r_ - q_ - 0.5 * sigma2**2) * dt
    vol1 = sigma1 * np.sqrt(dt)
    vol2 = sigma2 * np.sqrt(dt)

    S1 = np.empty((n_paths, n_times))
    S2 = np.empty((n_paths, n_times))
    S1[:, 0] = S0_1
    S2[:, 0] = S0_2

    for j in range(1, n_times):
        S1[:, j] = S1[:, j - 1] * np.exp(drift1 + vol1 * z1[:, j - 1])
        S2[:, j] = S2[:, j - 1] * np.exp(drift2 + vol2 * z2[:, j - 1])

    return S1, S2


def mc_mean_ci_95(x: np.ndarray) -> tuple[float, tuple[float, float], float]:
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    se = std / np.sqrt(x.size)
    return mean, (mean - 1.96 * se, mean + 1.96 * se), se


def bsm_put_price(S0: float, K: float, maturity: float, r_: float, q_: float, sigma: float) -> float:
    vol_sqrt_t = sigma * np.sqrt(maturity)
    d1 = (np.log(S0 / K) + (r_ - q_ + 0.5 * sigma**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(K * np.exp(-r_ * maturity) * norm.cdf(-d2) - S0 * np.exp(-q_ * maturity) * norm.cdf(-d1))


def bsm_put_value(S_t: np.ndarray, K: float, r_: float, q_: float, sigma: float, tau: float) -> np.ndarray:
    if tau <= 0:
        return np.maximum(K - S_t, 0.0)
    vol_sqrt_t = sigma * np.sqrt(tau)
    d1 = (np.log(S_t / K) + (r_ - q_ + 0.5 * sigma**2) * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return K * np.exp(-r_ * tau) * norm.cdf(-d2) - S_t * np.exp(-q_ * tau) * norm.cdf(-d1)


def forward_value(S_t: np.ndarray, K: float, r_: float, q_: float, tau: float) -> np.ndarray:
    return S_t * np.exp(-q_ * tau) - K * np.exp(-r_ * tau)


def survival_from_buckets(time_grid: np.ndarray, buckets: dict[str, float]) -> np.ndarray:
    lam01 = buckets["0-1"]
    lam13 = buckets["1-3"]
    lam35 = buckets["3-5"]

    def integ_hazard(t: float) -> float:
        if t <= 1.0:
            return lam01 * t
        if t <= 3.0:
            return lam01 + lam13 * (t - 1.0)
        return lam01 + 2.0 * lam13 + lam35 * (t - 3.0)

    return np.array([np.exp(-integ_hazard(t)) for t in time_grid])


def build_trade_values(S_sx: np.ndarray, S_ax: np.ndarray) -> dict[str, np.ndarray]:
    n_paths, n_times = S_sx.shape
    values = {
        "SX5E Forward": np.zeros((n_paths, n_times)),
        "AEX Forward": np.zeros((n_paths, n_times)),
        "SX5E Put": np.zeros((n_paths, n_times)),
        "AEX Put": np.zeros((n_paths, n_times)),
    }

    for j, t in enumerate(times):
        tau = T - t
        values["SX5E Forward"][:, j] = forward_value(S_sx[:, j], K_fwd_sx5e, r, q, tau) * Notional_sx5e
        values["AEX Forward"][:, j] = forward_value(S_ax[:, j], K_fwd_aex, r, q, tau) * Notional_aex
        values["SX5E Put"][:, j] = bsm_put_value(S_sx[:, j], K_put_sx5e, r, q, sigma_sx5e, tau) * Notional_sx5e
        values["AEX Put"][:, j] = bsm_put_value(S_ax[:, j], K_put_aex, r, q, sigma_aex, tau) * Notional_aex
    return values


def cva_from_ee(ee: np.ndarray, buckets: dict[str, float]) -> float:
    survival = survival_from_buckets(times, buckets)
    pd_bucket = survival[:-1] - survival[1:]
    df = np.exp(-r * times[1:])
    return float(LGD * np.sum(df * ee[1:] * pd_bucket))


def compute_portfolio_outputs(n_paths: int = 100_000, seed: int = 42) -> dict[str, object]:
    S_sx, S_ax = simulate_two_assets_gbm(S0_sx5e, S0_aex, r, q, sigma_sx5e, sigma_aex, rho, times, n_paths, seed)
    values = build_trade_values(S_sx, S_ax)

    # Q1 validation
    disc_T = np.exp(-r * T)
    S_T_sx = S_sx[:, -1]
    S_T_ax = S_ax[:, -1]
    mc_fwd_sx = mc_mean_ci_95(disc_T * (S_T_sx - K_fwd_sx5e))
    mc_fwd_ax = mc_mean_ci_95(disc_T * (S_T_ax - K_fwd_aex))
    theo_fwd_sx = S0_sx5e * np.exp(-q * T) - K_fwd_sx5e * np.exp(-r * T)
    theo_fwd_ax = S0_aex * np.exp(-q * T) - K_fwd_aex * np.exp(-r * T)

    mc_put_sx = mc_mean_ci_95(disc_T * np.maximum(K_put_sx5e - S_T_sx, 0.0))
    mc_put_ax = mc_mean_ci_95(disc_T * np.maximum(K_put_aex - S_T_ax, 0.0))
    bsm_put_sx = bsm_put_price(S0_sx5e, K_put_sx5e, T, r, q, sigma_sx5e)
    bsm_put_ax = bsm_put_price(S0_aex, K_put_aex, T, r, q, sigma_aex)

    ret_sx = np.log(S_sx[:, 1:] / S_sx[:, :-1]).reshape(-1)
    ret_ax = np.log(S_ax[:, 1:] / S_ax[:, :-1]).reshape(-1)
    rho_hat = float(np.corrcoef(ret_sx, ret_ax)[0, 1])
    z = np.arctanh(rho_hat)
    se_z = 1.0 / np.sqrt(ret_sx.size - 3)
    corr_ci = (float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z)))

    validation_df = pd.DataFrame(
        [
            {"Asset": "SX5E Forward", "Theoretical": theo_fwd_sx, "MC Mean": mc_fwd_sx[0], "CI Low": mc_fwd_sx[1][0], "CI High": mc_fwd_sx[1][1]},
            {"Asset": "AEX Forward", "Theoretical": theo_fwd_ax, "MC Mean": mc_fwd_ax[0], "CI Low": mc_fwd_ax[1][0], "CI High": mc_fwd_ax[1][1]},
            {"Asset": "SX5E Put", "Theoretical": bsm_put_sx, "MC Mean": mc_put_sx[0], "CI Low": mc_put_sx[1][0], "CI High": mc_put_sx[1][1]},
            {"Asset": "AEX Put", "Theoretical": bsm_put_ax, "MC Mean": mc_put_ax[0], "CI Low": mc_put_ax[1][0], "CI High": mc_put_ax[1][1]},
            {"Asset": "Return Correlation", "Theoretical": rho, "MC Mean": rho_hat, "CI Low": corr_ci[0], "CI High": corr_ci[1]},
        ]
    )

    ee_trades = {name: np.mean(np.maximum(v, 0.0), axis=0) for name, v in values.items()}
    standalone_cva_df = pd.DataFrame(
        [{"Contract": name, "Standalone CVA": cva_from_ee(ee, hazard_buckets)} for name, ee in ee_trades.items()]
    )

    portfolio_paths = sum(values.values())
    ee_net = np.mean(np.maximum(portfolio_paths, 0.0), axis=0)
    ee_unnet = sum(np.maximum(v, 0.0) for v in values.values()).mean(axis=0)
    cva_net = cva_from_ee(ee_net, hazard_buckets)
    cva_unnet = cva_from_ee(ee_unnet, hazard_buckets)
    netting_df = pd.DataFrame(
        [
            {"Metric": "Unnetted CVA", "Value": cva_unnet},
            {"Metric": "Netted CVA", "Value": cva_net},
            {"Metric": "Netting Benefit (EUR)", "Value": cva_unnet - cva_net},
            {"Metric": "Netting Benefit (%)", "Value": 100.0 * (cva_unnet - cva_net) / cva_unnet},
        ]
    )

    def compute_netted_cva(sig_sx: float, sig_ax: float, rho_new: float) -> float:
        stressed_sx, stressed_ax = simulate_two_assets_gbm(S0_sx5e, S0_aex, r, q, sig_sx, sig_ax, rho_new, times, n_paths, seed)
        stressed_values = build_trade_values(stressed_sx, stressed_ax)
        stressed_portfolio = sum(stressed_values.values())
        stressed_ee = np.mean(np.maximum(stressed_portfolio, 0.0), axis=0)
        return cva_from_ee(stressed_ee, hazard_buckets)

    cva_vol30 = compute_netted_cva(0.30, 0.30, rho)
    cva_rho40 = compute_netted_cva(sigma_sx5e, sigma_aex, 0.40)
    sensitivity_df = pd.DataFrame(
        [
            {"Scenario": "Baseline", "Netted CVA": cva_net, "Change EUR": 0.0, "Change %": 0.0},
            {"Scenario": "Volatility Stress", "Netted CVA": cva_vol30, "Change EUR": cva_vol30 - cva_net, "Change %": 100.0 * (cva_vol30 / cva_net - 1.0)},
            {"Scenario": "Correlation Stress", "Netted CVA": cva_rho40, "Change EUR": cva_rho40 - cva_net, "Change %": 100.0 * (cva_rho40 / cva_net - 1.0)},
        ]
    )

    def cva_with_vm_lag(m: int) -> float:
        ee = np.zeros(portfolio_paths.shape[1])
        for j in range(portfolio_paths.shape[1]):
            k = (j // m) * m
            ee[j] = np.maximum(portfolio_paths[:, j] - portfolio_paths[:, k], 0.0).mean()
        return cva_from_ee(ee, hazard_buckets)

    lag_months = np.arange(1, len(times))
    cva_vm = np.array([cva_with_vm_lag(int(m)) for m in lag_months])
    collateral_df = pd.DataFrame({"VM Lag (months)": lag_months, "CVA": cva_vm})

    def cva_with_initial_margin(initial_margin: float) -> float:
        ee = np.mean(np.maximum(portfolio_paths - initial_margin, 0.0), axis=0)
        return cva_from_ee(ee, hazard_buckets)

    im_levels = [0.0, 1e6, 10e6, 100e6]
    initial_margin_df = pd.DataFrame(
        [{"Initial Margin": im, "CVA": cva_with_initial_margin(im)} for im in im_levels]
    )

    def bump_bucket(buckets: dict[str, float], key: str, bump_size: float) -> dict[str, float]:
        bumped = buckets.copy()
        bumped[key] += bump_size
        return bumped

    bump = 10e-4
    dCVA = {}
    for bucket in hazard_buckets:
        dCVA[bucket] = cva_from_ee(ee_net, bump_bucket(hazard_buckets, bucket, bump)) - cva_net

    def cds_pv_components(maturity_years: float, buckets: dict[str, float]) -> tuple[float, float]:
        t_grid = np.linspace(0.0, maturity_years, int(maturity_years * steps_per_year) + 1)
        survival = survival_from_buckets(t_grid, buckets)
        pd_bucket = survival[:-1] - survival[1:]
        df = np.exp(-r * t_grid[1:])
        dt = t_grid[1] - t_grid[0]
        pv_prot = LGD * np.sum(df * pd_bucket)
        pv01 = np.sum(df * survival[1:] * dt)
        return float(pv_prot), float(pv01)

    def cds_par_spread(maturity_years: float, buckets: dict[str, float]) -> float:
        pv_prot, pv01 = cds_pv_components(maturity_years, buckets)
        return pv_prot / pv01

    def cds_mtm_protection_buyer(maturity_years: float, market_buckets: dict[str, float], fixed_spread: float) -> float:
        pv_prot, pv01 = cds_pv_components(maturity_years, market_buckets)
        return pv_prot - fixed_spread * pv01

    s1 = cds_par_spread(1.0, hazard_buckets)
    s3 = cds_par_spread(3.0, hazard_buckets)
    s5 = cds_par_spread(5.0, hazard_buckets)

    def cds_delta_mtm_for_bucket(maturity_years: float, fixed_spread: float, bucket_key: str) -> float:
        bumped = bump_bucket(hazard_buckets, bucket_key, bump)
        mtm_base = cds_mtm_protection_buyer(maturity_years, hazard_buckets, fixed_spread)
        mtm_bump = cds_mtm_protection_buyer(maturity_years, bumped, fixed_spread)
        return mtm_bump - mtm_base

    sens = pd.DataFrame(index=list(hazard_buckets.keys()), columns=["CDS1Y", "CDS3Y", "CDS5Y"], dtype=float)
    for bucket in hazard_buckets:
        sens.loc[bucket, "CDS1Y"] = cds_delta_mtm_for_bucket(1.0, s1, bucket)
        sens.loc[bucket, "CDS3Y"] = cds_delta_mtm_for_bucket(3.0, s3, bucket)
        sens.loc[bucket, "CDS5Y"] = cds_delta_mtm_for_bucket(5.0, s5, bucket)

    dCVA_vec = pd.Series(dCVA)
    N5 = -dCVA_vec["3-5"] / sens.loc["3-5", "CDS5Y"]
    N3 = (-dCVA_vec["1-3"] - N5 * sens.loc["1-3", "CDS5Y"]) / sens.loc["1-3", "CDS3Y"]
    N1 = (-dCVA_vec["0-1"] - N3 * sens.loc["0-1", "CDS3Y"] - N5 * sens.loc["0-1", "CDS5Y"]) / sens.loc["0-1", "CDS1Y"]

    hedge_df = pd.DataFrame(
        [
            {"Instrument": "1Y CDS", "Notional": N1},
            {"Instrument": "3Y CDS", "Notional": N3},
            {"Instrument": "5Y CDS", "Notional": N5},
        ]
    )

    hedge_check = pd.DataFrame(
        {
            "Bucket": list(hazard_buckets.keys()),
            "Delta CVA": [dCVA_vec[b] for b in hazard_buckets],
            "Delta Hedge": [
                N1 * sens.loc["0-1", "CDS1Y"] + N3 * sens.loc["0-1", "CDS3Y"] + N5 * sens.loc["0-1", "CDS5Y"],
                N3 * sens.loc["1-3", "CDS3Y"] + N5 * sens.loc["1-3", "CDS5Y"],
                N5 * sens.loc["3-5", "CDS5Y"],
            ],
        }
    )
    hedge_check["Residual"] = hedge_check["Delta CVA"] + hedge_check["Delta Hedge"]

    return {
        "validation": validation_df,
        "standalone": standalone_cva_df,
        "netting": netting_df,
        "sensitivity": sensitivity_df,
        "collateral": collateral_df,
        "initial_margin": initial_margin_df,
        "ee_profile": pd.DataFrame({"Time": times, "EE Netted": ee_net, "EE Unnetted": ee_unnet}),
        "hazard_sensitivity": pd.DataFrame({"Bucket": list(dCVA.keys()), "Delta CVA": list(dCVA.values())}),
        "cds_sensitivity": sens.reset_index(names="Bucket"),
        "hedge": hedge_df,
        "hedge_check": hedge_check,
    }


def save_figures(results: dict[str, object]) -> None:
    ee_profile = results["ee_profile"]
    plt.figure(figsize=(9, 5))
    plt.plot(ee_profile["Time"], ee_profile["EE Netted"], label="Netted EE")
    plt.plot(ee_profile["Time"], ee_profile["EE Unnetted"], label="Unnetted EE", linestyle="--")
    plt.xlabel("Time (years)")
    plt.ylabel("Exposure")
    plt.title("Expected Exposure Profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "expected_exposure_profiles.png", dpi=200)
    plt.close()

    sensitivity_df = results["sensitivity"]
    plt.figure(figsize=(8, 5))
    plt.bar(sensitivity_df["Scenario"], sensitivity_df["Netted CVA"], color=["#4c72b0", "#c44e52", "#55a868"])
    plt.ylabel("CVA (EUR)")
    plt.title("Netted CVA Under Stress Scenarios")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "netted_cva_stress_scenarios.png", dpi=200)
    plt.close()

    collateral_df = results["collateral"]
    plt.figure(figsize=(9, 5))
    plt.plot(collateral_df["VM Lag (months)"], collateral_df["CVA"])
    plt.xlabel("Variation Margin Update Lag (months)")
    plt.ylabel("CVA (EUR)")
    plt.title("CVA vs Variation Margin Update Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cva_vs_variation_margin_frequency.png", dpi=200)
    plt.close()


def save_outputs(results: dict[str, object]) -> None:
    for name, table in results.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(OUTPUTS_DIR / f"{name}.csv", index=False)


def run_analysis() -> dict[str, object]:
    ensure_dirs()
    results = compute_portfolio_outputs()
    save_outputs(results)
    save_figures(results)
    return results


def main() -> None:
    results = run_analysis()
    print("Validation")
    print(results["validation"].to_string(index=False))
    print()
    print("Netting summary")
    print(results["netting"].to_string(index=False))
    print()
    print("Stress sensitivity")
    print(results["sensitivity"].to_string(index=False))
    print()
    print("CDS hedge notionals")
    print(results["hedge"].to_string(index=False))


if __name__ == "__main__":
    main()
