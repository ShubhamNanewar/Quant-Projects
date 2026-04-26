from __future__ import annotations

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import binom, norm, stats, t


DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PRE_START = "2015-01-01"
PRE_END = "2016-12-31"
SAMPLE_START = "2017-01-01"
SAMPLE_END = "2026-03-31"
PORTFOLIO_VALUE = 1_000_000
TRADING_DAYS = 252
ALPHAS = (0.01, 0.05)
BACKTEST_ALPHA = 0.01
DF_CANDIDATES = (3, 4, 5, 6)
RISKY = ["ASML", "SHELL", "JPM_EUR", "STOXX50", "SP500_EUR"]
MODEL_ASSETS = RISKY + ["LOAN"]
EWMA_LAMBDA = 0.94
MULTI_DAY_HORIZONS = (1, 5, 10)


def ensure_dirs() -> None:
    for path in [DATA_DIR, PLOTS_DIR, OUTPUTS_DIR, PROJECT_ROOT / ".mplconfig", PROJECT_ROOT / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(DATA_DIR / "prices_clean.csv", index_col=0, parse_dates=True).sort_index()
    returns_raw = pd.read_csv(DATA_DIR / "returns_clean.csv", index_col=0, parse_dates=True).sort_index()

    simple_returns = pd.DataFrame(index=returns_raw.index)
    for col in MODEL_ASSETS:
        simple_returns[col] = returns_raw[col] if col == "LOAN" else np.expm1(returns_raw[col])

    return prices, simple_returns


def long_only_min_var_weights(ret_pre: pd.DataFrame) -> pd.Series:
    n = ret_pre.shape[1]
    cov_ann = ret_pre.cov().values * TRADING_DAYS

    def variance_objective(w: np.ndarray) -> float:
        return float(w @ cov_ann @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 0.3) for _ in range(n)]
    x0 = np.repeat(1.0 / n, n)
    result = minimize(variance_objective, x0=x0, bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=ret_pre.columns, name="min_var")


def tangency_weights(ret_pre: pd.DataFrame, rf_annual: float) -> pd.Series:
    n = ret_pre.shape[1]
    mu_ann = ret_pre.mean().values * TRADING_DAYS
    cov_ann = ret_pre.cov().values * TRADING_DAYS

    def negative_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu_ann)
        vol = float(np.sqrt(w @ cov_ann @ w))
        return -(ret - rf_annual) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 0.3) for _ in range(n)]
    x0 = np.repeat(1.0 / n, n)
    result = minimize(negative_sharpe, x0=x0, bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=ret_pre.columns, name="tangency")


def selected_portfolio_weights(simple_returns: pd.DataFrame) -> pd.Series:
    ret_pre = simple_returns.loc[PRE_START:PRE_END, RISKY]
    rf_annual = float(simple_returns.loc[PRE_START:PRE_END, "LOAN"].mean() * TRADING_DAYS)
    min_var = long_only_min_var_weights(ret_pre)
    tangency = tangency_weights(ret_pre, rf_annual)

    final_weights = pd.Series(
        {
            "ASML": 0.16,
            "SHELL": 0.08,
            "JPM_EUR": 0.24,
            "STOXX50": 0.08,
            "SP500_EUR": 0.24,
            "LOAN": 0.20,
        },
        name="selected",
    )

    summary = pd.concat([min_var, tangency, final_weights[RISKY]], axis=1)
    summary.columns = ["min_var", "tangency", "selected_risky_only"]
    summary.to_csv(OUTPUTS_DIR / "portfolio_weights_summary.csv")
    return final_weights


def portfolio_moments(weights: np.ndarray, mu_vec: np.ndarray, cov_mat: np.ndarray) -> tuple[float, float]:
    mu_p = float(weights @ mu_vec)
    sigma_p = float(np.sqrt(weights @ cov_mat @ weights))
    return mu_p, sigma_p


def normal_var_es(mu: float, sigma: float, alpha: float, notional: float) -> tuple[float, float]:
    z = norm.ppf(alpha)
    var = notional * (-mu - sigma * z)
    es = notional * (-mu + sigma * norm.pdf(z) / alpha)
    return var, es


def student_t_var_es(mu: float, sigma: float, alpha: float, nu: int, notional: float) -> tuple[float, float]:
    scale = sigma * np.sqrt((nu - 2) / nu)
    q = t.ppf(alpha, df=nu)
    pdf_q = t.pdf(q, df=nu)
    var = notional * (-mu - scale * q)
    es = notional * (-mu + scale * ((nu + q ** 2) / ((nu - 1) * alpha)) * pdf_q)
    return var, es


def historical_var_es(losses: pd.Series, alpha: float) -> tuple[float, float]:
    q = float(losses.quantile(1.0 - alpha))
    es = float(losses[losses >= q].mean())
    return q, es


def pooled_standardized_residuals(sample: pd.DataFrame) -> np.ndarray:
    z = (sample - sample.mean()) / sample.std(ddof=1)
    return z.stack().dropna().to_numpy()


def unit_variance_t_quantiles(n: int, nu: int) -> np.ndarray:
    probs = (np.arange(1, n + 1) - 0.5) / n
    return np.sqrt((nu - 2) / nu) * t.ppf(probs, df=nu)


def qq_rmse(z_values: np.ndarray, nu: int) -> float:
    empirical = np.sort(z_values)
    theoretical = unit_variance_t_quantiles(len(empirical), nu)
    return float(np.sqrt(np.mean((empirical - theoretical) ** 2)))


def select_student_t_df(ret_pre: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    z_pre = pooled_standardized_residuals(ret_pre)
    rows = [{"df": nu, "qq_rmse": qq_rmse(z_pre, nu)} for nu in DF_CANDIDATES]
    table = pd.DataFrame(rows).sort_values("qq_rmse").reset_index(drop=True)
    best_df = int(table.iloc[0]["df"])
    table.to_csv(OUTPUTS_DIR / "student_t_df_selection.csv", index=False)
    return best_df, table


def risk_table(sample: pd.DataFrame, weights: pd.Series, model_name: str, alphas: tuple[float, ...], nu: int | None = None) -> pd.DataFrame:
    mu = sample.mean()
    cov = sample.cov()
    rows: list[dict[str, float | str]] = []

    for asset, weight in weights.items():
        notional = weight * PORTFOLIO_VALUE
        mu_i = float(mu[asset])
        sigma_i = float(np.sqrt(cov.loc[asset, asset]))
        losses = -notional * sample[asset]

        for alpha in alphas:
            if model_name == "Historical":
                var_i, es_i = historical_var_es(losses, alpha)
            elif nu is None:
                var_i, es_i = normal_var_es(mu_i, sigma_i, alpha, notional)
            else:
                var_i, es_i = student_t_var_es(mu_i, sigma_i, alpha, nu, notional)
            rows.append({"model": model_name, "alpha": alpha, "entity": asset, "VaR_eur": var_i, "ES_eur": es_i})

    mu_p, sigma_p = portfolio_moments(weights.values, mu[weights.index].values, cov.loc[weights.index, weights.index].values)
    portfolio_returns = sample[weights.index] @ weights.values
    portfolio_losses = -PORTFOLIO_VALUE * portfolio_returns
    for alpha in alphas:
        if model_name == "Historical":
            var_p, es_p = historical_var_es(portfolio_losses, alpha)
        elif nu is None:
            var_p, es_p = normal_var_es(mu_p, sigma_p, alpha, PORTFOLIO_VALUE)
        else:
            var_p, es_p = student_t_var_es(mu_p, sigma_p, alpha, nu, PORTFOLIO_VALUE)
        rows.append({"model": model_name, "alpha": alpha, "entity": "Portfolio", "VaR_eur": var_p, "ES_eur": es_p})

    return pd.DataFrame(rows)


def historical_risk_table(sample: pd.DataFrame, weights: pd.Series, alphas: tuple[float, ...]) -> pd.DataFrame:
    return risk_table(sample, weights, "Historical", alphas)


def ewma_volatility(sample: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    arr = sample.values
    sigma = np.zeros_like(arr, dtype=float)
    sigma[0, :] = np.sqrt(np.maximum(np.var(arr, axis=0, ddof=1), 1e-12))
    for i in range(arr.shape[1]):
        for t_idx in range(1, arr.shape[0]):
            sigma[t_idx, i] = np.sqrt(lam * sigma[t_idx - 1, i] ** 2 + (1 - lam) * arr[t_idx - 1, i] ** 2)
    return pd.DataFrame(sigma, index=sample.index, columns=sample.columns)


def filtered_historical_table(sample: pd.DataFrame, weights: pd.Series, alphas: tuple[float, ...], lam: float = EWMA_LAMBDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    sigma = ewma_volatility(sample, lam=lam)
    z = sample / sigma.replace(0.0, np.nan)
    sigma_next = np.sqrt(lam * sigma.iloc[-1] ** 2 + (1 - lam) * sample.iloc[-1] ** 2)
    r_sim = z.mul(sigma_next, axis=1).dropna()

    rows: list[dict[str, float | str]] = []
    for asset, weight in weights.items():
        notional = weight * PORTFOLIO_VALUE
        losses = -notional * r_sim[asset]
        for alpha in alphas:
            var_i, es_i = historical_var_es(losses, alpha)
            rows.append({"model": "FHS-EWMA", "alpha": alpha, "entity": asset, "VaR_eur": var_i, "ES_eur": es_i})

    port_losses = -PORTFOLIO_VALUE * (r_sim[weights.index] @ weights.values)
    for alpha in alphas:
        var_p, es_p = historical_var_es(port_losses, alpha)
        rows.append({"model": "FHS-EWMA", "alpha": alpha, "entity": "Portfolio", "VaR_eur": var_p, "ES_eur": es_p})

    return pd.DataFrame(rows), sigma


def expanding_backtest(sample_all: pd.DataFrame, weights: pd.Series, alpha: float, nu: int | None = None, min_obs: int = 250) -> pd.DataFrame:
    out: list[dict[str, float | pd.Timestamp | bool]] = []
    sample_dates = sample_all.loc[SAMPLE_START:SAMPLE_END].index

    for date in sample_dates:
        history = sample_all.loc[sample_all.index < date, weights.index].dropna()
        if len(history) < min_obs:
            continue

        mu = history.mean().values
        cov = history.cov().values
        mu_p, sigma_p = portfolio_moments(weights.values, mu, cov)

        if nu is None:
            var_t, es_t = normal_var_es(mu_p, sigma_p, alpha, PORTFOLIO_VALUE)
        else:
            var_t, es_t = student_t_var_es(mu_p, sigma_p, alpha, nu, PORTFOLIO_VALUE)

        realized_return = float(weights.values @ sample_all.loc[date, weights.index].values)
        realized_loss = -PORTFOLIO_VALUE * realized_return
        out.append({"date": date, "VaR_eur": var_t, "ES_eur": es_t, "loss_eur": realized_loss})

    bt = pd.DataFrame(out).set_index("date")
    bt["hit"] = bt["loss_eur"] > bt["VaR_eur"]
    return bt


def expanding_backtest_hs_fhs(sample_all: pd.DataFrame, weights: pd.Series, alphas: tuple[float, ...], min_obs: int = 250, lam: float = EWMA_LAMBDA) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, np.ndarray], dict[float, np.ndarray], np.ndarray, pd.DatetimeIndex]:
    sample = sample_all.loc[SAMPLE_START:SAMPLE_END, weights.index].dropna()
    arr = sample.values
    n = len(sample)
    w = weights.values
    loss = -PORTFOLIO_VALUE * (arr @ w)
    sigma = ewma_volatility(sample, lam=lam).values
    z = arr / sigma

    hs_var = {a: np.full(n, np.nan) for a in alphas}
    hs_es = {a: np.full(n, np.nan) for a in alphas}
    fhs_var = {a: np.full(n, np.nan) for a in alphas}
    fhs_es = {a: np.full(n, np.nan) for a in alphas}

    for t_idx in range(min_obs - 1, n - 1):
        sig_t = np.sqrt(lam * sigma[t_idx] ** 2 + (1 - lam) * arr[t_idx] ** 2)
        r_sim_t = z[: t_idx + 1] * sig_t
        ls_fhs = np.sort(-PORTFOLIO_VALUE * (r_sim_t @ w))
        ls_hs = np.sort(loss[: t_idx + 1])
        m = t_idx + 1
        for alpha in alphas:
            k = int(np.ceil(m * alpha))
            if k < 1:
                k = 1
            var_hs = ls_hs[-k]
            var_fhs = ls_fhs[-k]
            es_hs = ls_hs[-k:].mean()
            es_fhs = ls_fhs[-k:].mean()
            hs_var[alpha][t_idx + 1] = var_hs
            hs_es[alpha][t_idx + 1] = es_hs
            fhs_var[alpha][t_idx + 1] = var_fhs
            fhs_es[alpha][t_idx + 1] = es_fhs

    bt_idx = np.arange(min_obs, n)
    return hs_var, hs_es, fhs_var, fhs_es, loss[bt_idx], sample.index[bt_idx]


def summarize_hs_fhs_backtests(hs_var: dict[float, np.ndarray], hs_es: dict[float, np.ndarray], fhs_var: dict[float, np.ndarray], fhs_es: dict[float, np.ndarray], bt_loss: np.ndarray, bt_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    residual_rows = []
    m_bt = len(bt_loss)
    for label, vd, ed in [("HS", hs_var, hs_es), ("FHS-EWMA", fhs_var, fhs_es)]:
        for alpha in ALPHAS:
            vt = vd[alpha][~np.isnan(vd[alpha])]
            et = ed[alpha][~np.isnan(ed[alpha])]
            current_losses = bt_loss[-len(vt):]
            hits = current_losses > vt
            actual = int(hits.sum())
            expected = len(vt) * alpha
            z_stat = (actual - expected) / np.sqrt(len(vt) * alpha * (1 - alpha))
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            kv = np.where(hits, (current_losses - et) / et, np.nan)
            kv = kv[~np.isnan(kv)]
            t_stat, t_p = (np.nan, np.nan) if len(kv) < 2 else stats.ttest_1samp(kv, 0.0)
            score = np.sum(np.abs((current_losses <= vt).astype(float) - alpha) * np.abs(current_losses - vt))
            rows.append(
                {
                    "model": label,
                    "alpha": alpha,
                    "expected_violations": expected,
                    "actual_violations": actual,
                    "binomial_z": z_stat,
                    "binomial_p": p_value,
                    "elicitability_score": score,
                }
            )
            residual_rows.append(
                {
                    "model": label,
                    "alpha": alpha,
                    "n_violations": len(kv),
                    "mean_k": float(np.nanmean(kv)) if len(kv) else np.nan,
                    "t_stat": t_stat,
                    "t_pvalue": t_p,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def fit_ccc_garch(sample: pd.DataFrame, weights: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    innovations = sample - sample.mean()
    assets = list(sample.columns)
    vol_df = pd.DataFrame(index=sample.index, columns=assets, dtype=float)
    next_vol = pd.Series(index=assets, dtype=float)

    for asset in assets:
        series = innovations[asset].dropna()
        model = arch_model(series, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = model.fit(disp="off")
        vol_df.loc[series.index, asset] = res.conditional_volatility
        omega = float(res.params.iloc[0])
        alpha = float(res.params.iloc[1])
        beta = float(res.params.iloc[2])
        a_last = float(series.iloc[-1])
        sigma_last = float(res.conditional_volatility.iloc[-1])
        next_vol[asset] = np.sqrt(max(omega + alpha * a_last ** 2 + beta * sigma_last ** 2, 1e-12))

    devol = innovations / vol_df
    corr = devol.corr().fillna(0.0)
    portfolio_sigma = pd.Series(index=sample.index, dtype=float)
    w = weights[assets].values.astype(float)
    for date in sample.index:
        delta = np.diag(vol_df.loc[date, assets].astype(float).values)
        sigma_t = delta @ corr.values @ delta
        portfolio_sigma.loc[date] = np.sqrt(max(w.T @ sigma_t @ w, 0.0))

    delta_next = np.diag(next_vol[assets].astype(float).values)
    sigma_next = delta_next @ corr.values @ delta_next
    port_sigma_next = float(np.sqrt(max(w.T @ sigma_next @ w, 0.0)))
    summary = pd.DataFrame(
        {
            "asset": assets + ["Portfolio"],
            "next_day_volatility": list(next_vol[assets].astype(float).values) + [port_sigma_next],
        }
    )
    path = pd.concat([vol_df, portfolio_sigma.rename("Portfolio")], axis=1)
    return path, summary


def garch_risk_tables(sample: pd.DataFrame, weights: pd.Series, garch_path: pd.DataFrame, garch_next: pd.DataFrame, alpha: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = norm.ppf(alpha)
    phi_z = norm.pdf(z)
    mu = sample.mean()
    assets = list(weights.index)
    asset_wealth = weights * PORTFOLIO_VALUE

    next_rows = []
    for asset in assets:
        sigma_hat = float(garch_next.loc[garch_next["asset"] == asset, "next_day_volatility"].iloc[0])
        var_i = asset_wealth[asset] * (-mu[asset] + z * sigma_hat)
        es_i = asset_wealth[asset] * (mu[asset] + sigma_hat * phi_z / (1 - alpha))
        next_rows.append({"entity": asset, "VaR_eur": var_i, "ES_eur": es_i})

    port_mu = float((weights * mu).sum())
    port_sigma_next = float(garch_next.loc[garch_next["asset"] == "Portfolio", "next_day_volatility"].iloc[0])
    next_rows.append(
        {
            "entity": "Portfolio",
            "VaR_eur": PORTFOLIO_VALUE * (-port_mu + z * port_sigma_next),
            "ES_eur": PORTFOLIO_VALUE * (port_mu + port_sigma_next * phi_z / (1 - alpha)),
        }
    )
    next_day = pd.DataFrame(next_rows)

    rolling = pd.DataFrame(index=garch_path.index)
    rolling["Portfolio_VaR_99"] = PORTFOLIO_VALUE * (-port_mu + z * garch_path["Portfolio"])
    rolling["Portfolio_ES_99"] = PORTFOLIO_VALUE * (port_mu + garch_path["Portfolio"] * phi_z / (1 - alpha))
    portfolio_returns = PORTFOLIO_VALUE * (sample[assets] @ weights.values)
    rolling["Portfolio_Return"] = portfolio_returns.reindex(rolling.index)

    loss_series = -portfolio_returns.reindex(rolling.index)
    yearly_rows = []
    years = sorted(set(rolling.index.year))
    alpha_c = 0.05
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp("2026-03-30") if year == 2026 else pd.Timestamp(f"{year+1}-01-01")
        mask = (rolling.index >= start) & (rolling.index <= end)
        losses = loss_series.loc[mask]
        var_series = rolling.loc[mask, "Portfolio_VaR_99"]
        es_series = rolling.loc[mask, "Portfolio_ES_99"]
        violations = losses > var_series
        ic_hat = int(violations.sum())
        t_len = len(violations)
        p = alpha
        expected = t_len * p
        c_l = binom.ppf(alpha_c / 2, t_len, p)
        c_u = binom.ppf(1 - alpha_c / 2, t_len, p)
        reject_var = (ic_hat < c_l) or (ic_hat > c_u)
        avg_shortfall = float(losses[violations].mean()) if ic_hat > 0 else 0.0
        avg_es = float(es_series.mean())
        yearly_rows.append(
            {
                "year": year,
                "violations": ic_hat,
                "expected": expected,
                "critical_low": c_l,
                "critical_high": c_u,
                "reject_var": reject_var,
                "avg_shortfall": avg_shortfall,
                "avg_es": avg_es,
            }
        )
    return next_day, rolling, pd.DataFrame(yearly_rows)


def make_non_overlapping_returns(r: pd.Series, horizon: int) -> pd.Series:
    r = pd.Series(r).dropna()
    n_full = len(r) // horizon
    trimmed = r.iloc[: n_full * horizon].to_numpy().reshape(n_full, horizon)
    block_returns = np.prod(1 + trimmed, axis=1) - 1
    block_dates = r.index[horizon - 1 : n_full * horizon : horizon]
    return pd.Series(block_returns, index=block_dates, name=f"{horizon}d_return")


def multi_day_var_table(sample: pd.DataFrame, weights: pd.Series, alphas: tuple[float, ...] = ALPHAS) -> pd.DataFrame:
    port_ret_1d = pd.Series(sample[weights.index].values @ weights.values, index=sample.index, name="portfolio_return")
    results = []
    for h in MULTI_DAY_HORIZONS:
        r_h = make_non_overlapping_returns(port_ret_1d, h)
        losses_h = -PORTFOLIO_VALUE * r_h
        for alpha in alphas:
            var_h, es_h = historical_var_es(losses_h, alpha)
            var_1d, _ = historical_var_es(-PORTFOLIO_VALUE * port_ret_1d, alpha)
            var_srot = var_1d * np.sqrt(h)
            results.append(
                {
                    "horizon_days": h,
                    "alpha": alpha,
                    "n_nonoverlap_obs": len(r_h),
                    "HS_VaR_eur": var_h,
                    "HS_ES_eur": es_h,
                    "SROT_VaR_eur": var_srot,
                    "difference_eur": var_srot - var_h,
                    "difference_pct_of_HS": 100 * (var_srot / var_h - 1),
                }
            )
    return pd.DataFrame(results)


def summarize_backtest(bt: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "obs": len(bt),
            "hits": int(bt["hit"].sum()),
            "hit_rate": float(bt["hit"].mean()),
            "avg_VaR_on_hits": float(bt.loc[bt["hit"], "VaR_eur"].mean()),
            "avg_shortfall": float(bt.loc[bt["hit"], "loss_eur"].mean()),
        }
    )


def build_stress_table(weights: pd.Series, sample: pd.DataFrame) -> pd.DataFrame:
    portfolio_returns = sample[weights.index] @ weights.values
    base_losses = -PORTFOLIO_VALUE * portfolio_returns
    base_var, base_es = historical_var_es(base_losses, 0.01)

    scenarios = [
        ("Equity/Index", "ASML", -0.40),
        ("Equity/Index", "ASML", -0.20),
        ("Equity/Index", "ASML", 0.20),
        ("Equity/Index", "ASML", 0.40),
        ("Equity/Index", "SHELL", -0.40),
        ("Equity/Index", "SHELL", -0.20),
        ("Equity/Index", "SHELL", 0.20),
        ("Equity/Index", "SHELL", 0.40),
        ("Equity/Index", "JPM_EUR", -0.40),
        ("Equity/Index", "JPM_EUR", -0.20),
        ("Equity/Index", "JPM_EUR", 0.20),
        ("Equity/Index", "JPM_EUR", 0.40),
        ("Equity/Index", "STOXX50", -0.40),
        ("Equity/Index", "STOXX50", -0.20),
        ("Equity/Index", "STOXX50", 0.20),
        ("Equity/Index", "STOXX50", 0.40),
        ("Equity/Index", "SP500_EUR", -0.40),
        ("Equity/Index", "SP500_EUR", -0.20),
        ("Equity/Index", "SP500_EUR", 0.20),
        ("Equity/Index", "SP500_EUR", 0.40),
        ("FX", "EURUSD", -0.10),
        ("FX", "EURUSD", 0.10),
        ("Rate", "EURIBOR3M", -0.03),
        ("Rate", "EURIBOR3M", -0.02),
        ("Rate", "EURIBOR3M", 0.02),
        ("Rate", "EURIBOR3M", 0.03),
    ]

    rows = []
    usd_block = weights["JPM_EUR"] + weights["SP500_EUR"]
    loan_weight = weights["LOAN"]
    mod_dur = 9.135099

    for factor_type, factor, shock in scenarios:
        if factor_type == "Equity/Index":
            instant_pnl = PORTFOLIO_VALUE * weights[factor] * shock
        elif factor_type == "FX":
            instant_pnl = PORTFOLIO_VALUE * usd_block * (-shock / (1.0 + shock))
        else:
            instant_pnl = -PORTFOLIO_VALUE * loan_weight * mod_dur * shock

        stressed_losses = base_losses - instant_pnl
        stressed_var, stressed_es = historical_var_es(stressed_losses, 0.01)
        rows.append(
            {
                "factor_type": factor_type,
                "factor": factor,
                "shock_label": shock_label(factor_type, shock),
                "instant_PnL_eur": instant_pnl,
                "stressed_VaR_eur": stressed_var,
                "stressed_ES_eur": stressed_es,
                "delta_VaR_eur": stressed_var - base_var,
                "delta_ES_eur": stressed_es - base_es,
            }
        )

    return pd.DataFrame(rows).sort_values(["factor_type", "factor", "shock_label"]).reset_index(drop=True)


def shock_label(factor_type: str, shock: float) -> str:
    if factor_type == "Rate":
        sign = "+" if shock > 0 else ""
        return f"{sign}{int(round(100 * shock))}pp"
    sign = "+" if shock > 0 else ""
    return f"{sign}{int(round(100 * shock))}%"


def save_plots(
    all_returns: pd.DataFrame,
    sample: pd.DataFrame,
    weights: pd.Series,
    qq_table: pd.DataFrame,
    best_df: int,
    bt_normal: pd.DataFrame,
    bt_t: pd.DataFrame,
    bt_hs_loss: np.ndarray,
    hs_var: dict[float, np.ndarray],
    fhs_var: dict[float, np.ndarray],
    bt_dates: pd.DatetimeIndex,
    garch_rolling: pd.DataFrame,
    multiday: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    cumulative = (1.0 + sample[weights.index] @ weights.values).cumprod()
    cumulative.plot(ax=ax, color="#2f6db3")
    ax.set_title("Portfolio Growth of €1")
    ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "portfolio_growth.png", dpi=160)
    plt.close(fig)

    z_pre = pooled_standardized_residuals(all_returns.loc[PRE_START:PRE_END, weights.index])
    z_sorted = np.sort(z_pre)
    n = len(z_sorted)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for idx, nu in enumerate(DF_CANDIDATES):
        ax = axes[idx]
        theoretical = unit_variance_t_quantiles(n, nu)
        ax.scatter(theoretical, z_sorted, s=2, alpha=0.4, color="#4c78a8")
        lims = [min(theoretical.min(), z_sorted.min()), max(theoretical.max(), z_sorted.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        rmse = float(qq_table.loc[qq_table["df"] == nu, "qq_rmse"].iloc[0])
        ax.set_title(f"t(df={nu}) RMSE={rmse:.4f}" + (" *" if nu == best_df else ""))
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Empirical quantiles")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "student_t_qq_plots.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, bt, title in [
        (axes[0], bt_normal, "Normal VaR Backtest"),
        (axes[1], bt_t, f"Student-t VaR Backtest (df={best_df})"),
    ]:
        ax.plot(bt.index, bt["loss_eur"], color="#4c78a8", linewidth=0.8, label="Actual loss")
        ax.plot(bt.index, bt["VaR_eur"], color="#e45756", linewidth=1.1, label="VaR")
        ax.scatter(bt.index[bt["hit"]], bt.loc[bt["hit"], "loss_eur"], color="black", s=14, label="Violations")
        ax.set_title(title)
        ax.set_ylabel("EUR")
        ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "var_backtest.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bt_normal.index, bt_normal["hit"].rolling(60).sum(), label="Normal")
    ax.plot(bt_t.index, bt_t["hit"].rolling(60).sum(), label=f"Student-t df={best_df}")
    ax.set_title("Rolling 60-Day VaR Violations")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "rolling_var_violations.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    for row, (label, vd) in enumerate([("HS", hs_var), ("FHS-EWMA", fhs_var)]):
        for col, alpha in enumerate(ALPHAS):
            ax = axes[row, col]
            vt = vd[alpha][~np.isnan(vd[alpha])]
            losses = bt_hs_loss[-len(vt):]
            dates = bt_dates[-len(vt):]
            hits = losses > vt
            ax.bar(dates, losses / 1e3, color="steelblue", alpha=0.5, width=1)
            ax.plot(dates, vt / 1e3, color="crimson", lw=1.2)
            ax.scatter(dates[hits], losses[hits] / 1e3, color="red", s=18)
            ax.set_title(f"{label}  α={(1-alpha)*100:.1f}%")
            ax.set_ylabel("Loss (EUR thousands)")
        plt.tight_layout()
    fig.savefig(PLOTS_DIR / "hs_fhs_backtest_violations.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(garch_rolling.index, garch_rolling["Portfolio_VaR_99"], label="CCC-GARCH VaR 99%", color="red")
    ax.plot(garch_rolling.index, garch_rolling["Portfolio_ES_99"], label="CCC-GARCH ES 99%", color="blue")
    ax.plot(garch_rolling.index, -garch_rolling["Portfolio_Return"], label="Portfolio Loss", color="green")
    ax.legend()
    ax.set_title("CCC-GARCH Portfolio VaR, ES, And Losses")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "garch_portfolio_risk.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, alpha in zip(axes, ALPHAS):
        sub = multiday[multiday["alpha"] == alpha].sort_values("horizon_days")
        ax.plot(sub["horizon_days"], sub["HS_VaR_eur"], marker="o", linewidth=2, label="Historical VaR")
        ax.plot(sub["horizon_days"], sub["SROT_VaR_eur"], marker="s", linewidth=2, linestyle="--", label="Scaled 1-day VaR")
        ax.set_title(f"Confidence level = {(1-alpha)*100:.1f}%")
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("VaR (EUR)")
        ax.set_xticks(list(MULTI_DAY_HORIZONS))
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "multiday_var_comparison.png", dpi=160)
    plt.close(fig)


def run_analysis() -> dict[str, pd.DataFrame | int]:
    ensure_dirs()
    prices, simple_returns = read_data()
    weights = selected_portfolio_weights(simple_returns)
    ret_pre = simple_returns.loc[PRE_START:PRE_END, weights.index]
    ret_sample = simple_returns.loc[SAMPLE_START:SAMPLE_END, weights.index]

    best_df, qq_table = select_student_t_df(ret_pre)
    risk_hist = risk_table(ret_sample, weights, "Historical", ALPHAS)
    risk_normal = risk_table(ret_sample, weights, "Normal", ALPHAS)
    risk_t = risk_table(ret_sample, weights, f"Student-t df={best_df}", ALPHAS, nu=best_df)
    risk_fhs, ewma_sigma = filtered_historical_table(ret_sample, weights, ALPHAS)

    bt_normal = expanding_backtest(simple_returns, weights, BACKTEST_ALPHA)
    bt_t = expanding_backtest(simple_returns, weights, BACKTEST_ALPHA, nu=best_df)
    hs_var, hs_es, fhs_var, fhs_es, bt_hs_loss, bt_dates = expanding_backtest_hs_fhs(simple_returns, weights, ALPHAS)
    hs_fhs_summary, hs_fhs_residuals = summarize_hs_fhs_backtests(hs_var, hs_es, fhs_var, fhs_es, bt_hs_loss, bt_dates)
    backtest_summary = pd.concat(
        {"Normal": summarize_backtest(bt_normal), f"Student-t df={best_df}": summarize_backtest(bt_t)},
        axis=1,
    ).T.reset_index().rename(columns={"index": "model"})
    backtest_summary = pd.concat([backtest_summary, hs_fhs_summary], ignore_index=True, sort=False)

    yearly_rows = []
    for label, bt in [("Normal", bt_normal), (f"Student-t df={best_df}", bt_t)]:
        yearly = bt.groupby(bt.index.year)["hit"].agg(["sum", "count"])
        yearly.columns = ["violations", "trading_days"]
        yearly["expected"] = yearly["trading_days"] * BACKTEST_ALPHA
        yearly["model"] = label
        yearly["year"] = yearly.index
        yearly_rows.append(yearly.reset_index(drop=True))
    yearly_violations = pd.concat(yearly_rows, ignore_index=True)

    es_rows = []
    for label, bt in [("Normal", bt_normal), (f"Student-t df={best_df}", bt_t)]:
        hits = bt.loc[bt["hit"]]
        yearly_es = hits.groupby(hits.index.year).agg(
            avg_predicted_ES=("ES_eur", "mean"),
            avg_realised_loss=("loss_eur", "mean"),
            n_violations=("hit", "count"),
        )
        yearly_es["model"] = label
        yearly_es["year"] = yearly_es.index
        es_rows.append(yearly_es.reset_index(drop=True))
    yearly_es = pd.concat(es_rows, ignore_index=True)

    stress_table = build_stress_table(weights, ret_sample)
    garch_path, garch_next = fit_ccc_garch(ret_sample, weights)
    garch_next_day, garch_rolling, garch_yearly = garch_risk_tables(ret_sample, weights, garch_path, garch_next)
    multiday = multi_day_var_table(ret_sample, weights)

    save_plots(
        simple_returns,
        simple_returns.loc[SAMPLE_START:SAMPLE_END],
        weights,
        qq_table,
        best_df,
        bt_normal,
        bt_t,
        bt_hs_loss,
        hs_var,
        fhs_var,
        bt_dates,
        garch_rolling,
        multiday,
    )

    risk_all = pd.concat([risk_hist, risk_normal, risk_t, risk_fhs], ignore_index=True)
    risk_all.to_csv(OUTPUTS_DIR / "var_es_table.csv", index=False)
    backtest_summary.to_csv(OUTPUTS_DIR / "backtest_summary.csv", index=False)
    hs_fhs_residuals.to_csv(OUTPUTS_DIR / "hs_fhs_es_residual_tests.csv", index=False)
    yearly_violations.to_csv(OUTPUTS_DIR / "yearly_violations.csv", index=False)
    yearly_es.to_csv(OUTPUTS_DIR / "yearly_es_shortfall.csv", index=False)
    stress_table.to_csv(OUTPUTS_DIR / "stress_scenarios.csv", index=False)
    ewma_sigma.to_csv(OUTPUTS_DIR / "ewma_volatility_paths.csv")
    garch_path.to_csv(OUTPUTS_DIR / "ccc_garch_volatility_paths.csv")
    garch_next_day.to_csv(OUTPUTS_DIR / "ccc_garch_next_day_risk.csv", index=False)
    garch_yearly.to_csv(OUTPUTS_DIR / "ccc_garch_yearly_backtest.csv", index=False)
    multiday.to_csv(OUTPUTS_DIR / "multiday_var_scaling.csv", index=False)

    return {
        "weights": weights.to_frame("weight"),
        "qq_table": qq_table,
        "risk_table": risk_all,
        "backtest_summary": backtest_summary,
        "hs_fhs_residuals": hs_fhs_residuals,
        "yearly_violations": yearly_violations,
        "yearly_es": yearly_es,
        "stress_table": stress_table,
        "garch_next_day": garch_next_day,
        "garch_yearly": garch_yearly,
        "multiday": multiday,
        "best_df": best_df,
    }


def main() -> None:
    results = run_analysis()
    print("Selected Student-t df:", results["best_df"])
    print("\nBacktest summary")
    print(results["backtest_summary"].round(4).to_string(index=False))
    portfolio_risk = results["risk_table"].query("entity == 'Portfolio'")
    print("\nPortfolio VaR/ES")
    print(portfolio_risk.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
