from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_industry_data() -> pd.DataFrame:
    sector = pd.read_csv(DATA_DIR / "48_Industry_Portfolios_Daily.csv", low_memory=False)
    sector = sector.rename(columns={sector.columns[0]: "date"})
    sector["date"] = sector["date"].astype(str).str.strip()
    sector = sector[sector["date"].str.fullmatch(r"\d{8}")].copy()
    sector["date"] = pd.to_datetime(sector["date"], format="%Y%m%d")

    columns = ["date", "Aero", "Guns", "Ships", "Oil", "Coal", "Util"]
    sector = sector[columns].copy()
    for column in columns[1:]:
        sector[column] = pd.to_numeric(sector[column], errors="coerce")
        sector.loc[sector[column] <= -99, column] = np.nan

    return sector.set_index("date").sort_index()


def load_market_data() -> pd.DataFrame:
    market = pd.read_csv(DATA_DIR / "SP_500_Data.csv")
    market["caldt"] = pd.to_datetime(market["caldt"])
    market["sprtrn"] = pd.to_numeric(market["sprtrn"], errors="coerce")
    market = market.rename(columns={"caldt": "date", "sprtrn": "market_ret"})
    return market.set_index("date").sort_index()


def construct_geopolitical_factor(
    start: str = "2000-01-03",
    end: str = "2009-12-31",
) -> pd.DataFrame:
    sector = load_industry_data().loc[start:end] / 100
    market = load_market_data().loc[start:end]

    factor = pd.concat([sector, market[["market_ret"]]], axis=1).dropna()
    factor["defense_ret"] = factor[["Aero", "Guns", "Ships"]].mean(axis=1)
    factor["energy_ret"] = factor[["Oil", "Coal", "Util"]].mean(axis=1)
    factor["gpr_long"] = 0.5 * factor["defense_ret"] + 0.5 * factor["energy_ret"]
    factor["gpr_factor"] = factor["gpr_long"] - factor["market_ret"]
    return factor


def calibrate_tail_shock(factor_returns: pd.Series, horizon_days: int = 126) -> float:
    factor_price = (1 + factor_returns).cumprod()
    rolling_return = factor_price / factor_price.shift(horizon_days) - 1
    return float(rolling_return.dropna().quantile(0.99))


def load_balance_sheet_data() -> pd.DataFrame:
    dashboard = pd.read_csv(DATA_DIR / "Dashboard_data.csv")
    dashboard = dashboard[["tic", "datadate", "atq", "ceqq"]].copy()
    dashboard["datadate"] = pd.to_datetime(dashboard["datadate"])
    dashboard["Book Leverage"] = dashboard["atq"] - dashboard["ceqq"]
    return dashboard


def load_bank_market_data() -> pd.DataFrame:
    bank = pd.read_csv(DATA_DIR / "Bank_data_full_2.csv", low_memory=False)
    bank["date"] = pd.to_datetime(bank["date"])
    bank["PRC"] = pd.to_numeric(bank["PRC"], errors="coerce")
    bank["SHROUT"] = pd.to_numeric(bank["SHROUT"], errors="coerce")
    bank["Market_Cap"] = bank["PRC"].abs() * bank["SHROUT"]
    return bank


def merge_daily_market_and_quarterly_book() -> dict[str, pd.DataFrame]:
    dashboard = load_balance_sheet_data()
    bank = load_bank_market_data()

    dashboard_dict = {
        ticker: group.drop(columns="tic").set_index("datadate").sort_index()
        for ticker, group in dashboard.groupby("tic")
    }
    bank_dict = {
        ticker: group.set_index("date").sort_index()
        for ticker, group in bank.groupby("TICKER")
    }

    merged: dict[str, pd.DataFrame] = {}
    common_tickers = set(dashboard_dict).intersection(bank_dict)

    for ticker in common_tickers:
        quarterly = dashboard_dict[ticker]
        daily = bank_dict[ticker].copy()
        daily["Book Leverage"] = quarterly["Book Leverage"].reindex(daily.index, method="ffill")
        daily = daily.dropna(subset=["Market_Cap", "Book Leverage"])
        merged[ticker] = daily

    return merged


def compute_loss_multiplier(beta_geo: pd.Series, stress_shock: float) -> pd.Series:
    return np.exp(beta_geo * np.log1p(stress_shock))


def compute_georisk(
    book_leverage: pd.Series,
    market_cap: pd.Series,
    beta_geo: pd.Series,
    prudential_ratio: float = 0.08,
    stress_shock: float | None = None,
) -> pd.DataFrame:
    if stress_shock is None:
        factor = construct_geopolitical_factor()
        stress_shock = calibrate_tail_shock(factor["gpr_factor"])

    loss_multiplier = compute_loss_multiplier(beta_geo, stress_shock)
    georisk = prudential_ratio * book_leverage - (1 - prudential_ratio) * market_cap * loss_multiplier
    marginal_georisk = (1 - prudential_ratio) * market_cap * (loss_multiplier - 1)

    return pd.DataFrame(
        {
            "Book Leverage": book_leverage,
            "Market_Cap": market_cap,
            "beta_geo": beta_geo,
            "stress_shock_xi": stress_shock,
            "LossMultiplier": loss_multiplier,
            "GeoRisk": georisk,
            "MarginalGeoRisk": marginal_georisk,
        }
    )


if __name__ == "__main__":
    factor = construct_geopolitical_factor()
    xi = calibrate_tail_shock(factor["gpr_factor"])
    print("99th percentile 6-month geopolitical stress shock:", round(xi, 6))
