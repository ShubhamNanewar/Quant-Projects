from __future__ import annotations

import numpy as np
import pandas as pd


def classify_driver(row: pd.Series) -> str:
    components = {
        "energy": abs(row.get("energy_spread", 0.0)),
        "congestion": abs(row.get("congestion_spread", 0.0)),
        "loss": abs(row.get("loss_spread", 0.0)),
    }
    total = sum(components.values())
    if total == 0:
        return "unclassified"
    driver, value = max(components.items(), key=lambda item: item[1])
    share = value / total
    if share >= 0.60:
        return f"{driver}_led"
    return "mixed"


def add_event_metrics(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_values("datetime_utc")
    frame["abs_spread"] = frame["spread"].abs()
    frame["spread_roll_mean_168h"] = frame["spread"].shift(1).rolling(168, min_periods=48).mean()
    frame["spread_roll_vol_168h"] = frame["spread"].shift(1).rolling(168, min_periods=48).std()
    frame["spread_zscore_168h"] = (
        (frame["spread"] - frame["spread_roll_mean_168h"]) / frame["spread_roll_vol_168h"]
    )
    frame["spread_percentile"] = frame["spread"].rank(pct=True)
    frame["driver"] = frame.apply(classify_driver, axis=1)
    return frame


def top_events(data: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = [
        "datetime_local",
        "total_lmp_da",
        "total_lmp_rt",
        "spread",
        "energy_spread",
        "congestion_spread",
        "loss_spread",
        "spread_zscore_168h",
        "driver",
    ]
    return data.sort_values("spread", ascending=False).head(n)[cols].reset_index(drop=True)


def event_summary(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.dropna(subset=["driver"]).copy()
    return (
        frame.groupby("driver", as_index=False)
        .agg(
            events=("spread", "size"),
            mean_spread=("spread", "mean"),
            p95_spread=("spread", lambda x: x.quantile(0.95)),
            mean_abs_spread=("abs_spread", "mean"),
        )
        .sort_values("mean_abs_spread", ascending=False)
    )

