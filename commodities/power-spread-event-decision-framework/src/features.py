from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "da_lmp_level",
    "da_lmp_roll_mean_24h",
    "da_lmp_roll_vol_24h",
    "spread_lag_24h",
    "spread_lag_48h",
    "spread_roll_mean_24h",
    "spread_roll_vol_24h",
    "spread_roll_mean_168h",
    "spread_roll_vol_168h",
]


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["hour"] = frame["datetime_local"].dt.hour
    frame["day_of_week"] = frame["datetime_local"].dt.dayofweek
    frame["month"] = frame["datetime_local"].dt.month
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    return frame


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = add_time_features(data).sort_values("datetime_utc")
    frame["spread_lag_24h"] = frame["spread"].shift(24)
    frame["spread_lag_48h"] = frame["spread"].shift(48)
    frame["spread_roll_mean_24h"] = frame["spread"].shift(1).rolling(24, min_periods=12).mean()
    frame["spread_roll_vol_24h"] = frame["spread"].shift(1).rolling(24, min_periods=12).std()
    frame["spread_roll_mean_168h"] = frame["spread"].shift(1).rolling(168, min_periods=48).mean()
    frame["spread_roll_vol_168h"] = frame["spread"].shift(1).rolling(168, min_periods=48).std()
    frame["da_lmp_level"] = frame["total_lmp_da"]
    frame["da_lmp_roll_mean_24h"] = (
        frame["total_lmp_da"].shift(1).rolling(24, min_periods=12).mean()
    )
    frame["da_lmp_roll_vol_24h"] = (
        frame["total_lmp_da"].shift(1).rolling(24, min_periods=12).std()
    )
    return frame.dropna().reset_index(drop=True)

