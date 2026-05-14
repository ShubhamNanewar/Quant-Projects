from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def hourly_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    hourly = train.groupby("hour")["spread"].mean()
    return test["hour"].map(hourly).fillna(train["spread"].mean())


def model_candidates() -> dict[str, object]:
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=3.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=20,
            max_depth=8,
            random_state=7,
            n_jobs=1,
        ),
    }


def walk_forward_backtest(
    frame: pd.DataFrame,
    features: list[str],
    initial_train_hours: int = 24 * 180,
    test_hours: int = 24 * 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    metrics = []
    importances = []
    models = model_candidates()
    start = initial_train_hours
    fold = 0

    while start + test_hours <= len(frame):
        train = frame.iloc[:start].copy()
        test = frame.iloc[start : start + test_hours].copy()

        base_pred = hourly_baseline(train, test)
        metrics.append(
            {"fold": fold, "model": "hourly_baseline", **regression_metrics(test["spread"], base_pred)}
        )
        predictions.append(
            test[["datetime_local", "spread", "total_lmp_da", "total_lmp_rt"]].assign(
                model="hourly_baseline", prediction=base_pred.to_numpy()
            )
        )

        for name, model in models.items():
            fitted = model.fit(train[features], train["spread"])
            pred = fitted.predict(test[features])
            metrics.append({"fold": fold, "model": name, **regression_metrics(test["spread"], pred)})
            predictions.append(
                test[["datetime_local", "spread", "total_lmp_da", "total_lmp_rt"]].assign(
                    model=name, prediction=pred
                )
            )
            if name == "random_forest":
                importances.append(
                    pd.DataFrame({"fold": fold, "feature": features, "importance": fitted.feature_importances_})
                )

        fold += 1
        start += test_hours

    if not predictions:
        raise ValueError("Not enough data for walk-forward validation.")

    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metrics),
        pd.concat(importances, ignore_index=True),
    )


def signal_summary(results: pd.DataFrame, model: str = "random_forest") -> dict[str, float]:
    subset = results.loc[results["model"] == model].copy()
    threshold = subset["prediction"].quantile(0.90)
    signal = subset.loc[subset["prediction"] >= threshold].copy()
    positive_total = subset["spread"].clip(lower=0).sum()
    return {
        "signal_count": float(len(signal)),
        "signal_share_of_hours": float(len(signal) / len(subset)),
        "mean_realized_spread_signal_hours": float(signal["spread"].mean()),
        "median_realized_spread_signal_hours": float(signal["spread"].median()),
        "positive_hit_rate_signal_hours": float((signal["spread"] > 0).mean()),
        "positive_hit_rate_all_hours": float((subset["spread"] > 0).mean()),
        "share_of_total_positive_spread_captured": float(
            signal["spread"].clip(lower=0).sum() / positive_total if positive_total else 0.0
        ),
    }

