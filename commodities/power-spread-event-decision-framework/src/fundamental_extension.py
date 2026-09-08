"""A fixed external-year forecasting experiment, not an execution backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from modeling import hourly_baseline, model_candidates, regression_metrics
from validation_audit import FEATURES, audit_features

PRICE_FEATURES = FEATURES + ['total_lmp_da']
LOAD_FEATURES = PRICE_FEATURES + ['load_da_system', 'load_da_pge', 'load_pge_share',
                                 'load_system_hourly_change', 'load_pge_hourly_change']


def build_panel(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Calendar/price histories lagged 48h; DAM prices and load are archived inputs.

    Load differences compare adjacent delivery hours, not successive forecast vintages.
    Missing forecast observations are excluded, never backfilled from future values.
    """
    frame = audit_features(prices)
    load = forecasts.copy()
    load['datetime_utc'] = pd.to_datetime(load.datetime_utc, utc=True)
    load = load.sort_values('datetime_utc')
    if load.datetime_utc.duplicated().any():
        raise ValueError('Duplicate demand forecast timestamps')
    load['load_pge_share'] = load.load_da_pge / load.load_da_system.replace(0, np.nan)
    continuous = load.datetime_utc.diff().eq(pd.Timedelta(hours=1))
    load['load_system_hourly_change'] = load.load_da_system.diff().where(continuous)
    load['load_pge_hourly_change'] = load.load_da_pge.diff().where(continuous)
    return frame.merge(load, on='datetime_utc', how='left', validate='one_to_one')


def fixed_year_predictions(frame: pd.DataFrame):
    local = frame.datetime_utc.dt.tz_convert('America/Los_Angeles')
    train = frame[(local < pd.Timestamp('2024-09-29', tz='America/Los_Angeles'))]
    calibration = frame[(local >= pd.Timestamp('2024-10-01', tz='America/Los_Angeles')) &
                        (local < pd.Timestamp('2024-12-30', tz='America/Los_Angeles'))]
    test = frame[(local >= pd.Timestamp('2025-01-01', tz='America/Los_Angeles')) &
                 (local < pd.Timestamp('2026-01-01', tz='America/Los_Angeles'))]
    splits = {'train': train, 'calibration': calibration, 'test': test}
    coverage = []
    for name, subset in splits.items():
        clean = subset.dropna(subset=LOAD_FEATURES+['spread'])
        coverage.append(dict(split=name, raw_hours=len(subset), retained_hours=len(clean),
                             excluded_missing_hours=len(subset)-len(clean)))
        splits[name] = clean
    train, calibration, test = (splits[k] for k in ('train','calibration','test'))
    if min(len(train),len(calibration),len(test)) == 0:
        raise ValueError('Each predefined split must contain complete observations')
    results, metrics = [], []
    for name, features in [('hourly_baseline', None), ('rf_price', PRICE_FEATURES),
                           ('rf_price_demand', LOAD_FEATURES)]:
        if features is None:
            cp = hourly_baseline(train, calibration)
            prediction = hourly_baseline(train, test).to_numpy()
        else:
            model = model_candidates()['random_forest']
            model.fit(train[features], train.spread)
            cp = model.predict(calibration[features])
            prediction = model.predict(test[features])
        threshold = float(np.quantile(np.abs(cp), .9))
        selected = (np.abs(prediction) >= threshold) & (prediction != 0)
        part = test[['datetime_utc','spread']].copy()
        part['model'] = name
        part['prediction'] = prediction
        part['threshold'] = threshold
        part['selected'] = selected
        part['direction'] = np.sign(prediction)
        part['signed_spread'] = part.direction * part.spread
        part['month'] = part.datetime_utc.dt.tz_convert('America/Los_Angeles').dt.strftime('%Y-%m')
        results.append(part)
        values = part.loc[part.selected,'signed_spread']
        metrics.append(dict(model=name, observations=len(test), threshold=threshold,
                            selected_hours=int(selected.sum()), selected_share=selected.mean(),
                            selected_positive_share=(values>0).mean(),
                            selected_mean_signed_spread=values.mean(),
                            selected_worst_signed_spread=values.min(),
                            selected_mean_less_1_usd_per_mwh=values.mean()-1,
                            **regression_metrics(test.spread,prediction)))
    return pd.concat(results,ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(coverage)
