"""Conservative hourly forecast audit, not an executable DA/RT trading backtest."""
from __future__ import annotations
import numpy as np
import pandas as pd
from modeling import model_candidates, hourly_baseline, regression_metrics

FEATURES = ['hour_sin', 'hour_cos', 'weekend', 'spread_lag48', 'spread_mean24',
            'spread_vol24', 'spread_mean168', 'spread_vol168', 'da_lag48']


def audit_features(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.copy()
    f['datetime_utc'] = pd.to_datetime(f['datetime_utc'], utc=True)
    f = f.sort_values('datetime_utc').reset_index(drop=True)
    if f.datetime_utc.duplicated().any() or not f.datetime_utc.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Expected unique, contiguous hourly UTC observations')
    local = f.datetime_utc.dt.tz_convert('America/Los_Angeles')
    f['hour'] = local.dt.hour
    f['hour_sin'] = np.sin(2*np.pi*f.hour/24)
    f['hour_cos'] = np.cos(2*np.pi*f.hour/24)
    f['weekend'] = (local.dt.dayofweek >= 5).astype(int)
    past = f.spread.shift(48)
    f['spread_lag48'] = past
    for w in (24, 168):
        f[f'spread_mean{w}'] = past.rolling(w, min_periods=w).mean()
        f[f'spread_vol{w}'] = past.rolling(w, min_periods=w).std()
    f['da_lag48'] = f.total_lmp_da.shift(48)
    return f.dropna(subset=FEATURES).reset_index(drop=True)


def audit_walk_forward(frame, train_hours=4320, calibration_hours=720, test_hours=720, embargo=48):
    """Fit on past labels; use a separate prior prediction window for thresholds.

    Features refresh hourly, while fitted models and thresholds stay fixed per fold.
    The 48-hour lag/embargo is an assumption, not verified publication availability.
    """
    if min(train_hours, calibration_hours, test_hours, embargo) < 1:
        raise ValueError('Window lengths must be positive')
    rows, metrics = [], []
    first = train_hours + calibration_hours + 2*embargo
    for fold, start in enumerate(range(first, len(frame), test_hours)):
        cal_end = start - embargo
        cal_start = cal_end - calibration_hours
        train = frame.iloc[:cal_start-embargo]
        cal = frame.iloc[cal_start:cal_end]
        test = frame.iloc[start:start+test_hours]
        for name, model in {'hourly_baseline': None, **model_candidates()}.items():
            if model is None:
                cp, pred = hourly_baseline(train, cal), hourly_baseline(train, test)
            else:
                model.fit(train[FEATURES], train.spread)
                cp, pred = model.predict(cal[FEATURES]), model.predict(test[FEATURES])
            threshold = max(0., float(np.quantile(cp, .9)))
            part = test[['datetime_utc','spread']].copy()
            part['prediction'] = np.asarray(pred)
            part['threshold'] = threshold
            part['selected'] = (part.prediction >= threshold) & (part.prediction > 0)
            part['model'], part['fold'] = name, fold
            part['train_end'] = train.datetime_utc.iloc[-1]
            part['calibration_end'] = cal.datetime_utc.iloc[-1]
            rows.append(part)
            metrics.append(dict(model=name, fold=fold, observations=len(test),
                                **regression_metrics(test.spread, np.asarray(pred))))
    if not rows:
        raise ValueError('Insufficient history for training and calibration')
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def diagnostic_summary(predictions):
    records=[]
    for name, group in predictions.groupby('model'):
        selected=group.loc[group.selected, 'spread']
        record=dict(model=name, observations=len(group), selected_hours=len(selected),
                    selected_share=len(selected)/len(group),
                    positive_share_all=(group.spread>0).mean(),
                    positive_share_selected=(selected>0).mean(),
                    mean_spread_selected=selected.mean(), worst_spread_selected=selected.min(),
                    mean_without_best_five=selected.sort_values().iloc[:-5].mean() if len(selected)>5 else np.nan,
                    **regression_metrics(group.spread, group.prediction))
        for hurdle in (1,5,10):
            record[f'mean_spread_less_{hurdle}_usd_per_mwh']=selected.mean()-hurdle
        records.append(record)
    return pd.DataFrame(records)
