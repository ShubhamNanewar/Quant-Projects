import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from fundamental_extension import build_panel, fixed_year_predictions


class FundamentalExtensionTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2024-01-01', '2025-01-05', freq='h', tz='UTC')
        t = np.arange(len(dates))
        self.prices = pd.DataFrame({'datetime_utc':dates, 'spread':np.sin(t/13),
                                    'total_lmp_da':30+np.cos(t/17)})
        self.forecasts = pd.DataFrame({'datetime_utc':dates, 'load_da_system':20000+100*np.sin(t/24),
                                      'load_da_pge':8000+50*np.sin(t/24)})

    def test_missing_forecast_not_backfilled(self):
        self.forecasts.loc[500,'load_da_pge'] = np.nan
        frame = build_panel(self.prices,self.forecasts)
        row = frame.loc[frame.datetime_utc==self.forecasts.datetime_utc.iloc[500]]
        self.assertTrue(row.load_da_pge.isna().all())

    def test_2025_targets_cannot_affect_frozen_predictions(self):
        frame = build_panel(self.prices,self.forecasts)
        changed = frame.copy()
        changed.loc[changed.datetime_utc.dt.year==2025,'spread'] = 123456
        def small_model():
            return {'random_forest':RandomForestRegressor(n_estimators=3, max_depth=3, random_state=7)}
        with patch('fundamental_extension.model_candidates',side_effect=small_model):
            a,_,_ = fixed_year_predictions(frame)
            b,_,_ = fixed_year_predictions(changed)
        pd.testing.assert_frame_equal(a[['prediction','threshold','selected']],b[['prediction','threshold','selected']])

if __name__=='__main__':
    unittest.main()
