import unittest
import numpy as np
import pandas as pd
from validation_audit import audit_features, audit_walk_forward, FEATURES

class TemporalIntegrityTests(unittest.TestCase):
    def setUp(self):
        n=900
        self.raw=pd.DataFrame({'datetime_utc':pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC'),
                              'spread':np.sin(np.arange(n)/7), 'total_lmp_da':np.cos(np.arange(n)/11)+30})
    def test_future_observations_do_not_change_past_features(self):
        changed=self.raw.copy()
        changed.loc[600:,['spread','total_lmp_da']]=9999
        a,b=audit_features(self.raw),audit_features(changed)
        cutoff=self.raw.datetime_utc.iloc[647]
        pd.testing.assert_frame_equal(a.loc[a.datetime_utc<=cutoff,FEATURES],b.loc[b.datetime_utc<=cutoff,FEATURES])
    def test_missing_hour_rejected(self):
        with self.assertRaises(ValueError): audit_features(self.raw.drop(index=100))
    def test_test_targets_do_not_change_predictions_or_thresholds(self):
        f=audit_features(self.raw)
        kwargs=dict(train_hours=200,calibration_hours=80,test_hours=150,embargo=48)
        first=200+80+96
        changed=f.copy()
        changed.loc[first:,'spread']=9999
        a,_=audit_walk_forward(f,**kwargs)
        b,_=audit_walk_forward(changed,**kwargs)
        # Only first fold: subsequent folds legitimately train on earlier test labels.
        cols=['prediction','threshold','selected']
        pd.testing.assert_frame_equal(a.loc[a.fold==0,cols],b.loc[b.fold==0,cols])
        self.assertTrue((a.train_end<a.calibration_end).all())
        self.assertTrue((a.calibration_end<a.datetime_utc).all())
if __name__=='__main__': unittest.main()
