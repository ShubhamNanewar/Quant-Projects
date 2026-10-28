import unittest
import pandas as pd
from desk_diagnostics import validate, summarize, cost_sensitivity

class DeskDiagnosticsTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame(dict(datetime_utc=["2025-11-02T08:00Z","2025-11-02T09:00Z","2025-11-02T10:00Z"],model=["m"]*3,spread=[-4,8,2],prediction=[-3,3,1],threshold=[2]*3,selected=[True,True,False],direction=[-1,1,1]))
    def test_direction_and_cost(self):
        d=validate(self.data());s=summarize(d,["model"]).iloc[0]
        self.assertEqual(s.mean_signed_spread,6)
        self.assertEqual(s.selected_hours,2)
        self.assertEqual(cost_sensitivity(d,(1,)).iloc[0].adjusted_mean,5)
    def test_dst_keeps_both_hours(self):
        d=validate(self.data());self.assertEqual(list(d.delivery_hour),[1,1,2]);self.assertEqual(len(d),3)
    def test_reject_duplicates(self):
        d=self.data()
        with self.assertRaises(ValueError):validate(pd.concat([d,d.iloc[:1]]))
    def test_reject_changed_selection(self):
        d=self.data();d.loc[2,"selected"]=True
        with self.assertRaises(ValueError):validate(d)
    def test_empty_selection(self):
        d=self.data();d["threshold"]=10;d["selected"]=False
        s=summarize(validate(d),["model"]).iloc[0]
        self.assertEqual(s.selected_hours,0);self.assertTrue(pd.isna(s.mean_signed_spread))
