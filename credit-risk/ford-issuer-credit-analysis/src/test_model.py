import unittest
from model import INPUT,forecast,bond_price
class ModelTests(unittest.TestCase):
 def test_base_bridge(self):
  r=forecast(INPUT['scenarios']['Base']);self.assertAlmostEqual(r['adjusted_fcf'],5.5);self.assertAlmostEqual(r['cash_before_funding'],23.46)
 def test_maturity_reconciliation(self):self.assertAlmostEqual(sum(INPUT['maturities'].values()),21.919)
 def test_no_credit_debt_in_cash_bridge(self):
  o=INPUT['opening'].copy();o['ford_credit_debt']=1000
  self.assertEqual(forecast(INPUT['scenarios']['Base'],o),forecast(INPUT['scenarios']['Base']))
 def test_draw_pays_interest(self):
  r=forecast(INPUT['scenarios']['Severe']);self.assertAlmostEqual(r['cash_after_funding'],20);self.assertGreater(r['required_draw'],6.74)
 def test_worse_cash_flow_more_funding(self):
  s=INPUT['scenarios']['Base'].copy();s['working_capital']=-10
  self.assertGreater(forecast(s)['required_draw'],forecast(INPUT['scenarios']['Base'])['required_draw'])
 def test_capacity_shortfall(self):
  o=INPUT['opening'].copy();o['durable_undrawn_facilities']=1
  r=forecast(INPUT['scenarios']['Severe'],o)
  self.assertGreater(r['unfunded_need'],0);self.assertLess(r['cash_after_funding'],20);self.assertEqual(r['actual_draw'],1)
 def test_bond_at_par(self):self.assertAlmostEqual(bond_price(.0325),100)
 def test_yield_price_direction(self):self.assertGreater(bond_price(.04),bond_price(.08))
 def test_dividend_action(self):
  s=INPUT['scenarios']['Downside'].copy();before=forecast(s);s['dividends']=0
  self.assertAlmostEqual(forecast(s)['cash_before_funding']-before['cash_before_funding'],2.4)
if __name__=='__main__':unittest.main()
