import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis as a

class NumericalChecks(unittest.TestCase):
    def test_pde_against_closed_form_across_moneyness(self):
        # A high-volatility, long-maturity case also exercises the far boundary.
        for strike, sigma, maturity in [(80, .2, 1), (100, .2, 1), (120, .2, 1), (100, .8, 2)]:
            expected = a.black_scholes_call_price(100, strike, .03, sigma, maturity)
            actual = a.crank_nicolson_call_price(100, strike, .03, sigma, maturity, 3600 if sigma > .5 else 900, 900, 12 if sigma > .5 else 3)
            # Expand the domain for the high-volatility case.
            self.assertLess(abs(actual-expected), .02)

    def test_single_interval_self_financing_identity(self):
        s, k, r, vol, t = 100, 100, .03, .2, 1
        terminal = a.simulate_gbm_paths(s, .05, vol, t, 1, 500, 7)[:, -1]
        delta = a.black_scholes_delta(s, k, r, vol, t)
        premium = a.black_scholes_call_price(s, k, r, vol, t)
        expected = delta*terminal + (premium-delta*s)*np.exp(r*t) - np.maximum(terminal-k, 0)
        actual, _ = a.simulate_delta_hedge_pnl(s,k,r,vol,t,1,500,.05,1,1,7)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_finer_hedging_reduces_error_under_model(self):
        coarse, _ = a.simulate_delta_hedge_pnl(100,100,.03,.2,1,4,20000,.03,1,1,7)
        fine, _ = a.simulate_delta_hedge_pnl(100,100,.03,.2,1,252,20000,.03,1,1,7)
        self.assertLess(fine.std(), .3*coarse.std())
        self.assertLess(abs(fine.mean()), 4*fine.std()/np.sqrt(len(fine)))

if __name__ == "__main__":
    unittest.main()
