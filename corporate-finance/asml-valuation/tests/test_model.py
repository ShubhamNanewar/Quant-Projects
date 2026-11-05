"""Financial identities and valuation behaviour, independent of workbook styling."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyse import historical, dcf


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hist=historical()

    def test_reported_statement_identities(self):
        for year,a in self.hist.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(a['Revenue']-a['Cost of sales'],a['Gross profit'],places=4)
                self.assertAlmostEqual(a['Gross profit']-a['R&D']-a['SG&A'],a['EBIT'],places=4)
                self.assertAlmostEqual(a['Assets']-a['Liabilities'],a['Equity'],places=4)

    def test_annual_controls(self):
        self.assertAlmostEqual(self.hist['2025']['Revenue'],32667.3)
        self.assertAlmostEqual(self.hist['2025']['Reported FCF'],11027.3)

    def test_discount_rate_and_growth_direction(self):
        self.assertGreater(dcf(wacc=.085)['per_share'],dcf(wacc=.105)['per_share'])
        self.assertGreater(dcf(terminal_growth=.04)['per_share'],dcf(terminal_growth=.02)['per_share'])

    def test_case_order(self):
        self.assertGreater(dcf(2)['per_share'],dcf(0)['per_share'])
        self.assertGreater(dcf(0)['per_share'],dcf(1)['per_share'])

    def test_cash_flow_bridge(self):
        for row in dcf()['forecast']:
            self.assertAlmostEqual(row['fcff'],row['nopat']+row['da']-row['capex']-row['delta_nwc'])

    def test_terminal_guard(self):
        with self.assertRaises(ValueError):
            dcf(wacc=.03,terminal_growth=.03)


if __name__=='__main__':
    unittest.main()
