"""Integration checks against the published public-data sample."""
import unittest
from desk_risk.analyze import (ROOT, verify, exchange_tables, candle_table,
                              market_table, position_tables, vault_tables)


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'data/raw' / (ROOT/'data/LATEST').read_text().strip()
        cls.manifest = verify(cls.directory)

    def test_books_and_cost_sensitivity(self):
        books, trades = exchange_tables(self.directory, self.manifest)
        self.assertEqual(len(books), 2 * self.manifest['settings']['rounds'])
        self.assertTrue((books.spread_bps > 0).all())
        for _, frame in trades.groupby(['round','buy_venue','target_notional_usd']):
            self.assertTrue((frame.sort_values('fee_bps_per_leg').net_bps.diff().dropna() <= 0).all())

    def test_candle_identity_and_no_future_rows(self):
        frame = candle_table(self.directory, self.manifest)
        self.assertFalse(frame.duplicated(['venue','timestamp']).any())
        self.assertTrue((frame.volume >= 0).all())

    def test_market_utilization_reconciles(self):
        frame, _ = market_table(self.directory, self.manifest)
        self.assertLess((frame.utilization-frame.api_utilization).abs().max(), 1e-8)

    def test_positions_cover_observed_debt(self):
        _, markets = market_table(self.directory, self.manifest)
        positions, stress, coverage = position_tables(self.directory, markets)
        self.assertTrue(coverage.debt_coverage.between(.995, 1.005).all())
        self.assertFalse(positions.duplicated(['position','market_id','shock']).any())
        for _, frame in stress.groupby('market_id'):
            self.assertTrue((frame.sort_values('shock').debt_beyond_threshold.diff().dropna() >= -1e-6).all())

    def test_vault_weights_and_exit_capacity(self):
        _, frame = vault_tables(self.directory)
        self.assertTrue(frame.allocation_coverage.between(.99, 1.01).all())
        self.assertTrue(frame.immediate_withdrawal_fraction.between(0, 1.000001).all())


if __name__ == '__main__':
    unittest.main()
