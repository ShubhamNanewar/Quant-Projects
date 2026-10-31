import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from desk_risk.metrics import (book_metrics, vwap, round_trip, debt_from_shares,
                              position_health, shocked_price, withdrawal_capacity)
from desk_risk.analyze import verify, parse_book
from desk_risk.collect import Capture


class BookTests(unittest.TestCase):
    def test_spread_and_depth(self):
        result = book_metrics([(99.99, 2), (99.9, 3)], [(100.01, 1), (100.1, 2)])
        self.assertAlmostEqual(result["spread_bps"], 2)
        self.assertAlmostEqual(result["bid_depth_10bps_usd"], 499.68)

    def test_partial_last_level(self):
        self.assertAlmostEqual(vwap([(100, 1), (102, 2)], 2), 101)

    def test_insufficient_depth_is_not_a_fill(self):
        self.assertIsNone(vwap([(100, 1)], 2))

    def test_fees_can_remove_positive_gross_edge(self):
        result = round_trip([(100, 5)], [(100.1, 5)], 2, 10, 10)
        self.assertGreater(result["gross_bps"], 0)
        self.assertLess(result["net_usd"], 0)

    def test_crossed_book_rejected(self):
        with self.assertRaises(ValueError):
            book_metrics([(101, 1)], [(100, 1)])

    def test_unsorted_book_rejected(self):
        with self.assertRaises(ValueError):
            book_metrics([(98, 1), (99, 1)], [(100, 1)])

    def test_nonfinite_price_rejected(self):
        with self.assertRaises(ValueError):
            book_metrics([(float('nan'), 1)], [(100, 1)])

    def test_zero_order_rejected(self):
        with self.assertRaises(ValueError):
            vwap([(100, 1)], 0)

    def test_kraken_errors_rejected(self):
        with self.assertRaises(ValueError):
            parse_book({"error": ["Unavailable"]}, "kraken")


class LendingTests(unittest.TestCase):
    def test_virtual_shares_round_up(self):
        self.assertEqual(debt_from_shares(1000001, 10, 1000000), 6)

    def test_no_debt(self):
        self.assertEqual(debt_from_shares(0, 100, 1), 0)
        self.assertFalse(position_health(0, 10**36, 0, 8*10**17)["liquidatable"])

    def test_decimal_normalisation(self):
        # Two BTC (8 decimals), 100,000 USDC/BTC, and a six-decimal loan token.
        result = position_health(2*10**8, 10**39, 150000*10**6, 86*10**16)
        self.assertEqual(result["collateral_value_raw"], 200000*10**6)
        self.assertAlmostEqual(result["health_factor"], 172000/150000)

    def test_exact_boundary_is_healthy(self):
        self.assertFalse(position_health(100, 10**36, 80, 8*10**17)["liquidatable"])
        self.assertTrue(position_health(100, 10**36, 81, 8*10**17)["liquidatable"])

    def test_shock_uses_full_integer_precision(self):
        price = 10**48 + 123456789
        self.assertEqual(shocked_price(price, 0), price)

    def test_more_severe_shock_never_improves_health(self):
        h = [position_health(100, shocked_price(10**36, s), 60, 8*10**17)["health_factor"]
             for s in [0, .1, .2, .3]]
        self.assertEqual(h, sorted(h, reverse=True))

    def test_withdrawal_capped_by_allocation_and_liquidity(self):
        self.assertEqual(withdrawal_capacity(30, 100, 90), 10)
        self.assertEqual(withdrawal_capacity(5, 100, 90), 5)
        self.assertEqual(withdrawal_capacity(5, 100, 101), 0)

    def test_invalid_shock(self):
        with self.assertRaises(ValueError):
            shocked_price(100, 1)


class CaptureTests(unittest.TestCase):
    def test_graphql_error_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Capture(Path(temp)/"run")
            response = Mock(status_code=200, headers={}, url="https://example.test")
            response.content = b'{"errors":[{"message":"bad query"}]}'
            response.json.return_value = json.loads(response.content)
            with patch("desk_risk.collect.requests.request", return_value=response):
                with self.assertRaises(ValueError):
                    capture.fetch("bad", response.url)
            self.assertTrue((capture.directory/"bad.json").exists())

    def test_corrupt_archive_fails_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"data.json").write_text('{}')
            (root/"manifest.json").write_text(json.dumps({"requests":[
                {"file":"data.json","sha256":"incorrect"}]}))
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == "__main__":
    unittest.main()
