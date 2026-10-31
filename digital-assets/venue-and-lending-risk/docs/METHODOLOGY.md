# Methods and assumptions

## 1. Exchange integration and data checks

Each adapter translates the exchange's native response into ordered bid and ask levels. Prices and quantities must be finite and positive. Empty, unsorted, crossed or locked books are rejected. The client detects HTTP errors, provider error fields and GraphQL errors even when the HTTP status is 200. Transient HTTP failures receive bounded retries. Successful and unsuccessful response bodies are kept for inspection.

The current collector stops if a required integration fails. It does not replace missing data with invented values or an old response. `data/LATEST` is updated only after the initial collection completes. Enrichment adds a second observation without overwriting the first.

One basis point is 0.01%. With best bid b and ask a, midpoint m = (a+b)/2 and spread = 10,000(a-b)/m. Displayed depth within 10 bps sums price times quantity for levels inside a midpoint band. The imbalance statistic is (bid depth - ask depth)/(bid depth + ask depth). These quantities describe displayed interest, not guaranteed liquidity.

## 2. Sizing a cross-venue pricing gap

The screen considers purchases on Coinbase followed by sales on Kraken, and the reverse. Both legs use the same ETH quantity. For a target notional N, quantity q = N / buyer midpoint. The calculation walks through the ask book to buy q and the bid book to sell q. If either book cannot fill q, the result is missing and explicitly marked insufficient depth.

For buy VWAP B, sell VWAP S and per-leg proportional fees f_b and f_s:

- Purchase cost = q B (1 + f_b).
- Sale proceeds = q S (1 - f_s).
- Net price difference in bps = 10,000 [S(1-f_s)/(B(1+f_b)) - 1].

The notional grid is $1,000, $10,000, $50,000 and $100,000. Fee scenarios are 0, 5, 10, 25, 40 and 60 bps per leg. This is a sensitivity grid, not a claim about an available fee tier. No maker rebate, gas cost or transfer cost is assumed. The paired trade presumes inventory is already funded on both venues. It does not model borrowing, account permissions, custody limits, transfer time, partial fills, stale quotes or an unfilled hedge leg.

A preliminary timing screen requires receipt skew at most two seconds, request latency at most two seconds, cache age at most two seconds, no Coinbase auction mode, and a Coinbase source age between minus one and two seconds. The negative allowance is for clock error. Kraken lacks a comparable top-level source timestamp in this response, so this screen cannot certify execution freshness. All proposals remain subject to a separate production gate.

## 3. Historical context

Hourly log returns are log(P_t/P_(t-1)). If the preceding bar is not exactly one hour earlier, the return is missing. The chart uses a 24-return rolling sample standard deviation, multiplied by sqrt(24 x 365), to describe annualised realised volatility in a continuously traded market. Annualisation does not turn a short estimate into a forecast. The two endpoints have different history lengths, and the plot shows that difference rather than filling missing observations.

## 4. Morpho market liquidity

Utilisation is total borrowed loan assets divided by supplied loan assets. The result is independently recomputed from raw integer balances and compared with the provider's utilisation. Available loan liquidity is supply minus borrow. High utilisation can make withdrawals difficult even when collateral coverage appears strong. APY is an annualised quoted rate, not a promised holding-period return.

The project records provider warnings and uses 95% utilisation as a review threshold. This threshold is a research choice, not a protocol limit or an estimate fitted to future data.

## 5. Borrower health and liquidation sensitivity

Morpho uses borrow shares. The pinned SharesMathLib converts shares into debt assets with an upward rounding convention:

`debt = ceil(borrowShares * (totalBorrowAssets + 1) / (totalBorrowShares + 1,000,000))`

The virtual quantities are part of the source implementation. Collateral value in loan-token base units is:

`value = floor(collateralRaw * oraclePriceRaw / 10^36)`

Maximum supported debt is:

`maxBorrow = floor(value * LLTVRaw / 10^18)`

The pinned `_isHealthy` implementation treats `maxBorrow >= debt` as healthy. Liquidation eligibility therefore requires debt strictly above maximum supported debt. A floating-point health factor is displayed for interpretation, but the classification uses integer arithmetic. This matters at equality and for small balances.

The stress grid reduces the oracle price by 0%, 5%, 10%, 20%, 30% and 40%, leaving debt and collateral quantity fixed. Shock multiplication also uses an exact integer ratio. No borrower adds collateral, repays debt or is liquidated during the shock. No interest accrues between the scenarios. The result is the amount of sampled debt that would move beyond the threshold under that static assumption.

The same percentage shock is not equally probable or economically comparable for BTC collateral and a principal token. It is a common sensitivity measure. A principal-token position can be highly sensitive to a small price decline while having very different expected price behaviour. This analysis does not assign probabilities, liquidation discounts, bad debt, recovery rates or loss estimates.

Each market is paginated at 100 positions per page, capped at 2,000 positions. The published sample ends with a short final page for every selected market. Debt coverage is reported independently. State timestamps differ across requests, so the result remains an indexed approximation rather than a same-block risk engine.

## 6. Vault curation diagnostics

For each vault, allocation weight is the loan assets supplied to a market divided by the vault's total assets. Positions are grouped by collateral contract address. The largest group weight and the Herfindahl index, sum of squared group weights, describe concentration. Address grouping does not consolidate common issuers, wrappers or correlated economic risks.

For a market in the withdrawal queue, immediately accessible assets are estimated as the smaller of the vault's allocation and the market's unborrowed loan assets. Sum these capacities and divide by vault total assets. The estimate assumes the vault is the next withdrawer and ignores competition, reallocation and idle balances. Results from separate vaults must not be added because they may compete for the same market liquidity. Timelocks and queue membership are reported because a curator may be unable to change allocations instantly.

Review thresholds are a largest collateral weight over 50% and immediate withdrawal capacity below 10%. They are prompts for investigation, not allocation recommendations.

## 7. Oracle dependencies and behaviour

The second observation records oracle configuration, base and quote feed addresses, feed decimals and ERC-4626-style conversion vault dependencies where exposed by the API. Human-readable price is:

`loan units per collateral = raw price / 10^36 * 10^(collateral decimals - loan decimals)`

Two observations allow a price-change calculation and a check that the indexed market-state timestamp advanced. They do not provide feed round timestamps, sequencer status, heartbeat configuration, proxy governance or deviation thresholds. Those checks remain mandatory before using an oracle to commit capital. The report deliberately labels feed freshness as unknown.

## 8. Tests and reproducibility

Unit tests cover fees, partial book consumption, insufficient liquidity, invalid books, API errors, checksum corruption, token decimals, virtual shares, rounding, exact liquidation boundaries and monotonic stress behaviour. Integration tests verify the archived books, utilisation reconciliation, borrower debt coverage and vault allocation reconciliation.

The first precision test exposed rounding when multiplying a very large oracle price with Decimal's default precision. The implementation was changed to integer numerator/denominator arithmetic. The test retains the large-integer example to prevent regression.

Rebuilding the tables and figures uses only saved responses. New collection runs have their own directories. The reference run has no fitted predictive model, training sample, strategy backtest or estimated alpha.
