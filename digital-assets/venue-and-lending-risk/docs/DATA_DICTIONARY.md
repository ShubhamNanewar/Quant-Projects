# Data dictionary and collection record

## Scope and selection

The reference collection is `20261031T103525Z`, recorded on 31 October 2026 in UTC. Every original response is retained under `data/raw/`. The manifest records the complete URL, HTTP method, GraphQL body where relevant, response time, HTTP date, cache age, byte count and SHA-256 checksum. No credentials or private account data are used.

Exchange selection is fixed in advance: Coinbase Exchange and Kraken, ETH against USD on both venues. ETH/USDT is deliberately excluded because a different quote asset would introduce conversion and stablecoin risk. These two mature venues provide a useful integration baseline; they are not a sample of early-stage exchanges.

Morpho selection is the 30 largest listed non-idle Ethereum markets ranked by supplied USD at retrieval. This favours established, large markets and should not be extrapolated to the long tail of protocols. Borrower analysis covers the first three USDC loan markets in that ranking. Vault analysis covers the five largest listed Ethereum USDC V1 vaults. V2 vaults and other chains are outside this release.

## Raw responses

| File pattern | Source | Shape and interpretation |
|---|---|---|
| `coinbase_product.json` | Coinbase public product endpoint | Instrument status, increments, base and quote identifiers |
| `kraken_product.json` | Kraken AssetPairs | Instrument naming, precision and trading parameters |
| `coinbase_book_*.json` | Coinbase level-2 book | Price and ETH quantity are decimal strings; third entry is aggregated order count; top-level sequence and exchange timestamp retained |
| `kraken_book_*.json` | Kraken Depth, 100 levels | Price, ETH quantity and per-level update time; a level's last change is not the timestamp of the entire book |
| `coinbase_hourly.json` | Coinbase candles, 3,600 seconds | Arrays ordered as time, low, high, open, close, volume |
| `kraken_hourly.json` | Kraken OHLC, 60 minutes | Time, open, high, low, close, VWAP, volume, trade count; final unfinished candle removed |
| `morpho_markets.json` | Morpho GraphQL | Market configuration, token addresses/decimals, oracle, reported state and warnings |
| `morpho_positions_<market>_<page>.json` | Morpho GraphQL | Borrower collateral, borrow shares, reported debt and health factor; pages of 100, maximum 20 per market |
| `morpho_vaults.json` | Morpho GraphQL | Vault balances, timelock, allocation amounts, supply caps, withdrawal-queue membership |
| `morpho_oracles_end.json` | Morpho GraphQL | Second observation, feed addresses, feed decimals and vault conversion dependencies |
| `contract_revision.json` and `*.sol.txt` | Official Morpho GitHub repository | Pinned contract source for checking the calculation, not proof of deployed bytecode equivalence |
| `contract_LICENSE.txt` | Official Morpho repository | Licence for the archived third-party Solidity source |

Public borrower addresses appear in the provider's raw position identifiers. The derived position table uses abbreviated hashes as presentation labels. No attempt is made to identify the owners.

## Normalised tables

| Table | Important fields and types | Units and caveats |
|---|---|---|
| `books.csv` | round: integer; venue: string; received_at: UTC ISO string; bid/ask/mid: float64; elapsed_ms: float64 | USD per ETH; spread in basis points; depth in USD; native timestamp missing for Kraken is unknown, not zero age |
| `opportunities.csv` | target_notional_usd, quantity_eth, buy_vwap, sell_vwap, gross_bps, net_bps, net_usd; gate booleans | Same ETH quantity on both legs; missing price means insufficient displayed depth; fees are scenarios, not verified account tiers |
| `candles.csv` | timestamp: epoch seconds; utc: timezone-aware string; OHLC, volume, log_return | USD prices, ETH volume; gaps invalidate the adjacent hourly return; no forward filling |
| `markets.csv` | market_id, token identifiers, lltv, utilization, supply_usd, borrow_usd, APYs, state_timestamp | Ratios and APYs are decimal fractions; USD valuations are provider estimates; price integer is scaled, not a human-readable price |
| `positions.csv` | hashed position, market_id, debt_loan_units, health_factor, shock, liquidatable | Debt in USDC token units, not assumed USD; one row per position and shock; mixed state timestamps retained |
| `borrower_stress.csv` | sampled debt, debt beyond threshold, breached position count | Threshold exposure, not loss, liquidation volume or a forecast |
| `position_coverage.csv` | positions, sampled_debt, market_borrow, debt_coverage | Completeness diagnostic against the same saved market state, not an atomic on-chain reconciliation |
| `vault_allocations.csv` | allocation_loan_units, weight, collateral_address, withdraw_queue | Weights use total vault assets; assets may share an issuer or economic risk despite different addresses |
| `vaults.csv` | largest_collateral_weight, collateral_hhi, immediate_withdrawal_fraction | Static estimates; competing withdrawals, reallocation, idle cash and execution ordering are not simulated |
| `oracles.csv` | two raw prices, normalised quote, feed addresses, conversion vaults, price change | Two observations do not establish update reliability; underlying feed freshness is explicitly unknown |
| `alerts.csv` | entity, rule, severity, value, threshold, action | Research review thresholds; no automated trading or capital movement |

## Integer precision

JSON BigInt fields may arrive as strings or integers. They are converted directly to Python integers. Borrow shares, collateral and oracle prices are never converted to floating point before health checks. The oracle scale is `10^36`; LLTV uses `10^18`. Loan and collateral token decimals are separate quantities. A USDC loan amount uses six decimals, while BTC wrappers usually use eight and ETH derivatives commonly use eighteen.

CSV files are presentation and analysis outputs. For exact recomputation of large integers, use the raw JSON. Spreadsheet software can silently round these values.

## Availability and timing

The book pair is requested concurrently, but the responses are not simultaneous market observations. Retrieval latency, native Coinbase quote age, HTTP cache age and receipt skew are stored separately. A small receipt skew does not establish that the two matching engines were observed at the same instant.

The Morpho API is an indexer. Market and borrower responses do not necessarily describe the same block. Market-state timestamps are not Chainlink heartbeat timestamps. Unchanged oracle prices can be legitimate between updates. None of these observations alone proves feed failure.

The historical candle sample is whatever the public endpoints returned at collection. The archive contains 349 completed Coinbase bars and 720 completed Kraken bars. These endpoints are not used to imply a long historical backtest. Both current and missing candles are handled explicitly.
