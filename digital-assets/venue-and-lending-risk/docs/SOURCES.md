# Sources

All project observations were retrieved on 31 October 2026. Complete request URLs and GraphQL bodies are included in the raw-data manifest.

## Exchange data

- [Coinbase Exchange order book](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book): level-2 aggregation and response fields.
- [Coinbase Exchange candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles): bar format and granularity.
- [Kraken order book](https://docs.kraken.com/api-reference/market-data/get-order-book): depth response and public API error format.
- [Kraken OHLC](https://docs.kraken.com/api-reference/market-data/get-ohlc-data): recent history and the unfinished final candle.
- [Kraken tradable pairs](https://docs.kraken.com/api-reference/market-data/get-tradable-asset-pairs): exchange identifiers and order precision.

## Morpho

- [API introduction](https://docs.morpho.org/developers/api/get-started/): public data access.
- [Morpho API announcement](https://morpho.org/blog/the-morpho-api-skip-indexing-start-building/): market, vault and position coverage.
- [Public GraphQL endpoint](https://api.morpho.org/graphql): the saved queries are the exact schema used by this collection. Older examples using `uniqueKey` and `oracleAddress` as Market output fields did not work; the current response uses `marketId` and nested `oracle`.
- [Collateral, LTV and health](https://docs.morpho.org/developers/borrow/concepts/ltv/): scaling and the financial interpretation of health.
- [Market data guidance](https://docs.morpho.org/developers/borrow/tutorials/get-data/): interest accrual, balances and indexed data limitations.
- [Pinned Morpho source](https://github.com/morpho-org/morpho-blue/tree/9cf6d5b844625ba108e36d1cf911f5d2faf7c136): exact arithmetic and boundary condition used here.

Some prose descriptions simplify the liquidation boundary as health factor at or below one. This implementation follows the pinned contract's integer comparison, which treats exact equality as healthy. It does not rely on that prose simplification.

## Reuse

The response archives are a small research snapshot from public endpoints, not a licensed commercial feed or an endorsed redistribution product. Source attribution is retained. Anyone extending collection or redistributing substantial datasets should check the providers' current terms. The archived Solidity remains subject to its upstream GPL-2.0-or-later licence, stored beside the source.
