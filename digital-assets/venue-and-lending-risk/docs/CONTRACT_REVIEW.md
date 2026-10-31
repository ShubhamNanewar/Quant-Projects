# Morpho contract reading notes

## Scope

This is a reading exercise tied to the risk calculations in this repository. It is not a security audit. The source is pinned to the official `morpho-org/morpho-blue` repository at revision `9cf6d5b844625ba108e36d1cf911f5d2faf7c136`. A source revision is not evidence that every sampled deployment has matching bytecode. That verification remains open.

Third-party source is archived without changes under `data/raw/20261031T103525Z/`. Its GPL-2.0-or-later notices and the upstream licence are preserved. The research code does not deploy or call state-changing contracts.

## What was traced

| Source location | Behaviour | Consequence for this project |
|---|---|---|
| `Morpho.sol`, `borrow`, lines 235-259 | Borrowing checks position health and available market liquidity | Collateral coverage and withdrawal liquidity are separate constraints |
| `Morpho.sol`, `withdraw`, lines 200-223 | Withdrawing supplied assets cannot leave borrowing above supply | Total supplied assets cannot be treated as immediately available cash |
| `Morpho.sol`, `withdrawCollateral`, lines 323-337 | Removing collateral requires the remaining position to be healthy | A static price shock differs from a borrower actively managing collateral |
| `Morpho.sol`, `liquidate`, lines 347-407 | Reads oracle price, checks that the position is unhealthy, calculates repayment and seized collateral; residual bad debt can reduce supply assets | Threshold exposure is not a loss estimate; liquidation and bad-debt mechanics would require a separate simulation |
| `Morpho.sol`, `_accrueInterest`, line 483 onward | Updates loan balances over time using the interest-rate model | Indexer snapshots at different times cannot be treated as an atomic live balance sheet |
| `Morpho.sol`, `_isHealthy`, lines 527-538 | Rounds borrow assets up, collateral value down and LLTV-adjusted capacity down; compares maximum debt with debt | Exact integer arithmetic and an equality-boundary test are necessary |
| `SharesMathLib.sol`, lines 20-43 | Defines virtual assets and shares and directional conversions | Direct division of raw shares is insufficient |
| `ConstantsLib.sol` | Defines the oracle price scale | Token decimals and oracle scaling must be treated separately |

[Read the pinned source](https://github.com/morpho-org/morpho-blue/blob/9cf6d5b844625ba108e36d1cf911f5d2faf7c136/src/Morpho.sol).

## Questions still open before capital allocation

1. Do chain ID, market ID and deployed bytecode match the reviewed configuration?
2. Which oracle feeds and conversion vaults does each market rely on? Who can change their implementations or parameters?
3. What timestamps, heartbeat limits and deviation thresholds apply to those feeds? How are incomplete or invalid rounds handled?
4. Can liquidators actually sell the collateral at the sizes implied by a stress event? Are there redemption gates, maturity constraints or issuer dependencies?
5. Are vault caps, curator permissions, guardian rights, timelocks and withdrawal queues consistent with the proposed allocation mandate?
6. How much of the quoted rate comes from sustainable borrower demand, and how much depends on rewards, leverage or temporary incentives?

The current project answers parts of the second and fifth questions through public configuration data. It does not claim to have answered all six.
