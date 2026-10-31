# Desk assessment and technical backlog

## Decision

**Proceed with read-only research. Do not approve live execution or a vault allocation on this evidence.**

The exchange adapters retrieve valid ETH/USD books and support transparent sizing calculations. In the published observation window, fees remove the small gross price differences. Three minutes of REST observations do not establish a repeatable opportunity. Both venues still need operational and execution review before any trading decision.

Morpho market, borrower and vault data support a useful first risk screen. Borrower debt and vault allocations reconcile closely to their saved totals. However, indexer timestamps differ, underlying oracle freshness is unknown and the contract reading has not been matched to deployed bytecode. A vault's quoted assets or APY do not resolve those questions.

## Proposal A: cross-venue ETH/USD price difference

- Size evaluated: $1,000 to $100,000 equivalent ETH per paired trade.
- Observation: at most 1.29 bps gross across the sampled directions and sizes.
- Cost scenario: at 10 bps per leg, even the best observation is negative by about 18.69 bps.
- Requirements: pre-funded inventory on each exchange; verified account fee tiers; price and quantity rounding; permissions; position limits; handling of partial fills and a failed second leg.
- Decision: no trade proposal from this sample. Keep the integration for longer observation and execution-quality research.

## Proposal B: Morpho USDC vault monitoring

- Observation: vaults with similar asset labels have materially different collateral concentration and current withdrawal capacity.
- Risk: multiple vaults can depend on the same markets and compete for the same available liquidity. Market health and vault liquidity are not interchangeable.
- Requirements: verified feed freshness, same-block debt and collateral, curator permissions, cap and queue change monitoring, token and issuer due diligence, and an allocation-specific mandate.
- Decision: publish monitoring outputs for review. Do not infer a recommended allocation or size from the highest APY.

## Prioritised work

| Priority | Work item | Proposed owner | Acceptance criteria |
|---|---|---|---|
| P0 | Reconcile Morpho state on a common Ethereum block | Research + DeFi engineering | Recompute debt, oracle quote and liquidation boundary from pinned-block RPC; explain every material difference from the indexer |
| P0 | Check oracle freshness and dependencies | DeFi engineering | Store feed round IDs and updatedAt, documented heartbeat rules, conversion-vault dependencies and failure handling |
| P0 | Validate exchange metadata and order constraints | Connectivity + Trading | Tests for tick and lot rounding, minimum size, product status, rate limits, rejection paths and exact account fee tier |
| P0 | Match contract source to deployments | DeFi engineering + Risk | Record verified bytecode, compiler settings, proxy dependencies and reviewed market configuration |
| P1 | Maintain WebSocket books | Connectivity | Snapshot bootstrap, sequencing, checksum or equivalent reconciliation, gap recovery and deterministic replay tests |
| P1 | Test execution controls in a permitted sandbox | Trading + Connectivity | Partial-fill handling, cancel/replace, limits, kill switch, reconciliation and audit trail; no live capital required |
| P1 | Track vault cap and queue changes | DeFi engineering + Operations | Detect curator or guardian changes and pending configurations; retain before/after records and acknowledgement |
| P1 | Add collateral-specific stress and liquidation costs | Research + Risk | Separate BTC, principal-token and stablecoin scenarios; include sale depth, repayment, incentive and recovery assumptions |
| P2 | Observe multiple sessions | Research | At least several days spanning different liquidity conditions; report outages and failed captures alongside valid data |
| P2 | Add perpetual funding and basis | Research + Connectivity | Reconcile contract multiplier, funding interval, index/mark definitions and hedge inventory costs |
| P2 | Add a small early-stage venue | Connectivity + Trading | Repeat documented integration checks; no unsupported assumption that mature-venue behaviour transfers |

These are proposed responsibilities for a desk workflow, not a claim that named teams participated in this personal project.

## Operational notes

The collector is an on-demand tool. It is not a daemon, does not send messages and does not place orders. Alerts are saved locally as a table. Any future service needs persistent state, deduplication, ownership, escalation and recovery procedures. A passed unit test is not a substitute for those controls.
