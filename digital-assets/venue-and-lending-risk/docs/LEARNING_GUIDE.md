# Working through the project

This project is intended to be understood and explained, not just listed on a CV. Rebuild the outputs, inspect the source and change an assumption before describing it in an interview.

## Start with the data

Open one book from each venue. Identify the price, quantity and timestamp fields. Compare the normalised table with those raw values. Explain why Coinbase's third level entry and Kraken's third level entry mean different things. Check the manifest and deliberately change a copy of a response to see the checksum test fail.

## Follow one proposed trade

Choose one observation and a $10,000 target. Calculate the ETH quantity from the buyer midpoint. Walk through asks and bids for exactly that quantity, then apply fees to each leg. Explain why a positive midpoint difference does not establish profit and why a transfer between exchanges is not part of the assumed paired execution.

## Follow one borrower

Select a cbBTC/USDC position. Note the collateral decimals, loan decimals, borrow shares and oracle scale. Reproduce the integer debt conversion and supported debt. Apply a 10% price shock. Distinguish a position becoming eligible for liquidation from a lender actually losing money.

## Read the Solidity beside the Python

Trace `_isHealthy`, `toAssetsUp` and `liquidate`. Explain why debt is rounded up and collateral capacity down. Find the exact equality rule. Then read the residual bad-debt adjustment and explain why this project stops at threshold exposure.

## Compare two vaults

Compare the largest collateral weight and immediate withdrawal estimate. Explain why a fully collateralised borrower book can coexist with limited lender withdrawal liquidity. Identify a shared market between vaults and explain why their withdrawal estimates cannot simply be added.

## Questions to answer before using this on a CV

- What did the API actually return, and what did you calculate yourself?
- Which inputs are raw integers and which are display values?
- Which timestamps are known, and which are missing?
- Why does the fee sensitivity use scenarios rather than claim a particular fee tier?
- Why is the principal-token stress not comparable in probability to a BTC shock?
- What would be required to turn the research collector into a reliable desk service?

A fair description is a public-data research integration with tested risk calculations. It is not professional exchange connectivity experience, a Solidity audit, live vault curation or deployed trading infrastructure.
