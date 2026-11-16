# Research protocol — frozen before first results

Question: can a regularised combination of return-history features improve on industry momentum, and what changes when risk and turnover enter portfolio construction?

Universe: the first (value-weighted-layout) block of the existing Kenneth French 48-industry daily snapshot. Verify the block boundary and unique dates. Use 1990–2024 only; no cherry-picking industries. The source has already appeared in another project, so this is a retrospective temporal holdout, not an untouched prospective experiment. No current constituent list is used.

Features: 21-day reversal, 63-day momentum, 252-day momentum skipping 21 days, 63-day volatility, 252-day maximum drawdown and 252-day beta to the equal-industry market proxy. Cross-sectional ranks are formed using only that date's 48 observations. Target: next-calendar-month return minus equal-industry return.

Timing: form features and estimate covariance through the penultimate trading day of the previous month. Hypothetically execute at its final close. Include the next month's complete close-to-close return. Use only training labels that have fully ended strictly before the feature date (one monthly observation is therefore embargoed).

1991–2004 initial development; 2005–2009 validation; 2010–2024 final evaluation. Expanding monthly refits. Select Ridge penalty from [0.1, 1, 10, 100] by validation mean monthly Spearman IC only. No final-period selection.

Four fixed variants: momentum rank weights; Ridge rank weights; momentum with covariance/turnover optimisation; Ridge with the same optimisation. Optimisation: monthly Ledoit-Wolf covariance from 252 daily observations, zero dollar exposure, zero estimated equal-industry-proxy beta, 100% gross cap, 5% individual cap, risk aversion 5, turnover penalty 10bp per unit absolute traded weight. Both optimiser signals use a 1% cross-sectional rank-score scale (a utility calibration, not a return forecast).

Base realised costs: 10bp per unit turnover plus 50bp annual charge on short exposure. Also report 0/5/25/50bp transaction costs, holding borrow constant; a separate no-frictions scenario; alternative 0/100/300bp short charges. Use actual drifted pre-trade weights, charge initial entry, and retain marked-to-market terminal positions (no closing liquidation). No cash yield/rebate or extra financing income.

Report all four variants, annual returns, drawdown, realised risk, rank IC, gross/net exposure, turnover, concentration, and paired 12-month moving-block-bootstrap confidence intervals (2,000 repetitions, seed 42). Cost stress reuses fixed holdings; it is not reoptimised. Daily P&L includes drift, costs and short charges. Sharpe uses zero hurdle and is not a cash-inclusive investor Sharpe.

Limits: research aggregates are not directly tradable; no constituent execution/borrow/capacity data; revised-vintage historical data are not point-in-time archives; estimated beta neutrality is not realised market neutrality. No claim of scalable alpha or live trading readiness. Negative results will be retained.
