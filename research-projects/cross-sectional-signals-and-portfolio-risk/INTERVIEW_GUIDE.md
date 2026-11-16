# Explaining the research

## A 60-second explanation

I studied whether a regularised combination of return-history features improves on simple industry momentum. I used 48 industry return series and separated model selection from a later 15-year evaluation. I compared both signals with simple rank weights and with identical covariance, beta and turnover controls. The Ridge model did not improve on momentum. The interesting result was that portfolio construction mattered, but it did not rescue a weaker signal. I also implemented a lagged, drift-aware backtest and documented why aggregate research returns are not equivalent to a tradable strategy.

## Questions to be able to answer yourself

1. **Why industries?** A stable, public, reproducible research cross-section avoids building a universe from today's surviving stocks. It sacrifices security-level detail and execution realism. Forty-eight assets do not establish enterprise scale.
2. **Why Ridge?** A transparent regularised linear benchmark tests whether combining correlated features adds information. Model complexity should earn its place against a simple baseline. Alpha is normalised by training rows so the penalty retains the same interpretation as data grow.
3. **Why a monthly target?** It separates short-run return noise from a slower cross-sectional signal and reduces hypothetical turnover. It says nothing about high-frequency alpha.
4. **Why rank features?** They limit sensitivity to extreme feature magnitudes without fitting a transform to future observations. They also discard magnitude information, which can be a disadvantage.
5. **Why embargo the previous month?** Its full return is not known at the feature cutoff. Including it as a training target would leak future information even if input features were lagged.
6. **Why not trade at the same close used for features?** An end-of-day observation cannot generally be used to guarantee an order at that same closing price. A full-session delay makes the assumed decision sequence explicit, but actual aggregate publication latency is still not established.
7. **Why optimise?** Equal dollar long/short weights can carry substantial common risk. Covariance and estimated beta constraints target that risk. They do not create information in the signal.
8. **Why Ledoit-Wolf?** Shrinkage stabilises a sample covariance estimate. It does not eliminate estimation error, especially in changing regimes.
9. **What is turnover?** Absolute traded weight relative to NAV, comparing target holdings to drifted positions. A 100% gross opening trade has turnover 1 and costs 0.1% at 10bp.
10. **Why do you track holdings?** Fixed monthly asset quantities imply weights drift with asset returns and portfolio NAV. Multiplying fixed weights by daily returns would implicitly rebalance for free.
11. **What happened?** Ridge had lower mean rank IC and higher turnover. With equal optimiser settings, its net Sharpe was about 0.30 versus momentum's 0.60. The measured difference favoured momentum; it is not evidence that ML generally fails.
12. **Are these alpha returns?** No. They are historical research-proxy P&Ls with illustrative costs and a zero cash hurdle. There is no licensed point-in-time security universe, execution or capacity analysis.
13. **Why can you not advertise beta-neutral?** Neutrality is to a fitted equal-industry proxy at rebalance, not all equity factors, the real market, or every day of the holding period.
14. **What would you improve?** Use genuinely point-in-time tradable instruments, real spreads/borrow, stronger risk-factor controls, and an unseen prospective evaluation. Do not retrospectively tune the same test period into a success.
15. **What did AI do?** AI assisted with implementation, documentation and checks. Reproduce the results, review each assumption and be explicit about that assistance. Do not claim expertise in code or mathematics you cannot explain.

## Suggested CV wording — after reviewing and reproducing

**Cross-Sectional Signals and Risk-Constrained Portfolio Construction | Python Project**

- Built an expanding-window research pipeline across 48 industry return portfolios, comparing Ridge forecasts with momentum over 180 evaluation months using lagged inputs and completed training labels.
- Implemented covariance-aware portfolio optimisation, drift-adjusted turnover, transaction/short-cost stresses and block-bootstrap comparisons; documented the model's underperformance and data/execution limitations.

Avoid: “Developed profitable Citadel strategy”, “generated alpha”, “institutional-grade execution”, “live beta-neutral trading”, or “large-scale multi-asset optimisation”. None is established by this work.
