# Explaining the project in an interview

## A 60-second explanation

"The question was whether past power-price behaviour could help identify delivery hours when real-time prices would exceed day-ahead prices. I used one year of CAISO hourly data and compared an hourly baseline, Ridge and Random Forest. The original ranking looked encouraging, but a more conservative evaluation fixed the selection threshold before each test window and delayed price features. Random Forest's selected hours had a 55% positive-spread rate, but their mean spread was only about 68 cents per MWh, before any costs. A one-dollar cost scenario erased that mean, and uncertainty included zero. My conclusion was not to trade the rule: forecast revisions and executable prices would be needed to test a stronger hypothesis."

Use this only after reviewing the implementation yourself and being able to reproduce and explain it. The September 2026 extension was developed with coding-assistant support; it is research code, not professional trading experience.

## Questions to be ready for

1. **Why is the DA price not simply a forecast?** It is an auction outcome shaped by bids, constraints and expectations. RT reflects its own market clearing. "Expectation versus correction" is intuition, not an identity about forecasting errors.
2. **What does 55% mean?** Of 189 selected evaluation hours, 55% had RT above DA. It is not the model's all-hour directional accuracy, a win rate on executed trades, or an annual return.
3. **Why can 55% still fail?** Gain/loss magnitudes, costs, and serial concentration matter. A hit rate does not establish positive expected net value.
4. **Why delay features 48 hours?** To test sensitivity to stale information and avoid using recent realized values casually. This is an explicit assumption, not a verified market publication schedule.
5. **Why use a separate calibration window?** It fixes the threshold before test observations. Using the entire test prediction distribution would select hours retrospectively.
6. **Is there still research bias?** Yes. This is one node and year, viewed during project development. The audit is chronological, but it is not a new, untouched external dataset. New years and nodes are needed.
7. **Why not choose the best November strategy?** That would use outcomes to choose a rule. November's results are a diagnostic of instability, not a reason to optimize on that month.
8. **What is the weather hypothesis?** Forecast revisions alter expected net load; supply flexibility and constraints determine the price effect. Weather itself is not in the dataset, so no meteorological cause is established.
9. **Does an evening ramp mean buy?** No. It can be anticipated and already priced in. Several summer evening hours have negative average RT-minus-DA spreads in this dataset.
10. **How would you trade in Europe?** First choose a specific product, forecast timestamp, tradable quote and exit/settlement rule. Obtain forecast vintages, liquidity and costs. Test in a paper environment before proposing deployment.
11. **Why this project for Northpool?** It demonstrates numerical analysis, uncertainty, market curiosity and accepting evidence against a trade. It does not imply prior European intraday execution.
12. **What would you add first?** Timestamped net-load forecast revisions and a precisely specified decision clock. Another complex ML model is lower priority.

## Reading sequence

Read `README.md`, `reports/validation_audit.md`, `src/validation_audit.py`, the tests, and then inspect individual rows of saved predictions. Recalculate mean selected spread, the cost hurdle and a few threshold decisions by hand. Read the linked CAISO and Modo methodology pages before referring to them in an interview.
