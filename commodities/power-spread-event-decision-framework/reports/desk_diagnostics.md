# From a forecast to a risk discussion

## Scope and information discipline

This extension examines the existing frozen 2025 forecasts. It does not retrain models, change the 2024-calibrated selection thresholds, introduce weather observations, or create an executable strategy. All breakdowns are **post-hoc descriptive diagnostics**, not independent evidence for a newly selected rule. They are designed to answer practical research questions: when does the model struggle, what adverse outcomes can accompany an apparently useful signal, and how sensitive is the conclusion to a cost assumption?

The source is `reports/tables/fundamental_extension/predictions.csv`: three models on the same 8,760 delivery hours. Delivery hours and seasons use America/Los_Angeles; repeated autumn DST hours retain distinct UTC timestamps. Winter groups January, February and December 2025 (not one contiguous winter). Summer is June-August; spring March-May; autumn September-November.

## Findings

The price-only RF retains its 5.3% MAE improvement over the hourly baseline. It selects 846 hours (9.66%) using the existing threshold, leaving 7,914 hours unselected. This selection is a research filter, not a validated trade/abstain policy.

| Price-only RF diagnostic | Result |
|---|---:|
| Mean selected signed spread | 4.08 USD/MWh |
| Worst selected signed spread | -179.98 USD/MWh |
| Mean of worst 5% of selected outcomes (43 observations) | -41.96 USD/MWh |
| Summer selected mean | 8.77 USD/MWh (112 hours) |
| Autumn selected mean | 0.99 USD/MWh (227 hours) |

The autumn mean falls just below zero after an illustrative 1 USD/MWh hurdle. This is not a reason to remove autumn from a backtest: we have already inspected 2025. A season exclusion would be a new hypothesis requiring untouched data. Seasons and hours describe calendar dependence; they do not identify causal weather, congestion or outage effects.

The lower-five-percent mean averages the worst ceil(0.05*n) selected signed spreads. It is an empirical tail description, not portfolio expected shortfall, because there are no positions, executable prices or capital assumptions.

## Cost sensitivity and comparison discipline

The report exports hypothetical hurdles of 0, 1, 2, 5 and 10 USD/MWh, subtracting the same hurdle from every selected observation. RF price's overall mean becomes 3.08 at 1 USD/MWh and -0.92 at 5 USD/MWh. These are scenario assumptions, not measured fees or slippage. Different models select different hours, so selected-hour means do not compare equal exposures. The baseline has a higher selected mean than RF: superior forecast MAE does not automatically imply superior economic decisions.

## What a trader would investigate next

- Inspect the exported worst selected hours alongside energy, congestion and loss components. Do not infer a weather or outage cause without timestamped evidence.
- Before acting on any forecast, establish the contract, executable quote, decision timestamp, available depth, entry/exit rules and position limits. The already-fixed day-ahead price is a reference, not an assumed available entry price.
- Treat missing/stale inputs, unexplained price jumps and unavailable quotes as reasons to pause and investigate. These are proposed operational controls, not controls tested in this dataset.
- Collect publication-vintage net-load/renewable forecast revisions and outage information for a genuinely new experiment. Hourly aggregation cannot substantiate five-minute trading or nodal execution expertise.
- Freeze any next experiment before looking at its evaluation outcomes. Do not optimise hour/season exclusions on these diagnostic tables.

## Reproduce and inspect

From the project directory:

```bash
PYTHONPATH=src python scripts/run_desk_diagnostics.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

Outputs: [overall](tables/desk_diagnostics/overall.csv), [delivery-hour breakdown](tables/desk_diagnostics/by_delivery_hour.csv), [season breakdown](tables/desk_diagnostics/by_season.csv), [hurdle sensitivity](tables/desk_diagnostics/cost_sensitivity.csv), [worst selected hours](tables/desk_diagnostics/worst_selected_hours.csv), and [input hash manifest](tables/desk_diagnostics/manifest.json).

Validation rejects duplicate model-hours, non-finite inputs, altered selection flags and direction mismatches. Unit tests cover signed-spread arithmetic, fixed hurdle subtraction, DST repeated hours, empty selections and malformed data. Existing temporal-integrity tests remain in place. The original [information-clock limitations](fundamental_extension.md) continue to apply.
