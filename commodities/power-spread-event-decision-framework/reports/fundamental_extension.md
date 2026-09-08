# Does adding demand data improve the forecast?

## Conclusion

**A measurable price-based forecasting improvement survives a separate year; added demand forecasts do not improve it further.** The price-only Random Forest reduces 2025 mean absolute error by **5.3%** versus an hourly baseline (7.955 versus 8.403 USD/MWh). Adding DAM system and regional demand forecasts produces MAE 8.015, slightly worse than the price-only model. The predefined primary criterion for incremental demand value is therefore **not met**.

The signed-spread selection diagnostics are positive in this experiment, including after an illustrative $1/MWh hurdle. That is evidence of a relationship in archived prices, **not executable profit**: the DA price was already fixed when the research forecast would be made, and cannot be assumed available as an entry price.

## What changed, and why

The 2024 audit remains in the repository. It asked whether stale price histories alone could select positive RT-minus-DA spreads. This extension asks a different, specified question: after the DA auction, do known DA prices and archived DAM demand forecasts help explain the direction and magnitude of later corrections?

The [experiment specification](../docs/fundamental_extension_protocol.md) and code were committed in `ef70afb` before the complete 2025 evaluation was run. We kept the same NP15 node, fixed the algorithms and thresholds using 2024, and reported all of 2025. There was no search over alternative nodes, favorable months or test-year thresholds. This is a retrospective external-year experiment, not a preregistered live trial.

## Data and information clock

- **Prices:** existing 8,784 CAISO NP15 hourly observations for 2024, plus 8,760 newly downloaded 2025 hours. Total: **17,544 hours**.
- **New fundamental inputs:** OASIS `SLD_FCST`, `DAM`, `SYS_FCST_DA_MW`, for `CA ISO-TAC` and `PGE-TAC`. There are 17,544 complete, unique delivery-hour forecasts for each area.
- **Real-time aggregation:** the 2025 panel has 12 distinct five-minute price observations per hour, averaged to hourly prices. This does not assert instrument-specific settlement equivalence.
- **Forecast context:** after the DA auction, one hour before delivery. Historical price features remain delayed 48 hours; current-delivery DA price is included as a known reference.
- **Availability limitation:** the archive contains delivery timestamps and market runs, not full publication/revision vintages. Availability is an explicit assumption. Revised historical inputs may differ from what was actually observable.

Added features are system load forecast, PGE-area load forecast, PGE/system share and adjacent-delivery-hour forecast changes. **They are not changes between successive forecast issues, nor actual weather data.** System/regional load forecasts do not measure nodal congestion, renewable uncertainty or supply outages.

## Fixed design

| Phase | Pacific delivery dates | Complete hours |
|---|---|---:|
| Training | Usable January history through 28 September 2024 | 6,312 |
| Calibration | 1 October through 29 December 2024 | 2,161 |
| Evaluation | Full calendar year 2025 | 8,760 |

There are 48-hour gaps before calibration and before evaluation. Initial feature warm-up removes 215 hours; no observations were excluded for missing demand features. DST explains the extra autumn calibration hour.

Models: hourly training-mean baseline; the existing Random Forest settings with delayed price features and DA price; the identical RF with demand features added. No test-year refitting. Each model selects positive or negative forecasts above the 90th percentile of its **absolute calibration predictions**, with signs carried into the signed-spread diagnostic. All three models use the same evaluation hours.

## Results

| Model | MAE | RMSE | Directional accuracy | Selected hours | Positive signed-spread rate | Mean signed spread | Less hypothetical $1/MWh |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hourly baseline | 8.403 | 17.341 | 59.61% | 1,095 | 76.62% | 4.299 | 3.299 |
| RF price | 7.955 | 17.117 | 59.35% | 846 | 72.81% | 4.077 | 3.077 |
| RF price + demand | 8.015 | 17.232 | 59.62% | 910 | 70.44% | 4.012 | 3.012 |

Errors and signed-spread values are USD/MWh. Selected-hour positive rates are not all-hour directional accuracy, executed-trade win rates or returns on capital. The different models select different sets of hours; their selected means are not an equal-exposure performance comparison.

![Frozen 2025 forecast evaluation](figures/fundamental_extension/2025_comparison.png)

Paired daily-bootstrap estimates for the **incremental demand model's MAE improvement**:

- Versus price-only RF: **-0.060 USD/MWh**, approximate 95% interval **[-0.167, 0.040]**. No demonstrated incremental improvement.
- Versus hourly baseline: **+0.389 USD/MWh**, interval **[0.252, 0.514]**. The combined model beats the baseline, but this cannot be credited to demand data because the price-only RF does better.

Intervals condition on the fitted models and assume independent UTC days. They do not account for longer dependence, data revisions or all researcher choices.

## What signal is present?

The selection is overwhelmingly **negative RT-minus-DA**: the hourly rule selects 1,095 negative-direction hours and no positive-direction hours; RF price selects 831 negative and 15 positive. Thus the observed pattern is largely hours where RT tends to settle below the already-known DA reference, not a broad discovery of profitable long and short opportunities.

Price RF has positive selected means after the illustrative hurdle in 11 of 12 months, but September is **-$3.65/MWh**. The worst selected signed spread is **-$179.98/MWh** for all three models. Averages do not eliminate tail risk. The hourly benchmark's strong selected-hour diagnostic also warns against attributing a calendar effect to ML sophistication.

## Research decision

1. Keep the price-based RF as a forecast benchmark: it improves magnitude error against the hourly baseline in a separate year.
2. Do not add the tested demand features to the preferred model merely because they sound more fundamental. They did not improve the specified metric over price-only RF.
3. Do not place trades from these results. To test an actual strategy, define an executable contract, decision timestamp, entry quote, exit/settlement and position limits; acquire publication vintages, quotes, depth and measured costs.
4. For further fundamental research, test timestamped net-load **forecast revisions**, renewables, outages and cross-border constraints. Freeze another protocol and reserve genuinely new data. Demand forecast levels alone are insufficient evidence of weather-driven edge.

For Northpool, this demonstrates hypothesis testing, numerical comparison and risk judgment. It does not establish European intraday execution experience. The [earlier audit](validation_audit.md) and this experiment use different information sets, selection directions and dates; their headline statistics must not be compared as if only data quality changed.

## Reproduce

```bash
pip install -r requirements-audit.txt
PYTHONPATH=src python scripts/fetch_fundamental_extension.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_fundamental_extension.py
```

Downloads are restartable and rate-limited. Raw chunks are excluded from Git. Saved [results](tables/fundamental_extension/), coverage and hashes in [manifest.json](tables/fundamental_extension/manifest.json) identify the evaluated data. Test-target perturbation confirms 2025 labels cannot affect frozen predictions or thresholds; missing forecasts are never backfilled.

Sources: [CAISO OASIS](https://oasis.caiso.com/), [day-ahead market overview](https://www.caiso.com/Documents/Presentation-Existing-Day-Ahead-Market-Overview.pdf), [demand-forecast report specification](https://www.caiso.com/documents/oasis-interfacespecification_v5_1_2clean_fall2017release.pdf). The separate [weather discussion](validation_audit.md#weather-and-fundamentals-hypotheses-not-fitted-results) links CAISO and Modo Energy background reading; no weather effects are claimed to have been measured here.
