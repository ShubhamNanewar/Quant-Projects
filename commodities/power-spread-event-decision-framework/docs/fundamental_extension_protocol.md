# Frozen specification: demand-informed forecast extension

Specified before inspecting full 2025 outcomes. This is one additional experiment, not a search across markets for the best-looking profit.

- Keep the NP15 node. Add public CAISO DAM demand forecasts for CA ISO-TAC and PGE-TAC; retain 2024 prices and fetch the full 2025 Pacific delivery year for external-year evaluation.
- Train on January-September 2024 usable history. Reserve October-December 2024 for threshold calibration; leave 48-hour gaps before calibration and the test year. Freeze fitted parameters and thresholds for 2025. No retraining or tuning on 2025.
- Compare an hourly training-mean baseline, the existing Random Forest configuration with lagged price features plus known DA price, and the same RF with system/regional DA demand forecasts, their ratio and adjacent-hour forecast changes added.
- Forecast context: after the DA auction, one hour before delivery. Price histories stay delayed 48 hours. Current-delivery DA price is a reference, not an executable entry. Demand forecasts are archived DAM values, without full revision/publication vintages. These are availability assumptions to verify, not proof of point-in-time execution.
- Select positive and negative forecasts symmetrically when absolute prediction exceeds the prior calibration 90th percentile. Evaluate signs, MAE/RMSE, selection rate, mean signed spread, worst signed spread and illustrative $1/MWh hurdle. No optimized position sizing.
- Primary success criterion: improved 2025 MAE versus both the hourly baseline and the price-only RF. A positive signed spread after a hypothetical hurdle is a separate diagnostic, never proof of executable profit.
- Report all three models, data gaps, missing forecast rows and every test month. Do not switch the node, period or threshold after seeing results.
- Demand forecast levels and differences between delivery hours are not forecast revisions. There is no weather series, renewable forecast or causal weather attribution in this experiment.

## Interpretation fixed in advance

This extension changes both the information set and the evaluation design relative to the first audit. Its results must not be presented as an improvement caused solely by demand data. The incremental demand effect is identified only by comparing the two RF models within this extension on exactly the same retained hours.

A positive signed spread can reflect a forecasting relationship to a DA price already fixed before the forecast. It does not prove that this DA price could be entered at the forecast timestamp. The funding, settlement, execution and participant requirements of a trade remain separate questions.

## Data sources and timing limits

CAISO OASIS `SLD_FCST`, version 1, market `DAM`, supplies `SYS_FCST_DA_MW` for `CA ISO-TAC` and `PGE-TAC`. `PRC_LMP` version 1 and `PRC_INTVL_LMP` version 3 provide NP15 prices. The forecast CSV identifies delivery time and market run but does not contain a complete publication/revision history.

[CAISO's day-ahead overview](https://www.caiso.com/Documents/Presentation-Existing-Day-Ahead-Market-Overview.pdf) describes the auction timetable. [CAISO's forecast report specification](https://www.caiso.com/documents/oasis-interfacespecification_v5_1_2clean_fall2017release.pdf) identifies the DAM demand-forecast report. These explain why DAM prices/forecasts can be relevant to an after-auction research view; they do not certify every historical downloaded value was available unchanged at the intended forecast timestamp.
