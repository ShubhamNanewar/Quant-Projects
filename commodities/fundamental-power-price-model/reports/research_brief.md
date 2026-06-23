# Research Brief: Fundamental Power Price Model

**Bidding zone:** Netherlands (NL), EIC `10YNL----------L`  
**Period:** 2023 (full calendar year, hourly)  
**Model version:** v1 — island dispatch, aggregate technology stack

---

## Research question
Can a bottom-up merit-order and economic-dispatch model reproduce the day-ahead clearing price of a European bidding zone from fundamentals alone, and where it cannot, what does the residual (actual price minus modelled price) tell us about market conditions?

---

## Methodology

### Supply stack
Six thermal technologies ordered by short-run marginal cost (SRMC):

```
SRMC_i = (P_fuel / η_i) + (P_CO2 · EF_i / η_i) + VOM_i
```

Technology parameters from DIW (Schröder et al. 2013) and ENTSO-E TYNDP 2022.

### Residual demand
Variable renewables (wind on/offshore, solar) and run-of-river hydro subtract from load first (near-zero SRMC, must-run). Nuclear treated as must-run. The thermal stack covers the remainder.

### Economic dispatch LP
For each hour, minimise total dispatch cost subject to demand balance and capacity bounds. The dual of the demand-balance constraint gives the system marginal price λ_t. Confirmed to agree with the analytic merit-order stacking (Method A).

---

## Results

> **[Fill in after running the model]**

| Metric | Value |
|---|---|
| MAE (EUR/MWh) | — |
| RMSE (EUR/MWh) | — |
| Correlation | — |
| Bias (mean signed error) | — |

### Conditional fit

> **[Insert conditional_metrics.csv table here after run]**

Key finding: tracking is strongest in [X] hours and breaks down in [Y] conditions, consistent with the known limitations (island assumption, no scarcity pricing).

### Marginal technology
Gas (CCGT) sets the price in [X]% of hours; hard coal in [Y]%; nuclear/renewables dominate in low-demand overnight hours.

### Fundamental residual (richness indicator)
Mean residual [positive/negative], indicating the market trades [above/below] fundamentals on average. Residual is most positive during [season/condition], consistent with [congestion/scarcity/risk premium].

### Fuel-switch price
The analytic gas-to-coal switching price averages [X] EUR/MWh_th for 2023 EUA and coal levels. Actual TTF averaged [Y], placing the market in [gas/coal-favoured] territory for [Z]% of hours. The dispatch model correctly reflects this regime in [A]% of gas-marginal hours.

---

## Limitations
See `docs/limitations.md`. The most material are:
1. Island-dispatch assumption (NL is heavily interconnected)
2. Fuel prices are daily public series, not firm hedged costs
3. No minimum-generation or ramp constraints
4. No scarcity pricing

---

## Conclusion
The merit-order model provides a useful **fundamental anchor** for the NL day-ahead price. The residual highlights systematic deviations that a trading desk would want to monitor: persistent positive residuals flag premia the stack does not explain (scarcity, congestion, risk). The model is not a price predictor; it is a diagnostic of how far the market is from competitive fundamentals.
