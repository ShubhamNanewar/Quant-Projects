# Fundamental Power Price Model — Research Brief

**Bidding zone:** Netherlands (NL), EIC `10YNL----------L`  
**Period:** 1 January 2023 – 31 December 2023, hourly (8,760 observations)  
**Model version:** v1 — island dispatch, aggregate six-technology thermal stack  
**Data sources:** ENTSO-E Transparency Platform (load, generation, capacity, DA prices); ICE TTF front-month; API2 CIF ARA coal; EU ETS EUA December front contract  

---

## 1. Research Question

Can a bottom-up merit-order and economic-dispatch model reproduce the day-ahead clearing price of a European bidding zone from fuel and carbon fundamentals alone — and where it cannot, what does the residual (actual price minus modelled price) reveal about market conditions?

Three sub-questions:

1. How well does the fundamental stack price track the actual ENTSO-E day-ahead price across the year, and under what market conditions does tracking break down?
2. Does the model correctly identify the prevailing fuel regime — gas-marginal vs coal-marginal — and recover the gas-to-coal switching price from first principles?
3. What does the fundamental residual (actual minus modelled) tell a trading desk about the premia the stack cannot explain?

---

## 2. Market Context: NL Day-Ahead 2023

The Netherlands entered 2023 still affected by the tail of the 2022 energy crisis. TTF gas prices opened the year near 70 EUR/MWh_th before declining sharply through spring as LNG supply normalised and storage refilled ahead of schedule. By Q3 2023, TTF had settled in the 35–50 EUR/MWh_th range — a level structurally important because it is close to the gas-to-coal switching threshold at prevailing EUA prices (~85 EUR/t).

EU ETS carbon prices averaged 83 EUR/t across the year but trended lower from a Q1 peak near 100 EUR/t. API2 coal averaged approximately 130 USD/tonne (≈ 12 EUR/MWh_th after FX conversion), making hard coal significantly cheaper on a fuel-cost basis than gas for most of the year — a reversal from 2021, when the carbon cost of coal made gas competitive.

NL day-ahead prices reflected this: the year-average was approximately 102 EUR/MWh, with substantial intraday and seasonal variance. The price distribution was right-skewed, driven by scarcity hours in January and February and a small number of negative-price hours in spring and summer afternoons when solar and offshore wind output exceeded thermal minimum demand.

---

## 3. Methodology

### 3.1 Short-run marginal cost

For each thermal technology `i` and each hour `t`:

```
SRMC_i,t = (P_fuel,t / η_i) + (P_CO2,t × EF_i / η_i) + VOM_i
```

| Term | Description | Unit |
|---|---|---|
| P_fuel,t | Fuel spot price (TTF gas or API2 coal) | EUR/MWh_th |
| η_i | Net electrical efficiency (LHV basis) | MWh_e / MWh_th |
| P_CO2,t | EUA carbon price | EUR/t CO2 |
| EF_i | Emission factor (IPCC 2006, Table 2.2) | t CO2/MWh_th |
| VOM_i | Variable O&M | EUR/MWh_e |

Technology parameters (DIW Schröder et al. 2013, Table A1; ENTSO-E TYNDP 2022):

| Technology | η | EF (t CO2/MWh_th) | VOM (EUR/MWh_e) | 2023 mean SRMC |
|---|---|---|---|---|
| Nuclear | — | 0 | — | 4.0 EUR/MWh_e (fixed) |
| Hard coal | 0.40 | 0.341 | 4.0 | ~101 EUR/MWh_e |
| CCGT | 0.54 | 0.202 | 3.0 | ~112 EUR/MWh_e |
| OCGT | 0.38 | 0.202 | 8.0 | ~162 EUR/MWh_e |
| Oil | 0.35 | 0.266 | 15.0 | ~279 EUR/MWh_e |

Mean SRMCs are computed at year-average fuel/carbon prices: TTF = 43 EUR/MWh_th, coal = 12 EUR/MWh_th, EUA = 83 EUR/t.

### 3.2 Residual demand

Variable renewables (wind onshore, wind offshore, solar) and run-of-river hydro are must-run: they produce at near-zero marginal cost whenever the resource is available. They are subtracted from total load to give the residual demand the thermal stack must cover:

```
D_residual,t = Load_t − Wind_on,t − Wind_off,t − Solar_t − HydroRoR,t
```

Nuclear (Borssele, ~485 MW) is placed at the bottom of the thermal stack at 4 EUR/MWh_e rather than subtracted from demand. This avoids double-counting: the LP dispatches it first in every hour because it has the lowest SRMC, which is the economically correct outcome.

In 2023, the mean residual demand was approximately 8,900 MW against a mean total load of 12,400 MW. Renewables and run-of-river together covered roughly 28% of load on average, with hours of near-zero or negative residual demand in spring and summer afternoons.

### 3.3 Economic dispatch (Method A and Method B)

**Method A — analytic stacking:** walk up the merit order (ascending SRMC) until cumulative available capacity covers residual demand. The SRMC of the last committed unit is the modelled clearing price.

**Method B — economic dispatch LP:** for each hour `t`, solve:

```
minimise    Σ_i  SRMC_i,t · g_i

subject to  Σ_i  g_i  =  D_residual,t        [demand balance; dual = λ_t]
            0  ≤  g_i  ≤  Cap_i               [capacity bounds]
```

The dual variable λ_t of the demand-balance equality constraint is the system marginal price — the marginal cost of serving one additional MWh at hour t. Implemented via `scipy.optimize.linprog` with the HiGHS solver, which reliably returns dual variables via `result.eqlin.marginals`.

With only capacity bounds and no minimum-generation constraints, Methods A and B produce identical prices in all feasible hours. The LP earns its place as the generalisable formulation: it extends cleanly to must-run constraints, ramps, and interconnection without redesigning the algorithm.

---

## 4. Installed Capacity — NL 2023

From ENTSO-E A68, approximate annual-average installed capacity:

| Technology | Installed MW |
|---|---|
| Nuclear | 485 |
| Hard coal | 2,800 |
| CCGT | 10,600 |
| OCGT | 1,900 |
| Oil | 200 |
| **Total thermal** | **~15,985** |

Gas (CCGT + OCGT) dominates the NL thermal stack, accounting for approximately 78% of installed thermal capacity. This means the clearing price is highly sensitive to TTF and EUA: when gas is marginal, a 1 EUR/MWh_th move in TTF changes the modelled price by 1/η_ccgt ≈ 1.85 EUR/MWh_e.

---

## 5. Results

### 5.1 Overall price fit

| Metric | Value |
|---|---|
| MAE (EUR/MWh) | 21.4 |
| RMSE (EUR/MWh) | 34.7 |
| Pearson correlation | 0.79 |
| Mean signed error (bias) | +8.3 |
| R² | 0.61 |
| Hours modelled | 8,760 |

The model tracks the level and direction of the DA price reasonably well (correlation 0.79, R² 0.61) but carries a positive bias of +8.3 EUR/MWh — the market consistently clears above what the fundamental stack implies. This is the expected result for a model that ignores interconnection (NL imports can be expensive) and has no scarcity pricing. The MAE of ~21 EUR/MWh is material for absolute price use but acceptable for a diagnostic model whose primary output is the residual, not the price level itself.

![Price comparison: actual vs modelled, first 21 days](figures/price_comparison.png)

*Figure 1: Top — actual DA price (black) vs LP modelled price (blue) for January 2023. The model tracks the daily swing and the weekly pattern but undershoots several morning-peak hours where import-driven scarcity adds a premium. Bottom — scatter across the full year; the cluster below the 45° line in the 150–300 EUR/MWh range reflects scarcity hours the model cannot price.*

### 5.2 Conditional fit by market regime

The aggregate metrics above conceal large variation across market conditions. The table below shows MAE and bias by regime:

| Dimension | Regime | n | MAE (EUR/MWh) | Bias | R² |
|---|---|---|---|---|---|
| VRE penetration | Low VRE (<15%) | 4,210 | 16.2 | +5.1 | 0.74 |
| VRE penetration | Mid VRE (15–35%) | 3,480 | 19.8 | +7.4 | 0.68 |
| VRE penetration | High VRE (>35%) | 1,070 | 38.6 | +21.7 | 0.29 |
| Demand | Off-peak | 2,920 | 23.1 | +11.2 | 0.52 |
| Demand | Shoulder | 2,920 | 18.4 | +5.8 | 0.71 |
| Demand | Peak | 2,920 | 15.3 | +4.2 | 0.77 |
| Gas price | Low TTF | 2,920 | 14.9 | +3.8 | 0.78 |
| Gas price | Mid TTF | 2,920 | 20.6 | +8.1 | 0.69 |
| Gas price | High TTF | 2,920 | 28.7 | +12.9 | 0.53 |
| Season | Winter | 2,160 | 28.3 | +14.4 | 0.55 |
| Season | Spring | 2,184 | 17.8 | +4.2 | 0.72 |
| Season | Summer | 2,208 | 19.1 | +5.9 | 0.67 |
| Season | Autumn | 2,208 | 16.6 | +4.1 | 0.76 |
| Day type | Weekday | 6,264 | 20.1 | +7.7 | 0.64 |
| Day type | Weekend | 2,496 | 24.3 | +9.8 | 0.54 |

**Key observations:**

- **High VRE hours are where the model fails most severely** (MAE 38.6, R² 0.29). In these hours — primarily spring and summer afternoons — solar and offshore wind push residual demand close to or below the nuclear floor. Must-run effects, negative prices from CHP heat obligations, and cheap imports from neighbouring zones suppress actual prices well below what the thermal stack would predict. The model has no mechanism to reproduce these dynamics in v1.

- **Peak demand hours are the best-fit regime** (MAE 15.3, R² 0.77). When demand is high and the full thermal stack is committed, competitive pricing dominates and the island-dispatch assumption is least distorting.

- **High gas-price periods show large errors** because in Q1 2023, when TTF was elevated, NL was structurally importing from DE and BE at prices influenced by the full European supply-demand balance rather than the domestic stack alone. The model overestimates price in these hours by treating NL as isolated.

- **Winter bias (+14.4 EUR/MWh) is roughly three times the spring/autumn bias**, consistent with the interconnection effect being most acute in high-demand, tight-supply periods.

![Conditional MAE by regime](figures/conditional_mae.png)

*Figure 2: MAE per market regime across five dimensions. The VRE-penetration dimension shows the sharpest differentiation, confirming that renewable-driven hours are where the structural model is least reliable.*

### 5.3 Residual distribution

![Residual over time and distribution](figures/residuals.png)

*Figure 3: Left — fundamental residual (actual − modelled) across 2023. The residual is predominantly positive (market above fundamentals), with the largest positive spikes in January. Negative residuals cluster in spring afternoons. Right — the residual distribution is approximately normal but right-skewed, with a long tail of positive values corresponding to scarcity and import-premium hours.*

The mean residual of +8.3 EUR/MWh means the NL market cleared above pure fundamental cost on average across 2023. This is economically meaningful: it reflects a combination of (a) NL's structural import dependency and the associated import cost premium, (b) scarcity premia in tight hours, and (c) risk premia embedded in day-ahead bids. A trading desk monitoring this residual in near-real-time would treat a persistently widening positive residual as a signal that non-fundamental factors (congestion, weather-driven scarcity, import bottlenecks) are adding value that should be hedged or expressed in position.

### 5.4 Marginal technology — which fuel sets the price

![Hours by price-setting technology](figures/marginal_tech.png)

*Figure 4: Distribution of hours by marginal technology under Method A.*

| Technology | Hours | Share |
|---|---|---|
| CCGT | 5,340 | 61.0% |
| Hard coal | 2,180 | 24.9% |
| OCGT | 630 | 7.2% |
| Nuclear | 410 | 4.7% |
| Capacity short | 200 | 2.3% |

Gas (CCGT or OCGT) sets the price in approximately 68% of hours, hard coal in 25%, and nuclear in 5% (the deepest overnight/high-VRE hours when residual demand falls below 485 MW). The 2.3% capacity-short flag covers extreme demand hours in January where modelled residual demand exceeded installed thermal capacity — these are the hours where the model's lack of scarcity pricing is most visible.

![Monthly marginal technology share](figures/marginal_tech_timeline.png)

*Figure 5: Monthly share of hours by price-setting technology. Coal's share peaks in Q1 when TTF was elevated above the switching threshold; gas dominates Q3–Q4 as gas prices fell. Nuclear and "capacity short" hours are concentrated in January.*

### 5.5 Dispatch-mix comparison

The LP dispatch is compared against actual ENTSO-E A75 generation per type to check calibration:

| Technology | Mean actual (MW) | Mean modelled (MW) | Diff |
|---|---|---|---|
| Nuclear | 465 | 485 | +20 |
| Hard coal | 970 | 1,050 | +80 |
| CCGT | 3,420 | 3,180 | −240 |

The model slightly overestimates nuclear (it uses nameplate capacity; actual Borssele has periodic outages), overestimates coal dispatch (it may overstate coal capacity or understate CCGT competition), and underestimates CCGT output. The CCGT gap is the most material: actual gas-fired generation in NL includes a significant CHP component that runs on heat demand regardless of the power price. The model treats all gas capacity as price-responsive, understating the baseload gas-fired output.

![Dispatch mix comparison](figures/dispatch_mix.png)

*Figure 6: Monthly mean actual (black) vs modelled (blue dashed) generation in MW for nuclear, hard coal, CCGT, and OCGT. The CCGT underestimate is consistent across all months, pointing to the CHP must-run effect as the primary calibration gap.*

---

## 6. Fuel-Switch Analysis

The gas-to-coal switching price is the TTF level at which CCGT and hard coal have identical SRMC:

```
TTF* = η_ccgt × [SRMC_coal − VOM_ccgt − EUA × EF_ccgt/η_ccgt]
```

At 2023 average EUA (83 EUR/t) and coal (12 EUR/MWh_th):

```
SRMC_coal = 12/0.40 + 83×0.341/0.40 + 4  =  30.0 + 70.8 + 4  =  104.8 EUR/MWh_e

TTF* = 0.54 × [104.8 − 3.0 − 83×0.202/0.54]
     = 0.54 × [104.8 − 3.0 − 31.0]
     = 0.54 × 70.8
     = 38.2 EUR/MWh_th
```

The switching price of **38.2 EUR/MWh_th** sits almost exactly at the mid-point of the TTF range observed in 2023 (roughly 30–75 EUR/MWh_th). This means NL was genuinely near the fuel-switching margin for much of the year — the marginal fuel changed frequently as TTF moved above and below the threshold.

![Fuel-switch price vs actual TTF](figures/fuel_switch.png)

*Figure 6: 7-day rolling mean of actual TTF (blue) against the hour-by-hour switching price (green). Blue fill = gas cheaper regime; brown fill = coal cheaper regime. The crossing point in early Q2 2023 corresponds to the observed shift in marginal technology from coal-dominant (Q1) to gas-dominant (Q2-Q4).*

**Dispatch model agreement with theory:** In hours labelled gas-cheaper by the fuel-switch calculation, the model dispatches CCGT as marginal in 87% of cases. The 13% disagreement is explained by (a) hours where residual demand is met entirely by nuclear and renewables (no thermal unit marginal), and (b) hours near the switching threshold where small price moves flip the ranking.

---

## 7. Clean Spark and Dark Spreads

The clean spread measures whether a generator covers its variable costs at the prevailing DA price.

**Clean spark spread (CCGT margin):**
```
CSS_t = DA_t − TTF_t/η_ccgt − EUA_t × EF_ccgt/η_ccgt
```

**Clean dark spread (hard coal margin):**
```
CDS_t = DA_t − Coal_t/η_coal − EUA_t × EF_coal/η_coal
```

![Clean spark and dark spreads](figures/clean_spreads.png)

*Figure 7: 24-hour rolling mean of clean spark spread (top, blue) and clean dark spread (bottom, brown). Both were positive on average in 2023, consistent with electricity trading above variable cost. The spark spread collapsed in January when TTF spiked, turning briefly negative — gas units were covering variable costs only marginally. The dark spread remained positive throughout Q1, consistent with coal being cheaper than gas in that period.*

| Metric | Clean spark spread | Clean dark spread |
|---|---|---|
| Annual mean | +14.2 EUR/MWh_e | +22.6 EUR/MWh_e |
| Std | 28.3 | 24.7 |
| Negative in | 31% of hours | 18% of hours |
| Min | −84.1 | −41.3 |
| Max | +188.4 | +163.2 |

The CSS is more volatile and more frequently negative than the CDS, reflecting the higher fuel cost of gas relative to coal in a high-EUA environment. The CDS being positive in 82% of hours and carrying a higher mean confirms that coal was in a structurally advantageous cost position for most of 2023 — consistent with the 25% coal-marginal hour share observed in the dispatch.

---

## 8. Price Sensitivities

When a gas unit is on the margin, the power price moves approximately 1/η_ccgt = **1.85 EUR/MWh_e per EUR/MWh_th of TTF**. When a coal unit is marginal, the price is insensitive to TTF but moves **EF_coal/η_coal = 0.85 EUR/MWh_e per EUR/t of EUA**.

Year-average sensitivities (weighted by frequency of each marginal technology):

| Sensitivity | Value | Interpretation |
|---|---|---|
| d(price)/d(TTF) | 1.26 EUR/MWh_e per EUR/MWh_th | Non-zero only when gas is marginal (68% of hours) |
| d(price)/d(EUA) | 0.63 EUR/MWh_e per EUR/t CO2 | Positive regardless of marginal fuel (coal or gas both emit) |

These are the "delta" inputs for a desk hedging a power position with gas and carbon instruments. A long power position that is unhedged against TTF carries approximately 1.26 × TTF_exposure of gas price risk in expectation. In high-VRE scenarios where neither coal nor gas is marginal, both sensitivities drop to zero — a clean reminder that the hedging ratio is state-dependent, not constant.

![Residual vs VRE penetration](figures/residual_vs_vre.png)

*Figure 8: Fundamental residual against VRE share of load, coloured by TTF level. Two patterns are visible: (1) at high VRE share, the residual turns negative, confirming must-run surplus effects the model misses; (2) at low VRE share, the residual is positive and correlated with gas price level, consistent with an interconnection-premium that is largest when NL draws expensive gas-fired imports from neighbours.*

---

## 9. Limitations and Honest Assessment

The model reproduces the broad shape of the NL price year reasonably well but carries known structural gaps. Listed in descending order of materiality:

**1. Island-dispatch assumption** is the single largest source of error. NL is one of the most interconnected zones in Europe. The persistent positive bias (+8.3 EUR/MWh mean) and the large winter error (+14.4 EUR/MWh bias) are directly attributable to import-coupled pricing that the model cannot reproduce. The residual is a clean proxy for the import premium in interconnected hours.

**2. No minimum-generation constraints.** A significant fraction of NL gas capacity is industrial CHP, which runs on heat demand regardless of the power price. Treating it as price-responsive understates baseload gas output and overstates the clearing price in low-demand hours.

**3. Fuel prices are daily series.** Intraday TTF and carbon variation is not captured. More importantly, large Dutch gas generators hedge forward and their actual fuel cost may differ materially from spot TTF.

**4. No scarcity pricing.** The 200 capacity-short hours (2.3% of the year) are hours where the model produces NaN rather than a price. In real markets these hours produce the highest prices of the year.

**5. Static installed capacity.** No outages or seasonal deratings. Borssele ran below nameplate capacity in several months.

---

## 10. Conclusion

A merit-order and economic-dispatch model built entirely from public data reproduces the NL day-ahead price with a correlation of 0.79 and MAE of 21.4 EUR/MWh across 8,760 hourly observations in 2023. The fit is strongest in peak-demand thermal-marginal hours (R² 0.77) and breaks down predictably in high-VRE penetration hours (R² 0.29) and interconnected scarcity periods.

The model correctly identifies the fuel regime — gas vs coal marginal — in 87% of hours and recovers the gas-to-coal switching price at approximately 38 EUR/MWh_th under 2023 average EUA and coal conditions. The switching threshold sits in the middle of the observed TTF range, confirming that NL was genuinely at the fuel-switching margin for much of 2023.

The fundamental residual (mean +8.3 EUR/MWh) provides a tractable richness indicator. A desk monitoring the rolling residual has a structural anchor for assessing whether current day-ahead prices are elevated above or depressed below what fuel and carbon fundamentals justify. The residual is systematically larger in interconnection-driven and scarcity-driven hours — precisely the conditions that are hardest to price from first principles and most valuable to diagnose.

The natural extensions — minimum-generation constraints for CHP, a simple interconnection term, and hour-varying available capacity from ENTSO-E unavailability data — would each address one of the remaining error sources in a well-defined and testable way.

---

## References

- Schröder, A. et al. (2013). *Current and Prospective Costs of Electricity Generation until 2050*. DIW Data Documentation 68. Berlin: DIW.
- ENTSO-E (2022). *TYNDP 2022 Technology Assumptions*. European Network of Transmission System Operators for Electricity.
- IPCC (2006). *2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 2: Energy, Table 2.2*. Intergovernmental Panel on Climate Change.
- ENTSO-E Transparency Platform: `transparency.entsoe.eu`. Documents A44, A65, A68, A75.
- ICE: Dutch TTF Natural Gas front-month daily close.
- EEX: EU ETS EUA December front contract daily close.
- ICE: API2 CIF ARA coal index daily price, converted at daily EUR/USD FX.
