# Model Limitations

Every limitation here is real and should be stated in any presentation of results. The model is useful precisely because its residual reveals what it cannot explain.

---

## 1. Island-dispatch assumption (most material for NL)
The LP treats the Netherlands as an electrically isolated system. In reality NL is one of the most interconnected bidding zones in continental Europe, with significant flows to/from DE, BE, GB, DK2, and NO2. The effect:
- In hours when NL imports cheap continental power, the model overestimates the clearing price (it dispatches expensive domestic units instead).
- In hours when NL imports expensive power (e.g. during European scarcity), the model underestimates the price.

**Mitigation in v1:** state it openly; the residual (actual minus modelled) should reveal systematic import-related bias. An extension would add a net-import term to the dispatch LP.

## 2. Fuel and carbon prices are daily public series
TTF gas, API2 coal, and EUA carbon are sourced as daily series and broadcast forward-filled to hourly. This misses:
- Intraday fuel price variation.
- Firm-level hedged positions and contract prices (actual fuel costs to generators may differ from spot).

**Effect:** the SRMC calculation is based on spot fuel costs, which is an approximation of the true offer cost. Standard in merit-order models for energy-market research; document the source for each series.

## 3. Aggregate technology stack (not plant-level)
The model uses one representative unit per technology (one efficiency, one capacity figure). Real stacks have many plants within each technology with varying efficiencies, ages, and outage profiles. A plant-level stack would require a REMIT or generator-level dataset and is a natural extension.

## 4. No minimum-generation constraints in v1
Nuclear and combined-heat-and-power (CHP) units typically have must-run minimum outputs. Without lower bounds on `g_i`, the LP can set nuclear to zero in low-demand hours, which is unrealistic. This can cause the model to underprice in low-demand / high-VRE hours. Minimum-generation constraints are listed as an optional extension in Section 5.5 of the spec.

## 5. Hourly resolution
The Netherlands day-ahead auction is hourly (SDAC). Some products and markets are moving to 15-minute resolution. The model is hourly throughout. Intraday and balancing price dynamics are not modelled.

## 6. No capacity payments or scarcity pricing
The LP has no value-of-lost-load (VOLL) term. When residual demand exceeds total installed capacity, the model has no price signal for scarcity. Real markets produce extreme prices in these hours; the model will underestimate them. Scarcity hours should be flagged separately in the backtest.

## 7. Static installed capacity
The model uses a single annual capacity figure from ENTSO-E A68. Unit outages, maintenance, and seasonal deratings are not captured. This adds noise to both the dispatch and the modelled price.
