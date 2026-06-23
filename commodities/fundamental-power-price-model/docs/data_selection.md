# Data Selection

## Bidding zone
**Netherlands (NL)**, EIC code `10YNL----------L`.

Chosen because it is the home market of Eneco, making the model directly relevant to the target role. A robustness run on **Germany / Luxembourg (DE-LU)**, EIC `10Y1001A1001A82H`, is available as an optional second pass: DE-LU has a richer thermal stack (lignite, coal, large CCGT fleet) and is the canonical merit-order case in the European power literature.

**Caveat:** The Netherlands is one of the most heavily interconnected bidding zones in continental Europe. Its physical flows with DE, BE, GB, DK2, and NO2 are substantial. The v1 model treats NL as an island (no net imports in the dispatch LP), which is a known simplification. The effect is that the model will underestimate prices when NL is structurally importing expensive power and overestimate them when cheap neighbouring power suppresses local prices. See `docs/limitations.md`.

## Period
Full calendar year 2023 (UTC), producing approximately 8,760 hourly observations. This mirrors the one-year hourly structure of the CAISO empirical project.

## Sources

| Data item | Source | Access | Notes |
|---|---|---|---|
| Actual total load | ENTSO-E Transparency (A65) | `entsoe-py: query_load` | Used as total demand input |
| Actual generation per type | ENTSO-E Transparency (A75) | `entsoe-py: query_generation` | Wind, solar, hydro, nuclear separated out |
| Installed capacity per type | ENTSO-E Transparency (A68) | `entsoe-py: query_installed_generation_capacity` | Used to size the supply stack |
| Day-ahead prices | ENTSO-E Transparency (A44) | `entsoe-py: query_day_ahead_prices` | Target series for validation |
| TTF gas price (EUR/MWh) | ICE / EEX daily front-month | Manual CSV (`data/raw/ttf_gas_eur_mwh.csv`) | Daily, broadcast to hourly |
| API2 coal (USD/t) | IEA / EEX daily | Manual CSV (`data/raw/api2_coal_usd_t.csv`) | Converted to EUR/MWh_th using 6.978 MWh_th/t |
| EUA carbon (EUR/t CO2) | EU ETS December front contract | Manual CSV (`data/raw/eua_carbon_eur_t.csv`) | Daily, broadcast to hourly |
| EUR/USD FX | Manual CSV or 1.08 constant | `data/raw/eurusd.csv` (optional) | Used for coal unit conversion only |

## Fuel data provenance and approximation
Fuel and carbon prices are **daily public series** broadcast to the hourly panel by forward-fill. This is a standard simplification in fundamental power-price modelling: intraday fuel price moves and firm-level hedged positions are not captured. The approximation is stated explicitly in `docs/limitations.md` and in the README.
