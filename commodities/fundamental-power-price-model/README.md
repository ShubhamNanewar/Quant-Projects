# Fundamental Power Price Model

A bottom-up merit-order and economic-dispatch model of the Dutch (NL) day-ahead electricity market. Derives the hourly clearing price from first principles — fuel costs, carbon price, plant efficiencies, and residual demand — and validates against actual ENTSO-E day-ahead prices.

**Structural counterpart** to the [CAISO Power Spread Event Decision Framework](https://github.com/ShubhamNanewar/Quant-Projects), which learns the price empirically. Together they illustrate the two ways a power-trading desk forms a price view: bottom-up fundamentals and data-driven signals.

---

## What the model does

1. **Merit-order stack** — computes the short-run marginal cost (SRMC) of six thermal technologies (nuclear, lignite, hard coal, CCGT, OCGT, oil) from hourly fuel and carbon prices.
2. **Residual demand** — subtracts must-run variable renewables (wind, solar) and run-of-river hydro from total load. Nuclear sits at the bottom of the thermal stack (lowest SRMC) and is dispatched by the LP rather than subtracted from demand, avoiding double-counting.
3. **Method A (analytic)** — walks up the merit order until cumulative capacity meets residual demand; the SRMC of the marginal unit is the modelled clearing price.
4. **Method B (LP)** — solves the economic dispatch linear programme using scipy HiGHS; the dual variable of the demand-balance equality constraint gives the system marginal price. Methods A and B agree; the LP is the generalisable OR formulation and the one that scales to extensions.
5. **Validation** — compares modelled price against actual ENTSO-E day-ahead prices across the year, with overall metrics (MAE, RMSE, bias, correlation) and conditional metrics by VRE penetration, demand level, gas-price regime, season, and day type. Includes a dispatch-mix comparison (modelled vs actual generation per type).
6. **Trader insights** — fundamental residual (actual minus modelled) as a richness/cheapness indicator; clean spark spread (CSS) and clean dark spread (CDS) — both including the carbon cost; gas-to-coal fuel-switch price recovery; hourly price sensitivities to gas and carbon.

---

## Repository structure

```
fundamental-power-price-model/
  data/
    raw/                  # cached ENTSO-E pulls and fuel/carbon CSVs (gitignored)
    processed/            # merged hourly panel and dispatch output (gitignored)
  docs/
    data_selection.md     # zone, period, sources
    market_math.md        # SRMC, merit order, residual demand
    model_formulation.md  # dispatch LP and its dual
    limitations.md        # island assumption, fuel-data approximations, etc.
  notebooks/
    fundamental_power_price_model.ipynb
  reports/
    research_brief.md     # narrative and results
    figures/              # generated plots
    conditional_metrics.csv
    fundamental_residual.csv
  scripts/
    run_analysis.py       # end-to-end pipeline
  src/
    data_fetch.py         # ENTSO-E wrappers + fuel/carbon loaders
    stack.py              # SRMC and merit-order construction
    dispatch.py           # Method A (analytic) + Method B (LP)
    backtest.py           # metrics and conditional buckets
    insights.py           # residuals, spreads, marginal-tech timeline
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your ENTSO-E API token
```bash
cp .env.example .env
# edit .env and paste your token (register at https://transparency.entsoe.eu/)
```

### 3. Add fuel and carbon price CSVs
Place the following daily series in `data/raw/` with columns `date,price`:

| File | Description |
|---|---|
| `ttf_gas_eur_mwh.csv` | Dutch TTF front-month gas price (EUR/MWh thermal) |
| `api2_coal_usd_t.csv` | API2 coal (USD/tonne) |
| `eua_carbon_eur_t.csv` | EU ETS EUA front contract (EUR/tonne CO2) |
| `eurusd.csv` | EUR/USD FX rate (optional; defaults to 1.08) |

### 4. Run the full pipeline
```bash
python scripts/run_analysis.py
```

Raw ENTSO-E data is cached to `data/raw/` on first run. Re-runs load from cache.

---

## Methodology

### SRMC formula
```
SRMC_i = (P_fuel_i / η_i) + (P_CO2 · EF_i / η_i) + VOM_i
```

### Dispatch LP (Method B)
```
minimise    Σ_i  SRMC_i · g_i
subject to  Σ_i  g_i  =  D_residual,t       ← dual λ_t = clearing price
            0 ≤ g_i ≤ Cap_i
```

Technology parameters (efficiency, emission factor, VOM) from Schröder et al. (2013) *DIW Data Documentation 68* and ENTSO-E TYNDP 2022. See `docs/market_math.md`.

---

## Limitations (stated openly)

- **Island dispatch:** the LP ignores interconnection. NL is heavily import-coupled; this is a binding approximation.
- **Fuel inputs:** daily public series (TTF, API2, EUA) broadcast to hourly. Intraday moves and firm hedges are not captured.
- **Aggregate stack:** one representative unit per technology; no plant-level outages or efficiency spread.
- **No scarcity pricing:** no value-of-lost-load term; extreme scarcity hours will be underpriced.
- **No minimum-generation constraints** in v1: nuclear and CHP minimum outputs are not modelled.

Full discussion in `docs/limitations.md`.

---

## Tech stack
- Python 3.10+
- `entsoe-py` — ENTSO-E Transparency Platform wrapper
- `pandas`, `numpy`
- `scipy.optimize.linprog` (HiGHS solver) — dispatch LP; used instead of PuLP because HiGHS reliably exposes dual variables via `result.eqlin.marginals`, which CBC does not always do
- `matplotlib`

---

## Honest framing

This model is a **fundamental anchor and diagnostic**, not a price forecaster. It answers "where should the price be, given fuel costs and the stack?" and surfaces where and why the market deviates. A persistently positive residual flags premia the model cannot explain (scarcity, congestion, risk premium) — exactly the signal a trading desk wants to monitor.

Numbers and metrics will be filled in once the pipeline has run on the full 2023 data.
