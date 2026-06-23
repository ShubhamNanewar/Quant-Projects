# Market Mathematics: SRMC, Merit Order, Residual Demand

## Short-run marginal cost

For a thermal generator using fuel `i`:

```
SRMC_i  =  (P_fuel_i / η_i)  +  (P_CO2 · EF_i / η_i)  +  VOM_i
```

| Symbol | Meaning | Unit |
|---|---|---|
| `P_fuel_i` | Fuel price | EUR / MWh_thermal |
| `η_i` | Net electrical efficiency (LHV basis) | MWh_e / MWh_th |
| `P_CO2` | EU ETS carbon price | EUR / t CO2 |
| `EF_i` | Emission factor | t CO2 / MWh_thermal |
| `VOM_i` | Variable operations & maintenance | EUR / MWh_e |

The resulting SRMC is in EUR / MWh_e and represents the additional cost of generating one more MWh of electricity from that unit.

## Technology parameters

Values from Schröder et al. (2013), *Current and Prospective Costs of Electricity Generation until 2050*, DIW Data Documentation 68, Table A1; and ENTSO-E TYNDP 2022 technology assumptions.

| Technology | η (LHV) | EF (t CO2/MWh_th) | VOM (EUR/MWh_e) | Fuel input |
|---|---|---|---|---|
| Nuclear | — | 0 | 4 | Fixed SRMC = 4 EUR/MWh_e |
| Lignite | 0.38 | 0.36 | 3 | API2 coal (proxy) |
| Hard coal | 0.40 | 0.34 | 4 | API2 coal |
| CCGT | 0.54 | 0.202 | 3 | TTF gas |
| OCGT | 0.38 | 0.202 | 8 | TTF gas |
| Oil | 0.35 | 0.266 | 15 | Fixed ~80 EUR/MWh_th |

## Merit order
Sorting all technologies by ascending SRMC gives the **supply stack**. Cheaper units are dispatched first. In a competitive electricity market, the clearing price equals the SRMC of the last (marginal) unit dispatched to meet demand.

## Residual demand

```
D_residual,t  =  Load_t  −  (Wind_on,t  +  Wind_off,t  +  Solar_t  +  HydroRoR,t  +  Nuclear_t)
```

Variable renewables (wind, solar) and run-of-river hydro have near-zero marginal cost and are **must-run**: they clear first. Nuclear is treated as must-run and subtracted from demand before the thermal dispatch. The thermal stack only needs to cover the residual.

## Gas-to-coal fuel-switch price
The analytic TTF gas price `p*` at which CCGT becomes cheaper than hard coal:

```
p*  =  [SRMC_coal  −  VOM_ccgt  −  (P_CO2 · EF_ccgt / η_ccgt)]  ·  η_ccgt
```

Implemented in `src/stack.py: fuel_switch_price_gas_vs_coal()`.
