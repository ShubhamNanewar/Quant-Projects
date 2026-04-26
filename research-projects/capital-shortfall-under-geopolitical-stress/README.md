# Capital Shortfall Under Geopolitical Stress

This project studies whether large banks remain solvent when geopolitical stress rises sharply and capital flows into defense and energy sectors. The main goal is not only to identify exposure, but to convert that exposure into a market-based capital-shortfall measure under a severe geopolitical stress scenario.

## Project Focus

The project has three connected layers:

- construct a tradable geopolitical stress factor from industry and market returns,
- estimate time-varying bank exposure with a DCC-GJR-GARCH style framework,
- translate that exposure into `GeoRisk` and `Marginal GeoRisk` capital-shortfall measures.

This is a research project rather than a standard assignment notebook. The economic question is whether geopolitical crises create a systematic solvency risk for large banks through falling equity values, rising volatility, tighter correlations, and the loss of diversification benefits.

## Repository Layout

```text
capital-shortfall-under-geopolitical-stress/
├── data/
│   ├── 48_Industry_Portfolios_Daily.csv
│   ├── Bank_data_full.csv
│   ├── Bank_data_full_2.csv
│   ├── Dashboard_data.csv
│   ├── SP_500_Data.csv
│   └── data_gpr_export.csv
├── notebooks/
│   ├── geopolitical_stress_dcc_gjrgarch.ipynb
│   └── geopolitical_bank_factor.ipynb
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Research Question

The central question is:

> When geopolitical stress rises sharply, do bank equity values deteriorate enough to create a meaningful capital shortfall?

The report argues that geopolitical shocks matter through the sovereign-bank nexus, funding-market stress, repricing of risky assets, and a flight to safety into sectors such as defense and energy. That means the right object to measure is not only a regression coefficient, but a stress-dependent capital gap.

## Why A Mimicking Portfolio?

Instead of using a text-based geopolitical-risk index directly as the main stress factor, the project constructs a return-based mimicking portfolio. That choice has a clear asset-pricing logic:

- stress testing requires a return factor that can be compared directly with bank equity returns,
- defense and energy stocks react quickly to conflict and security shocks,
- shorting the broad market strips out part of the general business-cycle component,
- the resulting factor behaves like a hedged market-based proxy for geopolitical stress.

The factor is designed to become positive when defense and energy sectors outperform the market, which is the kind of return pattern expected during periods of geopolitical anxiety.

## Construction Of The Geopolitical Stress Factor

Let

- $R^{\text{Aero}}_t$, $R^{\text{Guns}}_t$, $R^{\text{Ships}}_t$ be the daily defense-related industry returns,
- $R^{\text{Oil}}_t$, $R^{\text{Coal}}_t$, $R^{\text{Util}}_t$ be the daily energy-related industry returns,
- $R^{m}_t$ be the daily S&P 500 return.

The defense portfolio is

$$
R^{\text{Def}}_t = \frac{1}{3}\left(R^{\text{Aero}}_t + R^{\text{Guns}}_t + R^{\text{Ships}}_t\right),
$$

and the energy portfolio is

$$
R^{\text{Eng}}_t = \frac{1}{3}\left(R^{\text{Oil}}_t + R^{\text{Coal}}_t + R^{\text{Util}}_t\right).
$$

The long leg is a balanced allocation:

$$
R^{\text{Long}}_t = 0.5\,R^{\text{Def}}_t + 0.5\,R^{\text{Eng}}_t.
$$

The final geopolitical stress factor is

$$
\text{GPRFactor}_t = R^{\text{Long}}_t - R^m_t.
$$

So a positive factor realization means that defense and energy outperform the broad market on that day.

## Fixed-Beta Benchmark

Before moving to the dynamic model, the project uses a static regression benchmark:

$$
R_{i,t} = \beta_{i,m}R^m_t + \beta_{i,g}\text{GPRFactor}_t + \varepsilon_{i,t}.
$$

This fixed-beta regression is only a first pass. It is useful for screening banks with negative geopolitical exposure, but it is not sufficient for stress testing because it assumes constant covariance structure through time.

## Why GJR-GARCH?

Geopolitical shocks are not symmetric. Negative market returns usually increase future volatility more than positive returns of the same magnitude, and crisis periods often produce volatility clustering. A standard homoskedastic model misses that.

That is why the report uses a GJR-GARCH specification. For bank $i$, market $m$, and geopolitical factor $g$, the conditional variance dynamics are written as:

$$
\sigma^2_{i,t} = \omega_i + \alpha_i r^2_{i,t-1} + \gamma_i r^2_{i,t-1} I^-_{i,t-1} + \beta_i \sigma^2_{i,t-1},
$$

$$
\sigma^2_{m,t} = \omega_m + \alpha_m r^2_{m,t-1} + \gamma_m r^2_{m,t-1} I^-_{m,t-1} + \beta_m \sigma^2_{m,t-1},
$$

$$
\sigma^2_{g,t} = \omega_g + \alpha_g r^2_{g,t-1} + \gamma_g r^2_{g,t-1} I^+_{g,t-1} + \beta_g \sigma^2_{g,t-1}.
$$

Here:

- $I^-_{i,t-1}=1$ when the lagged bank return is negative,
- $I^-_{m,t-1}=1$ when the lagged market return is negative,
- $I^+_{g,t-1}=1$ when the lagged geopolitical factor is positive.

That last choice is economically important. In this setting, a large positive geopolitical-factor move corresponds to strong defense/energy performance relative to the market, which is exactly the stress regime of interest. So the asymmetry is attached to positive geopolitical shocks rather than negative ones.

## Why DCC?

Static correlations are too weak for this problem. During quiet periods, a bank may show modest relation with the stress factor. During conflict episodes, that relation can change sharply.

The DCC step models the conditional covariance matrix as time-varying. Let

$$
r_t =
\begin{bmatrix}
r_{i,t} \\
r_{m,t} \\
r_{g,t}
\end{bmatrix},
\qquad
H_t =
\begin{bmatrix}
\sigma^2_{i,t} & \rho_{im,t}\sigma_{i,t}\sigma_{m,t} & \rho_{ig,t}\sigma_{i,t}\sigma_{g,t} \\
\rho_{im,t}\sigma_{i,t}\sigma_{m,t} & \sigma^2_{m,t} & \rho_{mg,t}\sigma_{m,t}\sigma_{g,t} \\
\rho_{ig,t}\sigma_{i,t}\sigma_{g,t} & \rho_{mg,t}\sigma_{m,t}\sigma_{g,t} & \sigma^2_{g,t}
\end{bmatrix}.
$$

After standardizing the residuals, DCC updates the intermediate correlation matrix $Q_t$ as

$$
Q_t = (1-a-b)\bar{Q} + a\,e_{t-1}e_{t-1}^\prime + b\,Q_{t-1},
$$

where:

- $e_t$ is the vector of volatility-standardized residuals,
- $\bar{Q}$ is the unconditional correlation matrix of standardized residuals,
- $a$ measures the effect of the latest shock,
- $b$ measures correlation persistence.

The conditional correlation matrix is then obtained by normalization:

$$
R_t = \operatorname{diag}(Q_t)^{-1/2} Q_t \operatorname{diag}(Q_t)^{-1/2}.
$$

This is the key matrix step in the project. It gives a full time-varying correlation structure rather than a single constant correlation estimate.

## Dynamic Conditional Beta

Once the conditional correlations and volatilities are available, the bank's dynamic beta with respect to the market and the geopolitical factor is computed from the conditional covariance system.

Define the factor covariance matrix:

$$
\Sigma_{FF,t} =
\begin{bmatrix}
\sigma^2_{m,t} & \rho_{mg,t}\sigma_{m,t}\sigma_{g,t} \\
\rho_{mg,t}\sigma_{m,t}\sigma_{g,t} & \sigma^2_{g,t}
\end{bmatrix},
$$

and the covariance vector between bank $i$ and the two factors:

$$
\Sigma_{iF,t} =
\begin{bmatrix}
\rho_{im,t}\sigma_{i,t}\sigma_{m,t} \\
\rho_{ig,t}\sigma_{i,t}\sigma_{g,t}
\end{bmatrix}.
$$

Then the dynamic beta vector is

$$
\begin{bmatrix}
\beta^{m}_{i,t} \\
\beta^{g}_{i,t}
\end{bmatrix}
=
\Sigma_{FF,t}^{-1}\Sigma_{iF,t}.
$$

This matters because the geopolitical beta is not estimated from a single static coefficient. It is re-computed every date from the evolving covariance matrix.

## Stress Scenario Calibration

The report calibrates the geopolitical stress scenario from the factor itself.

First construct the factor price index:

$$
\text{FactorPrice}_t = \prod_{s=1}^{t}\left(1+\frac{\text{GPRFactor}_s}{100}\right).
$$

Then compute rolling six-month returns using 126 trading days:

$$
\text{SixMonthReturn}_t =
\frac{\text{FactorPrice}_t}{\text{FactorPrice}_{t-126}} - 1.
$$

The stress magnitude is the 99th percentile of that six-month distribution:

$$
\xi = Q_{0.99}\left(\{\text{SixMonthReturn}_t\}\right).
$$

This is meant to capture a severe but historically grounded tail event.

## Balance Sheet Inputs

To convert market exposure into solvency pressure, the project merges market data with balance-sheet data.

The key inputs are:

- $W_{i,t}$: market capitalization, computed from daily price times shares outstanding,
- $D_{i,t}$: book leverage or debt proxy, constructed from quarterly accounting variables and forward-filled to daily frequency,
- $k = 0.08$: prudential capital requirement.

The forward-fill step is necessary because market data are daily while the accounting data arrive quarterly.

## From Exposure To Capital Shortfall

The report maps the geopolitical beta into an equity-loss multiplier:

$$
\text{Loss}_{i,t} = \exp\left(\beta^g_{i,t}\ln(1+\xi)\right).
$$

This is the core stress translation step.

- If $\beta^g_{i,t}<0$ and $\xi>0$, the multiplier can imply equity deterioration under stress.
- Larger absolute negative beta means stronger sensitivity to geopolitical stress.

The total capital shortfall measure is:

$$
\text{GeoRisk}_{i,t} = kD_{i,t} - (1-k)W_{i,t}\text{Loss}_{i,t}.
$$

Interpretation:

- positive `GeoRisk` means capital shortfall,
- negative `GeoRisk` means capital surplus.

The marginal component that isolates the effect of the geopolitical shock is:

$$
\text{Marginal GeoRisk}_{i,t} = (1-k)W_{i,t}\left(\text{Loss}_{i,t} - 1\right).
$$

This strips out the bank's pre-existing leverage position and measures the incremental erosion caused by the geopolitical scenario itself.

## Why This Matters Economically

The whole purpose of the framework is to connect three things:

1. market repricing during geopolitical stress,
2. time-varying bank exposure to that repricing,
3. capital adequacy under a prudential threshold.

So the project is not just saying that some bank stocks fall during crises. It is asking whether the fall is large enough, conditional on leverage, to threaten solvency buffers.

## Code Notes

The main research workflow is in:

- [geopolitical_stress_dcc_gjrgarch.ipynb](/Users/shubhamnanewar/Documents/Playground/Projects/research-projects/capital-shortfall-under-geopolitical-stress/notebooks/geopolitical_stress_dcc_gjrgarch.ipynb)

That notebook contains:

- factor construction,
- static OLS screening,
- volatility standardization,
- DCC correlation reconstruction,
- dynamic beta calculation,
- market-data and balance-sheet merge,
- GeoRisk calculation.

Two practical notes are important when reviewing the code:

- the notebook is the primary implementation artifact for the dynamic model,
- some early exploratory cells were rough and are being cleaned so the review should focus on the DCC-GJR-GARCH and GeoRisk sections rather than the first-pass static regressions alone.

## What To Review First

If you want to verify the methodology quickly, start with:

1. the factor construction section in the notebook,
2. the GJR-GARCH and DCC sections,
3. the dynamic beta matrix calculation,
4. the GeoRisk and Marginal GeoRisk formulas.

That is where the actual research contribution is.
