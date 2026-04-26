# CVA Equity Derivatives

This project studies counterparty credit exposure for a small equity-derivatives portfolio consisting of two forwards and two puts on the SX5E and AEX indices. The main focus is how exposure evolves over time, how netting and collateral reduce CVA, and how CDS can hedge first-order spread risk.

## Project Focus

The project is organized in five blocks:

- validate the Monte Carlo market model,
- compute exposure and standalone trade CVA,
- compare unnetted and netted portfolio CVA,
- study sensitivity to volatility, correlation, and collateral,
- construct a bucketed CDS hedge for CVA spread risk.

## Repository Layout

```text
cva-equity-derivatives/
├── figures/
├── notebooks/
│   └── cva_equity_derivatives.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Market Model

The two equities follow correlated geometric Brownian motions under the risk-neutral measure:

$$
S_i(t+\Delta t)=S_i(t)\exp\left((r-q-\tfrac12\sigma_i^2)\Delta t+\sigma_i\sqrt{\Delta t}\,Z_i\right),
$$

with correlation

$$
\mathrm{Corr}(Z_1,Z_2)=\rho.
$$

This model is enough here because the assignment is about exposure and counterparty risk rather than smile-consistent pricing.

## Validation

The script checks three things:

1. forward martingale pricing,
2. Black-Scholes-Merton put pricing,
3. the target correlation between the two indices.

For the forwards, the theoretical value is

$$
V^{\text{fwd}}_0 = S_0 e^{-qT} - K e^{-rT}.
$$

For the puts, the Black-Scholes-Merton benchmark is

$$
P_0 = K e^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1),
$$

with

$$
d_1=\frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_2=d_1-\sigma\sqrt{T}.
$$

If those quantities line up with the Monte Carlo confidence intervals, the simulation engine is behaving correctly.

## Exposure

For each trade $i$, the positive exposure is

$$
E_i(t)=\max(V_i(t),0).
$$

Without netting, portfolio exposure is

$$
E_{\text{unnet}}(t)=\sum_i \max(V_i(t),0).
$$

With netting, the exposure becomes

$$
E_{\text{net}}(t)=\max\left(\sum_i V_i(t),0\right).
$$

The expected exposure profile is just the time-$t$ cross-sectional average of those positive exposures over Monte Carlo paths.

## CVA Formula

Using discrete time points $t_1,\dots,t_n$, CVA is approximated by

$$
\mathrm{CVA}=LGD\sum_{j=1}^{n} DF(t_j)\,EE(t_j)\,PD(t_{j-1},t_j),
$$

where:

- $EE(t_j)$ is the expected positive exposure,
- $DF(t_j)=e^{-rt_j}$,
- $PD(t_{j-1},t_j)=S(t_{j-1})-S(t_j)$ comes from the survival curve.

The hazard curve is piecewise constant over the buckets:

- $[0,1]$,
- $(1,3]$,
- $(3,5]$.

That gives a clean term structure for counterparty default risk.

## Why Netting Matters

This portfolio is naturally offsetting:

- long forwards gain when equities rise,
- long puts gain when equities fall.

Because of that, netting matters a lot. It allows positive MTM on one trade to be reduced by negative MTM on another before exposure is floored at zero. The project therefore shows both:

- standalone trade CVA,
- unnetted portfolio CVA,
- netted portfolio CVA.

That comparison is the cleanest way to show the economic value of a master netting agreement.

## Sensitivity Analysis

The project then stresses:

- volatility,
- cross-asset correlation.

Higher volatility widens the distribution of future portfolio values, and because exposure applies a positive-part operator,

$$
E(t)=\max(V_{\text{port}}(t),0),
$$

the effect is convex: more dispersion generally increases expected positive exposure and therefore increases CVA.

Lower correlation has the opposite effect in a netted portfolio. It improves diversification and raises the chance that one trade offsets another.

## Collateral

The collateral section studies both variation margin and initial margin.

If variation margin is updated every $M$ months, exposure becomes

$$
E(t_j)=\max\left(V_{\text{port}}(t_j)-V_{\text{port}}(t_k),0\right),
$$

where $t_k$ is the last collateral update date.

If an initial margin amount $IM$ is posted, the exposure becomes

$$
E(t)=\max(V_{\text{port}}(t)-IM,0).
$$

More frequent variation margin and larger initial margin both reduce the unsecured exposure that feeds into CVA.

## Credit Hedge With CDS

The last part computes first-order CVA sensitivity to bucketed hazard-rate bumps. For each hazard bucket, the project calculates

$$
\Delta \mathrm{CVA} = \mathrm{CVA}_{\text{bumped}} - \mathrm{CVA}_{\text{base}}.
$$

Then it computes the MTM sensitivity of `1Y`, `3Y`, and `5Y` CDS protection-buyer positions under the same hazard bump. The hedge notionals are chosen so that the CDS MTM changes offset the CVA sensitivity bucket by bucket.

That makes the hedge local and first-order:

- it is accurate for small spread moves,
- it must be rebalanced as the exposure profile and hazard curve change.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script writes:

- validation tables to `outputs/`,
- CVA tables to `outputs/`,
- collateral and hedge tables to `outputs/`,
- exposure and sensitivity figures to `figures/`.

## Main Takeaway

This project shows that CVA is driven by more than default probabilities alone. The market model, the portfolio payoff structure, the netting set, the collateral agreement, and the hedge design all matter. In this case, volatility is the strongest stress driver, netting gives a meaningful reduction in CVA, and collateral can reduce the charge substantially when updated frequently or supported by sufficient initial margin.
