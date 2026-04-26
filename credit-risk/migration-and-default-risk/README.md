# Migration And Default Risk

This project studies portfolio credit losses under rating migration and default using a one-factor Gaussian latent-variable model. The main focus is how asset correlation changes tail risk for concentrated and diversified credit portfolios.

## Project Focus

The project has two portfolio experiments:

- a concentrated portfolio with one issuer per rating bucket,
- a diversified portfolio with `100` issuers per rating bucket.

Across both cases, the key question is how joint credit deterioration changes as the common systematic factor becomes more important.

## Repository Layout

```text
migration-and-default-risk/
├── figures/
├── notebooks/
│   └── migration_and_default_risk.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Model Setup

The project uses a one-factor Gaussian migration model. For issuer $i$,

$$
X_i = \sqrt{\rho}\,Y + \sqrt{1-\rho}\,\varepsilon_i,
$$

where:

- $Y \sim N(0,1)$ is the systematic factor,
- $\varepsilon_i \sim N(0,1)$ is the idiosyncratic shock,
- $\rho \in [0,1]$ is the asset correlation.

This implies

$$
\mathrm{Corr}(X_i, X_j) = \rho \qquad (i \neq j).
$$

So larger $\rho$ means issuers react more similarly to the same macro shock, which increases the chance of clustered downgrades and defaults.

## Transition-Matrix Calibration

The one-period transition matrix gives probabilities

$$
p_{G \to r}
$$

for migration from initial rating $G$ to end-of-period rating $r$.

To map the latent variable $X$ into rating states, the project converts cumulative transition probabilities into Gaussian thresholds:

$$
\tau_{G,k} = \Phi^{-1}\left(\sum_{j=1}^{k} p_{G \to r_j}\right),
$$

where $\Phi^{-1}$ is the inverse standard normal CDF.

Then the migrated rating is determined by which threshold interval contains $X$.

This is the standard latent-threshold interpretation of migration risk: the credit state is a discretized version of an underlying continuous credit-quality variable.

## Valuation Step

Each rating bucket has:

- a current value $V_0(G)$,
- a next-period value $V_1(r)$ after migration to rating $r$.

If the issuer starts in bucket $G$ and ends in bucket $r$, the value factor is

$$
F = \frac{V_1(r)}{V_0(G)}.
$$

If a bucket weight is $w_G$ and total initial portfolio value is $PV_0$, then the end-of-period contribution is

$$
MV_{1,G} = w_G PV_0 \cdot F.
$$

For the diversified case with `100` issuers in each rating bucket, the bucket contribution is averaged across those issuers before aggregating across rating classes.

## Portfolio Loss And Risk Measures

Portfolio loss is defined as

$$
L = PV_0 - PV_1.
$$

The project reports:

$$
\text{VaR}_\alpha = \text{empirical } \alpha\text{-quantile of } L,
$$

and

$$
\text{ES}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha].
$$

I compute both `90%` and `99.5%` versions, because the interesting part of the problem is the right tail of the loss distribution rather than the mean.

## Why Correlation Matters

At low $\rho$, idiosyncratic shocks dominate, so one issuer can migrate badly without forcing the rest of the portfolio into the same state. Diversification remains effective.

At high $\rho$, the common factor drives most of the movement, so bad scenarios hit multiple issuers at once. This pushes probability mass into the loss tail and widens the gap between ordinary losses and extreme losses.

That is why the loss distribution changes shape so visibly across the four scenarios:

- `ρ = 0%`,
- `ρ = 33%`,
- `ρ = 66%`,
- `ρ = 100%`.

## Concentrated Vs Diversified Case

The diversified case keeps the same rating structure but adds `100` issuers per bucket. That changes the loss mechanism:

- idiosyncratic migration risk is diversified within the bucket,
- systematic migration risk remains,
- tail risk still grows with $\rho$ because the common factor cannot be diversified away.

So the comparison between Question 1 and Question 2 is economically useful:

- Question 1 shows the full effect of concentration,
- Question 2 isolates the portion of credit risk that survives even after within-bucket diversification.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script writes:

- risk-metric tables to `outputs/`,
- convergence diagnostics to `outputs/`,
- loss-distribution plots to `figures/`.

## Main Takeaway

The expected portfolio value stays close to the initial value across correlation scenarios, but tail risk does not. That is the important lesson here: migration and default risk behave mainly as a tail phenomenon.

For the weaker-credit portfolio, tail losses grow sharply as $\rho$ rises because defaults and downgrades cluster. Diversification helps a lot in the `100`-issuer case when correlation is low, but once the common factor dominates, the protection from diversification falls away quickly.
