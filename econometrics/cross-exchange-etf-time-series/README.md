# Cross-Exchange ETF Time-Series Modeling

This project studies how the same S&P 500 ETF behaves across three trading venues: `SPY5.P`, `SPY5.SIX`, and `SPY5l.CHIX`. The focus is on short-run return dynamics, cross-exchange dependence, cointegration in prices, and time-varying volatility.

## Project Focus

This project brings together several time-series tools in one coherent setting:

- ARMA model selection for daily closing returns,
- VAR forecasting for aligned intraday returns,
- VECM estimation for long-run price relationships,
- GARCH and EGARCH modeling for daily volatility,
- realized variance built directly from intraday returns.

The original assignment had good material but too much notebook-style trial and error. The version here keeps the core ideas while making the workflow easier to follow and the outputs easier to review.

## Repository Layout

```text
cross-exchange-etf-time-series/
├── data/
│   └── sp_g18.csv.gz
├── figures/
├── notebooks/
│   └── cross_exchange_time_series.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Modeling Idea

For returns, I work with log differences:

$$
r_t = 100 \cdot \left(\log P_t - \log P_{t-1}\right).
$$

The univariate mean model is an ARMA specification,

$$
r_t = \mu + \phi_1 r_{t-1} + \cdots + \phi_p r_{t-p} + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q} + \varepsilon_t.
$$

For the multivariate return system, the intraday dynamics are modeled with a VAR:

$$
y_t = c + A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t,
$$

where $y_t$ contains the aligned returns from the three venues.

Since price levels across venues should move together over time, I also estimate a VECM:

$$
\Delta y_t = \alpha \beta' y_{t-1} + \Gamma_1 \Delta y_{t-1} + \cdots + \Gamma_k \Delta y_{t-k} + u_t.
$$

Here, $\beta' y_{t-1}$ captures long-run equilibrium relations and $\alpha$ captures how each venue adjusts when prices drift away from that equilibrium.

Finally, daily conditional variance is modeled through GARCH-family processes:

$$
\varepsilon_t = \sigma_t z_t, \qquad z_t \sim \mathcal{N}(0,1),
$$

with volatility equations such as

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.
$$

## Data Design

The project uses the processed intraday file `sp_g18.csv.gz`, which already stacks minute-level prices for the three venues. I use different frequencies for different tasks:

- daily closing returns for ARMA and GARCH,
- 15-minute aligned returns for VAR,
- sampled 15-minute log prices for VECM,
- full intraday returns to construct daily realized variance.

This keeps the analysis computationally stable while preserving the economic structure of the problem.

## Main Workflow

### 1. Univariate Return Dynamics

I start with daily closing returns for `SPY5.SIX`, winsorize the most extreme observations, and estimate all ARMA$(p,q)$ combinations for $p,q \in \{0,1,2\}$. The models are compared using:

- AIC,
- BIC,
- log-likelihood,
- out-of-sample mean squared error,
- Ljung-Box diagnostics.

### 2. Cross-Exchange Return Spillovers

To compare the three venues jointly, I align prices on a 15-minute grid and estimate:

- `VAR(2)`,
- `VAR(5)`.

The point is to compare fit and out-of-sample forecasting error while keeping the economic question clear: which venue seems to move first, and how much extra lag structure is really useful?

### 3. Cointegration In Prices

Because the three ETF listings track the same underlying market, their log-price levels should not drift apart indefinitely. The VECM section checks:

- non-stationarity of price levels,
- cointegration rank,
- adjustment speeds across venues.

This section is more about long-run equilibrium than short-run forecasting.

### 4. Volatility And Realized Variation

Daily volatility is modeled with both GARCH and EGARCH variants. I then compare the fitted conditional variance against realized variance constructed from intraday squared returns:

$$
\widehat{\sigma}^2_{RV,t} = \sum_i r_{i,t}^2.
$$

That makes it possible to compare model-implied variance with a direct market-based volatility proxy.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script writes:

- model tables to `outputs/`,
- figures to `figures/`,
- a clean summary to the terminal.

## Main Results

The cleaned version gives a few clear takeaways:

- among the ARMA candidates, low-order specifications perform similarly, but a slightly richer model such as ARMA(1,2) can edge out the rest on AIC,
- `VAR(2)` and `VAR(5)` perform almost identically out of sample, which suggests little value in the extra lag depth,
- the three log-price series are non-stationary individually but remain cointegrated jointly,
- the VECM adjustment coefficients confirm that deviations from cross-venue equilibrium are corrected over time,
- the volatility comparison shows that GARCH-type models smooth market stress, while realized variance reacts more sharply to isolated intraday bursts.

## Presentation Notes

This project is presented as a compact empirical study rather than a full assignment report. The notebook provides a readable walkthrough, while the script is the reproducible core. That makes the logic, modeling choices, and results much easier to review on GitHub.
