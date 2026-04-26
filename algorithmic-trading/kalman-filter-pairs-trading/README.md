# Kalman Filter Pairs Trading

This project studies a pairs-trading strategy for `RF` and `SCHW` using a dynamic Kalman-filter hedge ratio instead of a fixed OLS regression. The objective is to let the spread relationship evolve over time and trade only the standardized mispricing implied by the state-space model.

## Project Focus

The core idea is simple: if two related assets move together in the long run, then temporary deviations from that relationship may mean-revert. A static hedge ratio can miss structural changes, especially around stress periods. The Kalman filter replaces the fixed slope with a time-varying state that updates every day as new prices arrive.

This is a cleaner algorithmic-trading setup than a one-shot regression because:

- the hedge ratio is allowed to drift,
- the signal is built from a model-based forecast error,
- uncertainty enters directly through the prediction variance,
- the trading rule can react to changing regimes without re-estimating a full regression window each time.

## Data

The script downloads adjusted close prices from Yahoo Finance for:

- `RF` = Regions Financial
- `SCHW` = Charles Schwab

Sample window:

- start: `2020-01-01`
- end: `2025-12-31`

The split is:

- first `75%` of observations: training period
- last `25%`: testing period

## State-Space Model

Let

$$
y_t = \log P^{RF}_t
$$

and

$$
x_t = \log P^{SCHW}_t.
$$

The observation equation is

$$
y_t = \alpha_t + \beta_t x_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, R).
$$

This says that the log price of `RF` is explained by the log price of `SCHW`, but both the intercept and the hedge ratio can change over time.

Define the latent state vector

$$
\theta_t =
\begin{bmatrix}
\alpha_t \\
\beta_t
\end{bmatrix}.
$$

The state transition equation is

$$
\theta_t = \theta_{t-1} + \omega_t, \qquad \omega_t \sim \mathcal{N}(0, Q).
$$

This is a random walk state model. It means:

- if the relation between the assets is stable, updates stay small,
- if the relation shifts, the filter can adapt gradually,
- no arbitrary rolling window is needed.

## Kalman Filter Recursion

Write the design vector as

$$
H_t = \begin{bmatrix} 1 & x_t \end{bmatrix}.
$$

If the filtered state at time $t-1$ is

$$
\theta_{t-1|t-1}, \qquad P_{t-1|t-1},
$$

then the prediction step is

$$
\theta_{t|t-1} = \theta_{t-1|t-1},
$$

$$
P_{t|t-1} = P_{t-1|t-1} + Q.
$$

The one-step-ahead forecast for $y_t$ is

$$
\hat y_{t|t-1} = H_t \theta_{t|t-1}.
$$

The forecast error is

$$
e_t = y_t - \hat y_{t|t-1}.
$$

Its variance is

$$
F_t = H_t P_{t|t-1} H_t^\top + R.
$$

The Kalman gain is

$$
K_t = P_{t|t-1} H_t^\top F_t^{-1}.
$$

The update step is then

$$
\theta_{t|t} = \theta_{t|t-1} + K_t e_t,
$$

$$
P_{t|t} = P_{t|t-1} - K_t H_t P_{t|t-1}.
$$

This is the key logic of the model:

- if the forecast error is small, the state moves little,
- if the forecast error is large and uncertainty is low, the update is stronger,
- if uncertainty is high, the filter is more cautious.

## Why the Signal Uses the Innovation

The raw spread is not enough because it ignores how uncertain the current hedge ratio estimate is. Instead, the strategy uses the standardized forecast error:

$$
z_t = \frac{e_t}{\sqrt{F_t}}.
$$

This is better than using a naive residual because:

- it scales the deviation by model uncertainty,
- it is directly tied to the state-space forecast,
- it gives a dimensionless trading signal across the whole sample.

Interpretation:

- a large positive z-score means `RF` looks rich relative to `SCHW`
- a large negative z-score means `RF` looks cheap relative to `SCHW`

## Trading Rule

The strategy uses a symmetric threshold.

Enter a short spread position when

$$
z_t \ge 1.25.
$$

Enter a long spread position when

$$
z_t \le -1.25.
$$

Exit when the z-score crosses back through zero.

The portfolio weights are based on the filtered hedge ratio:

$$
w^{RF}_t = \frac{1}{1 + |\beta_t|}, \qquad
w^{SCHW}_t = -\frac{\beta_t}{1 + |\beta_t|}.
$$

This keeps the position scaled and lets the short leg adapt with the evolving hedge ratio.

The daily gross return is

$$
r^{gross}_t = s_{t-1}\left(w^{RF}_t \Delta \log P^{RF}_t + w^{SCHW}_t \Delta \log P^{SCHW}_t \right),
$$

where

$$
s_{t-1} \in \{-1,0,1\}
$$

is the lagged trading signal.

The lag matters. It prevents look-ahead bias by ensuring that the signal computed at time $t-1$ is only applied to returns realized at time $t$.

## Transaction Costs

To keep the project focused on the Kalman framework, costs are modeled as a simple fixed turnover charge:

$$
\text{cost}_t = c \cdot |s_t - s_{t-1}|,
$$

with

$$
c = 5 \text{ bps}.
$$

So net return is

$$
r^{net}_t = r^{gross}_t - \text{cost}_t.
$$

This is intentionally simpler than a market-microstructure model because the main objective here is the state-space trading logic, not execution modeling.

## Outputs

The script saves:

- `outputs/prices.csv`
- `outputs/kalman_states.csv`
- `outputs/strategy_returns.csv`
- `outputs/performance_summary.csv`
- `outputs/strategy_diagnostics.csv`

It also saves:

- `figures/log_prices.png`
- `figures/state_paths.png`
- `figures/signals.png`
- `figures/cumulative_returns.png`

## Empirical Result

On the downloaded sample, the Kalman setup produces a dynamic hedge ratio with average absolute value close to `1.00`, which is consistent with the two stocks moving in a fairly similar scale over time.

The backtest is intentionally reported honestly:

- training gross return: about `100.4%`
- training net return: about `89.4%`
- testing gross return: about `-9.1%`
- testing net return: about `-10.2%`

This is useful rather than embarrassing. It shows that:

- the state-space logic is strong enough to produce a coherent in-sample mean-reversion strategy,
- the out-of-sample deterioration is real,
- the project is not just curve-fit presentation,
- the Kalman filter should be viewed as an adaptive estimation tool, not as a guarantee of stable alpha.

That is a better portfolio signal than a smoothed-over result, because the model logic remains clear even when the trading edge weakens.

## How to Run

From this project folder:

```bash
python3 src/analysis.py
```

## Files

- `src/analysis.py`: end-to-end download, filtering, backtest, and output generation
- `notebooks/kalman_filter_pairs_trading.ipynb`: notebook version for review
- `README.md`: mathematical setup and trading logic

## Takeaway

This project uses the Kalman filter as the core modeling device rather than as an add-on. The hedge ratio, forecast error, uncertainty estimate, and trading signal all come from the same state-space system. That makes the strategy internally consistent and easier to explain than a mixed workflow built from static regression plus ad hoc residual rules.
