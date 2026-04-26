# Multi-Asset VaR And Expected Shortfall

This project studies daily market risk for a multi-asset EUR portfolio containing European equities, US assets translated into EUR, an equity-index block, and a fixed-rate loan proxy. The final version now mirrors the full assignment rather than only a reduced subset of models.

The project compares five approaches:

- variance-covariance under normality,
- variance-covariance with Student-t tails,
- historical simulation,
- EWMA filtered historical simulation,
- CCC-GARCH volatility forecasting.

It also studies:

- one-day VaR and expected shortfall,
- multiday `1/5/10`-day VaR,
- square-root-of-time scaling,
- out-of-sample backtesting,
- stress testing.

## Project Focus

The core question is not just "what is the VaR number?" but whether the tail model is economically credible and statistically usable.

This matters because:

- a normal model is easy to estimate but often too thin-tailed,
- a Student-t model allows fatter tails but still keeps the covariance structure tractable,
- historical simulation is fully non-parametric but depends heavily on the chosen sample window,
- EWMA filtering adapts volatility before recycling past shocks,
- CCC-GARCH produces a conditional covariance forecast instead of a static covariance estimate.

The portfolio therefore becomes a good test case for comparing distributional assumptions, model risk, and out-of-sample backtest behavior.

## Portfolio Construction

The cleaned return file contains the final investable EUR return series:

- `ASML`
- `SHELL`
- `JPM_EUR`
- `STOXX50`
- `SP500_EUR`
- `LOAN`

The selected final weights are

$$ w = [0.16,\ 0.08,\ 0.24,\ 0.08,\ 0.24,\ 0.20]^\top. $$

which sum to one and represent a EUR `1,000,000` portfolio.

The risky-asset block was originally motivated by mean-variance optimization on the pre-sample period, while the `LOAN` block acts as the fixed-income component.

## Return And Loss Setup

If the daily simple return vector is $r_t$, then the portfolio return is

$$ r_{p,t} = w^\top r_t. $$

Daily EUR loss is defined as

$$ L_t = -V_0 r_{p,t}. $$

where $V_0 = 1{,}000{,}000$.

Large positive values of $L_t$ therefore correspond to bad days for the portfolio.

## Historical Simulation

Historical simulation uses the empirical loss distribution directly.

For confidence level $1-\alpha$, the one-day VaR is the empirical quantile

$$ VaR_{\alpha}^{HS} = Q_{1-\alpha}(L). $$

and expected shortfall is the mean of losses beyond that quantile:

$$ ES_{\alpha}^{HS} = E[L \;|\; L \ge VaR_{\alpha}^{HS}]. $$

This method is fully data-driven. It captures skewness and kurtosis automatically, but it assumes the future will look like the past sample.

## EWMA Filtered Historical Simulation

Filtered historical simulation first rescales returns by a time-varying volatility estimate and then re-inflates them using the current volatility forecast.

For each asset return $r_t$, the EWMA variance recursion is

$$ \sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2. $$

with

$$ \lambda = 0.94. $$

Standardized residuals are then

$$ z_t = \frac{r_t}{\sigma_t}. $$

The next-day volatility forecast is

$$ \sigma_{t+1|t} = \sqrt{\lambda \sigma_t^2 + (1-\lambda) r_t^2}. $$

Filtered historical scenarios are built as

$$ r_{t+1}^{(sim)} = z_t \sigma_{t+1|t}. $$

This keeps the empirical shock shape from the historical sample while updating the scale to current market volatility.

## CCC-GARCH

The GARCH block follows the constant-correlation setup used in the original notebook.

For each asset innovation $a_t$, the univariate GARCH(1,1) variance is

$$ \sigma_t^2 = \omega + \alpha a_{t-1}^2 + \beta \sigma_{t-1}^2. $$

After fitting each asset separately, the standardized innovations are

$$ \tilde a_t = \frac{a_t}{\sigma_t}. $$

Their sample correlation matrix is

$$ P = Corr(\tilde a_t). $$

The conditional covariance matrix is then reconstructed as

$$ \Sigma_t = D_t P D_t. $$

where $D_t$ is the diagonal matrix of conditional volatilities.

This is the CCC-GARCH logic:

- volatility is time-varying for each asset,
- correlations are held constant,
- the full portfolio covariance matrix updates every day through the diagonal volatility matrix.

Portfolio conditional volatility is

$$ \sigma_{p,t} = \sqrt{w^\top \Sigma_t w}. $$

That conditional volatility then feeds directly into the normal VaR and ES formulas.

## Normal Variance-Covariance Model

Let $\mu$ be the mean vector and $\Sigma$ the covariance matrix of daily returns. Then portfolio mean and volatility are

$$ \mu_p = w^\top \mu. $$

$$ \sigma_p = \sqrt{w^\top \Sigma w}. $$

Under a normal assumption, one-day VaR at tail probability $\alpha$ is

$$ VaR_{\alpha}^{N} = V_0(-\mu_p - \sigma_p z_\alpha). $$

where $z_\alpha = \Phi^{-1}(\alpha)$.

Expected shortfall becomes

$$ ES_{\alpha}^{N} = V_0\left(-\mu_p + \sigma_p \frac{\phi(z_\alpha)}{\alpha}\right). $$

This model is easy to estimate and transparent, but it tends to understate extreme losses when return tails are too heavy.

## Student-t Tail Model

To allow fatter tails, the project also uses a unit-variance Student-t distribution with degrees of freedom $\nu$.

The portfolio standard deviation is still estimated from the covariance matrix, but the tail quantile comes from the Student-t law. With

$$ q_\alpha = t_\nu^{-1}(\alpha). $$

and scale adjustment

$$ \tilde{\sigma}_p = \sigma_p \sqrt{\frac{\nu-2}{\nu}}. $$

the VaR formula becomes

$$ VaR_{\alpha}^{t} = V_0(-\mu_p - \tilde{\sigma}_p q_\alpha). $$

Expected shortfall is

$$ ES_{\alpha}^{t} = V_0\left(-\mu_p + \tilde{\sigma}_p \frac{\nu + q_\alpha^2}{(\nu-1)\alpha} f_\nu(q_\alpha)\right). $$

where $f_\nu$ is the Student-t density.

## Why The Degrees Of Freedom Are Chosen This Way

Rather than selecting $\nu$ arbitrarily, the project pools standardized residuals from the pre-sample period and compares them with theoretical unit-variance Student-t quantiles.

For each candidate $\nu \in \{3,4,5,6\}$, the fit is measured by

$$ RMSE(\nu) = \sqrt{\frac{1}{n}\sum_{i=1}^n \left(z_{(i)} - q_{(i)}^{(\nu)}\right)^2}. $$

where:

- $z_{(i)}$ are sorted empirical standardized residuals,
- $q_{(i)}^{(\nu)}$ are the corresponding theoretical Student-t quantiles.

The best-fitting value is the one with the smallest QQ-plot RMSE.

In this sample, the selected value is `df = 5`, which is a reasonable sign of heavier-than-normal tails without moving into extremely unstable tail estimates.

## Backtesting Logic

For each day in the out-of-sample period, the project re-estimates moments using an expanding window of past data only. This avoids look-ahead bias.

If the one-day-ahead VaR forecast for date $t$ is $VaR_{\alpha,t}$ and realized loss is $L_t$, then the violation indicator equals `1` when $L_t > VaR_{\alpha,t}$ and `0` otherwise.

The backtest checks:

- total hit count,
- hit rate,
- yearly violation clustering,
- whether predicted ES is close to realized average shortfall on hit days.

For the historical and filtered-historical models, the project also reports:

- binomial violation tests,
- ES residual t-tests,
- an elicitable VaR scoring rule.

That makes the comparison more operational than just counting breaches.

## Multiday VaR

The project also compares direct historical multiday VaR with square-root-of-time scaling.

For horizon $h$, non-overlapping compounded returns are constructed as

$$ R_{t,t+h} = \prod_{j=0}^{h-1}(1+r_{t+j}) - 1. $$

The direct historical VaR of these block returns is then compared with

$$ VaR_{h}^{SROT} = \sqrt{h}\,VaR_{1}. $$

This shows when iid-style scaling is too optimistic or too conservative relative to actual realized multiday tail losses.

## Stress Testing

The project also reports deterministic factor shocks for:

- equities and indices,
- EUR/USD,
- EURIBOR.

For the loan block, a duration approximation is used:

$$ \Delta P \approx - D_{mod} P \Delta y. $$

This connects the rate shock directly to the fixed-income mark-to-market effect.

## Main Empirical Result

The main model comparison is now clearer:

- the normal model is too optimistic in the tail,
- the Student-t model is more conservative,
- historical simulation gives a useful non-parametric benchmark,
- EWMA-filtered historical simulation improves the calibration of recent-volatility regimes,
- CCC-GARCH adds a conditional covariance view of risk rather than a single unconditional covariance estimate.

At the `1%` portfolio level, the notebook results are close to:

- Normal VaR: about `€22.6k`
- Normal ES: about `€26.0k`
- Student-t VaR: about `€25.4k`
- Student-t ES: about `€33.8k`

That ranking is exactly what we would expect if the sample has tail heaviness.

Backtesting supports that interpretation:

- normal VaR at the `1%` level gives `46` hits over `2366` observations,
- Student-t reduces that to `33`,
- historical simulation gives `27`,
- EWMA-filtered historical simulation gives `18`.

So the EWMA-filtered historical model is the most stable of the four one-day approaches in this sample. That is consistent with the idea that volatility updating matters as much as tail shape.

The multiday comparison is also useful:

- for `5` days, square-root-of-time scaling slightly overshoots direct historical VaR,
- for `10` days, the gap can flip sign depending on the confidence level,
- so naive horizon scaling should not be treated as mechanically reliable.

## Outputs

The script writes:

- `outputs/portfolio_weights_summary.csv`
- `outputs/student_t_df_selection.csv`
- `outputs/var_es_table.csv`
- `outputs/backtest_summary.csv`
- `outputs/hs_fhs_es_residual_tests.csv`
- `outputs/yearly_violations.csv`
- `outputs/yearly_es_shortfall.csv`
- `outputs/stress_scenarios.csv`
- `outputs/ewma_volatility_paths.csv`
- `outputs/ccc_garch_volatility_paths.csv`
- `outputs/ccc_garch_next_day_risk.csv`
- `outputs/ccc_garch_yearly_backtest.csv`
- `outputs/multiday_var_scaling.csv`

and figures:

- `figures/portfolio_growth.png`
- `figures/student_t_qq_plots.png`
- `figures/var_backtest.png`
- `figures/rolling_var_violations.png`
- `figures/hs_fhs_backtest_violations.png`
- `figures/garch_portfolio_risk.png`
- `figures/multiday_var_comparison.png`

## How To Run

From this project folder:

```bash
python3 src/analysis.py
```

## Files

- `src/analysis.py`: cleaned end-to-end risk analysis
- `notebooks/multi_asset_var_and_expected_shortfall.ipynb`: notebook version for review
- `README.md`: math, logic, and interpretation

## Takeaway

This project is stronger in its full form because it does not stop at one parametric model. It moves from static covariance to heavy-tailed parametrics, filtered historical scenarios, conditional GARCH volatility, horizon scaling, and stress testing in one coherent portfolio-risk workflow.
