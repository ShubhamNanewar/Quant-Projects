# Fama-French Panel Analysis Across Sectors

This project rebuilds an econometrics assignment into a cleaner panel-data study of stock returns. I use daily prices, Fama-French factors, and sector labels to study how excess returns behave across four consumer and materials-oriented sectors.

## Project Focus

This project brings together empirical finance and econometrics through:

- panel data construction from large equity price files,
- return alignment with external factor data,
- pooled and fixed-effects regressions,
- firm-level Fama-French loading estimation,
- cross-sector interpretation of market, size, and value exposure.

The original assignment idea is kept intact, but the presentation here is cleaner: the unit handling is consistent, the workflow is easier to follow, and the analysis is organized so the empirical results are clearer to review.

## Repository Layout

```text
fama-french-panel-analysis/
├── data/
│   ├── capmff_2010-2025_ff.csv
│   ├── capmff_2010-2025_prices.csv
│   └── capmff_2010-2025_sector.csv
├── figures/
├── notebooks/
│   └── panel_factor_analysis.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Empirical Idea

The daily stock return for firm $i$ at time $t$ is

$$
R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1.
$$

Using the risk-free rate $r_{f,t}$, the excess return becomes

$$
R^e_{i,t} = R_{i,t} - r_{f,t}.
$$

The main factor model is the three-factor Fama-French specification

$$
R^e_{i,t} = \alpha_i + \beta_{i,m}(MKT_t) + \beta_{i,s}(SMB_t) + \beta_{i,h}(HML_t) + \varepsilon_{i,t}.
$$

Here:

- $MKT_t$ is the market excess return,
- $SMB_t$ captures size exposure,
- $HML_t$ captures value exposure.

The cleaned analysis first estimates pooled exposures across the panel and then compares them with firm-level loadings and ticker fixed-effects results.

## Data Construction

The project keeps the same sector focus used in the original assignment:

- `Basic Materials`
- `Communication Services`
- `Consumer Cyclical`
- `Consumer Defensive`

To keep the panel usable, I retain firms with sufficiently high price coverage and then compute daily simple returns directly from adjusted price levels. The Fama-French series are converted from percent units to decimal units before regression, which is an important correction relative to the raw notebook workflow.

## Main Workflow

### 1. Panel Preparation

The script:

- loads daily prices, factor data, and sector labels,
- filters to the four sectors above,
- removes names with weak price coverage,
- computes daily returns,
- merges returns with Fama-French factors on common dates,
- forms excess returns.

### 2. Pooled Factor Regressions

I estimate:

- a CAPM benchmark,
- a Fama-French three-factor model,
- a sector-effects version with sector dummies.

This gives a clean baseline for whether the extra factors matter and whether average sector intercept shifts remain after controlling for them.

### 3. Ticker Fixed Effects

I then run a model with firm-level fixed effects:

$$
R^e_{i,t} = \alpha_i + \beta_m MKT_t + \beta_s SMB_t + \beta_h HML_t + \varepsilon_{i,t}.
$$

This absorbs time-invariant firm heterogeneity and keeps the focus on common factor exposures.

### 4. Firm-Level Loadings

For each stock with enough observations, I estimate its own three-factor regression. Those firm-level coefficients are then summarized by sector to show how exposures differ across groups.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script saves:

- regression tables in `outputs/`,
- factor-loading summaries in `outputs/`,
- figures in `figures/`.

## What Comes Out Of The Cleaned Version

The cleaned analysis produces a panel with roughly `476k` stock-day observations across `123` firms. A few of the main patterns are:

- the pooled three-factor model improves on CAPM, though only modestly in $R^2$ terms,
- the market factor remains the dominant exposure with an average loading near `0.90`,
- `SMB` and `HML` are both statistically meaningful once the units are handled correctly,
- consumer defensive stocks show meaningfully lower market beta than cyclical and materials names,
- basic materials and consumer cyclical firms tend to carry higher market and size exposure.

## Presentation Notes

This project is intentionally presented as a compact empirical study rather than a coursework dump. The notebook is there for readability and step-by-step explanation, while the script is the reproducible core. That makes it easier for someone reviewing the repository to understand both the econometric logic and the implementation quality quickly.
