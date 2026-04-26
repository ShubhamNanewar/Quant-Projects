# Housing Price Model Comparison

This project compares price-level and log-price modeling for housing valuation using a structured feature set of property characteristics and amenities. The main question is simple: should the target be modeled in levels or logs, and which regularized linear model performs best out of sample?

## Project Focus

This project combines a standard machine learning workflow with econometric interpretation:

- distribution checks for price versus log price,
- multivariate linear regression as a benchmark,
- Lasso and Ridge tuning without cross-validation,
- out-of-sample comparison under two data splits,
- Diebold-Mariano tests on competing prediction errors.

The original notebook only covered a small exploratory part of the assignment. The cleaned version completes the full workflow and turns it into a concise modeling project.

## Repository Layout

```text
housing-price-model-comparison/
├── data/
│   └── data_ASS_I_group_17.xlsx
├── figures/
├── notebooks/
│   └── housing_price_model_comparison.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Modeling Idea

Let $y_i$ denote the housing price and $x_i$ the feature vector containing area, bedrooms, and binary amenity indicators.

The linear benchmark is

$$
y_i = \beta_0 + x_i^\prime \beta + \varepsilon_i.
$$

The log-price version is

$$
\log(y_i) = \beta_0 + x_i^\prime \beta + \varepsilon_i.
$$

The regularized models solve:

$$
\min_{\beta_0,\beta} \sum_i (y_i - \beta_0 - x_i^\prime \beta)^2 + \lambda \sum_j |\beta_j|
$$

for Lasso, and

$$
\min_{\beta_0,\beta} \sum_i (y_i - \beta_0 - x_i^\prime \beta)^2 + \lambda \sum_j \beta_j^2
$$

for Ridge.

For the log-target case, predicted values are transformed back into price space before evaluation.

The prediction loss metrics are

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
$$

and

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2.
$$

To compare two forecasting models more formally, I also use a Diebold-Mariano-style test based on the loss differential

$$
d_i = e_{1,i}^2 - e_{2,i}^2,
$$

where $e_{j,i} = y_i - \hat{y}_{j,i}$. The test statistic is the sample mean of $d_i$ divided by its estimated standard error. In this project it is used as a simple holdout comparison of squared prediction errors.

## Data And Preprocessing

The dataset contains:

- `Price` as the target variable,
- `Area` and `No. of Bedrooms` as continuous housing characteristics,
- a large set of 0/1 amenity indicators.

The cleaned version checks:

- missing values,
- summary statistics,
- price and log-price distributions,
- normality with the Jarque-Bera test.

This matters because log-transforming the target may stabilize skewness, but it does not automatically improve prediction performance.

## Main Workflow

### 1. Distribution Check

I compare `Price` and `log(Price)` using:

- histograms,
- Jarque-Bera normality tests,
- a summary table for `Price`, `Area`, and one selected feature.

### 2. Benchmark Regression

The assignment asks for a regression table using log price as the dependent variable. I estimate the full linear model and report:

- the constant,
- the first coefficients,
- their standard errors,
- the training $R^2$.

This keeps the project connected to interpretation rather than treating it as a pure black-box exercise.

### 3. Model Tuning

Using the required holdout setup, I compare three models:

- linear regression,
- Lasso,
- Ridge.

Hyperparameters for Lasso and Ridge are tuned on the validation set only, without cross-validation, exactly as required by the assignment.

### 4. Out-of-Sample Evaluation

The main split uses:

- `60%` training,
- `20%` validation,
- `20%` testing.

The robustness split uses:

- `70%` training,
- `10%` validation,
- `20%` testing.

For both target definitions, I report:

- mean absolute error,
- mean squared error.

### 5. Prediction Error Tests

To compare competing models more formally, I apply a Diebold-Mariano-style test to the squared prediction errors. This is used for:

- the best versus second-best price model,
- the best versus second-best log-price model,
- the best price model versus the best log-price model.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script writes:

- assignment-style output tables to `outputs/`,
- diagnostic figures to `figures/`,
- a compact console summary.

## Main Results

The cleaned workflow gives a fairly clear result on this dataset:

- log price is less skewed than raw price, but normality is still rejected,
- the full linear model on raw prices is already a strong benchmark,
- Ridge performs slightly better than the plain linear model on the price target,
- the log-target models perform materially worse after transforming predictions back into price space,
- the robustness split does not change the recommendation in a meaningful way.

So the practical conclusion is straightforward: for this dataset, modeling `Price` directly is more useful than modeling `log(Price)`.

## Presentation Notes

This project is presented as a compact applied ML study rather than a half-finished class notebook. The notebook explains the logic step by step, while the script is the reproducible core. That makes it easier to review both the modeling choices and the conclusions quickly.
