# UK Default Classification

This project studies corporate default prediction for U.K. firms using a compact set of accounting and market-based features. The main question is whether more flexible classifiers improve meaningfully on a simple logistic benchmark once the class imbalance and model tuning are handled carefully.

## Project Focus

This project combines a standard credit-classification workflow with model comparison and interpretability:

- preprocessing with missing-value removal,
- class-imbalance diagnostics,
- logistic regression as a benchmark,
- tuned decision tree, neural network, and gradient boosting classifiers,
- a small two-feature decision tree for interpretability.

The original notebook contained the right ingredients, but the workflow was repetitive and notebook-heavy. The cleaned version keeps the empirical idea while presenting it more clearly.

## Repository Layout

```text
uk-default-classification/
├── data/
│   └── data.xlsx
├── figures/
├── notebooks/
│   └── uk_default_classification.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Modeling Idea

Let $y_i \in \{0,1\}$ indicate whether firm $i$ defaults. The main features are:

- `wkta`: working capital over total assets,
- `reta`: retained earnings over total assets,
- `ebitta`: earnings before interest and taxes over total assets,
- `mv`: market value type ratio.

The benchmark classifier is logistic regression:

$$
\mathbb{P}(y_i = 1 \mid x_i) = \frac{1}{1 + \exp(-(\beta_0 + x_i^\prime \beta))}.
$$

The project then compares this benchmark with:

- a tuned decision tree,
- a small neural network,
- gradient boosting,
- a reduced two-feature decision tree.

Because defaults are relatively rare, the evaluation is not based on accuracy alone. I use

$$
F_\beta = (1+\beta^2)\frac{\text{precision}\cdot\text{recall}}{\beta^2\cdot\text{precision}+\text{recall}}
$$

with $\beta = 2$, which puts more weight on recall than precision.

## Data And Preprocessing

The cleaned dataset contains:

- `1549` firm-year observations after removing missing values,
- a clear default imbalance,
- a large concentration of observations in year `2001`.

Instead of deleting extreme observations aggressively, I keep them because they are economically meaningful: many of the strongest outliers belong to defaulted firms and therefore carry signal rather than pure noise.

## Main Workflow

### 1. Exploratory Checks

I start with:

- missing-value counts,
- feature distributions,
- the correlation matrix between default status and the financial ratios,
- summary statistics in assignment-style table form.

This establishes the strong imbalance and the broad sign pattern one would expect: healthier firms generally exhibit lower default probability.

### 2. Main Model Comparison

Using `wkta`, `reta`, `ebitta`, and `mv`, I estimate:

- logistic regression,
- decision tree,
- neural network,
- gradient boosting.

The split is:

- `60%` training,
- `20%` validation,
- `20%` test.

To preserve the highly uneven year composition, the split is stratified by year rather than done as a fully naive random cut.

### 3. Hyperparameter Tuning

I tune the following:

- decision tree `max_depth`,
- neural network hidden-layer structure,
- gradient boosting `n_estimators`.

Each model is tuned on the validation set using the $F_2$ score, because missing defaults is more costly than generating extra false alarms.

### 4. Interpretable Small Tree

The final section estimates a reduced decision tree using only:

- `wkta`,
- `reta`.

This provides a more interpretable comparison against a two-feature logistic model and shows whether a simpler rule-based classifier can stay competitive.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script writes:

- summary tables to `outputs/`,
- tuning tables to `outputs/`,
- diagnostic and tree figures to `figures/`.

## Main Results

The cleaned workflow gives a fairly clear pattern:

- logistic regression already performs strongly because the signal in the four financial ratios is quite informative,
- the tuned neural network achieves the highest $F_2$ score among the main models,
- the tuned decision tree and gradient boosting models do not dominate as clearly as one might expect,
- the small two-feature decision tree remains interpretable, but it does not beat the corresponding two-feature logit model.

So the main conclusion is that more flexible machine learning models can help, but the gains over a good linear classification baseline are not dramatic in this dataset.

## Presentation Notes

This project is presented as a compact credit-risk classification study rather than a long assignment notebook. The script is the reproducible core, while the notebook gives a readable walkthrough of the modeling logic and results.
