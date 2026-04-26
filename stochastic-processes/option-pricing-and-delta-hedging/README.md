# Option Pricing and Delta Hedging

This project looks at the same market from two angles: pricing a European call option with different numerical methods, and managing the risk of that option through delta hedging. The focus is on how the methods compare, how close they stay to the Black-Scholes benchmark, and how hedging frequency affects the trader's profit and loss.

## What This Project Covers

- Estimating volatility from S&P 500 data.
- Pricing a call option with Black-Scholes.
- Pricing the same option with a Cox-Ross-Rubinstein binomial tree.
- Monte Carlo pricing with standard Euler and log-Euler discretizations.
- Finite-difference pricing with the Crank-Nicolson method.
- Simulating the P&L of a short-call delta hedge under different rebalancing frequencies.
- Checking how gamma exposure relates to hedging error.

## Repository Layout

```text
option-pricing-and-delta-hedging/
├── data/
│   └── SP500.csv
├── figures/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Mathematical Idea

The benchmark price comes from the Black-Scholes model, where the stock is assumed to follow geometric Brownian motion:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t.
$$

Under the risk-neutral measure, the drift becomes the risk-free rate $r$, which gives the closed-form European call price. That closed form is useful because it lets us judge how well the numerical methods converge.

The project then compares three numerical pricing approaches:

- a recombining binomial tree,
- Monte Carlo simulation,
- a Crank-Nicolson finite-difference solver.

For hedging, the key object is the Black-Scholes delta:

$$
\Delta = \frac{\partial C}{\partial S}.
$$

If a trader is short calls, they can hedge by buying shares according to the option delta and rebalancing through time. In practice that hedge is imperfect because rebalancing is discrete rather than continuous. This project measures that error directly through simulated P&L distributions.

## Main Workflow

### 1. Market Calibration

The script reads monthly S&P 500 data and estimates volatility from log returns. That volatility is annualized and used consistently across the pricing and hedging sections.

### 2. Pricing Comparison

The same call option is priced using:

- Black-Scholes,
- a CRR binomial tree,
- Euler Monte Carlo,
- log-Euler Monte Carlo,
- Crank-Nicolson finite differences.

This makes it easy to compare convergence, discretization error, and numerical stability.

### 3. Delta Hedging Simulation

The project then switches from valuation to risk management. It assumes a trader sells call options and delta-hedges the position over the life of the contract. The script compares:

- monthly hedging,
- weekly hedging,
- daily hedging,
- daily hedging under different assumed physical drifts.

The output is not just one number. It is a distribution of hedging outcomes, which is much closer to how the problem is viewed in practice.

### 4. Gamma and Hedging Error

To make the hedging results more interpretable, the script also tracks average gamma exposure. That helps explain why some scenarios produce larger residual P&L even when the position is delta-neutral at each rebalance date.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script prints pricing and hedging summaries and saves plots to `figures/`.

## Why I Kept This Project

This one is worth keeping in the portfolio because it shows more than formula application. It combines financial modeling, numerical methods, and trading-risk intuition in one project. That gives it stronger signal than a standard coursework notebook, especially for roles related to derivatives, quant research, or risk.
