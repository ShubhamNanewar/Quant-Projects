# CDS Hazard Rate Bootstrapping

This project strips default intensities from a CDS term structure using both a simplified hazard-rate approximation and an exact quarterly-premium bootstrap. The point is to see how market CDS spreads translate into survival probabilities, forward default rates, and contract-level valuation consistency.

## Project Focus

The workflow has two layers:

- a simple continuous-premium approximation that gives quick intuition,
- an exact quarterly-payment model with accrued premium and discounting.

The second model matters more in practice because CDS contracts do not pay continuously and because discounting changes the shape of the implied hazard curve.

## Repository Layout

```text
cds-hazard-rate-bootstrapping/
├── figures/
├── notebooks/
│   └── cds_hazard_rate_bootstrapping.ipynb
├── outputs/
├── requirements.txt
├── src/
│   └── analysis.py
└── README.md
```

## Simple Model

The simple model assumes:

- continuous premium payments,
- zero discounting or negligible timing effects,
- piecewise-constant hazard rates.

Under those assumptions, the fair CDS spread satisfies the approximation

$$
R(T) \approx \lambda_{\text{avg}}(T)\cdot LGD,
$$

so the average hazard rate is

$$
\lambda_{\text{avg}}(T) = \frac{R(T)}{LGD}.
$$

The cumulative hazard is

$$
H(T) = \lambda_{\text{avg}}(T)\,T,
$$

and the survival probability is

$$
Q(T) = \mathbb{P}(\tau > T) = e^{-H(T)}.
$$

From that, the forward hazard in bucket $(T_{i-1}, T_i]$ is

$$
\lambda_{\text{fwd}}(T_{i-1},T_i) =
\frac{H(T_i)-H(T_{i-1})}{T_i-T_{i-1}}.
$$

The unconditional default probability within the bucket is

$$
\mathbb{P}(T_{i-1}<\tau\le T_i)=Q(T_{i-1})-Q(T_i).
$$

This model is useful because it makes the term structure easy to interpret, even if it is not the most accurate pricing setup.

## Exact Model

The exact model removes the main simplifications:

- premiums are paid quarterly,
- accrued premium is paid if default occurs between dates,
- cash flows are discounted,
- hazard rates are bootstrapped one maturity bucket at a time.

Let payment dates be $t_1,\dots,t_N$ with spacing $\Delta t=0.25$. Then:

Regular premium leg:

$$
PV_{\text{regular}} = R\sum_{k=1}^{N}\Delta t \, Z(t_k)\,Q(t_k),
$$

Accrued premium leg:

$$
PV_{\text{accrued}} \approx
R\sum_{k=1}^{N}\frac{\Delta t}{2}\,Z(t_k^{mid})\left(Q(t_{k-1})-Q(t_k)\right),
$$

Protection leg:

$$
PV_{\text{protection}} \approx
LGD\sum_{k=1}^{N} Z(t_k^{mid})\left(Q(t_{k-1})-Q(t_k)\right).
$$

The par CDS condition is

$$
PV_{\text{regular}} + PV_{\text{accrued}} - PV_{\text{protection}} = 0.
$$

For each maturity, the project solves for the bucket hazard rate with root finding while keeping previously solved buckets fixed. That is the actual stripping step.

## Why Bootstrapping Changes The Numbers

Compared with the simple model, the exact model usually produces slightly different forward hazards because:

- future premium cash flows are discounted,
- premium accrual matters in default states,
- each bucket is solved recursively rather than backed out from a closed-form shortcut.

So the bootstrap hazard curve is the more internally consistent one for pricing and risk.

## Validation Logic

The project validates the stripped curve by re-pricing the 7-year CDS with the bootstrapped hazard rates. If the premium and protection legs offset almost exactly, then the stripped curve is internally consistent with the market quotes.

This is a more meaningful check than only reporting the hazard table, because it confirms that the calibration actually prices the contract at par.

## Running The Analysis

From the project root:

```bash
pip install -r requirements.txt
python3 src/analysis.py
```

The script writes:

- the simple-model table to `outputs/`,
- the exact-model bootstrap table to `outputs/`,
- the 7Y validation table to `outputs/`,
- the high-rate sensitivity rerun to `outputs/`,
- the hazard-term-structure plot to `figures/`.

## Main Takeaway

The simple model is good for intuition, but the exact model is the one to trust for pricing. Once discrete payments, accrual, and discounting are added, the hazard curve adjusts slightly across maturities, and those adjustments matter if the curve is later used for CVA, CDS valuation, or counterparty-risk applications.
