# Model Formulation: Economic Dispatch LP and Its Dual

## Method A — Analytic stacking

Walk up the merit-order stack (technologies sorted by ascending SRMC) until cumulative available capacity meets residual demand. The SRMC of the last committed unit is the modelled clearing price.

Simple, transparent, and produces the correct price for a single-period dispatch with no minimum-generation constraints.

## Method B — Economic dispatch linear programme

For each hour `t`, solve:

```
minimise    Σ_i  SRMC_i · g_i

subject to  Σ_i  g_i  =  D_residual,t          (demand balance)
            0  ≤  g_i  ≤  Cap_i                 for all i
```

**Decision variables:** `g_i` ∈ ℝ — generation (MWh) per technology `i` in hour `t`.

**Objective:** minimise total variable cost of meeting demand.

**Constraints:**
- Demand balance (equality): supply must exactly equal residual demand.
- Capacity bounds: generation is non-negative and bounded by installed MW.

### Dual and the system marginal price

The Lagrange multiplier (shadow price) `λ_t` of the demand-balance constraint is the **system marginal price** (SMP):

```
λ_t  =  ∂(min cost) / ∂(D_residual,t)
```

In words: the marginal cost of serving one additional MWh of demand at hour `t`. With binding capacity constraints only, the LP dual equals the SRMC of the marginal unit, so Methods A and B agree.

`λ_t` is extracted from PuLP via `prob.constraints["balance"].pi`.

### Why formulate as an LP if A and B agree?

1. The LP dual gives the clearing price cleanly and automatically, without walking the stack.
2. It is the industry-standard formulation and scales to larger problems.
3. Extensions (minimum generation, ramp constraints, interconnections) modify the LP naturally but cannot be grafted onto Method A without redesigning it.
4. A clean LP implementation on the portfolio is a direct signal of operations-research capability.

## Implementation

`src/dispatch.py`:
- `dispatch_analytic()` — Method A, one hour
- `dispatch_lp()` — Method B, one hour (PuLP, CBC solver)
- `run_dispatch()` — loops over the full panel, returns dispatched mix + price

Solver: PuLP with CBC (open-source, no licence needed). Can be swapped for Gurobi or CPLEX by changing the `prob.solve()` call.
