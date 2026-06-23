"""
Economic dispatch: Method A (analytic merit-order stacking) and Method B (LP).

Method A — Analytic stacking
    Walk up the merit order (ascending SRMC) until cumulative available capacity
    covers residual demand. The SRMC of the last committed unit is the modelled
    clearing price for that hour. Fast, transparent, gives the intuition.

Method B — Economic dispatch linear programme
    For each hour t:

        minimise    Σ_i  SRMC_i · g_i
        subject to  Σ_i  g_i  =  D_residual,t        (demand balance)
                    0  ≤  g_i  ≤  Cap_i               (capacity bounds)

    The shadow price (dual variable) of the demand-balance constraint equals the
    system marginal price — the marginal cost of serving one additional MWh of demand.

    Implementation note on dual variables
    --------------------------------------
    PuLP with its default CBC solver does not always expose dual variables via the
    .pi attribute. We use scipy.optimize.linprog instead, which always returns the
    dual solution in result.ineqlin (or result.eqlin for equality constraints).
    scipy.linprog uses the revised simplex method for small LPs (< ~1000 variables),
    which is exact and fast for the 6-variable problems here.

    The dual of the demand-balance equality constraint is extracted as:
        lambda_t = result.eqlin.marginals[0]

    This equals the SRMC of the marginal unit (the one whose capacity bound is neither
    0 nor fully binding) — consistent with Method A. Both methods are run and their
    prices are compared as a consistency check.

Methods agree because
    In a single-period LP with only capacity bounds (no minimum generation, no ramp
    constraints), the optimal solution is the same as analytic stacking. The LP earns
    its place as (a) the generalisable OR formulation, (b) the dual giving the price
    automatically, and (c) the basis for extensions (must-run, ramps, interconnection).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent

# Default installed capacity (MW) for the NL stack.
# Used when capacity is not passed explicitly.
# Source: ENTSO-E A68 for NL 2023, approximate annual average.
# The pipeline extracts this from the API (data_fetch.extract_capacity_from_entsoe);
# these defaults serve as a fallback and for standalone testing.
DEFAULT_CAPACITY = {
    "nuclear":    485,
    "lignite":      0,
    "hard_coal":  2800,
    "ccgt":       10600,
    "ocgt":        1900,
    "oil":          200,
}


# ---------------------------------------------------------------------------
# Method A: analytic stacking
# ---------------------------------------------------------------------------

def dispatch_analytic(
    residual_demand: float,
    srmc_row: pd.Series,
    capacity: dict,
) -> tuple[float, str, dict]:
    """
    Walk the merit order (ascending SRMC) and commit units until demand is met.

    If total installed capacity is insufficient (residual_demand > sum of capacities),
    the function exhausts all units and returns price = SRMC of the most expensive
    unit dispatched, with marginal_tech = 'capacity_short'. This flags scarcity hours.

    Parameters
    ----------
    residual_demand : float       — MWh to be covered by the thermal stack
    srmc_row        : pd.Series  — {tech: SRMC_EUR_MWh} for this hour
    capacity        : dict        — {tech: available_MW}

    Returns
    -------
    price         : float — modelled clearing price (EUR/MWh_e)
    marginal_tech : str   — technology that sets the price
    dispatch      : dict  — {tech: MW dispatched}
    """
    order = srmc_row.sort_values()
    remaining = float(residual_demand)
    dispatch  = {t: 0.0 for t in order.index}
    price     = np.nan
    marginal_tech = "capacity_short"

    for tech, cost in order.items():
        cap = float(capacity.get(tech, 0.0))
        if cap <= 0.0:
            continue
        gen = min(remaining, cap)
        dispatch[tech] = gen
        remaining -= gen
        price = cost
        marginal_tech = tech
        if remaining <= 1e-3:   # tolerance for floating-point residuals
            break

    if remaining > 1e-3:
        marginal_tech = "capacity_short"

    return price, marginal_tech, dispatch


# ---------------------------------------------------------------------------
# Method B: LP dispatch via scipy.linprog
# ---------------------------------------------------------------------------

def dispatch_lp(
    residual_demand: float,
    srmc_row: pd.Series,
    capacity: dict,
) -> tuple[float, dict]:
    """
    Solve the economic dispatch LP for one hour using scipy.optimize.linprog.

    The LP has one equality constraint (demand balance) and box constraints on g_i.
    scipy returns the dual of the equality constraint in result.eqlin.marginals[0],
    which is the system marginal price λ_t.

    Parameters
    ----------
    residual_demand : float
    srmc_row        : pd.Series  — {tech: SRMC}
    capacity        : dict       — {tech: MW}

    Returns
    -------
    price    : float — system marginal price (EUR/MWh_e); NaN if infeasible
    dispatch : dict  — {tech: MW dispatched}
    """
    techs = [t for t in srmc_row.index if capacity.get(t, 0.0) > 0.0]
    if not techs:
        return np.nan, {}

    c      = np.array([srmc_row[t] for t in techs], dtype=float)
    bounds = [(0.0, float(capacity[t])) for t in techs]

    # Equality constraint: sum(g_i) = D_residual
    A_eq = np.ones((1, len(techs)))
    b_eq = np.array([float(residual_demand)])

    result = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",   # HiGHS solver; available in scipy >= 1.7; always returns duals
    )

    if result.status != 0:
        return np.nan, {t: np.nan for t in techs}

    dispatch_out = {t: float(result.x[i]) for i, t in enumerate(techs)}

    # Dual of the equality constraint = system marginal price
    # result.eqlin.marginals is a 1-element array
    price = float(result.eqlin.marginals[0])

    return price, dispatch_out


# ---------------------------------------------------------------------------
# Full-panel dispatch loop
# ---------------------------------------------------------------------------

def run_dispatch(
    panel: pd.DataFrame,
    srmc_stack: pd.DataFrame,
    capacity: dict | None = None,
    method: str = "both",
) -> pd.DataFrame:
    """
    Run dispatch over every hour in the panel.

    Parameters
    ----------
    panel      : hourly panel from data_fetch.build_panel()
    srmc_stack : per-hour SRMC per technology from stack.build_stack()
    capacity   : {tech: MW}; uses DEFAULT_CAPACITY if None
    method     : 'A', 'B', or 'both'

    Returns
    -------
    DataFrame indexed by timestamp with columns:
        price_a          (Method A clearing price)
        marginal_tech_a  (technology that sets the price, Method A)
        price_b          (Method B LP dual = system marginal price)
        dispatched_{tech} for each technology (from Method B)

    Runtime: approximately 30–90 seconds for 8,760 hours depending on hardware.
    The LP has 6 variables and 1 equality constraint per hour — trivially small.
    """
    if capacity is None:
        capacity = DEFAULT_CAPACITY.copy()

    results = []
    n = len(panel)

    for i, (ts, row) in enumerate(panel.iterrows()):
        if i % 2000 == 0:
            print(f"  dispatch {i}/{n} hours ...")

        d        = float(row["residual_demand_mw"])
        srmc_row = srmc_stack.loc[ts]
        rec      = {"timestamp": ts}

        if method in ("A", "both"):
            pa, mt, disp_a = dispatch_analytic(d, srmc_row, capacity)
            rec["price_a"]         = pa
            rec["marginal_tech_a"] = mt

        if method in ("B", "both"):
            pb, disp_b = dispatch_lp(d, srmc_row, capacity)
            rec["price_b"] = pb
            for tech, mw in disp_b.items():
                rec[f"dispatched_{tech}"] = mw

        results.append(rec)

    df = pd.DataFrame(results).set_index("timestamp")

    if "price_a" in df.columns and "price_b" in df.columns:
        agreement = (df["price_a"] - df["price_b"]).abs()
        n_disagree = (agreement > 0.5).sum()
        if n_disagree > 0:
            print(f"  [warn] Methods A and B disagree in {n_disagree} hours "
                  f"(max diff = {agreement.max():.2f} EUR/MWh). "
                  f"Check for infeasible LP hours or rounding.")
        else:
            print(f"  Methods A and B agree in all hours (max diff = {agreement.max():.4f}).")

    return df


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    srmc_row = pd.Series({
        "nuclear":   4.0,
        "lignite":  35.0,
        "hard_coal": 65.0,
        "ccgt":      85.0,
        "ocgt":     140.0,
        "oil":      220.0,
    })
    cap = DEFAULT_CAPACITY.copy()
    demand = 9000.0

    pa, mt, disp = dispatch_analytic(demand, srmc_row, cap)
    pb, _         = dispatch_lp(demand, srmc_row, cap)

    print(f"Residual demand: {demand:,.0f} MW")
    print(f"Method A: {pa:.2f} EUR/MWh  (marginal tech: {mt})")
    print(f"Method B: {pb:.2f} EUR/MWh")
    print(f"Agree: {abs(pa - pb) < 0.01}")
    print("\nDispatch (Method A):")
    for t, mw in disp.items():
        if mw > 0:
            print(f"  {t:12s}: {mw:,.0f} MW  @ SRMC {srmc_row[t]:.1f}")
