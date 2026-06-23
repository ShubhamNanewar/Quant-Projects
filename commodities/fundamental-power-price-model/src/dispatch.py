"""
Economic dispatch: Method A (analytic stacking) and Method B (LP with dual price).

Method A: walk up the merit order until cumulative capacity meets residual demand.
Method B: linear programme whose demand-balance dual gives the system marginal price.

Both methods return the same clearing price when no minimum-generation constraints
are active. Method B is the generalisable OR formulation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import pulp

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Installed capacity (MW) per technology
# Source: ENTSO-E A68 via data_fetch.fetch_installed_capacity(), or fallback below.
# These are updated externally by build_capacity_table(); defaults here are
# NL 2023 approximate values from ENTSO-E Transparency (document in docs/).
# ---------------------------------------------------------------------------

DEFAULT_CAPACITY = {
    "nuclear":    485,
    "lignite":      0,
    "hard_coal":  3300,
    "ccgt":      12500,
    "ocgt":       2500,
    "oil":         200,
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
    Walk the merit order (ascending SRMC) until demand is met.

    Returns
    -------
    price       : float  — SRMC of the marginal unit (EUR/MWh_e)
    marginal_tech: str   — name of the marginal technology
    dispatch    : dict   — {tech: MW dispatched}
    """
    order = srmc_row.sort_values()
    remaining = residual_demand
    dispatch = {t: 0.0 for t in order.index}
    price = np.nan
    marginal_tech = "unmet"

    for tech, cost in order.items():
        cap = capacity.get(tech, 0.0)
        if cap <= 0:
            continue
        gen = min(remaining, cap)
        dispatch[tech] = gen
        remaining -= gen
        if remaining <= 0:
            price = cost
            marginal_tech = tech
            break
        price = cost  # last committed unit if demand is exactly met

    return price, marginal_tech, dispatch


# ---------------------------------------------------------------------------
# Method B: LP dispatch
# ---------------------------------------------------------------------------

def dispatch_lp(
    residual_demand: float,
    srmc_row: pd.Series,
    capacity: dict,
    hour_label: str = "",
) -> tuple[float, dict]:
    """
    Solve the economic dispatch LP for one hour.

    minimise   sum_i SRMC_i * g_i
    s.t.       sum_i g_i = D_residual   (dual = lambda = clearing price)
               0 <= g_i <= Cap_i

    Returns
    -------
    price    : float — shadow price of demand constraint (EUR/MWh_e)
    dispatch : dict  — {tech: MW}
    """
    prob = pulp.LpProblem(f"dispatch_{hour_label}", pulp.LpMinimize)

    techs = [t for t in srmc_row.index if capacity.get(t, 0) > 0]
    g = {t: pulp.LpVariable(f"g_{t}", lowBound=0, upBound=capacity[t]) for t in techs}

    # Objective
    prob += pulp.lpSum(srmc_row[t] * g[t] for t in techs)

    # Demand balance (named so we can read its dual)
    balance = pulp.LpConstraint(
        pulp.lpSum(g[t] for t in techs) - residual_demand,
        sense=pulp.LpConstraintEQ,
        name="balance",
        rhs=0,
    )
    prob += balance

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return np.nan, {t: np.nan for t in techs}

    dispatch = {t: pulp.value(g[t]) for t in techs}
    # Shadow price of the balance constraint
    price = prob.constraints["balance"].pi
    if price is None:
        price = np.nan

    return price, dispatch


# ---------------------------------------------------------------------------
# Full-panel dispatch
# ---------------------------------------------------------------------------

def run_dispatch(
    panel: pd.DataFrame,
    srmc_stack: pd.DataFrame,
    capacity: dict | None = None,
    method: str = "both",
) -> pd.DataFrame:
    """
    Run dispatch over every hour in panel.

    Parameters
    ----------
    panel       : hourly panel from data_fetch.build_panel()
    srmc_stack  : per-hour SRMC per technology from stack.build_stack()
    capacity    : dict of installed MW per technology (uses DEFAULT_CAPACITY if None)
    method      : 'A', 'B', or 'both'

    Returns
    -------
    DataFrame with columns:
        price_a, marginal_tech_a   (Method A)
        price_b                    (Method B)
        dispatched_{tech}          (from Method B)
    """
    if capacity is None:
        capacity = DEFAULT_CAPACITY.copy()

    results = []
    n = len(panel)

    for i, (ts, row) in enumerate(panel.iterrows()):
        if i % 1000 == 0:
            print(f"  dispatch hour {i}/{n} ...")

        d = row["residual_demand_mw"]
        srmc_row = srmc_stack.loc[ts]

        rec = {"timestamp": ts}

        if method in ("A", "both"):
            pa, mt, disp_a = dispatch_analytic(d, srmc_row, capacity)
            rec["price_a"] = pa
            rec["marginal_tech_a"] = mt

        if method in ("B", "both"):
            pb, disp_b = dispatch_lp(d, srmc_row, capacity, hour_label=str(i))
            rec["price_b"] = pb
            for tech, mw in disp_b.items():
                rec[f"dispatched_{tech}"] = mw

        results.append(rec)

    df = pd.DataFrame(results).set_index("timestamp")
    return df


if __name__ == "__main__":
    # Quick sanity check
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from stack import TECH_PARAMS, build_stack

    # Tiny synthetic example
    srmc_row = pd.Series({
        "nuclear":   4.0,
        "lignite":  30.0,
        "hard_coal": 60.0,
        "ccgt":      80.0,
        "ocgt":     130.0,
        "oil":      200.0,
    })
    cap = DEFAULT_CAPACITY.copy()
    demand = 8000.0

    pa, mt, _ = dispatch_analytic(demand, srmc_row, cap)
    pb, _     = dispatch_lp(demand, srmc_row, cap)
    print(f"Method A: {pa:.1f} EUR/MWh ({mt})")
    print(f"Method B: {pb:.1f} EUR/MWh")
    print(f"Methods agree: {abs(pa - pb) < 0.01}")
