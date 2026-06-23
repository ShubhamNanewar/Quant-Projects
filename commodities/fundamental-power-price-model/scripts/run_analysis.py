"""
End-to-end pipeline for the Fundamental Power Price Model.

Steps:
  1. Fetch / load data        data_fetch.build_panel()
  2. Build SRMC stack         stack.build_stack()
  3. Dispatch (A + B)         dispatch.run_dispatch()
  4. Validate against actual  backtest.run_backtest()
  5. Trader insights          insights.run_insights()

Run time: ~2-5 minutes on first run (ENTSO-E fetch + 8,760 LP solves).
Subsequent runs load from cache and complete in under 30 seconds.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data_fetch import build_panel, fetch_generation, BIDDING_ZONE, START, END
from stack import build_stack
from dispatch import run_dispatch
from backtest import run_backtest
from insights import run_insights


def main():
    sep = "=" * 60
    print(sep)
    print("  Fundamental Power Price Model -- NL day-ahead 2023")
    print(sep)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    print("\n[1/5] Data")
    panel_path = ROOT / "data" / "processed" / "panel.parquet"
    cap_path   = ROOT / "data" / "processed" / "capacity.csv"

    if panel_path.exists() and cap_path.exists():
        print(f"  Loading cached panel from {panel_path}")
        panel    = pd.read_parquet(panel_path)
        capacity = pd.read_csv(cap_path, index_col=0)["installed_mw"].to_dict()
    else:
        panel, capacity = build_panel()

    print(f"  Period : {panel.index[0]} -> {panel.index[-1]}")
    print(f"  Rows   : {len(panel):,}  Columns: {list(panel.columns)}")
    print(f"\n  Installed capacity (MW):")
    for tech, mw in sorted(capacity.items()):
        print(f"    {tech:12s}: {mw:,.0f}")

    # ------------------------------------------------------------------
    # Step 2: SRMC stack
    # ------------------------------------------------------------------
    print("\n[2/5] Building SRMC stack ...")
    srmc_stack = build_stack(panel)

    print("  Sample SRMCs for first hour:")
    first_srmc = srmc_stack.iloc[0].sort_values()
    for tech, cost in first_srmc.items():
        print(f"    {tech:12s}: {cost:.1f} EUR/MWh_e")

    # ------------------------------------------------------------------
    # Step 3: Dispatch
    # ------------------------------------------------------------------
    print("\n[3/5] Running dispatch (Method A + B) ...")
    dispatch_path = ROOT / "data" / "processed" / "dispatch.parquet"

    if dispatch_path.exists():
        print(f"  Loading cached dispatch from {dispatch_path}")
        dispatch = pd.read_parquet(dispatch_path)
    else:
        dispatch = run_dispatch(panel, srmc_stack, capacity=capacity, method="both")
        dispatch.to_parquet(dispatch_path)
        print(f"  Saved -> {dispatch_path}")

    # ------------------------------------------------------------------
    # Step 4: Backtest
    # ------------------------------------------------------------------
    print("\n[4/5] Backtest -- price fit and dispatch-mix comparison ...")

    try:
        gen_actual = fetch_generation(BIDDING_ZONE, START, END)
    except Exception as e:
        print(f"  [warn] Could not load generation data for mix comparison: {e}")
        gen_actual = None

    bt = run_backtest(panel, dispatch, gen_actual=gen_actual)

    # ------------------------------------------------------------------
    # Step 5: Insights
    # ------------------------------------------------------------------
    print("\n[5/5] Trader insights ...")
    ins = run_insights(panel, dispatch)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  Done. Outputs in reports/")
    print(f"  Overall MAE  : {bt['overall']['mae']:.1f} EUR/MWh")
    print(f"  Overall bias : {bt['overall']['bias']:+.1f} EUR/MWh")
    print(f"  Correlation  : {bt['overall']['corr']:.3f}")
    print(f"  Mean residual: {ins['residual'].mean():+.2f} EUR/MWh")
    print(sep)


if __name__ == "__main__":
    main()
