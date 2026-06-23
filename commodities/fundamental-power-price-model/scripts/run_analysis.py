"""
End-to-end pipeline:
  1. Fetch / load data   (data_fetch)
  2. Build SRMC stack    (stack)
  3. Dispatch (A + B)    (dispatch)
  4. Backtest            (backtest)
  5. Trader insights     (insights)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_fetch import build_panel
from stack import build_stack
from dispatch import run_dispatch, DEFAULT_CAPACITY
from backtest import run_backtest
from insights import run_insights


def main():
    print("=" * 60)
    print("Fundamental Power Price Model — NL day-ahead")
    print("=" * 60)

    # Step 1: data
    processed = ROOT / "data" / "processed" / "panel.parquet"
    if processed.exists():
        import pandas as pd
        print(f"\n[1] Loading cached panel from {processed}")
        panel = pd.read_parquet(processed)
    else:
        print("\n[1] Building panel (fetching from ENTSO-E + fuel CSVs) ...")
        panel = build_panel()

    print(f"    Panel shape: {panel.shape}")
    print(f"    Date range:  {panel.index[0]} → {panel.index[-1]}")

    # Step 2: SRMC stack
    print("\n[2] Building SRMC stack ...")
    srmc_stack = build_stack(panel)
    print(f"    Stack shape: {srmc_stack.shape}")
    print(f"    Technologies: {list(srmc_stack.columns)}")

    # Capacity (could load from ENTSO-E A68 here; using defaults for v1)
    capacity = DEFAULT_CAPACITY.copy()
    print(f"\n    Capacity (MW): {capacity}")

    # Step 3: dispatch
    print("\n[3] Running dispatch (Method A + B) ...")
    dispatch = run_dispatch(panel, srmc_stack, capacity=capacity, method="both")
    print(f"    Dispatch shape: {dispatch.shape}")

    out_dispatch = ROOT / "data" / "processed" / "dispatch.parquet"
    dispatch.to_parquet(out_dispatch)
    print(f"    Saved to {out_dispatch}")

    # Step 4: backtest
    print("\n[4] Backtest ...")
    bt = run_backtest(panel, dispatch)

    # Step 5: insights
    print("\n[5] Trader insights ...")
    run_insights(panel, dispatch)

    print("\n" + "=" * 60)
    print("Done. Check reports/ for figures and metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
