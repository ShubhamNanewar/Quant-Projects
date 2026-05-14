from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from caiso_fetch import load_or_fetch_caiso
from event_analysis import add_event_metrics, event_summary, top_events
from features import FEATURE_COLUMNS, build_feature_frame
from modeling import signal_summary, walk_forward_backtest
from visualization import (
    save_component_decomposition,
    save_da_rt_context,
    save_hourly_heatmap,
    save_model_comparison,
    save_signal_capture,
    save_spread_distribution,
)


DATA_PATH = ROOT / "data" / "raw" / "caiso_th_np15_da_rt_2024.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_or_fetch_caiso(DATA_PATH)
    events = add_event_metrics(raw)
    frame = build_feature_frame(events)

    predictions, metrics, importances = walk_forward_backtest(frame, FEATURE_COLUMNS)

    metric_summary = metrics.groupby("model", as_index=False).mean(numeric_only=True)
    importance_summary = (
        importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    signal = signal_summary(predictions)

    events.to_csv(TABLE_DIR / "event_diagnostics.csv", index=False)
    top_events(events).to_csv(TABLE_DIR / "top_spread_events.csv", index=False)
    event_summary(events).to_csv(TABLE_DIR / "event_driver_summary.csv", index=False)
    predictions.to_csv(TABLE_DIR / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(TABLE_DIR / "walk_forward_metrics_by_fold.csv", index=False)
    metric_summary.to_csv(TABLE_DIR / "model_metrics_summary.csv", index=False)
    importance_summary.to_csv(TABLE_DIR / "feature_importance.csv", index=False)

    with (TABLE_DIR / "signal_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Signal Summary\n\n")
        for key, value in signal.items():
            handle.write(f"- {key}: {value:.4f}\n")

    save_da_rt_context(events, FIGURE_DIR / "01_da_rt_context.png")
    save_spread_distribution(events, FIGURE_DIR / "02_spread_distribution.png")
    save_component_decomposition(events, FIGURE_DIR / "03_component_decomposition.png")
    save_hourly_heatmap(frame, FIGURE_DIR / "04_hour_week_heatmap.png")
    save_model_comparison(metrics, FIGURE_DIR / "05_model_comparison.png")
    save_signal_capture(predictions, FIGURE_DIR / "06_signal_capture.png")

    print("Analysis complete.")
    print(f"Rows: {len(raw):,}")
    print(metric_summary.to_string(index=False))
    print(signal)


if __name__ == "__main__":
    main()
