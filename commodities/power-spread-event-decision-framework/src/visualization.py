from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_da_rt_context(data: pd.DataFrame, path: Path) -> None:
    set_style()
    daily = (
        data.set_index("datetime_local")[["total_lmp_da", "total_lmp_rt", "spread"]]
        .resample("D")
        .mean()
        .rolling(7, min_periods=1)
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(daily["datetime_local"], daily["total_lmp_da"], label="Day-ahead LMP", linewidth=1.8)
    axes[0].plot(daily["datetime_local"], daily["total_lmp_rt"], label="Real-time LMP", linewidth=1.8)
    axes[0].set_title("DA and RT prices, 7-day rolling daily average")
    axes[0].set_ylabel("$/MWh")
    axes[0].legend()
    axes[1].bar(daily["datetime_local"], daily["spread"], color="#6a9fb5", width=1.0)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Average daily RT - DA spread")
    axes[1].set_ylabel("$/MWh")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_spread_distribution(data: pd.DataFrame, path: Path) -> None:
    set_style()
    lo, hi = data["spread"].quantile([0.01, 0.99])
    fig, ax = plt.subplots()
    sns.histplot(data["spread"].clip(lo, hi), bins=70, kde=True, color="#2f6f8f", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Distribution of RT - DA spreads, clipped at 1st/99th percentiles")
    ax.set_xlabel("RT LMP - DA LMP ($/MWh)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_component_decomposition(data: pd.DataFrame, path: Path, n: int = 12) -> None:
    set_style()
    top = data.nlargest(n, "spread").sort_values("spread")
    labels = top["datetime_local"].dt.strftime("%b %d %H:%M")
    fig, ax = plt.subplots(figsize=(11, 7))
    left = pd.Series(0.0, index=top.index)
    for col, label, color in [
        ("energy_spread", "Energy", "#4c78a8"),
        ("congestion_spread", "Congestion", "#f58518"),
        ("loss_spread", "Loss", "#54a24b"),
    ]:
        ax.barh(labels, top[col], left=left, label=label, color=color)
        left = left + top[col]
    ax.set_title("Top positive spread events: component decomposition")
    ax.set_xlabel("RT component - DA component ($/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hourly_heatmap(data: pd.DataFrame, path: Path) -> None:
    set_style()
    pivot = data.pivot_table(index="hour", columns="day_of_week", values="spread", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.3,
        cbar_kws={"label": "$/MWh"},
        ax=ax,
    )
    ax.set_title("Average DA/RT spread by hour and weekday")
    ax.set_xlabel("Day of week, Monday=0")
    ax.set_ylabel("Hour")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    set_style()
    summary = metrics.groupby("model", as_index=False)[["mae", "rmse", "directional_accuracy"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, title in zip(
        axes, ["mae", "rmse", "directional_accuracy"], ["MAE", "RMSE", "Directional accuracy"]
    ):
        sns.barplot(data=summary, x="model", y=metric, color="#6a9fb5", ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_signal_capture(results: pd.DataFrame, path: Path, model: str = "random_forest") -> None:
    set_style()
    subset = results.loc[results["model"] == model].sort_values("datetime_local").copy()
    threshold = subset["prediction"].quantile(0.90)
    subset["signal_positive_spread"] = subset["spread"].where(subset["prediction"] >= threshold, 0).clip(lower=0)
    subset["positive_spread"] = subset["spread"].clip(lower=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(subset["datetime_local"], subset["positive_spread"].cumsum(), label="All positive spread")
    ax.plot(
        subset["datetime_local"],
        subset["signal_positive_spread"].cumsum(),
        label="Top-decile signal positive spread",
    )
    ax.set_title("Positive spread captured by top-decile signal")
    ax.set_ylabel("Cumulative positive spread ($/MWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

