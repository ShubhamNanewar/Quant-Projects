from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


CAISO_OASIS_URL = "https://oasis.caiso.com/oasisapi/SingleZip"


def oasis_timestamp(timestamp: str | pd.Timestamp) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%dT%H:%M-0000")


def read_oasis_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("CAISO OASIS response did not contain a CSV file.")
        with archive.open(csv_names[0]) as handle:
            return pd.read_csv(handle)


def fetch_oasis(params: dict[str, str], version: int) -> pd.DataFrame:
    response = requests.get(
        CAISO_OASIS_URL,
        params={"resultformat": "6", "version": str(version), **params},
        timeout=90,
    )
    response.raise_for_status()
    return read_oasis_zip(response.content)


def fetch_day_ahead_lmp(start: pd.Timestamp, end: pd.Timestamp, node: str) -> pd.DataFrame:
    return fetch_oasis(
        {
            "queryname": "PRC_LMP",
            "market_run_id": "DAM",
            "node": node,
            "startdatetime": oasis_timestamp(start),
            "enddatetime": oasis_timestamp(end),
        },
        version=1,
    )


def fetch_real_time_lmp(start: pd.Timestamp, end: pd.Timestamp, node: str) -> pd.DataFrame:
    return fetch_oasis(
        {
            "queryname": "PRC_INTVL_LMP",
            "market_run_id": "RTM",
            "node": node,
            "startdatetime": oasis_timestamp(start),
            "enddatetime": oasis_timestamp(end),
        },
        version=3,
    )


def standardize_lmp(data: pd.DataFrame, market: str) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [column.lower() for column in frame.columns]
    frame["datetime_utc"] = pd.to_datetime(
        frame["intervalstarttime_gmt"], utc=True, errors="coerce"
    )
    frame["node"] = frame["node"].astype(str)
    value_col = "mw" if "mw" in frame.columns else "value"
    frame["value"] = pd.to_numeric(frame[value_col], errors="coerce")

    pivot = (
        frame.pivot_table(
            index=["datetime_utc", "node"],
            columns="xml_data_item",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename_map = {
        "LMP_PRC": f"total_lmp_{market}",
        "LMP_ENE_PRC": f"system_energy_price_{market}",
        "LMP_CONG_PRC": f"congestion_price_{market}",
        "LMP_LOSS_PRC": f"marginal_loss_price_{market}",
    }
    pivot = pivot.rename(columns=rename_map)
    keep = ["datetime_utc", "node"] + [col for col in rename_map.values() if col in pivot]
    return pivot[keep].dropna(subset=[f"total_lmp_{market}"])


def build_caiso_dataset(
    output_path: Path,
    start: str = "2024-01-01T08:00",
    end: str = "2025-01-01T08:00",
    node: str = "TH_NP15_GEN-APND",
    chunk_days: int = 28,
    sleep_seconds: float = 5.0,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    da_parts: list[pd.DataFrame] = []
    rt_parts: list[pd.DataFrame] = []
    cursor = start_ts
    while cursor < end_ts:
        window_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        print(f"Fetching {cursor.date()} to {window_end.date()} for {node}")
        da_parts.append(fetch_day_ahead_lmp(cursor, window_end, node))
        time.sleep(sleep_seconds)
        rt_parts.append(fetch_real_time_lmp(cursor, window_end, node))
        time.sleep(sleep_seconds)
        cursor = window_end

    da = standardize_lmp(pd.concat(da_parts, ignore_index=True), "da")
    rt = standardize_lmp(pd.concat(rt_parts, ignore_index=True), "rt")
    rt["datetime_utc"] = rt["datetime_utc"].dt.floor("h")
    rt_hourly = rt.groupby(["datetime_utc", "node"], as_index=False).mean(numeric_only=True)

    merged = pd.merge(da, rt_hourly, on=["datetime_utc", "node"], how="inner")
    merged = merged[(merged["datetime_utc"] >= start_ts) & (merged["datetime_utc"] <= end_ts)]
    merged["datetime_local"] = (
        merged["datetime_utc"].dt.tz_convert("US/Pacific").dt.tz_localize(None)
    )
    merged["spread"] = merged["total_lmp_rt"] - merged["total_lmp_da"]
    merged["energy_spread"] = (
        merged["system_energy_price_rt"] - merged["system_energy_price_da"]
    )
    merged["congestion_spread"] = (
        merged["congestion_price_rt"] - merged["congestion_price_da"]
    )
    merged["loss_spread"] = (
        merged["marginal_loss_price_rt"] - merged["marginal_loss_price_da"]
    )
    merged = merged.sort_values("datetime_utc").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def load_or_fetch_caiso(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["datetime_utc", "datetime_local"])
    return build_caiso_dataset(path)

