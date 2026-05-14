# Data Selection

## Objective

The project needs data that can answer one market question:

```text
When does real-time power price materially correct the day-ahead expectation?
```

To answer this properly, the dataset must include:

- day-ahead prices,
- real-time prices,
- same location,
- same delivery timestamp,
- enough history for regimes and validation,
- price components where possible.

## Selected Source

The default source is CAISO OASIS.

Selected market location:

```text
TH_NP15_GEN-APND
```

Selected period:

```text
2024-01-01T08:00Z to 2025-01-01T08:00Z
```

Selected reports:

```text
PRC_LMP       market_run_id = DAM
PRC_INTVL_LMP market_run_id = RTM
```

## Why CAISO OASIS

CAISO is selected because it gives a clean research setup:

- it provides day-ahead and real-time LMPs,
- it provides energy, congestion, and loss components,
- it is free and reproducible,
- it allows event-level diagnosis,
- it avoids relying on private vendor data.

## What This Data Can Explain

The data can explain:

- DA/RT spread behavior,
- energy-driven versus congestion-driven events,
- hourly and monthly spread structure,
- extreme events,
- rolling spread regimes,
- whether simple features rank positive spread events.

## What This Data Cannot Fully Explain

A professional trading desk would also want:

- load forecast,
- actual load,
- renewable forecast,
- actual renewable output,
- temperature and weather forecast revisions,
- generator outages,
- transmission outages,
- fuel prices,
- ancillary service prices,
- forward curves,
- liquidity and position limits.

This project is therefore a first research layer. It is built so those drivers can be added later.

## Data Quality Checks

Before analysis, check:

- timestamps are UTC and ordered,
- DA and RT prices are matched on same timestamp and node,
- RT interval prices are converted to hourly averages,
- LMP component math reconciles approximately,
- extreme prices are investigated rather than blindly removed.

Important:

```text
large DA/RT spread does not automatically mean bad data
```

It may mean real-time system stress, congestion, or scarcity.
