# Decision Framework

This document describes how I interpret the spread analysis as a research screen.

## Step 1: Identify The Spread

Calculate:

```text
Spread_t = RT_LMP_t - DA_LMP_t
```

Questions:

- Is RT above DA?
- Is the spread large versus history?
- Is it large in absolute terms or only relative to a quiet period?

## Step 2: Classify The Event Driver

Decompose:

```text
Spread = EnergySpread + CongestionSpread + LossSpread
```

Driver classification:

- energy-led event,
- congestion-led event,
- mixed event,
- loss/minor component event.

This matters because the next step depends on what caused the spread.

## Step 3: Check Market Regime

Use:

- rolling spread mean,
- rolling spread volatility,
- lagged spread,
- hour of day,
- month,
- recent DA price level.

Questions:

- Are spreads clustering?
- Is volatility elevated?
- Is the event in an hour that historically has wider spreads?
- Is DA price already high, suggesting the market priced some stress?

## Step 4: Score The Signal

The model ranks hours by predicted spread.

Useful criteria:

```text
predicted_spread > top_decile_threshold
spread_volatility is acceptable
component driver is interpretable
historical downside is not excessive
```

## Step 5: Interpret The Decision

Possible research labels:

```text
High positive RT correction risk
Neutral / no edge
Congestion event watch
Energy scarcity watch
Avoid: model unclear or volatility too high
```

## Example Interpretation

If:

```text
predicted spread is high
recent spread is positive
rolling volatility is elevated but not extreme
event type historically energy-led
hour is evening peak
```

Then the research interpretation may be:

```text
Real-time price has elevated risk of settling above day-ahead.
The thesis is system-tightness rather than local congestion.
```

If:

```text
predicted spread is high
congestion component dominates historical similar events
```

Then:

```text
The thesis is location/grid constraint risk, not broad system scarcity.
```

## What Would Improve The Decision

Before any live trading interpretation, add:

- load forecast error,
- renewable forecast error,
- weather forecast revisions,
- outages,
- gas prices,
- constraint data,
- liquidity and transaction cost estimates.
