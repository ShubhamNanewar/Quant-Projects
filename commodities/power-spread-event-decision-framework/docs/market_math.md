# Market Math

## Spread Definition

The core target is:

```text
Spread_t = RT_LMP_t - DA_LMP_t
```

Interpretation:

- positive spread: real-time settled above day-ahead,
- negative spread: real-time settled below day-ahead.

Market intuition:

```text
Spread_t ≈ Realized system stress - Expected system stress
```

## LMP Decomposition

Locational marginal price decomposes into:

```text
LMP = Energy + Congestion + Loss
```

Therefore:

```text
RT_LMP - DA_LMP
= (RT_Energy - DA_Energy)
+ (RT_Congestion - DA_Congestion)
+ (RT_Loss - DA_Loss)
```

## Component Interpretation

### Energy Spread

```text
EnergySpread_t = RT_Energy_t - DA_Energy_t
```

High positive value can indicate system-wide scarcity, higher real-time marginal generation cost, or real-time supply-demand tightness.

### Congestion Spread

```text
CongestionSpread_t = RT_Congestion_t - DA_Congestion_t
```

High positive value can indicate a transmission constraint binding more severely in real time than in day-ahead.

### Loss Spread

```text
LossSpread_t = RT_Loss_t - DA_Loss_t
```

Usually smaller, but still part of the nodal price.

## Event Size

Absolute spread:

```text
AbsSpread_t = |Spread_t|
```

Rolling abnormality:

```text
Z_t = (Spread_t - rolling_mean_168h) / rolling_std_168h
```

Percentile rank:

```text
Percentile_t = empirical_rank(Spread_t)
```

An event is interesting when:

```text
Spread_t is large
and Z_t is high
and one component explains most of the move
```

## Model Target

Regression target:

```text
y_t = Spread_t
```

Classification-style target:

```text
I(Spread_t > threshold)
```

Ranking target:

```text
rank hours by predicted positive spread risk
```

In practice, ranking is often more useful than exact price prediction.

