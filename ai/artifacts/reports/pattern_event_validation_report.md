# Pattern Event Validation Report

Pattern event rows are used only after training for reference analysis.
They are not training features, target labels, or label correction sources.

- Pattern rows: `1655`
- Matched rows: `1655`
- Match rate: `1.000000`
- Unmatched rows: `0`
- Mismatch case rows: `3447`

## Prediction Distribution For Matched Pattern Rows
- `normal`: 4
- `caution`: 149
- `warning`: 505
- `high`: 997

## Prediction Distribution For Rows Without Pattern Event
- `normal`: 74030
- `caution`: 42031
- `warning`: 8301
- `high`: 3447

## By Event Type
- `daily_noise`: normal=4, caution=149, warning=504, high=21
- `impact_noise`: normal=0, caution=0, warning=0, high=101
- `repeated_vibration`: normal=0, caution=0, warning=1, high=875

## By Severity
- `critical`: normal=0, caution=0, warning=0, high=82
- `high`: normal=0, caution=0, warning=0, high=593
- `low`: normal=4, caution=149, warning=474, high=0
- `medium`: normal=0, caution=0, warning=31, high=322

## By Pattern Label
- `night_repeated_impact`: normal=0, caution=0, warning=0, high=91
- `no_pattern`: normal=2, caution=44, warning=140, high=10
- `recurring_noise`: normal=2, caution=105, warning=364, high=21
- `vibration_cluster`: normal=0, caution=0, warning=1, high=875

## By Is Meaningful
- `0`: normal=4, caution=149, warning=504, high=21
- `1`: normal=0, caution=0, warning=1, high=976

## Interpretation
Differences are review signals only. Pattern event columns are previous pattern-classification outputs, not verified ground truth.
