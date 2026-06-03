# LightGBM Noise Risk Training Report

## Scope
- Training dataset: `training_dataset_feature_spec_2_0_0.csv`
- Pattern event CSV is used for post-training validation only.
- Target `noise_risk_level` is an internal pseudo-label, not legal or externally verified ground truth.

## Dataset
- Feature spec version: `2.0.0`
- Target column: `noise_risk_level`
- Feature count: `17`
- Train rows: `90624`
- Validation rows: `19420`
- Test rows: `19420`

## Test Metrics
- Accuracy: `1.000000`
- Macro F1: `1.000000`
- Weighted F1: `1.000000`

## Feature Importance Top 10
- `vibration_level_category`: gain=1870742.6698, split=2117
- `impact_count_in_window`: gain=470482.1579, split=4914
- `sound_level`: gain=201381.5517, split=4568
- `vibration_raw`: gain=47554.9759, split=5784
- `vibration_acc_mps2`: gain=326.7627, split=1237
- `sound_over_impact_leq`: gain=317.8019, split=3257
- `repeated_impact_flag`: gain=30.7588, split=8
- `sound_over_impact_lmax`: gain=0.0140, split=759
- `sound_over_airborne_leq`: gain=0.0130, split=1183
- `hour_of_day`: gain=0.0070, split=168

## Pattern Validation
- Pattern rows: `1655`
- Matched rows: `1655`
- Match rate: `1.000000`
- Mismatch case rows: `3447`

## Limitations
- `noise_risk_level` is generated from current internal alert rules.
- Pattern event labels are previous pattern-classification outputs and are not used for training.
- This model reproduces the current alert-rule pseudo-labels; it is not a verified inter-floor-noise ground-truth model.
- Future supervised labels from real complaints, user feedback, or administrator verification are required.
