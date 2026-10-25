# Implementation Plan — Feature Selection & ML Pipeline from `data_final.csv`

## Phase 1 — Load and Validate

1. Load `data_final.csv`.
2. Parse `date` as datetime.
3. Sort by:

   ```text
   symbol → date
   ```
4. Check:

   * Duplicate `symbol + date`
   * Missing values
   * Infinite values
   * Constant columns
   * Data types
   * Unexpected values
5. Save a validation report.

---

## Phase 2 — Define Columns

### Never use as ML features

```text
date
symbol
sector
target_hit
sl_hit
timeout
outcome
days_to_outcome
return_7d_after_signal
max_gain_7d
max_loss_7d
mfe
mae
```

### Use as target

```text
outcome
```

Convert to:

```text
SUCCESS = 1
FAILURE = 0
```

For the first ML experiment, exclude `TIMEOUT` or define its treatment explicitly before training.

### Candidate feature columns

Use the remaining market-information columns from `data_final.csv`, including:

```text
OHLC
Volume
MA features
Bollinger features
RSI features
Return features
Price-structure features
Candle features
Volume features
Volatility features
MA-relationship features
Support/resistance features
NIFTY features
Sector features
Relative-strength features
```

Do **not** manually remove candidate features yet.

---

# Phase 3 — Create Signal Dataset

1. Filter:

   ```text
   base_signal == 1
   ```
2. Each row represents one candidate trade.
3. Keep:

   ```text
   symbol
   date
   all candidate features
   outcome
   ```
4. Verify that every feature represents information available **on or before the signal date**.

---

# Phase 4 — Remove Data Leakage

For every candidate column, verify:

```text
Could this value have been known at the moment the trade signal occurred?
```

If **NO → permanently remove it.**

Specifically remove any feature calculated using:

```text
future prices
future volume
future returns
target/SL results
future 7-day movement
future MFE/MAE
future outcome
```

Also verify rolling calculations use:

```text
current day + previous days
```

and never:

```text
future days
```

---

# Phase 5 — Remove Technically Useless Columns

Remove columns satisfying:

### 1. Constant

```text
nunique == 1
```

### 2. Almost constant

Remove if:

```text
>99.5% of rows contain the same value
```

unless it is a deliberately meaningful binary feature.

### 3. Excessive missingness

Remove if:

```text
missing percentage > 30%
```

For remaining missing values:

```text
impute using training data only
```

### 4. Invalid numeric values

Replace:

```text
+inf
-inf
```

with missing values, then handle them.

---

# Phase 6 — Remove Redundant Features

Calculate feature-to-feature correlation using **training data only**.

For numeric features:

```text
|correlation| >= 0.95
```

→ treat as highly redundant.

For each highly correlated group:

1. Keep the feature that is:

   * More interpretable
   * More stable
   * Less redundant with other features
2. Remove the others.

Example:

```text
MA50
Close/MA50
Distance_from_MA50
```

If they contain essentially the same information, don't automatically keep all three.

Repeat for:

```text
BB features
MA features
return features
volatility features
volume features
```

---

# Phase 7 — Build Baseline Models

Before feature selection, train models using the remaining candidate features.

Use:

```text
1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
```

Use the **same train/validation/test periods** for all models.

---

# Phase 8 — Time-Based Dataset Split

Never randomly shuffle.

Use chronological splitting.

Example:

```text
TRAIN       VALIDATION       TEST
---------   ------------     --------
oldest 60%  next 20%         newest 20%
```

The exact percentages can be adjusted, but the order must remain chronological.

All preprocessing must be fitted using **training data only**.

---

# Phase 9 — Univariate Feature Analysis

For every candidate feature:

1. Divide training observations into quantiles/bins.
2. Calculate success rate for each bin.
3. Calculate:

   * Mean outcome
   * Success rate
   * Number of observations
   * Mean return/R if available
4. Repeat on validation data.

Keep a feature as **potentially useful** if:

```text
training relationship exists
AND
validation relationship remains directionally consistent
AND
each important bin has sufficient observations
```

Do not keep a feature simply because training performance improves.

---

# Phase 10 — Feature Importance

For each model calculate:

### Logistic Regression

* Standardized coefficient magnitude

### Random Forest

* Feature importance
* Permutation importance

### XGBoost

* Feature importance
* Permutation importance

### LightGBM

* Feature importance
* Permutation importance

Create a combined feature-ranking table:

```text
feature
logistic_rank
rf_rank
xgb_rank
lgbm_rank
permutation_rank
average_rank
```

---

# Phase 11 — Stability Test

A feature should not be considered reliable because it is important in only one model.

For each feature, check:

```text
Important in model 1?
Important in model 2?
Important in model 3?
Important in model 4?
```

Then check across different time periods.

Prefer features that show importance:

```text
across multiple models
+
across multiple time periods
```

---

# Phase 12 — Feature Ablation Test

Create multiple feature sets.

### Set A

```text
All surviving features
```

### Set B

```text
Top 75% features
```

### Set C

```text
Top 50% features
```

### Set D

```text
Top 30% features
```

### Set E

```text
Top 20 features
```

Train all four models on every set.

Compare **validation**, not training, performance.

---

# Phase 13 — Remove Features That Hurt Generalization

A feature is a candidate for removal when:

```text
Including feature → validation performance decreases
```

and the decrease is:

```text
consistent across multiple models
AND
consistent across multiple validation periods
```

Also remove features where:

```text
training importance = high
validation importance = low
```

because this indicates possible overfitting.

Do **not** remove a feature because one model considers it unimportant.

---

# Phase 14 — Walk-Forward Feature Selection

Repeat the feature-selection process across multiple historical windows.

Example:

```text
Window 1:
Train → Validate

Window 2:
Train → Validate

Window 3:
Train → Validate

Window 4:
Train → Validate
```

For each feature record:

```text
times useful
times harmful
times neutral
```

Prefer features that remain useful across windows.

---

# Phase 15 — Final Feature Set

A feature becomes part of the final ML feature set only if it passes:

```text
✓ No data leakage
✓ Not constant
✓ Acceptable missingness
✓ Not excessively redundant
✓ Has historical relationship with outcome
✓ Validation relationship is consistent
✓ Useful across multiple models OR demonstrably complementary
✓ Stable across time
✓ Does not consistently reduce out-of-sample performance
```

Create:

```text
final_features.txt
```

containing only the selected columns.

---

# Phase 16 — Train Final Models

Using only `final_features`:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

Train on historical training data.

Tune hyperparameters using **training/validation data only**.

Do not touch the final test set during tuning.

---

# Phase 17 — Ensemble

Each model outputs:

```text
P(success)
```

Example:

```text
Logistic   = 0.68
Random     = 0.73
XGBoost    = 0.79
LightGBM   = 0.76
```

Calculate ensemble probability:

```text
ensemble_probability =
mean(model_probabilities)
```

Also test weighted averaging only if validation data shows improvement.

---

# Phase 18 — Determine TAKE/SKIP Threshold

Test thresholds such as:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
```

For each threshold calculate:

```text
Number of trades
Win rate
Average R
Expectancy
Profit factor
Total R
Maximum drawdown
Longest losing streak
```

Select the threshold using **validation data**, not test data.

---

# Phase 19 — Final Unseen Test

Freeze:

```text
features
models
hyperparameters
ensemble method
probability threshold
```

Then run once on the untouched test period.

Report:

```text
Baseline strategy performance
vs
ML-filtered strategy performance
```

Compare:

* Number of trades
* Win rate
* Average R
* Expectancy
* Profit factor
* Total R
* Maximum drawdown
* Losing streak
* Average holding period

---

# Phase 20 — Final Acceptance Criteria

The ML strategy is accepted only if:

```text
ML-filtered strategy > Base strategy
```

on **unseen data**, preferably across multiple walk-forward periods.

The final selected features should be the intersection of:

```text
Predictive
+
Stable
+
Non-leaky
+
Non-redundant
+
Out-of-sample useful
```

Final pipeline:

```text
data_final.csv
      ↓
Validation
      ↓
Leakage removal
      ↓
Useless-column removal
      ↓
Redundancy removal
      ↓
Candidate features
      ↓
Time split
      ↓
Univariate analysis
      ↓
4 ML models
      ↓
Feature importance
      ↓
Permutation importance
      ↓
Feature ablation
      ↓
Walk-forward stability
      ↓
FINAL FEATURES
      ↓
4 final models
      ↓
Probability ensemble
      ↓
Threshold optimization
      ↓
Unseen test
      ↓
Final strategy
```
