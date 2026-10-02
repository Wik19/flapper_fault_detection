---
name: leakage-check
description: Audit data processing and model evaluation scripts for data leakage, invalid splits, or improper scaling.
argument-hint: "[path/to/script.py]"
---
Audit the file provided in `$ARGUMENTS` against these strict checks:

1. Scaler & Normalization Scope:
   - Are `StandardScaler`, `MinMaxScaler`, or PCA fitted ONLY on `X_train`?
   - Ensure `.fit_transform()` is used on train, and ONLY `.transform()` on val/test.

2. Split Validity:
   - Does any feature engineering (e.g., rolling window stats, FFT filtering) happen across split boundaries?
   - If time-series: Is window overlap isolated within separate folds?

3. Leakage via Target:
   - Is target encoding or threshold selection using validation/test signals?

Report findings as a clean table with status: [PASS], [WARNING], or [FAIL] with line numbers and remediation steps.