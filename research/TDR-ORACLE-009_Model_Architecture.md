# Oracle-V Technology Decision Record — Model Architecture Selection

**TDR ID:** TDR-ORACLE-009
**Date:** 2026-02-21
**Status:** APPROVED (pre-training freeze)
**Spec Reference:** Part IV §6.1 (architecture is Research-stage decision)

---

## Decision

**Primary model:** XGBoost (gradient boosted trees)
**Sanity check:** Logistic Regression (L2-regularised)
**Model topology:** Separate binary classifiers per {underlying, horizon, target}
**Total models:** 8 (2 underlyings × 2 horizons × 2 targets)

---

## Problem Characteristics

| Dimension | Value | Implication |
|---|---|---|
| Training samples | ~300–633 per fold (expanding) | Too small for deep learning |
| Features | 15 (after R1 drops) | Moderate dimensionality |
| Task | Binary classification (expansion/compression) | Standard supervised learning |
| Class balance | ~17% expansion, ~24% compression | Mild-to-moderate imbalance |
| Signal strength | Max |r| ≈ 0.18 (R1c) | Weak individual signals |
| Temporal structure | Walk-forward mandatory | No random splits |
| Label autocorrelation | Lag-1 up to 0.573 (R1j) | Persistence is strong baseline |
| Calibration requirement | ECE ≤ 0.13 (§7.2.6) | Calibration is first-class constraint |

## Architecture Rationale

### XGBoost (Primary)

- Excels at combining many weak signals via feature interactions
- Handles missing values natively (critical: `riv_change_3d` at 67–83% coverage)
- Provides built-in feature importance + compatible with SHAP TreeExplainer
- Strong regularisation controls (depth, learning rate, subsample)
- Well-calibrated with post-hoc isotonic regression
- Proven performance on small tabular datasets

### Logistic Regression (Sanity Check)

- Naturally calibrated (outputs are probabilities by construction)
- If logistic regression matches XGBoost, the signal is linear and simpler is better
- Fast to train, fully interpretable
- Requires imputation for missing values (median + missing indicator)

### Rejected Alternatives

| Architecture | Reason for Rejection |
|---|---|
| Deep learning (MLP, LSTM, Transformer) | ~300–633 samples is far too small; overfitting guaranteed; Governance §10.2 complexity guard (√n ≈ 30 parameters max) |
| CatBoost | Only advantage is native categorical handling; only categorical feature is `day_of_week` (5 values) — XGBoost handles via ordinal encoding |
| Random Forest | Less effective than boosting for weak signal aggregation; no native missing value handling |
| Shared-backbone multi-horizon | Adds complexity without benefit at this data size; risks cross-horizon leakage |

## Model Topology Decision

**Separate binary classifiers** per {underlying, horizon, target}.

| Dimension | Options Considered | Decision | Rationale |
|---|---|---|---|
| Multi-class vs binary | 3-class (exp/comp/stable) vs 2 separate binary | Separate binary | Spec treats expansion and compression independently (§6.2.2); different base rates, different feature importance |
| Multi-horizon | Shared model vs separate | Separate per horizon | Avoids cross-horizon leakage; T=2 and T=5 may have different predictive dynamics |
| Cross-underlying | Shared model vs separate | Separate per underlying | NIFTY and BANKNIFTY have different expiry structures, different volatility dynamics |

## Hyperparameter Strategy

### XGBoost Search Space (Constrained)

| Parameter | Range | Rationale |
|---|---|---|
| `max_depth` | [2, 3, 4] | Forces broad structure discovery; prevents micro-fitting on ~500 rows |
| `n_estimators` | [50, 100, 150, 200, 300] | With early stopping; upper bound prevents overfitting |
| `learning_rate` | [0.02, 0.05, 0.1, 0.15] | ≥ 0.02 floor prevents excessive boosting rounds |
| `subsample` | [0.7, 0.8, 0.9, 1.0] | Row subsampling for variance reduction |
| `colsample_bytree` | [0.6, 0.7, 0.8, 0.9, 1.0] | Feature subsampling |
| `min_child_weight` | [3, 5, 10, 20] | Minimum leaf size — prevents splits on tiny groups |
| `reg_alpha` (L1) | [0, 0.01, 0.1, 1.0] | Sparsity regularisation |
| `reg_lambda` (L2) | [1.0, 2.0, 5.0] | Ridge regularisation |
| `scale_pos_weight` | [1.0] | **START UNWEIGHTED** — add only if recall collapses |

### Logistic Regression Search Space

| Parameter | Range | Rationale |
|---|---|---|
| `C` | [0.01, 0.1, 1.0, 10.0] | Inverse regularisation strength |
| `penalty` | ['l2'] | Standard ridge |

### Tuning Method

- **Random search:** 50 iterations per model
- **Inner validation:** TimeSeriesSplit(n_splits=3) within each fold's training window
- **Scoring:** neg_brier_score (primary), then check ECE post-hoc
- **Fixed random seed:** 42 (reproducibility)

## Calibration Strategy

- **Post-hoc calibration:** Isotonic regression on out-of-fold predictions
- **Rationale:** Better than Platt scaling for tree models on small datasets (non-parametric)
- **Implementation:** Fit calibrator on validation fold predictions; report both raw and calibrated metrics
- **Start unweighted:** Add class weighting only if recall collapses below 20% for the minority class

## Feature Importance

- **Method:** SHAP TreeExplainer (XGBoost); coefficient magnitude (Logistic Regression)
- **Stability assessment:** Kendall rank correlation of SHAP rankings across folds (Governance §10.2, exit criterion 9)
- **Deliverable:** Per-fold importance + cross-fold stability metric

## Missing Value Handling

| Model | Strategy |
|---|---|
| XGBoost | Native handling (learns optimal split direction for missing) |
| Logistic Regression | Median imputation (from training fold only) + binary missing indicator features |

## Evaluation Metrics

| Metric | Purpose | Threshold (§7.2.6) |
|---|---|---|
| Brier score | Primary accuracy | ≤ 0.22 (validation); ≥ 0.02 below best baseline |
| ECE | Calibration quality | ≤ 0.13 |
| AUC-ROC | Discrimination | ≥ 0.62 |
| Log-loss | Calibration sensitivity | Reported (no threshold) |
| Bootstrap 95% CI | Confidence interval on Brier | CI must not overlap best baseline |

## Dependencies

- `xgboost` — gradient boosting implementation
- `scikit-learn` — logistic regression, calibration, metrics, search
- `shap` — feature importance (SHAP TreeExplainer)

## Risks

1. **Overfitting on small folds:** Fold 1 has only 300 training observations. Monitor train-val gap.
2. **BANKNIFTY expansion medium:** Folds 5–6 had insufficient positive samples in R3. May fail for this combination.
3. **Calibration instability:** Isotonic regression on <60 validation samples may overfit. Monitor ECE variance across folds.

---

*This TDR must be committed to Git before any model training begins.*
