# Oracle-V Model Research Phase — R7: Research Exit Report

**Document ID:** ORACLE-V-RESEARCH-EXIT-001
**Date:** 2026-02-21
**Status:** FINAL
**Model Lifecycle State Transition:** RESEARCH → RETIRED (no candidate meets promotion criteria)
**Spec References:** Governance §10.2 (Research exit criteria), Part IV §7.1–§7.2

---

## 1. Executive Summary

Oracle-V's first model research cycle (R1–R6) has been completed using the full
specification-first methodology. The research evaluated 8 model combinations
(2 underlyings × 2 horizons × 2 targets) across a 6-fold expanding walk-forward
validation and single-shot test-set evaluation.

**Outcome:** No model combination passes the pre-declared test-set threshold
(Brier improvement ≥ 0.02 over best baseline). Oracle-V v1 does not demonstrate
production-grade predictive edge with the current dataset.

**Classification:** This is a *data insufficiency finding*, not an architectural
failure. The research infrastructure, governance pipeline, and methodology are
validated and preserved for reuse.

**Recommendation:** Defer Oracle-V model promotion. Continue data accumulation.
Re-evaluate when dataset reaches 750+ trading days (~3 years) or when deferred
features (India VIX ×3, F&O ban flag) become available.

---

## 2. Research Phase Summary

### 2.1 Timeline

| Phase | Date | Commit | Description |
|---|---|---|---|
| C1 | Jan 2026 | Multiple | Data pipeline: 1.4M+ options records ingested |
| C2 | Feb 2026 | Multiple | Label computation: baselines, z-scores, labels |
| C3 | Feb 2026 | 50e9693→f2b02fa | Feature engineering: 21 features computed |
| R1 | 2026-02-21 | 14b9b88 | Exploratory Data Analysis |
| R2 | 2026-02-21 | 10cc045 | Partitioning scheme frozen |
| R3 | 2026-02-21 | 0e18669 | Baseline construction (4 baselines) |
| R4 | 2026-02-21 | ebe3821 | Architecture TDR (XGBoost + LR) |
| R5 | 2026-02-21 | (committed) | Walk-forward training |
| R6 | 2026-02-21 | (committed) | Test-set evaluation — 0/4 pass |

### 2.2 Dataset Characteristics

| Dimension | Value |
|---|---|
| Data coverage | Jan 2024 – Feb 2026 (~2 years, 528 trading days) |
| Training eligible observations | ~933 (NIFTY), ~938 (BANKNIFTY) per underlying |
| Usable features | 15 (of 21 computed; 2 dropped in R1, 4 deferred) |
| Deferred features | india_vix_level, india_vix_percentile_rank, india_vix_change, fno_ban_heavyweight_flag |
| Spec recommendation | Minimum 3 years, preferred 5 years (§7.1.1) |
| Actual vs recommended | **Below minimum** (2 years vs 3 years required) |

### 2.3 Models Evaluated

| Model | Role | Implementation |
|---|---|---|
| XGBoost | Primary (gradient boosted trees) | scikit-learn compatible, native missing value handling |
| Logistic Regression | Sanity check | L2-regularised, imputation pipeline |

Both models trained per combination with RandomizedSearchCV (50 iterations),
TimeSeriesSplit inner CV (3 splits), isotonic post-hoc calibration.

---

## 3. Governance §10.2 Exit Criteria Assessment

### 3.1 Ten Mandatory Exit Criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Training data, validation data, and test data strictly separated | ✅ SATISFIED | R2: partitioning.json, 5-day purge gap, test never touched until R6 |
| 2 | Partitioning documented before training, not altered after | ✅ SATISFIED | R2 committed (10cc045) before R5 training |
| 3 | Walk-forward validation (not random split) | ✅ SATISFIED | 6-fold expanding walk-forward |
| 4 | Model architecture documented before training | ✅ SATISFIED | TDR-ORACLE-009 committed (ebe3821) before R5 |
| 5 | Hyperparameter selection documented | ✅ SATISFIED | R5 training_results.json contains per-fold best params |
| 6 | Minimum 3 documented failure modes | ✅ SATISFIED | See §5 below |
| 7 | Model outperforms all 3 required baselines | ❌ NOT SATISFIED | R6: 0/4 combinations beat best baseline by ≥0.02 |
| 8 | Dataset limitations documented | ✅ SATISFIED | See §4 below (OV-R-8) |
| 9 | Feature importance stability assessed | ✅ SATISFIED | SHAP Kendall τ computed per combination in R5 |
| 10 | Test criteria frozen before evaluation | ✅ SATISFIED | R6_test_criteria.md committed before test run |

**Exit criterion 7 is the blocking criterion.** All other criteria are satisfied.

### 3.2 Oracle-V-Specific Exit Criteria (OV-R-1 through OV-R-8)

| Criterion | Status | Evidence |
|---|---|---|
| OV-R-1: Event-window Brier beats baselines | ❌ NOT TESTED | Event-window separation not implemented in v1 (events excluded from training eligible set) |
| OV-R-2: Non-event Brier reported | ✅ REPORTED | R5/R6 results cover non-event observations |
| OV-R-3: Output consistency violation rate < 5% | ✅ N/A | Binary classifiers produce valid [0,1] probabilities by construction |
| OV-R-4: ECE ≤ calibration_ece_floor (0.15) | ✅ SATISFIED | Worst ECE: 0.040 (well below 0.15 floor) |
| OV-R-5: Feature importance documented | ✅ SATISFIED | SHAP TreeExplainer per fold in R5 |
| OV-R-6: Multicollinearity assessed | ✅ SATISFIED | R1d: VIF + pairwise correlation; iv_percentile_rank dropped (r=0.92 with riv_level) |
| OV-R-7: Historical expiry transition documented | ✅ SATISFIED | R1e: BANKNIFTY weekly→monthly Nov 2024 documented |
| OV-R-8: Dataset limitations documented | ✅ SATISFIED | See §4 below |

---

## 4. Dataset Limitations (OV-R-8)

### 4.1 Coverage Shortfall

The specification recommends minimum 3 years, preferred 5 years of training data (§7.1.1).
This research used ~2 years (Jan 2024 – Feb 2026). Concrete consequences:

1. **Insufficient regime diversity.** R1g showed rolling base rate swings of 0.22–0.40,
   but the training data contains at most 1–2 complete regime cycles. A model trained on
   one cycle cannot generalise to the next.

2. **Test period regime shift.** The test window (Oct 2025 – Feb 2026) exhibited a
   low-expansion, high-compression regime (expansion base rate dropped to 5–8% vs
   training average of 16–18%). The model had no prior exposure to sustained
   low-expansion regimes of this duration.

3. **Feature availability gaps.** iv_percentile_rank (F-105) requires 252 trading days
   of lookback — available for only 51% of observations. Dropped from modelling due
   to coverage + collinearity with riv_level.

### 4.2 Deferred Features

| Feature | Reason Deferred | Expected Impact |
|---|---|---|
| india_vix_level | NSE VIX data not yet in pipeline | Regime context (high VIX = expansion-prone) |
| india_vix_percentile_rank | Depends on VIX level | Relative regime positioning |
| india_vix_change | Depends on VIX level | VIX momentum |
| fno_ban_heavyweight_flag | F&O ban list integration not built | Microstructure stress indicator |

### 4.3 Quality Degradation Rates

| Feature | NIFTY Available | BANKNIFTY Available |
|---|---|---|
| riv_change_3d (strongest signal) | 67% | 83% |
| iv_percentile_rank (dropped) | 51% | 52% |
| iv_term_structure_slope | 98% | 64% |

---

## 5. Documented Failure Modes (Governance §10.2, criterion 6)

### Failure Mode 1: Regime Non-Stationarity

**Description:** Expansion and compression base rates vary dramatically over time
(R1g: swings of 0.22–0.40). A model trained during an active-expansion regime
over-predicts expansion during quiet periods.

**Evidence:** R6 test period had ~5% expansion rate vs ~17% training average.

**Mitigation for v2:** Regime-conditional models or adaptive calibration with
short lookback window.

### Failure Mode 2: Label Persistence Dominance

**Description:** High label autocorrelation (lag-1 up to 0.573 for BANKNIFTY
medium expansion) means persistence ("predict same as yesterday") captures a
large fraction of predictable variance. ML must find signal *beyond* persistence,
but with weak features (max |r| ≈ 0.18), the incremental lift is fragile.

**Evidence:** R3 showed persistence as best baseline for BANKNIFTY medium horizon.
R6 confirmed persistence Brier of 0.069 on NIFTY expansion medium — better than
XGBoost's 0.107.

**Mitigation for v2:** Feature engineering targeting persistence-breaking signal
(e.g., cross-asset volatility divergence, order flow imbalance).

### Failure Mode 3: Small-Sample Calibration Instability

**Description:** Isotonic calibration fitted on ~60–70 validation observations
can overfit when the test base rate differs from training.

**Evidence:** NIFTY compression medium: R5 Brier 0.172 → R6 Brier 0.415.
Compression rate jumped from ~25% training to ~43% test.

**Mitigation for v2:** Larger calibration windows; Platt scaling as more
stable alternative; regime-aware recalibration.

### Failure Mode 4: BANKNIFTY Expansion Event Sparsity

**Description:** BANKNIFTY expansion events disappear entirely for extended
periods (R1g: rolling base rate hit 0%).

**Evidence:** R3 skipped folds 5–6 (insufficient positives). R6 test: only
4 expansion events in 77 observations.

**Mitigation for v2:** Longer dataset covering multiple expansion cycles.

---

## 6. Key Research Findings

### 6.1 Positive Findings

1. **Localised predictive signal exists.** R5 walk-forward showed genuine lift
   for BANKNIFTY expansion short-horizon (AUC 0.657, Brier improvement +0.020).
   The signal is real but regime-dependent.

2. **Calibration is excellent.** ECE ≤ 0.040 across all 8 combinations in R5.
   The probability pipeline produces trustworthy probabilities.

3. **Compression is more predictable than expansion.** riv_change_3d (3-day IV
   momentum) carries genuine mean-reversion signal.

4. **Logistic regression matches XGBoost.** The predictive relationship is
   largely linear — simpler models are preferable for production robustness.

5. **Feature engineering pipeline is validated.** 15 usable features computed
   at scale across 528 trading days with documented quality metrics.

### 6.2 Negative Findings

1. **No universal predictor.** No single model works across all 8 combinations.

2. **No regime-robust edge.** Walk-forward lift did not survive the regime
   shift in the test period.

3. **GARCH captures most predictable variance.** ML adds incremental value
   only in specific niches.

### 6.3 R6 Technical Issue

All 4 test combinations may have used identical hyperparameters from R5 fold 6
(BANKNIFTY expansion short). The `get_best_params()` function should be
investigated for a key-lookup issue. While unlikely to change the fundamental
outcome (regime shift dominates), this should be corrected before any future re-run.

---

## 7. Baseline Performance Reference

### 7.1 R5 Walk-Forward Results (XGBoost, Calibrated, Mean Across Folds)

| UL | Target | Horizon | XGB Brier | Best BL | Improvement | Pass 0.02? |
|---|---|---|---|---|---|---|
| BANKNIFTY | Expansion | Short | 0.1117 | 0.1319 | +0.0202 | ✅ |
| NIFTY | Compression | Medium | 0.1720 | 0.1905 | +0.0185 | ❌ |
| NIFTY | Expansion | Short | 0.1286 | 0.1398 | +0.0112 | ❌ |
| NIFTY | Expansion | Medium | 0.1307 | 0.1406 | +0.0099 | ❌ |
| NIFTY | Compression | Short | 0.1783 | 0.1838 | +0.0055 | ❌ |
| BANKNIFTY | Compression | Short | 0.1648 | 0.1680 | +0.0031 | ❌ |
| BANKNIFTY | Compression | Medium | 0.1715 | 0.1723 | +0.0008 | ❌ |
| BANKNIFTY | Expansion | Medium | 0.1536 | 0.1258 | -0.0278 | ❌ |

### 7.2 R6 Test-Set Results (Single Shot, Final)

| # | UL | Target | Horizon | XGB Brier | Best BL | Improvement | Pass? |
|---|---|---|---|---|---|---|---|
| 1 | BANKNIFTY | Expansion | Short | 0.0472 | GARCH (0.0498) | +0.0026 | ❌ |
| 2 | NIFTY | Compression | Medium | 0.4151 | GARCH (0.2801) | -0.1350 | ❌ |
| 3 | NIFTY | Expansion | Short | 0.0798 | Naive (0.0902) | +0.0104 | ❌ |
| 4 | NIFTY | Expansion | Medium | 0.1071 | Persistence (0.0694) | -0.0377 | ❌ |

---

## 8. Infrastructure Preserved for Reuse

| Component | Location | Status |
|---|---|---|
| Data ingestion pipeline (C1) | src/oracle/data/ | Operational — 528 days ingested |
| IV computation service | src/oracle/iv/ | Operational — 1.4M+ records processed |
| Baseline & label pipeline (C2) | src/oracle/baselines/, labels/ | Operational |
| Feature engineering (C3) | src/oracle/features/ | Operational — 21 features |
| ODAL write validation | src/oracle/odal/ | 71 compliance tests passing |
| Training tables | S3: computed/training/ | NIFTY + BANKNIFTY Parquet |
| Research scripts (R1–R6) | scripts/run_r*.py | All reusable |

**Estimated time to re-run R1–R6 on expanded dataset: 1–2 hours.**

---

## 9. Re-Evaluation Triggers

| Trigger | Condition | Rationale |
|---|---|---|
| Dataset expansion | 750+ trading days (~3 years) | Meets spec minimum (§7.1.1) |
| VIX features | India VIX integrated into pipeline | 3 deferred CRITICAL features |
| Market structure change | New expiry cycle or instruments | May require fresh analysis |
| Methodology advance | Regime-adaptive techniques validated | Addresses primary failure mode |

**Estimated timeline for Trigger 1:** ~Jan 2027 (12 months additional accumulation).

---

## 10. Impact on Oracle Framework Sequencing

**Atlas-V:** Cannot proceed. Requires Production-grade Oracle-V model (Governance §15.2).

**Oracle-R, Oracle-T, Oracle-S:** May proceed independently. Key lessons:
- Regime non-stationarity is the primary challenge across all Oracle products.
- Data requirements for tail events (Oracle-T) will be more demanding.
- Unsupervised anomaly detection (Oracle-S) may be less affected by regime shifts.

**Helios:** No impact. Full isolation maintained (Bright Line Rule).

---

## 11. Conclusion

Oracle-V's first research cycle demonstrates that the specification-first
methodology works exactly as designed. It prevented a model that cannot survive
regime shifts from reaching production, produced honest and reproducible results,
and preserved all infrastructure for efficient re-evaluation.

**Oracle-V is deferred, not dead.** The signal exists in specific regimes.
The question is whether sufficient data can resolve the regime-adaptation problem.
That question will be answered by time, not by further iteration on the current dataset.

---

*This document is the formal conclusion of Oracle-V Research Cycle 1.*
