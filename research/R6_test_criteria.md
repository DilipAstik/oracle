# Oracle-V Model Research — R6: Test Set Evaluation Criteria

**Date Frozen:** 2026-02-21
**Status:** FROZEN — must not be altered after creation (Governance §10.2)

---

## Purpose

This document freezes the success criteria for single-shot test-set evaluation
BEFORE any test-set predictions are generated. Per Governance §10.2, test criteria
must be documented and committed prior to test evaluation to prevent post-hoc
reinterpretation.

## Combinations Under Test

Based on R5 walk-forward results (Tier 1 + Tier 2 only):

| # | Underlying | Target | Horizon | R5 Brier | R5 Improvement | Tier |
|---|---|---|---|---|---|---|
| 1 | BANKNIFTY | Expansion | Short | 0.1117 | +0.0202 | Tier 1 |
| 2 | NIFTY | Compression | Medium | 0.1720 | +0.0185 | Tier 2 |
| 3 | NIFTY | Expansion | Short | 0.1286 | +0.0112 | Tier 2 |
| 4 | NIFTY | Expansion | Medium | 0.1307 | +0.0099 | Tier 2 |

### Excluded from Test (Tier 3 — No Demonstrable Edge)

- NIFTY compression short (improvement 0.0055)
- BANKNIFTY compression short (improvement 0.0031)
- BANKNIFTY compression medium (improvement 0.0008)
- BANKNIFTY expansion medium (improvement -0.0278)

## Success Criteria

| Metric | Threshold | Rationale |
|---|---|---|
| Brier score | ≤ 0.22 | Spec §7.2.6 Tier 2 absolute ceiling |
| ECE | ≤ 0.13 | Spec §7.2.6 calibration requirement |
| AUC-ROC | ≥ 0.58 | Relaxed from 0.62 for test set (smaller sample) |
| Baseline outperformance (Brier) | ≥ 0.02 | Spec §7.2.5 — primary gate |

## Outcome Interpretation (Pre-Declared)

| Test Result | Interpretation | Action |
|---|---|---|
| Improvement ≥ 0.02 | **Confirmed production candidate** | Proceed to Atlas product spec |
| Improvement 0.01–0.02 | **Research-valid, not yet tradable** | Document; revisit with more data |
| Improvement < 0.01 or negative | **Validation noise — discard** | Exclude from v1; defer to v2 |

## Methodology

1. **Training data:** ALL non-test observations (trade dates before 2025-10-27)
2. **Hyperparameters:** Best params from R5 last fold (fold 6, most data)
3. **Calibration:** Isotonic regression fitted on R5 last fold validation predictions
4. **Test data:** Frozen hold-out (2025-10-27 to 2026-02-19, ~80 trading days)
5. **Baselines:** Recomputed on test set using training-set parameters
6. **Single shot:** No iteration. Results are final regardless of outcome.

## Pre-Test Checklist

- [ ] This document committed to Git before any test predictions
- [ ] R5 training_results.json contains all needed hyperparameters
- [ ] No code changes to model architecture since R5 commit
- [ ] Test set has never been accessed by any training or validation code

---

*This document is immutable from the point of Git commit.*
