# Oracle-V Model Research — R2: Partitioning Scheme

**Generated:** 2026-02-21 04:54:09
**Status:** FROZEN — must not be altered after training begins (Governance §10.2)

---

## Specification Compliance

| Requirement | Spec Reference | This Scheme |
|---|---|---|
| Walk-forward mandatory | §7.2.2 | ✅ Expanding walk-forward, 6 folds |
| Random splits prohibited | §7.2.2 | ✅ Chronological only |
| Purge gap = max(T) | §7.2.2 | ✅ 5 trading days |
| min_walkforward_folds ≥ 5 | §7.2.2 | ✅ 6 folds |
| Test set held out completely | Governance §10.2 | ✅ Final 15%, single-shot |
| Documented before training | Governance §10.2 | ✅ This document |

## Design Principles

1. **Mechanical expanding walk-forward** — no manual regime-aware fold placement
2. **Percentage-based boundaries** on trading day index — fully reproducible
3. **Feature preprocessing refit** inside each fold — no global statistics leakage
4. **Regime analysis** is a post-evaluation diagnostic, not a partition design input

## Feature Set (R1 Decisions)

**Modelling features: 15**

- `day_of_week`
- `days_to_current_expiry`
- `is_monthly_expiry_week`
- `days_to_next_holiday`
- `riv_level`
- `riv_change_1d`
- `riv_change_3d`
- `iv_term_structure_slope`
- `iv_skew_25d`
- `iv_rv_spread_5d`
- `iv_rv_spread_10d`
- `pcr_volume`
- `pcr_oi`
- `oi_change_net_1d`
- `volume_zscore_session`

**Dropped features (Specification-Constrained Omission):**

- `time_of_day_bucket` — Constant value (19), zero variance, infinite VIF
- `iv_percentile_rank` — 49% UNAVAILABLE + r=0.92 with `riv_level` (§5.4.1)

## Walk-Forward Fold Structure

| Fold | Train Period | Train Days | Purge | Val Period | Val Days | Train Eligible | Val Eligible |
|---|---|---|---|---|---|---|---|
| 1 | 2024-01-01 → 2024-09-23 | 179 | 5d | 2024-10-01 → 2025-01-07 | 67 | 300 | 114 |
| 2 | 2024-01-01 → 2024-11-28 | 224 | 5d | 2024-12-06 → 2025-03-12 | 67 | 372 | 122 |
| 3 | 2024-01-01 → 2025-01-30 | 268 | 5d | 2025-02-07 → 2025-05-21 | 67 | 453 | 121 |
| 4 | 2024-01-01 → 2025-04-08 | 313 | 5d | 2025-04-21 → 2025-07-23 | 67 | 534 | 125 |
| 5 | 2024-01-01 → 2025-05-15 | 336 | 5d | 2025-05-23 → 2025-08-26 | 67 | 575 | 130 |
| 6 | 2024-01-01 → 2025-06-27 | 367 | 5d | 2025-07-07 → 2025-10-10 | 67 | 633 | 131 |

## Test Hold-Out

| Period | 2025-10-27 → 2026-02-19 |
|---|---|
| Trading days | 80 |
| Eligible observations | 143 |
| Usage | Single-shot evaluation (R6) — NEVER touched before then |

## Purge Gap Rationale

The purge gap of 5 trading days equals max(T) = 5 (the longest prediction horizon).
This prevents any label computed from data overlapping the validation period from
entering training. A label at time t requires data at t+T to resolve — training data
within T days of the validation boundary could contain labels computed from data that
overlaps with the validation period.

## Pre-Training Checklist

Before any model training begins, verify:

- [ ] This document has been committed to Git
- [ ] `partitioning.json` matches this document
- [ ] No training code accesses test set dates
- [ ] Feature preprocessing pipeline refits per fold
- [ ] Class weighting: START UNWEIGHTED (add only if recall collapses)

---
*Document frozen at generation time. Any modification requires formal amendment.*