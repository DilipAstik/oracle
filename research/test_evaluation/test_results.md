# Oracle-V Model Research — R6: Test Set Evaluation Results

**Generated:** 2026-02-21 07:33:34
**Type:** Single-shot hold-out evaluation (Oct 2025 – Feb 2026)
**Criteria:** Frozen in R6_test_criteria.md before evaluation

---

## Results Summary

| # | UL | Target | Horizon | Tier | Test Brier | 95% CI | Best BL (Brier) | Improvement | Pass? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BANKNIFTY | expansion | horizon_short | 1 | 0.0472 | [0.0177, 0.0852] | garch (0.0498) | +0.0026 | ❌ FAIL |
| 2 | NIFTY | compression | horizon_medium | 2 | 0.4151 | [0.3285, 0.5157] | garch (0.2801) | -0.1350 | ❌ FAIL |
| 3 | NIFTY | expansion | horizon_short | 2 | 0.0798 | [0.0295, 0.1356] | naive (0.0902) | +0.0104 | ❌ FAIL |
| 4 | NIFTY | expansion | horizon_medium | 2 | 0.1071 | [0.0427, 0.1852] | persistence (0.0694) | -0.0377 | ❌ FAIL |

## Detailed Metrics

| # | Brier | ECE | AUC-ROC | Log-Loss | Brier ≤ 0.22 | ECE ≤ 0.13 | AUC ≥ 0.58 |
|---|---|---|---|---|---|---|---|
| 1 | 0.0472 | 0.0274 | 0.7586 | 0.1697 | ✅ | ✅ | ✅ |
| 2 | 0.4151 | 0.3696 | 0.4264 | 1.8501 | ❌ | ❌ | ❌ |
| 3 | 0.0798 | 0.0375 | 0.6000 | 0.2866 | ✅ | ✅ | ✅ |
| 4 | 0.1071 | 0.0645 | 0.4776 | 0.5846 | ✅ | ✅ | ❌ |

## XGBoost vs Logistic Regression (Test Set)

| # | XGB Brier | LR Brier | Winner |
|---|---|---|---|
| 1 | 0.0472 | 0.0591 | XGBoost |
| 2 | 0.4151 | 0.3091 | Logistic |
| 3 | 0.0798 | 0.0855 | XGBoost |
| 4 | 0.1071 | 0.0646 | Logistic |

## Test Set Baseline Detail

| # | Naive | Persistence | GARCH | RIV Momentum | Best |
|---|---|---|---|---|---|
| 1 | 0.0670 | 0.0779 | 0.0498 | 0.0662 | garch |
| 2 | 0.2878 | 0.3333 | 0.2801 | 0.3255 | garch |
| 3 | 0.0902 | 0.1549 | 0.0980 | 0.0987 | naive |
| 4 | 0.0785 | 0.0694 | 0.0858 | 0.0875 | persistence |

## Walk-Forward (R5) vs Test (R6) Comparison

| # | R5 Brier | R6 Brier | Δ | Interpretation |
|---|---|---|---|---|
| 1 | 0.1117 | 0.0472 | -0.0645 | Improved (check for luck) |
| 2 | 0.1720 | 0.4151 | +0.2431 | Degraded (expected some) |
| 3 | 0.1286 | 0.0798 | -0.0488 | Improved (check for luck) |
| 4 | 0.1307 | 0.1071 | -0.0236 | Improved (check for luck) |

---

## Final Verdict

**0 of 4 combinations pass the 0.02 improvement threshold.**


### Deferred to v2 (Below Threshold)

- BANKNIFTY expansion horizon_short — improvement +0.0026
- NIFTY compression horizon_medium — improvement -0.1350
- NIFTY expansion horizon_short — improvement +0.0104
- NIFTY expansion horizon_medium — improvement -0.0377

*Report generated: 2026-02-21 07:33:34. Results are final — no iteration permitted.*