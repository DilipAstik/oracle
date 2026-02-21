# Oracle-V Model Research — R5: Training Results

**Generated:** 2026-02-21 05:24:59
**Models:** XGBoost (primary), Logistic Regression (sanity check)
**Features:** 15 (post-R1 drops)
**Calibration:** Isotonic regression on validation fold

---

## Summary Table (XGBoost, Calibrated)

| UL | Target | Horizon | Brier (mean±std) | ECE | AUC-ROC | Gap | Best BL | Improvement | Pass? |
|---|---|---|---|---|---|---|---|---|---|
| NIFTY | expansion | horizon_medium | 0.1307±0.0528 | 0.0115 | 0.5722 | 0.0329 | naive (0.1406) | 0.0099 | ❌ |
| NIFTY | expansion | horizon_short | 0.1286±0.0399 | 0.0175 | 0.6163 | 0.0360 | naive (0.1398) | 0.0112 | ❌ |
| NIFTY | compression | horizon_medium | 0.1720±0.0148 | 0.0128 | 0.6013 | -0.0200 | riv_momentum (0.1905) | 0.0185 | ❌ |
| NIFTY | compression | horizon_short | 0.1783±0.0195 | 0.0122 | 0.5503 | -0.0155 | garch (0.1838) | 0.0055 | ❌ |
| BANKNIFTY | expansion | horizon_medium | 0.1536±0.0287 | 0.0293 | 0.5503 | 0.0178 | persistence (0.1258) | -0.0278 | ❌ |
| BANKNIFTY | expansion | horizon_short | 0.1117±0.0437 | 0.0215 | 0.6573 | 0.0529 | naive (0.1319) | 0.0202 | ✅ |
| BANKNIFTY | compression | horizon_medium | 0.1715±0.0319 | 0.0400 | 0.6057 | -0.0024 | persistence (0.1723) | 0.0008 | ❌ |
| BANKNIFTY | compression | horizon_short | 0.1648±0.0246 | 0.0344 | 0.5836 | 0.0008 | riv_momentum (0.1680) | 0.0031 | ❌ |

## SHAP Feature Importance Stability

| UL | Target | Horizon | Kendall τ (mean) | Interpretation |
|---|---|---|---|---|
| NIFTY | expansion | horizon_medium | 0.161 | ⚠️ Unstable |
| NIFTY | expansion | horizon_short | 0.507 | 🟡 Moderate |
| NIFTY | compression | horizon_medium | 0.189 | ⚠️ Unstable |
| NIFTY | compression | horizon_short | 0.168 | ⚠️ Unstable |
| BANKNIFTY | expansion | horizon_medium | 0.406 | 🟡 Moderate |
| BANKNIFTY | expansion | horizon_short | 0.250 | ⚠️ Unstable |
| BANKNIFTY | compression | horizon_medium | 0.276 | ⚠️ Unstable |
| BANKNIFTY | compression | horizon_short | 0.444 | 🟡 Moderate |

## XGBoost vs Logistic Regression

| UL | Target | Horizon | XGB Brier | LR Brier | Winner |
|---|---|---|---|---|---|
| NIFTY | expansion | horizon_medium | 0.1307 | 0.1215 | Logistic |
| NIFTY | expansion | horizon_short | 0.1286 | 0.1049 | Logistic |
| NIFTY | compression | horizon_medium | 0.1720 | 0.1602 | Logistic |
| NIFTY | compression | horizon_short | 0.1783 | 0.1522 | Logistic |
| BANKNIFTY | expansion | horizon_medium | 0.1536 | 0.1302 | Logistic |
| BANKNIFTY | expansion | horizon_short | 0.1117 | 0.1109 | Logistic |
| BANKNIFTY | compression | horizon_medium | 0.1715 | 0.1660 | Logistic |
| BANKNIFTY | compression | horizon_short | 0.1648 | 0.1513 | Logistic |

---

## Validation Thresholds (§7.2.6)

| Metric | Threshold | Status |
|---|---|---|
| Brier ≤ 0.22 | Worst: 0.1783 | ✅ |
| ECE ≤ 0.13 | Worst: 0.0400 | ✅ |
| AUC-ROC ≥ 0.62 | Worst: 0.5503 | ❌ |

*Report generated at completion of R5.*