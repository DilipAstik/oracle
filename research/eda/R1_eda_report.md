# Oracle-V Model Research — R1: Exploratory Data Analysis Report

**Generated:** 2026-02-21 04:32:11
**Phase:** Model Research, Sub-phase R1
**Spec References:** Part IV §7.1.1, Governance §10.2, OV-R-5/6/7

---

## R1a: Label Distribution Analysis

**Purpose:** Verify class balance; compute naive baseline rates.

| Underlying | Horizon | N Total | N Eligible | Expansion | Compression | Stable |
|---|---|---|---|---|---|---|
| NIFTY | horizon_medium | 528 | 478 | 16.9% | 25.5% | 57.5% |
| NIFTY | horizon_short | 528 | 455 | 18.0% | 24.8% | 57.1% |
| BANKNIFTY | horizon_medium | 528 | 467 | 16.9% | 23.3% | 59.7% |
| BANKNIFTY | horizon_short | 528 | 471 | 16.3% | 22.3% | 61.4% |


## R1b: Feature Distribution Profiling


### NIFTY (n=933 eligible rows)

| Feature | Available% | Mean | Std | Min | Max | Skew | VALID | DEGRADED | UNAVAIL |
|---|---|---|---|---|---|---|---|---|---|
| day_of_week | 100.0% | 1.9550 | 1.4074 | 0.0000 | 4.0000 | 0.06 | 933 | 0 | 0 |
| time_of_day_bucket | 100.0% | 19.0000 | 0.0000 | 19.0000 | 19.0000 | 0.00 | 933 | 0 | 0 |
| days_to_current_expiry | 100.0% | 1.8971 | 1.3554 | 0.0000 | 4.0000 | 0.06 | 933 | 0 | 0 |
| is_monthly_expiry_week | 100.0% | 0.2519 | 0.4343 | 0.0000 | 1.0000 | 1.15 | 933 | 0 | 0 |
| days_to_next_holiday | 100.0% | 7.1447 | 3.5263 | 0.0000 | 10.0000 | -0.79 | 933 | 0 | 0 |
| riv_level | 100.0% | 0.1263 | 0.0297 | 0.0711 | 0.2891 | 1.34 | 933 | 0 | 0 |
| riv_change_1d | 88.7% | -0.0012 | 0.0136 | -0.0625 | 0.1703 | 2.76 | 773 | 55 | 105 |
| riv_change_3d | 66.8% | -0.0024 | 0.0221 | -0.1836 | 0.1648 | 0.00 | 585 | 38 | 310 |
| iv_percentile_rank | 51.4% | 0.3412 | 0.2917 | 0.0000 | 0.9960 | 0.65 | 480 | 0 | 453 |
| iv_term_structure_slope | 97.9% | 0.0024 | 0.0144 | -0.0891 | 0.1091 | 2.57 | 913 | 0 | 20 |
| iv_skew_25d | 100.0% | 0.0371 | 0.0192 | -0.0536 | 0.1659 | 1.26 | 933 | 0 | 0 |
| iv_rv_spread_5d | 98.9% | 0.0176 | 0.0613 | -0.4589 | 0.2183 | -3.38 | 923 | 0 | 10 |
| iv_rv_spread_10d | 97.9% | 0.0092 | 0.0549 | -0.3102 | 0.1657 | -2.77 | 913 | 0 | 20 |
| pcr_volume | 100.0% | 0.9060 | 0.1489 | 0.3606 | 1.6546 | 0.69 | 933 | 0 | 0 |
| pcr_oi | 100.0% | 0.9902 | 0.2096 | 0.4460 | 1.8038 | 0.39 | 933 | 0 | 0 |
| oi_change_net_1d | 100.0% | 19700438.1618 | 26092960.3231 | -4919475.0000 | 132092925.0000 | 1.95 | 933 | 0 | 0 |
| volume_zscore_session | 96.6% | -0.0210 | 0.9248 | -1.2248 | 3.4714 | 1.44 | 901 | 0 | 32 |


### BANKNIFTY (n=938 eligible rows)

| Feature | Available% | Mean | Std | Min | Max | Skew | VALID | DEGRADED | UNAVAIL |
|---|---|---|---|---|---|---|---|---|---|
| day_of_week | 100.0% | 1.9787 | 1.4095 | 0.0000 | 4.0000 | 0.04 | 938 | 0 | 0 |
| time_of_day_bucket | 100.0% | 19.0000 | 0.0000 | 19.0000 | 19.0000 | 0.00 | 938 | 0 | 0 |
| days_to_current_expiry | 100.0% | 7.1386 | 6.3377 | 0.0000 | 24.0000 | 0.69 | 938 | 0 | 0 |
| is_monthly_expiry_week | 100.0% | 0.2452 | 0.4304 | 0.0000 | 1.0000 | 1.19 | 938 | 0 | 0 |
| days_to_next_holiday | 100.0% | 7.0928 | 3.5612 | 0.0000 | 10.0000 | -0.77 | 938 | 0 | 0 |
| riv_level | 100.0% | 0.1482 | 0.0375 | 0.0870 | 0.3650 | 1.63 | 938 | 0 | 0 |
| riv_change_1d | 93.5% | -0.0005 | 0.0130 | -0.1270 | 0.1124 | 0.78 | 817 | 60 | 61 |
| riv_change_3d | 82.9% | -0.0013 | 0.0199 | -0.1534 | 0.1117 | -0.34 | 703 | 75 | 160 |
| iv_percentile_rank | 51.5% | 0.2914 | 0.2719 | 0.0000 | 0.9722 | 0.82 | 483 | 0 | 455 |
| iv_term_structure_slope | 63.9% | 0.0035 | 0.0224 | -0.1050 | 0.1395 | 2.48 | 599 | 0 | 339 |
| iv_skew_25d | 100.0% | 0.0307 | 0.0239 | -0.1256 | 0.1630 | -1.10 | 938 | 0 | 0 |
| iv_rv_spread_5d | 99.0% | 0.0235 | 0.0797 | -0.6198 | 0.2383 | -4.01 | 929 | 0 | 9 |
| iv_rv_spread_10d | 98.0% | 0.0159 | 0.0693 | -0.4135 | 0.2471 | -3.20 | 919 | 0 | 19 |
| pcr_volume | 100.0% | 0.8951 | 0.1311 | 0.3800 | 1.6130 | -0.04 | 938 | 0 | 0 |
| pcr_oi | 100.0% | 0.9189 | 0.1663 | 0.3768 | 1.8249 | -0.08 | 938 | 0 | 0 |
| oi_change_net_1d | 100.0% | 3370262.5533 | 5962563.9070 | -2403390.0000 | 45126255.0000 | 3.58 | 938 | 0 | 0 |
| volume_zscore_session | 96.8% | 0.0268 | 1.0365 | -0.8356 | 4.1685 | 2.33 | 908 | 0 | 30 |


## R1c: Feature-Label Correlation + Mutual Information

**Methods:** Point-biserial correlation (linear), Mutual Information (non-linear).


### NIFTY


**Target: label_expansion**

| Feature | Corr (r) | p-value | MI Score | Signal |
|---|---|---|---|---|
| day_of_week | 0.1010 | 0.0020 | 0.0124 | 🟡 Moderate |
| time_of_day_bucket | nan | nan | 0.0021 | ⚪ Weak |
| days_to_current_expiry | 0.0016 | 0.9609 | 0.0099 | ⚪ Weak |
| is_monthly_expiry_week | 0.0712 | 0.0297 | 0.0000 | ⚪ Weak |
| days_to_next_holiday | -0.0613 | 0.0611 | 0.0000 | ⚪ Weak |
| riv_level | 0.0525 | 0.1088 | 0.0163 | ⚪ Weak |
| riv_change_1d | 0.0205 | 0.5557 | 0.0000 | ⚪ Weak |
| riv_change_3d | 0.0138 | 0.7311 | 0.0000 | ⚪ Weak |
| iv_percentile_rank | 0.0656 | 0.1512 | 0.0140 | ⚪ Weak |
| iv_term_structure_slope | 0.0583 | 0.0785 | 0.0000 | ⚪ Weak |
| iv_skew_25d | 0.0076 | 0.8176 | 0.0000 | ⚪ Weak |
| iv_rv_spread_5d | 0.0850 | 0.0098 | 0.0049 | 🟡 Moderate |
| iv_rv_spread_10d | 0.0629 | 0.0576 | 0.0134 | ⚪ Weak |
| pcr_volume | 0.0519 | 0.1133 | 0.0044 | ⚪ Weak |
| pcr_oi | 0.0267 | 0.4160 | 0.0000 | ⚪ Weak |
| oi_change_net_1d | 0.1157 | 0.0004 | 0.0124 | 🟡 Moderate |
| volume_zscore_session | 0.0591 | 0.0763 | 0.0105 | ⚪ Weak |


**Target: label_compression**

| Feature | Corr (r) | p-value | MI Score | Signal |
|---|---|---|---|---|
| day_of_week | -0.0007 | 0.9820 | 0.0122 | ⚪ Weak |
| time_of_day_bucket | nan | nan | 0.0043 | ⚪ Weak |
| days_to_current_expiry | -0.0452 | 0.1674 | 0.0071 | ⚪ Weak |
| is_monthly_expiry_week | 0.0217 | 0.5086 | 0.0030 | ⚪ Weak |
| days_to_next_holiday | 0.0112 | 0.7325 | 0.0000 | ⚪ Weak |
| riv_level | 0.0161 | 0.6233 | 0.0253 | 🟡 Moderate |
| riv_change_1d | -0.1202 | 0.0005 | 0.0257 | 🟡 Moderate |
| riv_change_3d | -0.1828 | 0.0000 | 0.0089 | 🟢 Strong |
| iv_percentile_rank | -0.0341 | 0.4559 | 0.0000 | ⚪ Weak |
| iv_term_structure_slope | -0.1312 | 0.0001 | 0.0110 | 🟡 Moderate |
| iv_skew_25d | -0.0005 | 0.9871 | 0.0032 | ⚪ Weak |
| iv_rv_spread_5d | -0.0765 | 0.0201 | 0.0000 | ⚪ Weak |
| iv_rv_spread_10d | -0.0505 | 0.1273 | 0.0150 | ⚪ Weak |
| pcr_volume | -0.0303 | 0.3555 | 0.0000 | ⚪ Weak |
| pcr_oi | -0.0055 | 0.8668 | 0.0000 | ⚪ Weak |
| oi_change_net_1d | -0.0229 | 0.4841 | 0.0127 | ⚪ Weak |
| volume_zscore_session | -0.0184 | 0.5806 | 0.0370 | 🟡 Moderate |


### BANKNIFTY


**Target: label_expansion**

| Feature | Corr (r) | p-value | MI Score | Signal |
|---|---|---|---|---|
| day_of_week | 0.0820 | 0.0120 | 0.0055 | 🟡 Moderate |
| time_of_day_bucket | nan | nan | 0.0135 | ⚪ Weak |
| days_to_current_expiry | -0.1065 | 0.0011 | 0.0302 | 🟡 Moderate |
| is_monthly_expiry_week | 0.0050 | 0.8789 | 0.0223 | 🟡 Moderate |
| days_to_next_holiday | -0.1259 | 0.0001 | 0.0241 | 🟡 Moderate |
| riv_level | 0.1537 | 0.0000 | 0.0307 | 🟢 Strong |
| riv_change_1d | 0.1115 | 0.0009 | 0.0306 | 🟡 Moderate |
| riv_change_3d | 0.0873 | 0.0149 | 0.0241 | 🟡 Moderate |
| iv_percentile_rank | 0.1508 | 0.0009 | 0.0238 | 🟢 Strong |
| iv_term_structure_slope | -0.0742 | 0.0694 | 0.0215 | 🟡 Moderate |
| iv_skew_25d | -0.0263 | 0.4208 | 0.0141 | ⚪ Weak |
| iv_rv_spread_5d | 0.0877 | 0.0075 | 0.0269 | 🟡 Moderate |
| iv_rv_spread_10d | 0.0013 | 0.9677 | 0.0244 | 🟡 Moderate |
| pcr_volume | 0.0650 | 0.0467 | 0.0009 | ⚪ Weak |
| pcr_oi | -0.0006 | 0.9863 | 0.0228 | 🟡 Moderate |
| oi_change_net_1d | 0.1675 | 0.0000 | 0.0237 | 🟢 Strong |
| volume_zscore_session | -0.0521 | 0.1166 | 0.0335 | 🟡 Moderate |


**Target: label_compression**

| Feature | Corr (r) | p-value | MI Score | Signal |
|---|---|---|---|---|
| day_of_week | 0.0227 | 0.4883 | 0.0034 | ⚪ Weak |
| time_of_day_bucket | nan | nan | 0.0000 | ⚪ Weak |
| days_to_current_expiry | -0.0231 | 0.4793 | 0.0203 | 🟡 Moderate |
| is_monthly_expiry_week | 0.0090 | 0.7827 | 0.0065 | ⚪ Weak |
| days_to_next_holiday | 0.0072 | 0.8246 | 0.0000 | ⚪ Weak |
| riv_level | 0.0393 | 0.2293 | 0.0125 | ⚪ Weak |
| riv_change_1d | -0.1120 | 0.0009 | 0.0320 | 🟡 Moderate |
| riv_change_3d | -0.1702 | 0.0000 | 0.0654 | 🟢 Strong |
| iv_percentile_rank | 0.0563 | 0.2171 | 0.0190 | ⚪ Weak |
| iv_term_structure_slope | -0.0330 | 0.4200 | 0.0253 | 🟡 Moderate |
| iv_skew_25d | -0.0209 | 0.5223 | 0.0269 | 🟡 Moderate |
| iv_rv_spread_5d | 0.0198 | 0.5461 | 0.0482 | 🟡 Moderate |
| iv_rv_spread_10d | -0.0115 | 0.7272 | 0.0452 | 🟡 Moderate |
| pcr_volume | -0.1052 | 0.0013 | 0.0129 | 🟡 Moderate |
| pcr_oi | -0.0271 | 0.4079 | 0.0226 | 🟡 Moderate |
| oi_change_net_1d | -0.0560 | 0.0863 | 0.0004 | ⚪ Weak |
| volume_zscore_session | -0.0463 | 0.1633 | 0.0095 | ⚪ Weak |


## R1d: Multicollinearity Assessment (OV-R-6)

**Spec Reference:** §5.4.1 — Flag pairs with |correlation| > 0.85.


### NIFTY (n=306 complete rows)

**⚠️ Flagged pairs (|r| > 0.85, per §5.4.1):**

| Feature A | Feature B | Correlation |
|---|---|---|
| riv_level | iv_percentile_rank | 0.9161 |

**Notable pairs (0.70 < |r| ≤ 0.85):**

- days_to_current_expiry ↔ volume_zscore_session = -0.8148

**Variance Inflation Factor (VIF):**

| Feature | VIF | Flag |
|---|---|---|
| day_of_week | 1.24 | ✅ |
| time_of_day_bucket | inf | ⚠️ HIGH |
| days_to_current_expiry | 4.02 | ✅ |
| is_monthly_expiry_week | 1.17 | ✅ |
| days_to_next_holiday | 1.23 | ✅ |
| riv_level | 9.95 | 🟡 |
| riv_change_1d | 1.59 | ✅ |
| riv_change_3d | 1.97 | ✅ |
| iv_percentile_rank | 7.41 | 🟡 |
| iv_term_structure_slope | 2.16 | ✅ |
| iv_skew_25d | 1.95 | ✅ |
| iv_rv_spread_5d | 1.86 | ✅ |
| iv_rv_spread_10d | 1.90 | ✅ |
| pcr_volume | 1.44 | ✅ |
| pcr_oi | 1.67 | ✅ |
| oi_change_net_1d | 1.72 | ✅ |
| volume_zscore_session | 4.88 | ✅ |


### BANKNIFTY (n=245 complete rows)

**⚠️ Flagged pairs (|r| > 0.85, per §5.4.1):**

| Feature A | Feature B | Correlation |
|---|---|---|
| riv_level | iv_percentile_rank | 0.9097 |

**Notable pairs (0.70 < |r| ≤ 0.85):**

- days_to_current_expiry ↔ is_monthly_expiry_week = -0.7727
- is_monthly_expiry_week ↔ volume_zscore_session = 0.7342
- iv_rv_spread_5d ↔ iv_rv_spread_10d = 0.7752

**Variance Inflation Factor (VIF):**

| Feature | VIF | Flag |
|---|---|---|
| day_of_week | 1.18 | ✅ |
| time_of_day_bucket | inf | ⚠️ HIGH |
| days_to_current_expiry | 3.59 | ✅ |
| is_monthly_expiry_week | 4.19 | ✅ |
| days_to_next_holiday | 1.77 | ✅ |
| riv_level | 7.92 | 🟡 |
| riv_change_1d | 1.65 | ✅ |
| riv_change_3d | 2.05 | ✅ |
| iv_percentile_rank | 7.58 | 🟡 |
| iv_term_structure_slope | 2.22 | ✅ |
| iv_skew_25d | 2.56 | ✅ |
| iv_rv_spread_5d | 3.56 | ✅ |
| iv_rv_spread_10d | 3.42 | ✅ |
| pcr_volume | 1.15 | ✅ |
| pcr_oi | 1.45 | ✅ |
| oi_change_net_1d | 1.26 | ✅ |
| volume_zscore_session | 2.70 | ✅ |


## R1e: Temporal Structure & Regime Breaks (OV-R-7)

**Known break:** BANKNIFTY weekly→monthly expiry, November 2024.


### NIFTY — Daily eligible observations

- Mean per day: 1.9
- Min per day: 1
- Max per day: 2
- Days with 0 eligible: 0


### BANKNIFTY — Daily eligible observations

- Mean per day: 1.9
- Min per day: 1
- Max per day: 2
- Days with 0 eligible: 0


## R1f: Quality & Exclusion Rate Analysis

**Guardrail:** §2.1.7 — exclusion rate must be <15% per horizon.


### NIFTY

**horizon_medium:** 50/528 excluded (9.5%) — ✅ OK

  - RIV_T_DEGRADED: 44
  - B6_NOT_USABLE: 6
**horizon_short:** 73/528 excluded (13.8%) — ✅ OK

  - RIV_T_DEGRADED: 70
  - B6_NOT_USABLE: 3


### BANKNIFTY

**horizon_medium:** 61/528 excluded (11.6%) — ✅ OK

  - RIV_T_DEGRADED: 50
  - B6_NOT_USABLE: 6
  - NO_FORWARD_RIV: 5
**horizon_short:** 57/528 excluded (10.8%) — ✅ OK

  - RIV_T_DEGRADED: 52
  - B6_NOT_USABLE: 3
  - NO_FORWARD_RIV: 2


## R1g: Rolling Base Rate Stability

**Method:** 60-trading-day rolling mean of expansion/compression rates.

**Red flag:** >10–15% swing between regime periods.

- NIFTY/horizon_medium/Expansion: rolling range = 0.033–0.300, swing = 0.267 ⚠️
- NIFTY/horizon_medium/Compression: rolling range = 0.091–0.467, swing = 0.376 ⚠️
- NIFTY/horizon_short/Expansion: rolling range = 0.050–0.400, swing = 0.350 ⚠️
- NIFTY/horizon_short/Compression: rolling range = 0.150–0.367, swing = 0.217 ⚠️
- BANKNIFTY/horizon_medium/Expansion: rolling range = 0.000–0.400, swing = 0.400 ⚠️
- BANKNIFTY/horizon_medium/Compression: rolling range = 0.050–0.383, swing = 0.333 ⚠️
- BANKNIFTY/horizon_short/Expansion: rolling range = 0.033–0.317, swing = 0.284 ⚠️
- BANKNIFTY/horizon_short/Compression: rolling range = 0.050–0.367, swing = 0.317 ⚠️


## R1h: Label Noise Estimation

**Method:** Count observations where |z_score| is in the 'ambiguous zone' near the classification threshold.

**Implication:** High density near threshold → structural ceiling on accuracy.


### NIFTY


**horizon_medium** (n=478):

| |z| Band | Count | % | Interpretation |
|---|---|---|---|
| 1.0–1.3 | 51 | 10.7% | Safe below threshold |
| 1.3–1.5 | 36 | 7.5% | Near threshold (below) |
| 1.5–1.7 | 21 | 4.4% | Near threshold (above) |
| 1.7–2.0 | 17 | 3.6% | Safe above threshold |

**Ambiguous zone (1.3 ≤ |z| < 1.7):** 74 observations (15.5%) — ✅ Acceptable

**horizon_short** (n=455):

| |z| Band | Count | % | Interpretation |
|---|---|---|---|
| 1.0–1.3 | 56 | 12.3% | Safe below threshold |
| 1.3–1.5 | 22 | 4.8% | Near threshold (below) |
| 1.5–1.7 | 28 | 6.2% | Near threshold (above) |
| 1.7–2.0 | 25 | 5.5% | Safe above threshold |

**Ambiguous zone (1.3 ≤ |z| < 1.7):** 75 observations (16.5%) — ✅ Acceptable

### BANKNIFTY


**horizon_medium** (n=467):

| |z| Band | Count | % | Interpretation |
|---|---|---|---|
| 1.0–1.3 | 46 | 9.9% | Safe below threshold |
| 1.3–1.5 | 27 | 5.8% | Near threshold (below) |
| 1.5–1.7 | 24 | 5.1% | Near threshold (above) |
| 1.7–2.0 | 18 | 3.9% | Safe above threshold |

**Ambiguous zone (1.3 ≤ |z| < 1.7):** 69 observations (14.8%) — ✅ Acceptable

**horizon_short** (n=471):

| |z| Band | Count | % | Interpretation |
|---|---|---|---|
| 1.0–1.3 | 57 | 12.1% | Safe below threshold |
| 1.3–1.5 | 28 | 5.9% | Near threshold (below) |
| 1.5–1.7 | 19 | 4.0% | Near threshold (above) |
| 1.7–2.0 | 27 | 5.7% | Safe above threshold |

**Ambiguous zone (1.3 ≤ |z| < 1.7):** 74 observations (15.7%) — ✅ Acceptable


## R1i: Information Decay Check

**Method:** Train logistic regression on first 12 months, evaluate on next 6 months, roll forward.

**Red flag:** AUC collapses after first roll → signal is regime-fragile.


### NIFTY

Insufficient data for information decay check.


### BANKNIFTY



## R1j: Label Temporal Autocorrelation

**Purpose:** Check if consecutive labels are quasi-duplicates (inflates apparent walk-forward performance).


### NIFTY


**horizon_medium / label_expansion:**
  Lag-1 autocorr: 0.380, Lag-5: -0.100 ⚠️ High persistence

**horizon_medium / label_compression:**
  Lag-1 autocorr: 0.317, Lag-5: 0.064 ⚠️ High persistence

**horizon_short / label_expansion:**
  Lag-1 autocorr: 0.152, Lag-5: 0.006 ✅

**horizon_short / label_compression:**
  Lag-1 autocorr: 0.344, Lag-5: 0.052 ⚠️ High persistence

### BANKNIFTY


**horizon_medium / label_expansion:**
  Lag-1 autocorr: 0.573, Lag-5: -0.038 ⚠️ High persistence

**horizon_medium / label_compression:**
  Lag-1 autocorr: 0.461, Lag-5: 0.024 ⚠️ High persistence

**horizon_short / label_expansion:**
  Lag-1 autocorr: 0.301, Lag-5: -0.038 ⚠️ High persistence

**horizon_short / label_compression:**
  Lag-1 autocorr: 0.391, Lag-5: -0.040 ⚠️ High persistence


---

## Deferred Features (Excluded from Modelling)

The following features are always UNAVAILABLE and are excluded from all R1 analyses and subsequent modelling:

- `india_vix_level`
- `india_vix_percentile_rank`
- `india_vix_change`
- `fno_ban_heavyweight_flag`

These will be included when their data sources are acquired (India VIX download, F&O ban list integration).
