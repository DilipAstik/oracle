# Oracle-V Model Research — R3: Baseline Results

**Generated:** 2026-02-21 05:03:45
**Purpose:** Establish performance floor that ML model must beat (§7.2.5)

## Outperformance Requirement

ML model Brier score must be at least **0.02 below** each baseline's Brier score
on validation data, with statistical significance at 5% (Diebold-Mariano test).

**Primary comparator:** Persistence baseline (given high label autocorrelation from R1j).

---

## NIFTY


### Expansion


**horizon_medium** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1406 ± 0.0424 | 0.0768 | 0.5000 | 0.4571 |
| Persistence | 0.1805 ± 0.0836 | 0.1805 | 0.6586 | 2.9094 |
| Garch | 0.1430 ± 0.0378 | 0.0979 | 0.5498 | 0.4621 |
| Riv Momentum | 0.1494 ± 0.0286 | 0.1125 | 0.4926 | 0.4777 |

**Best baseline:** naive (Brier 0.1406)
**ML target:** Brier ≤ 0.1206 (best - 0.02 margin)

**horizon_short** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1398 ± 0.0288 | 0.0744 | 0.5000 | 0.4558 |
| Persistence | 0.2334 ± 0.0844 | 0.2334 | 0.5620 | 3.7618 |
| Garch | 0.1430 ± 0.0277 | 0.0957 | 0.5463 | 0.4623 |
| Riv Momentum | 0.1468 ± 0.0195 | 0.1155 | 0.5185 | 0.4736 |

**Best baseline:** naive (Brier 0.1398)
**ML target:** Brier ≤ 0.1198 (best - 0.02 margin)

### Compression


**horizon_medium** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1920 ± 0.0236 | 0.0512 | 0.5000 | 0.5738 |
| Persistence | 0.2561 ± 0.0328 | 0.2561 | 0.6560 | 4.1275 |
| Garch | 0.1964 ± 0.0265 | 0.0887 | 0.5034 | 0.5894 |
| Riv Momentum | 0.1905 ± 0.0253 | 0.1108 | 0.6394 | 0.6688 |

**Best baseline:** riv_momentum (Brier 0.1905)
**ML target:** Brier ≤ 0.1705 (best - 0.02 margin)

**horizon_short** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1862 ± 0.0253 | 0.0339 | 0.5000 | 0.5599 |
| Persistence | 0.2756 ± 0.0348 | 0.2756 | 0.6223 | 4.4425 |
| Garch | 0.1838 ± 0.0242 | 0.0554 | 0.5735 | 0.5527 |
| Riv Momentum | 0.1839 ± 0.0244 | 0.0866 | 0.6281 | 0.5500 |

**Best baseline:** garch (Brier 0.1838)
**ML target:** Brier ≤ 0.1638 (best - 0.02 margin)

## BANKNIFTY


### Expansion


**horizon_medium** (4 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1594 ± 0.0323 | 0.0584 | 0.5000 | 0.4998 |
| Persistence | 0.1258 ± 0.0563 | 0.1258 | 0.7911 | 2.0277 |
| Garch | 0.1689 ± 0.0294 | 0.1333 | 0.4122 | 0.5261 |
| Riv Momentum | 0.1823 ± 0.0252 | 0.1394 | 0.5101 | 0.5468 |

**Best baseline:** persistence (Brier 0.1258)
**ML target:** Brier ≤ 0.1058 (best - 0.02 margin)

**horizon_short** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1319 ± 0.0305 | 0.0759 | 0.5000 | 0.4374 |
| Persistence | 0.1689 ± 0.0560 | 0.1689 | 0.6222 | 2.7219 |
| Garch | 0.1397 ± 0.0295 | 0.1185 | 0.4898 | 0.4573 |
| Riv Momentum | 0.1357 ± 0.0309 | 0.1187 | 0.5502 | 0.4437 |

**Best baseline:** naive (Brier 0.1319)
**ML target:** Brier ≤ 0.1119 (best - 0.02 margin)

### Compression


**horizon_medium** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1857 ± 0.0302 | 0.0519 | 0.5000 | 0.5588 |
| Persistence | 0.1723 ± 0.0471 | 0.1723 | 0.7522 | 2.7776 |
| Garch | 0.1933 ± 0.0294 | 0.1312 | 0.4650 | 0.5929 |
| Riv Momentum | 0.1923 ± 0.0390 | 0.1121 | 0.4656 | 0.6878 |

**Best baseline:** persistence (Brier 0.1723)
**ML target:** Brier ≤ 0.1523 (best - 0.02 margin)

**horizon_short** (6 folds):

| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |
|---|---|---|---|---|
| Naive | 0.1731 ± 0.0287 | 0.0521 | 0.5000 | 0.5311 |
| Persistence | 0.2116 ± 0.0382 | 0.2116 | 0.6796 | 3.4111 |
| Garch | 0.1846 ± 0.0283 | 0.1110 | 0.4550 | 0.5690 |
| Riv Momentum | 0.1680 ± 0.0343 | 0.0894 | 0.6343 | 0.5195 |

**Best baseline:** riv_momentum (Brier 0.1680)
**ML target:** Brier ≤ 0.1480 (best - 0.02 margin)

---

## Per-Fold Detail (Brier Scores)


**NIFTY / expansion / horizon_medium:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 60 | 15 | 0.1893 | 0.3333 | 0.1892 | 0.1831 |
| 2 | 65 | 14 | 0.1691 | 0.2154 | 0.1611 | 0.1745 |
| 3 | 65 | 16 | 0.1858 | 0.2000 | 0.1861 | 0.1741 |
| 4 | 65 | 8 | 0.1193 | 0.1538 | 0.1239 | 0.1307 |
| 5 | 66 | 5 | 0.0917 | 0.0909 | 0.0968 | 0.1108 |
| 6 | 67 | 5 | 0.0887 | 0.0896 | 0.1009 | 0.1229 |

**NIFTY / expansion / horizon_short:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 54 | 13 | 0.1828 | 0.3704 | 0.1856 | 0.1778 |
| 2 | 57 | 12 | 0.1673 | 0.2807 | 0.1693 | 0.1666 |
| 3 | 56 | 10 | 0.1507 | 0.2857 | 0.1493 | 0.1477 |
| 4 | 60 | 6 | 0.1081 | 0.1667 | 0.1189 | 0.1241 |
| 5 | 64 | 7 | 0.1118 | 0.1562 | 0.1080 | 0.1329 |
| 6 | 64 | 8 | 0.1181 | 0.1406 | 0.1267 | 0.1319 |

**NIFTY / compression / horizon_medium:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 60 | 12 | 0.1600 | 0.3000 | 0.1559 | 0.1532 |
| 2 | 65 | 16 | 0.1884 | 0.2615 | 0.1961 | 0.1880 |
| 3 | 65 | 14 | 0.1692 | 0.2615 | 0.1800 | 0.1639 |
| 4 | 65 | 20 | 0.2230 | 0.2769 | 0.2322 | 0.2203 |
| 5 | 66 | 20 | 0.2206 | 0.2424 | 0.2269 | 0.2180 |
| 6 | 67 | 17 | 0.1906 | 0.1940 | 0.1871 | 0.1996 |

**NIFTY / compression / horizon_short:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 54 | 13 | 0.1830 | 0.2593 | 0.1765 | 0.1938 |
| 2 | 57 | 11 | 0.1578 | 0.2105 | 0.1555 | 0.1507 |
| 3 | 56 | 13 | 0.1784 | 0.3214 | 0.1829 | 0.1705 |
| 4 | 60 | 14 | 0.1790 | 0.3000 | 0.1762 | 0.1787 |
| 5 | 64 | 15 | 0.1794 | 0.2812 | 0.1773 | 0.1792 |
| 6 | 64 | 22 | 0.2397 | 0.2812 | 0.2344 | 0.2303 |

**BANKNIFTY / expansion / horizon_medium:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 55 | 12 | 0.1706 | 0.2182 | 0.1722 | 0.2034 |
| 2 | 59 | 7 | 0.1193 | 0.1017 | 0.1288 | 0.1653 |
| 3 | 60 | 17 | 0.2056 | 0.1167 | 0.2115 | 0.2101 |
| 4 | 60 | 10 | 0.1421 | 0.0667 | 0.1631 | 0.1502 |

**BANKNIFTY / expansion / horizon_short:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 57 | 11 | 0.1578 | 0.2807 | 0.1569 | 0.1723 |
| 2 | 62 | 9 | 0.1324 | 0.1935 | 0.1470 | 0.1471 |
| 3 | 64 | 14 | 0.1710 | 0.1562 | 0.1713 | 0.1650 |
| 4 | 63 | 11 | 0.1455 | 0.1270 | 0.1646 | 0.1385 |
| 5 | 62 | 4 | 0.0856 | 0.1129 | 0.0972 | 0.0862 |
| 6 | 63 | 6 | 0.0993 | 0.1429 | 0.1014 | 0.1053 |

**BANKNIFTY / compression / horizon_medium:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 55 | 6 | 0.1222 | 0.1455 | 0.1351 | 0.1127 |
| 2 | 59 | 18 | 0.2150 | 0.1186 | 0.2231 | 0.2248 |
| 3 | 60 | 15 | 0.1881 | 0.1500 | 0.2222 | 0.1929 |
| 4 | 60 | 17 | 0.2043 | 0.2333 | 0.1881 | 0.2244 |
| 5 | 58 | 16 | 0.2009 | 0.2414 | 0.2010 | 0.2167 |
| 6 | 62 | 15 | 0.1834 | 0.1452 | 0.1902 | 0.1824 |

**BANKNIFTY / compression / horizon_short:**

| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |
|---|---|---|---|---|---|---|
| 1 | 57 | 6 | 0.1198 | 0.1404 | 0.1263 | 0.1033 |
| 2 | 62 | 14 | 0.1753 | 0.1935 | 0.2049 | 0.1551 |
| 3 | 64 | 13 | 0.1621 | 0.2656 | 0.1749 | 0.1604 |
| 4 | 63 | 14 | 0.1730 | 0.2222 | 0.1910 | 0.1880 |
| 5 | 62 | 17 | 0.2013 | 0.2258 | 0.2019 | 0.2085 |
| 6 | 63 | 18 | 0.2073 | 0.2222 | 0.2085 | 0.1925 |

---

## Interpretation Guide

- **Naive baseline** sets the floor — any model must beat random prediction
- **Persistence baseline** is the primary hurdle (Expert Review recommendation)
  due to high label autocorrelation (R1j: lag-1 up to 0.573)
- **GARCH baseline** captures volatility clustering — beating GARCH demonstrates
  ML adds value beyond conditional variance dynamics
- **RIV momentum** is a falsification test — if ML cannot beat a hand-crafted
  threshold rule on `riv_change_3d`, Oracle-V adds no value

*Frozen at generation time.*