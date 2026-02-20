# Feature ID Reconciliation — Oracle-V

**Date:** 2026-02-20  
**Rule:** Data Contract §4.4 is the canonical authority for feature IDs and column names.

## Why This Exists

Three spec documents assign different IDs to the same features:
- Part III §5.2 — computation methods
- Appendix C — earlier catalogue (superseded)
- Data Contract §4.4 — persisted schema (authoritative)

## Key Conflicts Resolved

**Family 1:** Appendix C puts `iv_term_structure_ratio` at OV-F-107. Data Contract puts
`india_vix_level` at OV-F-107. Data Contract wins — VIX features are OV-F-107/108/109.

**Family 4:** Appendix C has `max_oi_strike_distance` (F-403), `atm_oi_change_1d` (F-404),
`total_oi_change_1d` (F-405). Data Contract has `oi_change_net_1d` (F-403),
`volume_zscore_session` (F-404), `fno_ban_heavyweight_flag` (F-405). Different features entirely.

**Family 5:** Appendix C swaps F-502/F-503/F-504 assignments and uses `trading_days_in_month`
for F-505. Data Contract uses `days_to_next_holiday` — a different computation.

**Family 3 count:** §5.2.6 table says "6 Core" — counting error. §5.2.7 explicitly lists
F-305 and F-307 as Extension. Correct: 5 Core + 2 Extension.

## Implementation Authority

`config/feature_registry.yaml` is the machine-readable source of truth.
All C3 code references this registry.
