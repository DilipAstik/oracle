"""
Oracle-V Feature Engineering — Phase C3

Feature families are computed independently, then assembled into
the training-ready feature matrix by assembler.py.

Module structure:
    temporal.py         — Family 5: 5 temporal/calendar features
    microstructure.py   — Family 4: 5 options microstructure features
    iv_surface.py       — Family 1: 9 IV surface features (6 immediate + 3 VIX deferred)
    realised_implied.py — Family 2: 2 realised-implied spread features
    event_context.py    — Family 3: 5 event context features (all deferred)
    assembler.py        — Joins family outputs with C2 labels → training table

Authority: Data Contract §4.4 for IDs and column names.
           config/feature_registry.yaml for machine-readable mapping.
           Part III §5.2.x for computation methods.

Created: 2026-02-20 (C3a)
"""

# Quality states for feature computation
QUALITY_VALID = "VALID"
QUALITY_DEGRADED = "DEGRADED"
QUALITY_UNAVAILABLE = "UNAVAILABLE"

# Unavailability reasons (for audit/lineage)
REASON_SOURCE_MISSING = "SOURCE_MISSING"
REASON_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
REASON_COMPUTATION_FAILED = "COMPUTATION_FAILED"

# Historical reference time bucket (14:00 IST = bucket 19 in 15-min grid)
# Bucket 0 = 9:15-9:30, Bucket 1 = 9:30-9:45, ..., Bucket 19 = 14:00-14:15
HISTORICAL_TIME_BUCKET = 19

# Holiday proximity cap (Tier 2 parameter, bounds [5, 20])
HOLIDAY_HORIZON_MAX = 10
