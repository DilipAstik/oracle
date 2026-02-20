"""
Oracle-V Feature Engineering — Family 5: Temporal Features
==========================================================

Phase: C3a
Authority: Data Contract §4.4 (IDs/columns), Part III §5.2.5 (computation)
Features: 5 Core (1 CRITICAL, 4 STANDARD)

    OV-F-501  day_of_week            STANDARD  SHARED     0=Mon..4=Fri
    OV-F-502  time_of_day_bucket     STANDARD  SHARED     15-min bucket (historical: constant 19)
    OV-F-503  days_to_current_expiry CRITICAL  ORACLE_V   Trading days to nearest standard expiry
    OV-F-504  is_monthly_expiry_week STANDARD  ORACLE_V   1 if within 5 trading days of monthly expiry
    OV-F-505  days_to_next_holiday   STANDARD  SHARED     Trading days to next NSE holiday (cap 10)

Output: DataFrame keyed on (underlying, trade_date) with feature columns + quality columns.
        Temporal features are horizon-independent — assembler joins to both horizons.

Dependencies:
    - NSETradingCalendar (Phase A) — is_trading_day, next_expiry, next_monthly_expiry,
      trading_days_between, holidays_in_range

Design notes:
    - All features use data available at observation time (no future leakage)
    - Deterministic: same inputs always produce same values
    - trading_days_between(a, b) counts strictly between a and b (exclusive of both)
    - For OV-F-503: days_to_expiry = 0 on expiry day, otherwise between(t, exp) + 1
    - For OV-F-505: days_to_holiday = between(t, holiday) — no +1 (holiday is not a trading day)

Created: 2026-02-20
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from datetime import date, timedelta
from typing import Any

import pandas as pd

from oracle.compute.features import (
    HISTORICAL_TIME_BUCKET,
    HOLIDAY_HORIZON_MAX,
    QUALITY_VALID,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Feature computation functions
# ─────────────────────────────────────────────────────────

def compute_day_of_week(trade_date: date) -> int:
    """OV-F-501: Day of week encoding. 0=Monday through 4=Friday."""
    return trade_date.weekday()


def compute_time_of_day_bucket() -> int:
    """OV-F-502: Time of day bucket.

    Historical batch: constant 19 (14:00 IST reference time).
    Bucket 0 = 9:15-9:30, Bucket 19 = 14:00-14:15.
    """
    return HISTORICAL_TIME_BUCKET


def compute_days_to_current_expiry(
    trade_date: date,
    underlying: str,
    calendar: Any,
) -> int:
    """OV-F-503: Trading days to nearest standard expiry. [CRITICAL]

    Part III §5.2.5:
        NIFTY: nearest weekly expiry (Tuesday post-Sep-2025, Thursday pre-Sep-2025)
        BANKNIFTY: nearest monthly expiry (weekly discontinued Nov 2024)
        On expiry day: returns 0.

    Uses calendar.next_expiry(d, instrument) and trading_days_between(a, b).
    trading_days_between counts strictly between (exclusive of both endpoints),
    so we add 1 to include the expiry day in the count.
    """
    nearest_expiry = calendar.next_expiry(trade_date, underlying)

    if trade_date == nearest_expiry:
        return 0

    # between(a, b) is exclusive of both → add 1 to include expiry day
    return calendar.trading_days_between(trade_date, nearest_expiry) + 1


def compute_is_monthly_expiry_week(
    trade_date: date,
    underlying: str,
    calendar: Any,
) -> int:
    """OV-F-504: Is monthly expiry week flag.

    Part III §5.2.5:
        "1 if prediction time t is within the final 5 trading days
        before reference expiry, 0 otherwise."

    On expiry day itself, flag = 1 (days_to = 0, which is <= 5).
    """
    next_monthly = calendar.next_monthly_expiry(trade_date, underlying)

    if trade_date == next_monthly:
        return 1  # Expiry day itself

    days_to = calendar.trading_days_between(trade_date, next_monthly) + 1
    return 1 if days_to <= 5 else 0


def compute_days_to_next_holiday(
    trade_date: date,
    holidays_sorted: list[date],
    calendar: Any,
    cap: int = HOLIDAY_HORIZON_MAX,
) -> int:
    """OV-F-505: Trading days to next NSE holiday.

    Part III §5.2.5:
        "Trading days from prediction time t to the next exchange-declared
        trading holiday. Capped at holiday_horizon_max (default 10)."

    A holiday is a weekday that is NOT a trading day.
    trading_days_between(t, holiday) counts trading days strictly between —
    no +1 needed since the holiday itself is not a trading day.
    """
    # Find next holiday strictly after trade_date
    idx = bisect_right(holidays_sorted, trade_date)
    if idx >= len(holidays_sorted):
        return cap

    next_holiday = holidays_sorted[idx]

    # Count trading days between trade_date and next_holiday
    # (exclusive of both — but holiday isn't a trading day anyway)
    days_to = calendar.trading_days_between(trade_date, next_holiday)
    return min(days_to, cap)


# ─────────────────────────────────────────────────────────
# Main computation entry point
# ─────────────────────────────────────────────────────────

def compute_temporal_features(
    underlying: str,
    trade_dates: list[date],
    calendar: Any,
    holidays_sorted: list[date] | None = None,
) -> pd.DataFrame:
    """Compute all Family 5 temporal features for one underlying.

    Args:
        underlying: "NIFTY" or "BANKNIFTY"
        trade_dates: List of trading dates to compute features for.
        calendar: NSETradingCalendar instance (must support next_expiry,
                  next_monthly_expiry, trading_days_between, holidays_in_range).
        holidays_sorted: Pre-computed sorted list of NSE holidays (weekday non-trading
                         days). If None, extracted from calendar over dataset range + buffer.

    Returns:
        DataFrame with columns:
            underlying, trade_date,
            day_of_week, time_of_day_bucket, days_to_current_expiry,
            is_monthly_expiry_week, days_to_next_holiday,
            quality_OV_F_501..quality_OV_F_505
    """
    logger.info(
        "Computing Family 5 (Temporal) features for %s: %d trade dates",
        underlying, len(trade_dates),
    )

    trade_dates_sorted = sorted(trade_dates)

    # Extract holidays if not provided
    if holidays_sorted is None:
        buffer_end = trade_dates_sorted[-1] + timedelta(days=90)
        holidays_sorted = calendar.holidays_in_range(trade_dates_sorted[0], buffer_end)
        logger.info("  Extracted %d NSE holidays from calendar", len(holidays_sorted))

    holidays_sorted = sorted(holidays_sorted)

    # ── Compute features ──
    rows = []
    for td in trade_dates_sorted:
        f501 = compute_day_of_week(td)
        f502 = compute_time_of_day_bucket()
        f503 = compute_days_to_current_expiry(td, underlying, calendar)
        f504 = compute_is_monthly_expiry_week(td, underlying, calendar)
        f505 = compute_days_to_next_holiday(td, holidays_sorted, calendar)

        rows.append({
            "underlying": underlying,
            "trade_date": td,
            # Feature values
            "day_of_week": f501,
            "time_of_day_bucket": f502,
            "days_to_current_expiry": f503,
            "is_monthly_expiry_week": f504,
            "days_to_next_holiday": f505,
            # Per-feature quality flags — all VALID for temporal
            "quality_OV_F_501": QUALITY_VALID,
            "quality_OV_F_502": QUALITY_VALID,
            "quality_OV_F_503": QUALITY_VALID,
            "quality_OV_F_504": QUALITY_VALID,
            "quality_OV_F_505": QUALITY_VALID,
        })

    df = pd.DataFrame(rows)

    # ── Enforce types ──
    for col in ["day_of_week", "time_of_day_bucket", "days_to_current_expiry",
                 "is_monthly_expiry_week", "days_to_next_holiday"]:
        df[col] = df[col].astype(int)

    # ── Validation assertions ──
    assert df["day_of_week"].between(0, 4).all(), "day_of_week out of range [0,4]"
    assert (df["time_of_day_bucket"] == HISTORICAL_TIME_BUCKET).all(), \
        "Historical time_of_day_bucket must be constant"
    assert (df["days_to_current_expiry"] >= 0).all(), \
        "days_to_current_expiry must be non-negative"
    assert df["is_monthly_expiry_week"].isin([0, 1]).all(), \
        "is_monthly_expiry_week must be binary"
    assert df["days_to_next_holiday"].between(0, HOLIDAY_HORIZON_MAX).all(), \
        "days_to_next_holiday out of range"

    # ── Log summary stats ──
    n = len(df)
    logger.info("  OV-F-501 day_of_week distribution: %s",
                df["day_of_week"].value_counts().sort_index().to_dict())
    logger.info("  OV-F-503 days_to_current_expiry: range [%d, %d], mean %.1f",
                df["days_to_current_expiry"].min(),
                df["days_to_current_expiry"].max(),
                df["days_to_current_expiry"].mean())
    n_ew = df["is_monthly_expiry_week"].sum()
    logger.info("  OV-F-504 is_monthly_expiry_week: %d/%d (%.1f%%)",
                n_ew, n, 100 * n_ew / n)
    logger.info("  OV-F-505 days_to_next_holiday: range [%d, %d], mean %.1f, at_cap=%d",
                df["days_to_next_holiday"].min(),
                df["days_to_next_holiday"].max(),
                df["days_to_next_holiday"].mean(),
                (df["days_to_next_holiday"] == HOLIDAY_HORIZON_MAX).sum())

    logger.info("  Family 5 complete: %d rows, all VALID", n)
    return df
