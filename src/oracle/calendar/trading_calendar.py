"""NSE Trading Calendar — holiday awareness, trading day navigation, and expiry calendar.

Specification References:
    - Data Contract §2.5: Trading Day Definition
    - Data Contract §3.2: Reference Expiry Selection
    - Data Contract §6: TDR-ORACLE-005 (Thursday→Tuesday expiry regime transition)
    - Product Spec Part III §5.2.5: Temporal features (OV-F-503, OV-F-505)
    - Product Spec Part III §5.3.5: Historical Expiry Day Transition

Design Principles:
    - This module is the SOLE source of truth for trading day determination.
    - Holiday data is loaded from curated JSON (data/event_calendar/nse_holidays.json).
    - No empirical thresholds are pre-populated (Design Doctrine §G.2 Principle 1).
    - All dates are Python date objects (not datetime) — time-of-day is handled elsewhere.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional


class ExpiryRegime(Enum):
    """Expiry day regime per TDR-ORACLE-005.

    Pre-September 2025: Thursday expiry (THURSDAY regime).
    Post-September 2025: Tuesday expiry (TUESDAY regime).
    The transition date is the first Tuesday expiry in September 2025.
    """
    THURSDAY = "THURSDAY"
    TUESDAY = "TUESDAY"


# --- Constants ---
# Regime transition: NSE moved weekly expiry from Thursday to Tuesday
# effective September 2025. The exact cutover is the first weekly expiry
# in September 2025.
_REGIME_TRANSITION_DATE = date(2025, 9, 1)

# BANKNIFTY weekly expiry discontinued November 2024 per SEBI directive.
_BANKNIFTY_WEEKLY_DISCONTINUATION = date(2024, 11, 1)

# Weekday constants (Monday=0 ... Sunday=6)
_MONDAY = 0
_TUESDAY = 1
_THURSDAY = 3
_SATURDAY = 5
_SUNDAY = 6


class NSETradingCalendar:
    """NSE trading calendar with holiday awareness and expiry computation.

    This class is the sole source of truth for:
        - Whether a given date is a trading day
        - Trading day counting between dates
        - Trading day navigation (next/previous)
        - NSE derivatives expiry dates (weekly and monthly)
        - Expiry regime awareness (Thursday vs Tuesday)

    Usage:
        cal = NSETradingCalendar()
        cal.is_trading_day(date(2026, 1, 26))  # False (Republic Day)
        cal.trading_days_between(date(2026, 1, 1), date(2026, 1, 31))  # int
        cal.next_expiry(date(2026, 3, 1), instrument="NIFTY")  # date
    """

    def __init__(self, holidays_path: Optional[Path] = None) -> None:
        """Initialise calendar from curated holiday JSON.

        Args:
            holidays_path: Path to nse_holidays.json. If None, uses the
                default location relative to the oracle repository root.

        Raises:
            FileNotFoundError: If the holiday file does not exist.
            ValueError: If the holiday file is malformed.
        """
        if holidays_path is None:
            # Default: relative to this file's location in src/oracle/calendar/
            # Repo root is three levels up: calendar/ -> oracle/ -> src/ -> repo/
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            holidays_path = repo_root / "data" / "event_calendar" / "nse_holidays.json"

        self._holidays_path = Path(holidays_path)
        self._holiday_set: set[date] = set()
        self._holiday_descriptions: dict[date, str] = {}
        self._loaded_years: set[int] = set()
        self._load_holidays()

    def _load_holidays(self) -> None:
        """Load and parse holiday data from JSON file."""
        if not self._holidays_path.exists():
            raise FileNotFoundError(
                f"Holiday calendar not found: {self._holidays_path}. "
                "Ensure data/event_calendar/nse_holidays.json exists."
            )

        with open(self._holidays_path, "r") as f:
            raw = json.load(f)

        if "holidays" not in raw:
            raise ValueError(
                f"Holiday file missing 'holidays' key: {self._holidays_path}"
            )

        for year_str, entries in raw["holidays"].items():
            year = int(year_str)
            self._loaded_years.add(year)
            for entry in entries:
                d = date.fromisoformat(entry["date"])
                self._holiday_set.add(d)
                self._holiday_descriptions[d] = entry.get("description", "")

    # --- Core Trading Day Functions ---

    def is_trading_day(self, d: date) -> bool:
        """Check if a date is an NSE trading day.

        A trading day is a weekday (Mon-Fri) that is NOT in the NSE holiday list.

        Args:
            d: The date to check.

        Returns:
            True if d is a trading day, False otherwise.

        Raises:
            ValueError: If the date's year is not in the loaded holiday calendar.
        """
        self._validate_year(d.year)
        if d.weekday() >= _SATURDAY:  # Saturday or Sunday
            return False
        return d not in self._holiday_set

    def trading_days_between(self, a: date, b: date) -> int:
        """Count trading days between two dates (exclusive of both endpoints).

        Per Data Contract §2.5: counts business days between a and b,
        skipping NSE-declared holidays and weekends.

        Convention: Returns the count of trading days strictly between a and b.
        If a == b or a and b are adjacent with no trading days between them,
        returns 0.

        For the common use case "trading days from t to expiry", call:
            trading_days_between(t_date, expiry_date)
        This gives the number of full trading days remaining (excluding today,
        excluding expiry day). To include one endpoint, add 1.

        Args:
            a: Start date (exclusive).
            b: End date (exclusive).

        Returns:
            Number of trading days strictly between a and b.
            If b <= a, returns 0 (or negative count is avoided).
        """
        if b <= a:
            return 0

        count = 0
        current = a + timedelta(days=1)
        while current < b:
            if current.weekday() < _SATURDAY and current not in self._holiday_set:
                count += 1
            current += timedelta(days=1)
        return count

    def trading_days_from(self, start: date, end: date) -> int:
        """Count trading days from start to end, inclusive of start, exclusive of end.

        This is the natural interpretation for "trading days from prediction
        time t to expiry" — includes today if it's a trading day.

        Args:
            start: Start date (inclusive if trading day).
            end: End date (exclusive).

        Returns:
            Number of trading days in [start, end).
        """
        if end <= start:
            return 0

        count = 0
        current = start
        while current < end:
            if current.weekday() < _SATURDAY and current not in self._holiday_set:
                count += 1
            current += timedelta(days=1)
        return count

    def next_trading_day(self, d: date) -> date:
        """Return the next trading day strictly after d.

        Args:
            d: Reference date.

        Returns:
            The first trading day after d.
        """
        candidate = d + timedelta(days=1)
        while not self._is_trading_day_unchecked(candidate):
            candidate += timedelta(days=1)
        return candidate

    def prev_trading_day(self, d: date) -> date:
        """Return the previous trading day strictly before d.

        Args:
            d: Reference date.

        Returns:
            The most recent trading day before d.
        """
        candidate = d - timedelta(days=1)
        while not self._is_trading_day_unchecked(candidate):
            candidate -= timedelta(days=1)
        return candidate

    # --- Expiry Calendar Functions ---

    def expiry_regime(self, d: date) -> ExpiryRegime:
        """Determine the expiry regime for a given date.

        Per TDR-ORACLE-005:
            - Before September 2025: THURSDAY regime
            - From September 2025 onwards: TUESDAY regime

        Args:
            d: The date to classify.

        Returns:
            ExpiryRegime.THURSDAY or ExpiryRegime.TUESDAY
        """
        if d < _REGIME_TRANSITION_DATE:
            return ExpiryRegime.THURSDAY
        return ExpiryRegime.TUESDAY

    def expiry_weekday(self, d: date) -> int:
        """Return the standard expiry weekday for a given date's regime.

        Args:
            d: Date to determine regime for.

        Returns:
            Weekday integer (0=Monday): 3 for THURSDAY regime, 1 for TUESDAY regime.
        """
        return _THURSDAY if self.expiry_regime(d) == ExpiryRegime.THURSDAY else _TUESDAY

    def next_weekly_expiry(self, d: date, instrument: str = "NIFTY") -> Optional[date]:
        """Find the next weekly expiry on or after date d.

        Per Product Spec Part III §5.2.5:
            - NIFTY: weekly expiry exists in both regimes
            - BANKNIFTY: weekly expiry discontinued November 2024;
              returns None for BANKNIFTY after discontinuation.

        The expiry is adjusted if the scheduled day is a holiday
        (moved to preceding trading day per NSE convention).

        Args:
            d: Reference date.
            instrument: "NIFTY" or "BANKNIFTY".

        Returns:
            The next weekly expiry date, or None if not applicable.
        """
        instrument = instrument.upper()

        if instrument == "BANKNIFTY" and d >= _BANKNIFTY_WEEKLY_DISCONTINUATION:
            return None  # No weekly expiry for BANKNIFTY post Nov 2024

        target_weekday = self.expiry_weekday(d)
        candidate = d

        # Find the next occurrence of the target weekday on or after d
        days_ahead = (target_weekday - candidate.weekday()) % 7
        if days_ahead == 0 and not self._is_trading_day_unchecked(candidate):
            # If today is the expiry weekday but is a holiday, this week's
            # expiry was moved earlier — look for next week
            days_ahead = 7
        candidate = candidate + timedelta(days=days_ahead)

        # Adjust for holidays: if the scheduled expiry is a holiday,
        # move to the preceding trading day
        return self._adjust_expiry_for_holiday(candidate)

    def next_monthly_expiry(self, d: date, instrument: str = "NIFTY") -> date:
        """Find the next monthly expiry on or after date d.

        Monthly expiry is the last [Tuesday/Thursday] of the month,
        adjusted for holidays.

        Args:
            d: Reference date.
            instrument: "NIFTY" or "BANKNIFTY" (both have monthly expiry).

        Returns:
            The next monthly expiry date.
        """
        target_weekday = self.expiry_weekday(d)

        # Start from d's month
        year, month = d.year, d.month

        for _ in range(3):  # Look up to 3 months ahead (safety bound)
            last_target = self._last_weekday_of_month(year, month, target_weekday)
            adjusted = self._adjust_expiry_for_holiday(last_target)
            if adjusted >= d:
                return adjusted
            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

        # Should never reach here
        raise RuntimeError(f"Could not find monthly expiry within 3 months of {d}")

    def next_expiry(self, d: date, instrument: str = "NIFTY") -> date:
        """Find the next applicable expiry on or after date d.

        For NIFTY: returns the nearest weekly expiry (which includes monthly).
        For BANKNIFTY post Nov 2024: returns the nearest monthly expiry.

        Args:
            d: Reference date.
            instrument: "NIFTY" or "BANKNIFTY".

        Returns:
            The next expiry date.
        """
        instrument = instrument.upper()

        if instrument == "BANKNIFTY" and d >= _BANKNIFTY_WEEKLY_DISCONTINUATION:
            return self.next_monthly_expiry(d, instrument)

        weekly = self.next_weekly_expiry(d, instrument)
        if weekly is not None:
            return weekly
        return self.next_monthly_expiry(d, instrument)

    def is_monthly_expiry(self, expiry_date: date) -> bool:
        """Check if an expiry date is a monthly expiry.

        Monthly expiry = last [Tuesday/Thursday] of its month
        (possibly adjusted for holidays).

        Args:
            expiry_date: The expiry date to check.

        Returns:
            True if the expiry is the monthly expiry for its month.
        """
        target_weekday = self.expiry_weekday(expiry_date)
        last_target = self._last_weekday_of_month(
            expiry_date.year, expiry_date.month, target_weekday
        )
        adjusted = self._adjust_expiry_for_holiday(last_target)
        return expiry_date == adjusted

    # --- Holiday Information ---

    def holiday_description(self, d: date) -> Optional[str]:
        """Get the description of a holiday, or None if not a holiday."""
        return self._holiday_descriptions.get(d)

    def holidays_in_range(self, start: date, end: date) -> list[date]:
        """Return sorted list of holidays in [start, end] inclusive."""
        return sorted(
            d for d in self._holiday_set
            if start <= d <= end
        )

    def trading_days_in_month(self, year: int, month: int) -> int:
        """Count trading days in a given month.

        Per feature OV-F-505 (Appendix C): total trading days in
        the current calendar month.

        Args:
            year: Calendar year.
            month: Calendar month (1-12).

        Returns:
            Integer count of trading days in the month.
        """
        first = date(year, month, 1)
        if month == 12:
            last = date(year, 12, 31)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)

        count = 0
        current = first
        while current <= last:
            if current.weekday() < _SATURDAY and current not in self._holiday_set:
                count += 1
            current += timedelta(days=1)
        return count

    @property
    def loaded_years(self) -> set[int]:
        """Return the set of years for which holiday data is loaded."""
        return self._loaded_years.copy()

    # --- Internal Helpers ---

    def _validate_year(self, year: int) -> None:
        """Raise ValueError if year is not in loaded holiday data."""
        if year not in self._loaded_years:
            raise ValueError(
                f"Year {year} not in loaded holiday calendar. "
                f"Loaded years: {sorted(self._loaded_years)}. "
                "Update data/event_calendar/nse_holidays.json."
            )

    def _is_trading_day_unchecked(self, d: date) -> bool:
        """Check trading day without year validation (for navigation)."""
        if d.weekday() >= _SATURDAY:
            return False
        return d not in self._holiday_set

    def _adjust_expiry_for_holiday(self, scheduled: date) -> date:
        """If a scheduled expiry falls on a non-trading day, move to preceding trading day.

        Per Data Contract §3.2: "When a scheduled expiry falls on a trading
        holiday, the exchange-designated revised date applies" — typically
        the preceding trading day.

        Args:
            scheduled: The originally scheduled expiry date.

        Returns:
            The adjusted expiry date (same date if it's a trading day).
        """
        if self._is_trading_day_unchecked(scheduled):
            return scheduled
        return self.prev_trading_day(scheduled)

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
        """Find the last occurrence of a weekday in a given month.

        Args:
            year: Calendar year.
            month: Calendar month (1-12).
            weekday: Target weekday (0=Monday, 1=Tuesday, ..., 6=Sunday).

        Returns:
            The last date in the month that falls on the target weekday.
        """
        if month == 12:
            last_day = date(year, 12, 31)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Walk backwards from last day of month to find the target weekday
        offset = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=offset)
