"""Spec compliance tests for NSE Trading Calendar.

Tests verify compliance with:
    - Data Contract §2.5 (Trading Day Definition)
    - Data Contract §3.2 (Reference Expiry Selection)
    - Data Contract §6 TDR-ORACLE-005 (Expiry regime transition)
    - Product Spec Part III §5.2.5 (Temporal features)
    - Product Spec Part III §5.3.5 (Historical Expiry Day Transition)
"""

import pytest
from datetime import date
from oracle.calendar.trading_calendar import NSETradingCalendar, ExpiryRegime


@pytest.fixture
def cal() -> NSETradingCalendar:
    """Provide a calendar instance for all tests."""
    return NSETradingCalendar()


class TestTradingDayDetermination:
    """Data Contract §2.5: Trading days skip weekends and NSE holidays."""

    def test_weekday_non_holiday_is_trading_day(self, cal: NSETradingCalendar) -> None:
        # 2026-01-27 is a Tuesday, not a holiday
        assert cal.is_trading_day(date(2026, 1, 27)) is True

    def test_saturday_is_not_trading_day(self, cal: NSETradingCalendar) -> None:
        assert cal.is_trading_day(date(2026, 1, 17)) is False  # Saturday

    def test_sunday_is_not_trading_day(self, cal: NSETradingCalendar) -> None:
        assert cal.is_trading_day(date(2026, 1, 18)) is False  # Sunday

    def test_republic_day_is_not_trading_day(self, cal: NSETradingCalendar) -> None:
        assert cal.is_trading_day(date(2026, 1, 26)) is False  # Republic Day

    def test_diwali_is_not_trading_day(self, cal: NSETradingCalendar) -> None:
        # Diwali-Balipratipada 2026-11-10
        assert cal.is_trading_day(date(2026, 11, 10)) is False

    def test_christmas_is_not_trading_day(self, cal: NSETradingCalendar) -> None:
        assert cal.is_trading_day(date(2025, 12, 25)) is False

    def test_unknown_year_raises_valueerror(self, cal: NSETradingCalendar) -> None:
        with pytest.raises(ValueError, match="Year 2019"):
            cal.is_trading_day(date(2019, 6, 15))


class TestTradingDaysCounting:
    """Data Contract §2.5: trading_days_between counts correctly."""

    def test_same_date_returns_zero(self, cal: NSETradingCalendar) -> None:
        d = date(2026, 3, 2)
        assert cal.trading_days_between(d, d) == 0

    def test_adjacent_trading_days(self, cal: NSETradingCalendar) -> None:
        # Mon 2026-03-02 to Tue 2026-03-03 (Holi) — 0 days between
        # because Tue is a holiday and we count strictly between
        assert cal.trading_days_between(date(2026, 3, 2), date(2026, 3, 3)) == 0

    def test_across_weekend(self, cal: NSETradingCalendar) -> None:
        # Fri 2026-01-23 to Mon 2026-01-26 (Republic Day)
        # Strictly between: Sat 24, Sun 25 — neither trading days → 0
        assert cal.trading_days_between(date(2026, 1, 23), date(2026, 1, 26)) == 0

    def test_full_week_no_holidays(self, cal: NSETradingCalendar) -> None:
        # Mon 2026-02-02 to Mon 2026-02-09: between = Tue-Fri = 4 trading days
        assert cal.trading_days_between(date(2026, 2, 2), date(2026, 2, 9)) == 4

    def test_week_with_holiday(self, cal: NSETradingCalendar) -> None:
        # Week containing Holi (2026-03-03, Tuesday)
        # Mon 2026-03-02 to Fri 2026-03-06: between = Tue(hol), Wed, Thu = 2
        assert cal.trading_days_between(date(2026, 3, 2), date(2026, 3, 6)) == 2

    def test_b_before_a_returns_zero(self, cal: NSETradingCalendar) -> None:
        assert cal.trading_days_between(date(2026, 3, 10), date(2026, 3, 5)) == 0

    def test_trading_days_in_january_2026(self, cal: NSETradingCalendar) -> None:
        assert cal.trading_days_in_month(2026, 1) == 20


class TestTradingDayNavigation:
    """Navigation functions for next/prev trading day."""

    def test_next_from_friday_skips_weekend(self, cal: NSETradingCalendar) -> None:
        # Fri 2026-01-16 → next = Mon 2026-01-19
        assert cal.next_trading_day(date(2026, 1, 16)) == date(2026, 1, 19)

    def test_next_from_saturday(self, cal: NSETradingCalendar) -> None:
        assert cal.next_trading_day(date(2026, 1, 17)) == date(2026, 1, 19)

    def test_next_skips_holiday(self, cal: NSETradingCalendar) -> None:
        # 2026-01-23 (Fri) → next should skip weekend + Mon 26 (Republic Day) → Tue 27
        assert cal.next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)

    def test_prev_from_monday_skips_weekend(self, cal: NSETradingCalendar) -> None:
        assert cal.prev_trading_day(date(2026, 1, 19)) == date(2026, 1, 16)

    def test_prev_skips_holiday(self, cal: NSETradingCalendar) -> None:
        # prev of 2026-01-27 (Tue) → skip Mon 26 (Republic Day) → Fri 23
        assert cal.prev_trading_day(date(2026, 1, 27)) == date(2026, 1, 23)


class TestExpiryRegime:
    """TDR-ORACLE-005: Thursday→Tuesday regime transition at September 2025."""

    def test_pre_september_2025_is_thursday(self, cal: NSETradingCalendar) -> None:
        assert cal.expiry_regime(date(2025, 8, 31)) == ExpiryRegime.THURSDAY

    def test_september_2025_onwards_is_tuesday(self, cal: NSETradingCalendar) -> None:
        assert cal.expiry_regime(date(2025, 9, 1)) == ExpiryRegime.TUESDAY

    def test_2022_is_thursday(self, cal: NSETradingCalendar) -> None:
        assert cal.expiry_regime(date(2022, 6, 15)) == ExpiryRegime.THURSDAY

    def test_2026_is_tuesday(self, cal: NSETradingCalendar) -> None:
        assert cal.expiry_regime(date(2026, 6, 15)) == ExpiryRegime.TUESDAY

    def test_expiry_weekday_thursday_regime(self, cal: NSETradingCalendar) -> None:
        assert cal.expiry_weekday(date(2024, 6, 1)) == 3  # Thursday

    def test_expiry_weekday_tuesday_regime(self, cal: NSETradingCalendar) -> None:
        assert cal.expiry_weekday(date(2026, 6, 1)) == 1  # Tuesday


class TestExpiryCalendar:
    """Data Contract §3.2: Expiry selection and holiday adjustment."""

    def test_nifty_weekly_expiry_tuesday_regime(self, cal: NSETradingCalendar) -> None:
        # From 2026-03-04 (Wed), next weekly expiry should be Tue 2026-03-10
        expiry = cal.next_weekly_expiry(date(2026, 3, 4), "NIFTY")
        assert expiry is not None
        assert expiry == date(2026, 3, 10)
        assert expiry.weekday() == 1  # Tuesday

    def test_nifty_weekly_expiry_thursday_regime(self, cal: NSETradingCalendar) -> None:
        # From 2024-03-04 (Mon), next weekly expiry should be Thu 2024-03-07
        expiry = cal.next_weekly_expiry(date(2024, 3, 4), "NIFTY")
        assert expiry is not None
        assert expiry == date(2024, 3, 7)
        assert expiry.weekday() == 3  # Thursday

    def test_banknifty_no_weekly_post_november_2024(self, cal: NSETradingCalendar) -> None:
        # BANKNIFTY weekly discontinued Nov 2024
        assert cal.next_weekly_expiry(date(2026, 3, 1), "BANKNIFTY") is None

    def test_banknifty_has_weekly_pre_november_2024(self, cal: NSETradingCalendar) -> None:
        # Before Nov 2024, BANKNIFTY had weekly expiry
        expiry = cal.next_weekly_expiry(date(2024, 3, 4), "BANKNIFTY")
        assert expiry is not None

    def test_banknifty_uses_monthly_post_november_2024(self, cal: NSETradingCalendar) -> None:
        # next_expiry for BANKNIFTY should return monthly
        expiry = cal.next_expiry(date(2026, 3, 1), "BANKNIFTY")
        assert cal.is_monthly_expiry(expiry)

    def test_monthly_expiry_is_last_tuesday_2026(self, cal: NSETradingCalendar) -> None:
        # March 2026: last Tuesday = 2026-03-31 (Mahavir Jayanti — holiday!)
        # Should adjust to preceding trading day = 2026-03-30 (Mon)
        expiry = cal.next_monthly_expiry(date(2026, 3, 1))
        assert expiry == date(2026, 3, 30)

    def test_monthly_expiry_holiday_adjustment(self, cal: NSETradingCalendar) -> None:
        # The above test also verifies holiday adjustment:
        # 2026-03-31 is Mahavir Jayanti → moved to 2026-03-30
        assert cal.is_trading_day(date(2026, 3, 31)) is False
        expiry = cal.next_monthly_expiry(date(2026, 3, 25))
        assert expiry == date(2026, 3, 30)
        assert cal.is_trading_day(expiry) is True

    def test_expiry_on_trading_day_not_adjusted(self, cal: NSETradingCalendar) -> None:
        # 2026-02-24 is last Tuesday of Feb — not a holiday
        expiry = cal.next_monthly_expiry(date(2026, 2, 1))
        assert expiry == date(2026, 2, 24)
        assert expiry.weekday() == 1  # Tuesday


class TestHolidayInfo:
    """Verify holiday metadata access."""

    def test_holiday_description(self, cal: NSETradingCalendar) -> None:
        desc = cal.holiday_description(date(2026, 1, 26))
        assert desc is not None
        assert "Republic" in desc

    def test_non_holiday_returns_none(self, cal: NSETradingCalendar) -> None:
        assert cal.holiday_description(date(2026, 1, 27)) is None

    def test_holidays_in_range(self, cal: NSETradingCalendar) -> None:
        holidays = cal.holidays_in_range(date(2026, 1, 1), date(2026, 1, 31))
        assert len(holidays) == 2  # Jan 15 + Jan 26
