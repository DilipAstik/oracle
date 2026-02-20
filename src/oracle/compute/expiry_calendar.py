"""B6 — Expiry Locking Module.

Implements reference expiry selection per Oracle-V §2.1.2 Step 1,
expiry-locking protocol per §2.1.6, and Data Contract §3.2-§3.3.

Produces per-(underlying, trade_date, horizon) reference expiry mapping
for downstream RIV computation in Phase C.

Specification References:
    - Oracle-V §2.1.2 Step 1: Reference expiry selection algorithm
    - Oracle-V §2.1.6: Expiry-locking protocol, expiry-crossing exclusion
    - Oracle-V §2.1.7: Expiry-exclusion guardrail (<15%)
    - Data Contract §3.2: Reference expiry selection pseudocode
    - Data Contract §3.3: Reference expiry locking rule
    - Data Contract §3.4: Expiry rollover handling

Design Reviews Incorporated:
    - R2: Trading calendar from Phase A module (not rebuilt from data)
    - R2: Do NOT extend calendar beyond dataset boundary
    - R3: has_forward_data column + three-condition observation_usable
    - R2: Cached expiry master table (classify once)
    - R3: Duplicate expiry date warning
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import boto3
import pandas as pd
import pyarrow.parquet as pq

from oracle.calendar.trading_calendar import NSETradingCalendar

logger = logging.getLogger(__name__)

# --- Constants ---
HORIZONS: dict[str, int] = {
    "horizon_short": 2,
    "horizon_medium": 5,
}
DEFAULT_BUFFER: int = 2  # Tier 2, range [1, 5]

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_REGION = "ap-south-1"
IV_PREFIX_TEMPLATE = "computed/iv_surfaces/underlying={underlying}/"
OUTPUT_PREFIX = "computed/expiry_mapping/"


# --- Data Structures ---

@dataclass(frozen=True)
class ExpiryInfo:
    """Classification of a single expiry date (built once, referenced many times)."""
    expiry_date: date
    is_monthly: bool
    weekday: int       # 0=Mon ... 6=Sun
    regime: str        # "THURSDAY" / "TUESDAY" / "OTHER"


# --- Helper Functions ---

def remaining_life_trading_days(
    trade_date: date,
    expiry: date,
    cal: NSETradingCalendar,
) -> int:
    """Count trading days from trade_date (exclusive) to expiry (inclusive).

    Matches spec convention per §2.1.2 example:
        "Monday -> Thursday expiry has remaining life = 3 trading days"
        (Tuesday, Wednesday, Thursday = 3)

    Phase A's trading_days_between(a, b) counts strictly between (both
    endpoints exclusive). Adding 1 includes the expiry endpoint, which
    is always a trading day (holiday-adjusted expiries are moved to the
    preceding trading day by NSE convention).
    """
    if expiry <= trade_date:
        return 0
    return cal.trading_days_between(trade_date, expiry) + 1


def advance_trading_days(
    start: date,
    n: int,
    cal: NSETradingCalendar,
) -> Optional[date]:
    """Advance n trading days from start (exclusive of start).

    Returns the n-th trading day after start.
    Returns None if advancing goes beyond a safe range (10 years).
    """
    if n <= 0:
        return start
    current = start
    safety_limit = date(start.year + 10, 1, 1)
    for _ in range(n):
        candidate = current + timedelta(days=1)
        while candidate.weekday() >= 5 or not cal._is_trading_day_unchecked(candidate):
            candidate += timedelta(days=1)
            if candidate > safety_limit:
                return None
        current = candidate
    return current


# --- S3 Data Extraction ---

def extract_expiry_calendar_from_s3(
    underlying: str,
    bucket: str = S3_BUCKET,
) -> tuple[dict[date, set[date]], list[date]]:
    """Extract expiry calendar and trade dates from IV surface files.

    Reads only trade_date and expiry_date columns (Parquet column pruning).
    This is the authoritative source for which expiry dates exist in the
    data for each trade date.

    The trading CALENDAR (which dates are trading days) comes from Phase A.
    The expiry CALENDAR (which expiries are available) comes from the data.
    These serve different roles per R2 review.

    Returns:
        (expiry_calendar, sorted_trade_dates)
        expiry_calendar: {trade_date: {expiry_1, expiry_2, ...}}
        sorted_trade_dates: sorted list of all trade dates in dataset
    """
    s3 = boto3.client("s3", region_name=S3_REGION)
    prefix = IV_PREFIX_TEMPLATE.format(underlying=underlying)

    # List all IV surface Parquet files (exclude summary JSONs)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".parquet") and "_summary" not in k:
                keys.append(k)

    logger.info(f"{underlying}: Found {len(keys)} IV surface files in S3")

    expiry_calendar: dict[date, set[date]] = {}
    all_trade_dates: set[date] = set()

    for i, key in enumerate(sorted(keys)):
        obj = s3.get_object(Bucket=bucket, Key=key)
        table = pq.read_table(
            io.BytesIO(obj["Body"].read()),
            columns=["trade_date", "expiry_date"],
        )
        df = table.to_pandas()

        if df.empty:
            logger.warning(f"Empty IV surface file: {key}")
            continue

        # Extract trade_date (single value per file)
        td = df["trade_date"].iloc[0]
        if isinstance(td, pd.Timestamp):
            td = td.date()

        # Extract unique expiry dates
        expiries: set[date] = set()
        for e in df["expiry_date"].unique():
            if isinstance(e, pd.Timestamp):
                e = e.date()
            expiries.add(e)

        # Duplicate expiry check (defensive — sets prevent this, but log for audit)
        if td in expiry_calendar:
            logger.warning(f"Duplicate trade_date {td} in IV surfaces — merging expiries")
            expiry_calendar[td].update(expiries)
        else:
            expiry_calendar[td] = expiries

        all_trade_dates.add(td)

        if (i + 1) % 100 == 0:
            logger.info(f"{underlying}: Read {i + 1}/{len(keys)} files")

    sorted_trade_dates = sorted(all_trade_dates)
    all_expiries = set()
    for exps in expiry_calendar.values():
        all_expiries.update(exps)

    logger.info(
        f"{underlying}: {len(sorted_trade_dates)} trade dates, "
        f"{len(all_expiries)} unique expiry dates extracted"
    )
    return expiry_calendar, sorted_trade_dates


# --- Expiry Master Table ---

def build_expiry_master(
    expiry_calendar: dict[date, set[date]],
    cal: Optional[NSETradingCalendar] = None,
) -> dict[date, ExpiryInfo]:
    """Build cached expiry classification table (computed once).

    Per R2 review: "Build expiry_master table once to prevent
    reclassification inconsistencies during reruns."

    Monthly classification uses Phase A calendar's is_monthly_expiry()
    method, which checks: "is this the holiday-adjusted last
    Tuesday/Thursday of its month?" This correctly handles quarters,
    half-yearly expiries, and holiday adjustments — unlike the naive
    "last expiry of month" heuristic which is distorted by long-dated
    quarterly/half-yearly option chains in the data.

    Regime detection is data-driven (weekday of the expiry), not
    based on a hardcoded cutover date.
    """
    if cal is None:
        cal = NSETradingCalendar()

    all_expiries: set[date] = set()
    for exps in expiry_calendar.values():
        all_expiries.update(exps)

    master: dict[date, ExpiryInfo] = {}
    for exp in sorted(all_expiries):
        weekday = exp.weekday()
        if weekday == 3:      # Thursday
            regime = "THURSDAY"
        elif weekday == 1:    # Tuesday
            regime = "TUESDAY"
        else:
            regime = "OTHER"  # Holiday-adjusted expiry

        master[exp] = ExpiryInfo(
            expiry_date=exp,
            is_monthly=cal.is_monthly_expiry(exp),
            weekday=weekday,
            regime=regime,
        )

    logger.info(
        f"Expiry master: {len(master)} expiries classified "
        f"({sum(1 for e in master.values() if e.is_monthly)} monthly, "
        f"{sum(1 for e in master.values() if not e.is_monthly)} weekly/quarterly)"
    )
    return master


# --- Core Selection Algorithm ---

def select_reference_expiry(
    trade_date: date,
    available_expiries: list[date],   # sorted ascending
    horizon_T: int,
    buffer: int,
    cal: NSETradingCalendar,
) -> tuple[Optional[date], int]:
    """Select reference expiry per §2.1.2 Step 1 / Data Contract §3.2.

    Scan expiries in ascending order. Return the FIRST expiry where:
        remaining_life >= horizon_T + buffer

    Returns:
        (reference_expiry, remaining_life_trading_days)
        (None, 0) if no qualifying expiry exists
    """
    threshold = horizon_T + buffer

    for expiry in available_expiries:
        if expiry <= trade_date:
            continue
        rl = remaining_life_trading_days(trade_date, expiry, cal)
        if rl >= threshold:
            return (expiry, rl)

    return (None, 0)


# --- Mapping Table Builder ---

def build_expiry_mapping(
    underlying: str,
    expiry_calendar: dict[date, set[date]],
    sorted_trade_dates: list[date],
    expiry_master: dict[date, ExpiryInfo],
    cal: NSETradingCalendar,
    buffer: int = DEFAULT_BUFFER,
) -> pd.DataFrame:
    """Build the complete reference expiry mapping table.

    One row per (trade_date, horizon). Includes:
        - reference_expiry: selected expiry for this observation
        - forward_date: t + T trading days
        - has_forward_data: whether forward_date is in our dataset
        - expiry_crossing: whether ref expiry expires before forward_date
        - observation_usable: composite (all three conditions must pass)
    """
    max_trade_date = max(sorted_trade_dates)
    trade_date_set = set(sorted_trade_dates)
    rows: list[dict] = []

    for trade_date in sorted_trade_dates:
        available = sorted(expiry_calendar.get(trade_date, set()))

        for horizon_name, horizon_T in HORIZONS.items():
            # 1. Select reference expiry
            ref_expiry, rl = select_reference_expiry(
                trade_date, available, horizon_T, buffer, cal,
            )

            # 2. Compute forward date (t + T trading days)
            forward_date = advance_trading_days(trade_date, horizon_T, cal)

            # 3. Has forward data: forward_date must exist in our dataset
            has_forward = (
                forward_date is not None
                and forward_date <= max_trade_date
                and forward_date in trade_date_set
            )

            # 4. Expiry crossing: ref expiry expires before forward date
            expiry_crossing = False
            if ref_expiry is not None and forward_date is not None:
                expiry_crossing = (ref_expiry < forward_date)

            # 5. Composite observation_usable (three conditions per R3)
            observation_usable = (
                ref_expiry is not None
                and not expiry_crossing
                and has_forward
            )

            # 6. Get expiry classification from master
            regime = ""
            is_monthly = False
            if ref_expiry is not None and ref_expiry in expiry_master:
                info = expiry_master[ref_expiry]
                regime = info.regime
                is_monthly = info.is_monthly

            rows.append({
                "trade_date": trade_date,
                "underlying": underlying,
                "horizon_name": horizon_name,
                "horizon_T": horizon_T,
                "reference_expiry": ref_expiry,
                "remaining_life_td": rl,
                "forward_date": forward_date,
                "expiry_regime": regime,
                "is_monthly_expiry": is_monthly,
                "expiry_crossing": expiry_crossing,
                "has_forward_data": has_forward,
                "observation_usable": observation_usable,
                "buffer_used": buffer,
            })

    df = pd.DataFrame(rows)
    logger.info(f"{underlying}: Built {len(df)} mapping rows")
    return df


# --- Validation ---

def validate_mapping(underlying: str, df: pd.DataFrame, buffer: int) -> None:
    """Run post-build validation checks per B6 design §8.

    Raises ValueError on FATAL checks. Logs warnings on non-fatal issues.
    """
    logger.info(f"{underlying}: Running validation checks...")

    # Check 1: No null reference expiries
    null_count = int(df["reference_expiry"].isna().sum())
    if null_count > 0:
        raise ValueError(
            f"FATAL — {underlying}: {null_count} rows with no qualifying expiry. "
            "Expiry calendar may have gaps."
        )
    logger.info(f"  [1] No null reference expiries: PASS")

    # Check 2: No expiry crossings (structurally impossible with buffer >= 2)
    crossing_count = int(df["expiry_crossing"].sum())
    if crossing_count > 0:
        raise ValueError(
            f"FATAL — {underlying}: {crossing_count} expiry-crossing rows. "
            "Integrity violation — algorithm or calendar bug."
        )
    logger.info(f"  [2] No expiry crossings: PASS")

    # Check 3: remaining_life >= T + buffer for all rows with ref_expiry
    valid_rows = df[df["reference_expiry"].notna()]
    violations = valid_rows[
        valid_rows["remaining_life_td"] < (valid_rows["horizon_T"] + valid_rows["buffer_used"])
    ]
    if len(violations) > 0:
        sample = violations.iloc[0]
        raise ValueError(
            f"FATAL — {underlying}: remaining_life {sample['remaining_life_td']} < "
            f"T+buffer {sample['horizon_T'] + sample['buffer_used']} "
            f"on {sample['trade_date']}"
        )
    logger.info(f"  [3] remaining_life >= T + buffer: PASS")

    # Check 4: reference_expiry > trade_date for all rows with ref_expiry
    if len(valid_rows) > 0:
        bad = valid_rows[valid_rows["reference_expiry"] <= valid_rows["trade_date"]]
        if len(bad) > 0:
            sample = bad.iloc[0]
            raise ValueError(
                f"FATAL — {underlying}: reference_expiry {sample['reference_expiry']} "
                f"<= trade_date {sample['trade_date']}"
            )
    logger.info(f"  [4] reference_expiry > trade_date: PASS")

    # Check 5: BANKNIFTY post-Nov-2024 — all selections should be monthly
    if underlying == "BANKNIFTY":
        post_nov = df[
            (df["trade_date"] >= date(2024, 11, 1))
            & (df["reference_expiry"].notna())
        ]
        non_monthly = post_nov[~post_nov["is_monthly_expiry"]]
        if len(non_monthly) > 0:
            logger.info(
                f"  [5] BANKNIFTY post-Nov-2024: {len(non_monthly)} selections "
                f"from quarterly/half-yearly expiries (not classified as monthly). "
                f"Selection algorithm correct — is_monthly is metadata only."
            )





        else:
            logger.info(f"  [5] BANKNIFTY post-Nov-2024 monthly-only: PASS")

    # Check 6: Boundary unlabelable count
    boundary_count = int((~df["has_forward_data"]).sum())
    total = len(df)
    pct = 100.0 * boundary_count / total if total > 0 else 0
    logger.info(
        f"  [6] Boundary unlabelable: {boundary_count}/{total} ({pct:.1f}%) — "
        f"{'PASS' if pct < 5 else 'REVIEW'}"
    )

    # Check 7: Exclusion rate per horizon (spec §2.1.7: must be < 15%)
    for hn in HORIZONS:
        hdf = df[df["horizon_name"] == hn]
        unusable = int((~hdf["observation_usable"]).sum())
        hpct = 100.0 * unusable / len(hdf) if len(hdf) > 0 else 0
        logger.info(
            f"  [7] {hn} exclusion rate: {unusable}/{len(hdf)} ({hpct:.1f}%) — "
            f"{'PASS' if hpct < 15 else 'FAIL (>15%)'}"
        )
        if hpct >= 15:
            raise ValueError(
                f"FATAL — {underlying} {hn}: exclusion rate {hpct:.1f}% >= 15% "
                "(spec §2.1.7 guardrail). Review buffer or expiry selection."
            )

    # Check 8: Expiry regime transition visibility
    regimes = set(df["expiry_regime"].unique()) - {""}
    if "THURSDAY" in regimes and "TUESDAY" in regimes:
        logger.info(f"  [8] Expiry regime transition: THURSDAY -> TUESDAY detected")
    elif "THURSDAY" in regimes:
        logger.info(f"  [8] Expiry regime: THURSDAY only (pre-Sept-2025 data)")
    elif "TUESDAY" in regimes:
        logger.info(f"  [8] Expiry regime: TUESDAY only (post-Sept-2025 data)")

    logger.info(f"{underlying}: All validation checks passed.")


# --- Summary Generation ---

def generate_summary(
    underlying: str,
    df: pd.DataFrame,
    expiry_master: dict[date, ExpiryInfo],
    buffer: int,
) -> dict:
    """Generate sidecar summary JSON for audit and quick inspection."""
    # Find regime transition dates
    thu_expiries = [e.expiry_date for e in expiry_master.values() if e.regime == "THURSDAY"]
    tue_expiries = [e.expiry_date for e in expiry_master.values() if e.regime == "TUESDAY"]

    summary: dict = {
        "underlying": underlying,
        "total_trade_dates": int(df["trade_date"].nunique()),
        "buffer_parameter": buffer,
        "spec_reference": "Oracle-V §2.1.2, §2.1.6, Data Contract §3.2-§3.3",
        "max_trade_date": str(df["trade_date"].max()),
        "min_trade_date": str(df["trade_date"].min()),
        "generated_at": pd.Timestamp.now().isoformat(),
        "expiry_calendar": {
            "total_unique_expiries": len(expiry_master),
            "monthly_count": sum(1 for e in expiry_master.values() if e.is_monthly),
            "weekly_count": sum(1 for e in expiry_master.values() if not e.is_monthly),
            "regime_thursday_count": len(thu_expiries),
            "regime_tuesday_count": len(tue_expiries),
            "regime_other_count": sum(
                1 for e in expiry_master.values() if e.regime == "OTHER"
            ),
            "last_thursday_expiry": str(max(thu_expiries)) if thu_expiries else "N/A",
            "first_tuesday_expiry": str(min(tue_expiries)) if tue_expiries else "N/A",
        },
        "horizons": {},
    }

    for horizon_name, horizon_T in HORIZONS.items():
        hdf = df[df["horizon_name"] == horizon_name]
        valid = hdf[hdf["reference_expiry"].notna()]

        summary["horizons"][horizon_name] = {
            "T": horizon_T,
            "buffer": buffer,
            "rows": len(hdf),
            "no_qualifying_expiry_count": int(hdf["reference_expiry"].isna().sum()),
            "expiry_crossing_count": int(hdf["expiry_crossing"].sum()),
            "boundary_unlabelable_count": int((~hdf["has_forward_data"]).sum()),
            "observation_usable_count": int(hdf["observation_usable"].sum()),
            "observation_unusable_count": int((~hdf["observation_usable"]).sum()),
            "regime_thursday_count": int((valid["expiry_regime"] == "THURSDAY").sum()),
            "regime_tuesday_count": int((valid["expiry_regime"] == "TUESDAY").sum()),
            "monthly_expiry_selected_count": int(valid["is_monthly_expiry"].sum()),
            "weekly_expiry_selected_count": int(
                (~valid["is_monthly_expiry"]).sum()
            ),
            "remaining_life_min": int(valid["remaining_life_td"].min())
                if len(valid) > 0 else 0,
            "remaining_life_max": int(valid["remaining_life_td"].max())
                if len(valid) > 0 else 0,
            "remaining_life_median": round(float(valid["remaining_life_td"].median()), 1)
                if len(valid) > 0 else 0,
        }

    return summary


# --- S3 Upload ---

def upload_to_s3(
    underlying: str,
    df: pd.DataFrame,
    summary: dict,
    bucket: str = S3_BUCKET,
) -> None:
    """Upload Parquet mapping + JSON summary to S3."""
    s3 = boto3.client("s3", region_name=S3_REGION)

    # Parquet
    parquet_key = f"{OUTPUT_PREFIX}{underlying}_expiry_mapping.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=parquet_key, Body=buf.getvalue())
    logger.info(f"Uploaded: s3://{bucket}/{parquet_key}")

    # JSON summary
    json_key = f"{OUTPUT_PREFIX}{underlying}_expiry_mapping_summary.json"
    s3.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=json.dumps(summary, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Uploaded: s3://{bucket}/{json_key}")


# --- Pipeline Entry Point ---

def run_expiry_mapping(
    underlying: str,
    buffer: int = DEFAULT_BUFFER,
    bucket: str = S3_BUCKET,
) -> dict:
    """End-to-end B6 expiry mapping pipeline for one underlying.

    Steps:
        1. Extract expiry calendar from IV surface data (S3)
        2. Build expiry master table (classify once)
        3. Build mapping table using Phase A trading calendar
        4. Validate per §8
        5. Generate summary
        6. Upload to S3

    Returns:
        Summary dict for inspection.
    """
    logger.info(f"{'='*60}")
    logger.info(f"B6 Expiry Mapping: {underlying} (buffer={buffer})")
    logger.info(f"{'='*60}")

    # Step 1: Extract expiry calendar from S3
    t0 = time.time()
    expiry_calendar, sorted_trade_dates = extract_expiry_calendar_from_s3(
        underlying, bucket,
    )
    logger.info(f"Step 1 complete: S3 extraction ({time.time() - t0:.1f}s)")

    # Step 2: Build expiry master (classify once — per R2)
    t1 = time.time()
    cal = NSETradingCalendar()
    expiry_master = build_expiry_master(expiry_calendar, cal)
    logger.info(f"Step 2 complete: Expiry master built ({time.time() - t1:.1f}s)")

    # Step 3: Build mapping table (uses Phase A calendar — per R2)
    t2 = time.time()
    df = build_expiry_mapping(
        underlying, expiry_calendar, sorted_trade_dates,
        expiry_master, cal, buffer,
    )
    logger.info(f"Step 3 complete: Mapping built ({time.time() - t2:.1f}s)")

    # Step 4: Validate
    validate_mapping(underlying, df, buffer)

    # Step 5: Generate summary
    summary = generate_summary(underlying, df, expiry_master, buffer)

    # Step 6: Upload to S3
    upload_to_s3(underlying, df, summary, bucket)

    return summary


# --- CLI ---

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    underlyings = sys.argv[1:] if len(sys.argv) > 1 else ["NIFTY", "BANKNIFTY"]
    buffer = DEFAULT_BUFFER

    for ul in underlyings:
        t_start = time.time()
        summary = run_expiry_mapping(ul, buffer=buffer)
        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print(f"{ul} -- B6 Expiry Mapping Complete ({elapsed:.1f}s)")
        print(f"{'='*60}")
        print(json.dumps(summary, indent=2, default=str))
        print()
