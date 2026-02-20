"""C1 — Daily RIV Computation.

Computes Reference Implied Volatility (RIV) for every unique
(trade_date, reference_expiry) pair from the B6 expiry mapping.

This is a MEASUREMENT layer — it records observations with quality
annotations but never drops rows. Filtering/exclusion decisions
belong to C2 (baseline + labeling).

Specification References:
    - Oracle-V §2.1.2 Step 2: ATM strike identification
    - Oracle-V §2.1.2 Step 3: RIV = (IV_call + IV_put) / 2
    - Oracle-V §2.1.2: Data quality filters (range, consistency)
    - Data Contract §3.1: ATM strike rules (lower on tie)
    - Data Contract §3.5: RIV computation
    - Data Contract §3.6: No interpolation (v1)

Design Reviews Incorporated:
    - R2: C1 measures reality — never drops rows, records all metrics
    - R1/R3: Spec-mandated quality flags computed and recorded
    - All: Standalone C1 output to S3 (separate from C2)
    - Synthesis: atm_source tag (R2) + DEGRADED flag (R1/R3)
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# --- Constants ---
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_REGION = "ap-south-1"
IV_PREFIX_TEMPLATE = "computed/iv_surfaces/underlying={underlying}/"
EXPIRY_MAPPING_KEY_TEMPLATE = "computed/expiry_mapping/{underlying}_expiry_mapping.parquet"
OUTPUT_PREFIX = "computed/riv_series/"

# Quality filter parameters (Tier 2, §2.1.2)
RIV_MIN = 0.05          # 5% annualised
RIV_MAX = 0.65          # 65% annualised
MAX_CP_DIVERGENCE = 0.02  # 2.0 percentage points (in decimal: 0.02)

# Note: IV values in our Parquet files are stored as decimals (e.g., 0.12 = 12%)


# --- Data Structures ---

@dataclass(frozen=True)
class RIVResult:
    """Result of RIV computation for one (trade_date, reference_expiry)."""
    trade_date: date
    underlying: str
    reference_expiry: date
    spot_price: Optional[float]
    atm_strike: Optional[int]
    iv_call_atm: Optional[float]
    iv_put_atm: Optional[float]
    riv: Optional[float]
    cp_divergence: Optional[float]
    atm_source: str        # TRADED_BOTH / TRADED_CALL_ONLY / TRADED_PUT_ONLY / SETTLEMENT_BOTH
    riv_quality: str       # VALID / DEGRADED / MISSING
    quality_reason: str    # all_passed / CP_DIVERGENCE / OUT_OF_RANGE / NO_TRADED_ATM / PARTIAL_TRADED / MISSING_SIDE / NO_CONVERGED
    atm_call_traded: bool
    atm_put_traded: bool
    atm_call_oi: int
    atm_put_oi: int


# --- ATM Strike Identification ---

def identify_atm_strike(
    surface: pd.DataFrame,
    spot_price: float,
) -> tuple[Optional[int], str]:
    """Identify ATM strike per §2.1.2 Step 2 / Data Contract §3.1.

    Strategy:
      1. Try traded-only strikes (volume > 0) first
      2. Fall back to all CONVERGED strikes if no traded near ATM
      3. ATM = min |strike - spot|, lower strike on tie

    Args:
        surface: IV surface filtered to reference_expiry + CONVERGED status.
        spot_price: Underlying index level.

    Returns:
        (atm_strike, source) where source describes which path was taken.
        (None, "NO_CONVERGED") if no usable strikes exist.
    """
    if surface.empty:
        return (None, "NO_CONVERGED")

    # Unique strikes with their traded status
    strikes = surface.groupby("strike").agg(
        has_traded_call=("traded", lambda x: any(
            x[(surface.loc[x.index, "option_type"] == "CE")].values
        )),
        has_traded_put=("traded", lambda x: any(
            x[(surface.loc[x.index, "option_type"] == "PE")].values
        )),
    ).reset_index()

    # Path 1: traded strikes (at least one side traded)
    traded = strikes[strikes["has_traded_call"] | strikes["has_traded_put"]]

    if not traded.empty:
        # Find nearest to spot among traded strikes
        traded = traded.copy()
        traded["abs_diff"] = (traded["strike"] - spot_price).abs()
        min_diff = traded["abs_diff"].min()
        candidates = traded[traded["abs_diff"] == min_diff]
        atm = int(candidates["strike"].min())  # lower on tie

        # Determine source detail
        row = strikes[strikes["strike"] == atm].iloc[0]
        if row["has_traded_call"] and row["has_traded_put"]:
            return (atm, "TRADED_BOTH")
        elif row["has_traded_call"]:
            return (atm, "TRADED_CALL_ONLY")
        else:
            return (atm, "TRADED_PUT_ONLY")

    # Path 2: fallback to all CONVERGED strikes
    all_strikes = surface["strike"].unique()
    if len(all_strikes) == 0:
        return (None, "NO_CONVERGED")

    diffs = np.abs(all_strikes - spot_price)
    min_diff = diffs.min()
    candidates = all_strikes[diffs == min_diff]
    atm = int(min(candidates))  # lower on tie
    return (atm, "SETTLEMENT_BOTH")


# --- RIV Computation ---

def compute_riv_for_pair(
    iv_surface_day: pd.DataFrame,
    reference_expiry: date,
    underlying: str,
    trade_date: date,
) -> RIVResult:
    """Compute RIV for one (trade_date, reference_expiry) pair.

    Per §2.1.2 Steps 2-3:
      1. Filter surface to reference_expiry + CONVERGED
      2. Identify ATM strike
      3. Extract IV_call, IV_put at ATM
      4. RIV = (IV_call + IV_put) / 2
      5. Apply quality filters, record all metrics

    C1 NEVER drops rows — every pair gets a result with quality annotation.
    """
    # Get spot price (consistent across all rows for this trade_date)
    spot_price = float(iv_surface_day["underlying_price"].iloc[0])

    # Filter to reference expiry + CONVERGED
    mask = (
        (iv_surface_day["expiry_date"] == reference_expiry)
        & (iv_surface_day["iv_status"] == "CONVERGED")
    )
    surface = iv_surface_day[mask].copy()

    if surface.empty:
        return RIVResult(
            trade_date=trade_date, underlying=underlying,
            reference_expiry=reference_expiry, spot_price=spot_price,
            atm_strike=None, iv_call_atm=None, iv_put_atm=None,
            riv=None, cp_divergence=None, atm_source="NO_CONVERGED",
            riv_quality="MISSING", quality_reason="NO_CONVERGED",
            atm_call_traded=False, atm_put_traded=False,
            atm_call_oi=0, atm_put_oi=0,
        )

    # Step 2: ATM strike
    atm_strike, atm_source = identify_atm_strike(surface, spot_price)

    if atm_strike is None:
        return RIVResult(
            trade_date=trade_date, underlying=underlying,
            reference_expiry=reference_expiry, spot_price=spot_price,
            atm_strike=None, iv_call_atm=None, iv_put_atm=None,
            riv=None, cp_divergence=None, atm_source=atm_source,
            riv_quality="MISSING", quality_reason="NO_CONVERGED",
            atm_call_traded=False, atm_put_traded=False,
            atm_call_oi=0, atm_put_oi=0,
        )

    # Step 3: Extract call and put IV at ATM strike
    atm_options = surface[surface["strike"] == atm_strike]
    call_rows = atm_options[atm_options["option_type"] == "CE"]
    put_rows = atm_options[atm_options["option_type"] == "PE"]

    has_call = len(call_rows) > 0
    has_put = len(put_rows) > 0

    if not has_call or not has_put:
        # One or both sides missing at ATM
        iv_call = float(call_rows["iv"].iloc[0]) if has_call else None
        iv_put = float(put_rows["iv"].iloc[0]) if has_put else None
        return RIVResult(
            trade_date=trade_date, underlying=underlying,
            reference_expiry=reference_expiry, spot_price=spot_price,
            atm_strike=atm_strike, iv_call_atm=iv_call, iv_put_atm=iv_put,
            riv=None, cp_divergence=None, atm_source=atm_source,
            riv_quality="MISSING", quality_reason="MISSING_SIDE",
            atm_call_traded=bool(call_rows["traded"].iloc[0]) if has_call else False,
            atm_put_traded=bool(put_rows["traded"].iloc[0]) if has_put else False,
            atm_call_oi=int(call_rows["open_interest"].iloc[0]) if has_call else 0,
            atm_put_oi=int(put_rows["open_interest"].iloc[0]) if has_put else 0,
        )

    # Both sides present
    iv_call = float(call_rows["iv"].iloc[0])
    iv_put = float(put_rows["iv"].iloc[0])
    riv = (iv_call + iv_put) / 2.0
    cp_div = abs(iv_call - iv_put)

    call_traded = bool(call_rows["traded"].iloc[0])
    put_traded = bool(put_rows["traded"].iloc[0])
    call_oi = int(call_rows["open_interest"].iloc[0])
    put_oi = int(put_rows["open_interest"].iloc[0])

    # Quality flag cascade (first failing check wins)
    riv_quality = "VALID"
    quality_reason = "all_passed"

    # Check 1: ATM source (no-trade fallback)
    if atm_source == "SETTLEMENT_BOTH":
        riv_quality = "DEGRADED"
        quality_reason = "NO_TRADED_ATM"
    elif atm_source in ("TRADED_CALL_ONLY", "TRADED_PUT_ONLY"):
        riv_quality = "DEGRADED"
        quality_reason = "PARTIAL_TRADED"

    # Check 2: cp_divergence (only upgrade severity, never downgrade)
    if riv_quality == "VALID" and cp_div > MAX_CP_DIVERGENCE:
        riv_quality = "DEGRADED"
        quality_reason = "CP_DIVERGENCE"

    # Check 3: Range filter
    if riv_quality == "VALID" and (riv < RIV_MIN or riv > RIV_MAX):
        riv_quality = "DEGRADED"
        quality_reason = "OUT_OF_RANGE"

    return RIVResult(
        trade_date=trade_date, underlying=underlying,
        reference_expiry=reference_expiry, spot_price=spot_price,
        atm_strike=atm_strike, iv_call_atm=iv_call, iv_put_atm=iv_put,
        riv=riv, cp_divergence=cp_div, atm_source=atm_source,
        riv_quality=riv_quality, quality_reason=quality_reason,
        atm_call_traded=call_traded, atm_put_traded=put_traded,
        atm_call_oi=call_oi, atm_put_oi=put_oi,
    )


# --- S3 I/O ---

def load_expiry_mapping(underlying: str, bucket: str = S3_BUCKET) -> pd.DataFrame:
    """Load B6 expiry mapping from S3."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    key = EXPIRY_MAPPING_KEY_TEMPLATE.format(underlying=underlying)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    logger.info(f"Loaded expiry mapping: {len(df)} rows for {underlying}")
    return df


def get_iv_surface_key(underlying: str, trade_date: date) -> str:
    """Construct IV surface S3 key for a given trade date."""
    return (
        f"computed/iv_surfaces/underlying={underlying}/"
        f"year={trade_date.year}/month={trade_date.month:02d}/"
        f"{underlying}_iv_{trade_date.strftime('%Y%m%d')}.parquet"
    )


def load_iv_surface(
    underlying: str,
    trade_date: date,
    s3_client,
    bucket: str = S3_BUCKET,
) -> Optional[pd.DataFrame]:
    """Load IV surface for one trade date from S3."""
    key = get_iv_surface_key(underlying, trade_date)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        return table.to_pandas()
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"IV surface not found: {key}")
        return None


# --- Batch Pipeline ---

def run_riv_pipeline(
    underlying: str,
    bucket: str = S3_BUCKET,
) -> dict:
    """End-to-end C1 RIV computation pipeline for one underlying.

    Steps:
        1. Load B6 expiry mapping
        2. Extract unique (trade_date, reference_expiry) pairs
        3. For each pair, load IV surface and compute RIV
        4. Validate
        5. Upload to S3

    Returns:
        Summary dict for inspection.
    """
    logger.info(f"{'='*60}")
    logger.info(f"C1 RIV Computation: {underlying}")
    logger.info(f"{'='*60}")

    s3 = boto3.client("s3", region_name=S3_REGION)

    # Step 1: Load expiry mapping
    mapping = load_expiry_mapping(underlying, bucket)

    # Step 2: Extract unique (trade_date, reference_expiry) pairs
    # Both horizons may share the same ref_expiry — deduplicate
    valid_mapping = mapping[mapping["reference_expiry"].notna()].copy()
    pairs = (
        valid_mapping[["trade_date", "reference_expiry"]]
        .drop_duplicates()
        .sort_values("trade_date")
    )
    logger.info(f"Unique (trade_date, reference_expiry) pairs: {len(pairs)}")

    # Step 3: Compute RIV for each pair
    results: list[RIVResult] = []
    # Cache IV surfaces by trade_date (multiple expiries on same day)
    surface_cache: dict[date, Optional[pd.DataFrame]] = {}

    t0 = time.time()
    for i, (_, row) in enumerate(pairs.iterrows()):
        td = row["trade_date"]
        if isinstance(td, pd.Timestamp):
            td = td.date()
        ref_exp = row["reference_expiry"]
        if isinstance(ref_exp, pd.Timestamp):
            ref_exp = ref_exp.date()

        # Load IV surface (cached per trade_date)
        if td not in surface_cache:
            surface_cache[td] = load_iv_surface(underlying, td, s3, bucket)

        iv_surface = surface_cache[td]

        if iv_surface is None:
            results.append(RIVResult(
                trade_date=td, underlying=underlying,
                reference_expiry=ref_exp, spot_price=None,
                atm_strike=None, iv_call_atm=None, iv_put_atm=None,
                riv=None, cp_divergence=None, atm_source="NO_DATA",
                riv_quality="MISSING", quality_reason="NO_IV_SURFACE",
                atm_call_traded=False, atm_put_traded=False,
                atm_call_oi=0, atm_put_oi=0,
            ))
            continue

        result = compute_riv_for_pair(iv_surface, ref_exp, underlying, td)
        results.append(result)

        # Free cache for dates we've passed (memory management)
        if len(surface_cache) > 5:
            oldest = min(surface_cache.keys())
            if oldest < td:
                del surface_cache[oldest]

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(pairs)} pairs ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    logger.info(f"RIV computation complete: {len(results)} results ({elapsed:.1f}s)")

    # Convert to DataFrame
    df = pd.DataFrame([r.__dict__ for r in results])

    # Step 4: Validate
    validate_riv(underlying, df, len(pairs))

    # Step 5: Generate summary
    summary = generate_riv_summary(underlying, df)

    # Step 6: Upload to S3
    upload_riv_to_s3(underlying, df, summary, bucket)

    return summary


# --- Validation ---

def validate_riv(underlying: str, df: pd.DataFrame, expected_pairs: int) -> None:
    """Post-build validation checks for C1."""
    logger.info(f"{underlying}: Running C1 validation...")

    # Check 1: Coverage — every pair has a row
    actual = len(df)
    if actual != expected_pairs:
        logger.warning(
            f"  [1] Coverage: expected {expected_pairs}, got {actual} — MISMATCH"
        )
    else:
        logger.info(f"  [1] Coverage: {actual}/{expected_pairs} — PASS")

    # Check 2: MISSING rate
    missing = int((df["riv_quality"] == "MISSING").sum())
    missing_pct = 100.0 * missing / actual if actual > 0 else 0
    logger.info(
        f"  [2] MISSING rate: {missing}/{actual} ({missing_pct:.1f}%) — "
        f"{'PASS' if missing_pct < 5 else 'WARNING'}"
    )

    # Check 3: DEGRADED rate
    degraded = int((df["riv_quality"] == "DEGRADED").sum())
    deg_pct = 100.0 * degraded / actual if actual > 0 else 0
    logger.info(f"  [3] DEGRADED rate: {degraded}/{actual} ({deg_pct:.1f}%) — INFO")

    # Check 4: VALID rate
    valid = int((df["riv_quality"] == "VALID").sum())
    val_pct = 100.0 * valid / actual if actual > 0 else 0
    logger.info(f"  [4] VALID rate: {valid}/{actual} ({val_pct:.1f}%) — INFO")

    # Check 5: RIV sanity range (wider than filter — raw sanity)
    riv_vals = df[df["riv"].notna()]["riv"]
    if len(riv_vals) > 0:
        out_of_sanity = int(((riv_vals < 0.02) | (riv_vals > 0.80)).sum())
        logger.info(
            f"  [5] RIV outside [2%, 80%] sanity: {out_of_sanity} — "
            f"{'PASS' if out_of_sanity == 0 else 'WARNING'}"
        )
        logger.info(
            f"  [5] RIV stats: min={riv_vals.min():.4f} "
            f"median={riv_vals.median():.4f} max={riv_vals.max():.4f}"
        )

    # Check 6: Day-over-day RIV changes
    valid_riv = df[df["riv"].notna()].sort_values("trade_date")
    if len(valid_riv) > 1:
        changes = valid_riv["riv"].diff().dropna().abs()
        large_changes = int((changes > 0.05).sum())  # >5 ppts
        logger.info(
            f"  [6] Day-over-day changes > 5 ppts: {large_changes}/{len(changes)} — "
            f"{'PASS' if large_changes < len(changes) * 0.05 else 'WARNING'}"
        )

    # Check 7: Degradation breakdown
    if degraded > 0:
        reasons = df[df["riv_quality"] == "DEGRADED"]["quality_reason"].value_counts()
        logger.info(f"  [7] DEGRADED breakdown:")
        for reason, count in reasons.items():
            logger.info(f"      {reason}: {count}")

    logger.info(f"{underlying}: C1 validation complete.")


# --- Summary ---

def generate_riv_summary(underlying: str, df: pd.DataFrame) -> dict:
    """Generate sidecar summary JSON."""
    riv_vals = df[df["riv"].notna()]["riv"]

    summary = {
        "underlying": underlying,
        "total_pairs": len(df),
        "quality_distribution": {
            "VALID": int((df["riv_quality"] == "VALID").sum()),
            "DEGRADED": int((df["riv_quality"] == "DEGRADED").sum()),
            "MISSING": int((df["riv_quality"] == "MISSING").sum()),
        },
        "degraded_breakdown": (
            df[df["riv_quality"] == "DEGRADED"]["quality_reason"]
            .value_counts().to_dict()
            if (df["riv_quality"] == "DEGRADED").any() else {}
        ),
        "atm_source_distribution": df["atm_source"].value_counts().to_dict(),
        "riv_stats": {
            "count": int(riv_vals.count()),
            "min": round(float(riv_vals.min()), 6) if len(riv_vals) > 0 else None,
            "median": round(float(riv_vals.median()), 6) if len(riv_vals) > 0 else None,
            "max": round(float(riv_vals.max()), 6) if len(riv_vals) > 0 else None,
            "mean": round(float(riv_vals.mean()), 6) if len(riv_vals) > 0 else None,
        },
        "parameters": {
            "riv_min": RIV_MIN,
            "riv_max": RIV_MAX,
            "max_cp_divergence": MAX_CP_DIVERGENCE,
        },
        "spec_reference": "Oracle-V §2.1.2 Steps 2-3, Data Contract §3.1-§3.6",
        "date_range": {
            "min": str(df["trade_date"].min()),
            "max": str(df["trade_date"].max()),
        },
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    return summary


# --- S3 Upload ---

def upload_riv_to_s3(
    underlying: str,
    df: pd.DataFrame,
    summary: dict,
    bucket: str = S3_BUCKET,
) -> None:
    """Upload RIV Parquet + JSON summary to S3."""
    s3 = boto3.client("s3", region_name=S3_REGION)

    # Parquet
    parquet_key = f"{OUTPUT_PREFIX}{underlying}_riv_daily.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=parquet_key, Body=buf.getvalue())
    logger.info(f"Uploaded: s3://{bucket}/{parquet_key}")

    # JSON
    json_key = f"{OUTPUT_PREFIX}{underlying}_riv_daily_summary.json"
    s3.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=json.dumps(summary, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Uploaded: s3://{bucket}/{json_key}")


# --- CLI ---

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    underlyings = sys.argv[1:] if len(sys.argv) > 1 else ["NIFTY", "BANKNIFTY"]

    for ul in underlyings:
        t_start = time.time()
        summary = run_riv_pipeline(ul)
        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print(f"{ul} -- C1 RIV Computation Complete ({elapsed:.1f}s)")
        print(f"{'='*60}")
        print(json.dumps(summary, indent=2, default=str))
        print()
