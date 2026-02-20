"""C2 — Baseline, σ_daily, Z-Scores & Labels.

The first INTERPRETIVE layer in Oracle's pipeline. Reads the C1 RIV
observation series and B6 expiry mapping to produce labeled training
observations with z-scores and expansion/compression/stable labels.

Critical Spec Rules Enforced:
    1. Baseline uses only VALID RIV from the SAME reference expiry (§2.1.3)
    2. σ_daily from consecutive-day pairs only — no skipped weekdays (§2.1.4)
    3. σ_floor = 0.35 ppts prevents z-score explosion (§2.1.4)
    4. Forward RIV uses LOCKED expiry, not re-selected (§2.1.6, DC §3.3)
    5. √T horizon scaling on z-score denominator (§2.1.4)
    6. Labels: expansion z≥1.5, compression z≤-1.2 (§2.1.5)
    7. DEGRADED baseline does NOT exclude — only MISSING excludes (§2.1.3)
    8. Three-field observation status per R2 review

Design Reviews Incorporated:
    - R2: Layer separation (C2 is pure statistical consumer of C1)
    - R2: Three-field observation status (available/computable/eligible)
    - R1/R3: Interpretation C for consecutive-day (no skipped weekdays)
    - All: Exclude DEGRADED RIV from baseline (spec §2.1.3)
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oracle.calendar.trading_calendar import NSETradingCalendar

logger = logging.getLogger(__name__)

# --- Constants ---
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_REGION = "ap-south-1"

# Horizons (from B6)
HORIZONS = {
    "horizon_short": {"T": 2, "L": 5},
    "horizon_medium": {"T": 5, "L": 10},
}

# Tier 2 parameters (§2.1.3–§2.1.5)
MIN_DATA_FRACTION = 0.6
SIGMA_FLOOR = 0.0035        # 0.35 percentage points in decimal
Z_EXPAND = 1.5
Z_COMPRESS = 1.2


# --- Consecutive-Day Logic (§2.1.4, Interpretation C) ---

def is_consecutive_for_sigma(d1: date, d2: date) -> bool:
    """Check if d1→d2 is a valid pair for σ_daily computation.

    Per §2.1.4 and spec example: "consecutive trading day" means
    no weekdays (Mon-Fri) are skipped between d1 and d2.

    Examples:
        Mon→Tue: 0 weekdays between → True
        Fri→Mon: 0 weekdays between (Sat/Sun not weekdays) → True
        Tue→Thu (Wed holiday): 1 weekday (Wed) between → False
        Mon→Wed (Tue holiday): 1 weekday (Tue) between → False
    """
    gap_days = (d2 - d1).days
    if gap_days <= 0:
        return False
    weekdays_between = sum(
        1 for i in range(1, gap_days)
        if (d1 + timedelta(days=i)).weekday() < 5
    )
    return weekdays_between == 0


# --- Baseline & σ_daily Computation ---

def compute_baseline_and_sigma(
    trade_date: date,
    reference_expiry: date,
    L: int,
    riv_lookup: dict[tuple[date, date], tuple[float, str]],
    valid_dates_for_expiry: list[date],
    cal: NSETradingCalendar,
) -> dict:
    """Compute Baseline_RIV(t) and σ_daily(t) per §2.1.3–§2.1.4.

    Args:
        trade_date: Observation date t.
        reference_expiry: Locked reference expiry.
        L: Lookback period in trading days.
        riv_lookup: (trade_date, ref_expiry) → (riv, quality).
        valid_dates_for_expiry: Sorted dates where this expiry has VALID RIV.
        cal: Trading calendar.

    Returns dict with:
        baseline_riv, sigma_daily, sigma_daily_effective, sigma_floor_binding,
        baseline_quality, baseline_n_valid, baseline_n_required,
        n_consecutive_pairs
    """
    min_required = math.ceil(L * MIN_DATA_FRACTION)

    # Collect VALID RIV values in [t-L, t] for locked expiry
    # "t-L" means L trading days back from t
    window_start = trade_date
    for _ in range(L):
        window_start = cal.prev_trading_day(window_start)

    # Filter valid_dates_for_expiry to [window_start, trade_date]
    window_dates = [
        d for d in valid_dates_for_expiry
        if window_start <= d <= trade_date
    ]
    window_rivs = []
    for d in window_dates:
        key = (d, reference_expiry)
        if key in riv_lookup:
            riv_val, quality = riv_lookup[key]
            if quality == "VALID" and riv_val is not None:
                window_rivs.append((d, riv_val))

    n_valid = len(window_rivs)

    if n_valid == 0:
        return {
            "baseline_riv": None,
            "sigma_daily": None,
            "sigma_daily_effective": None,
            "sigma_floor_binding": False,
            "baseline_quality": "MISSING",
            "baseline_n_valid": 0,
            "baseline_n_required": min_required,
            "n_consecutive_pairs": 0,
        }

    # Baseline = mean of valid RIV values
    riv_values = [r for _, r in window_rivs]
    baseline = float(np.mean(riv_values))

    # σ_daily = stdev of consecutive-day RIV changes
    # Both values in the pair must be VALID (already filtered above)
    # and the pair must be consecutive (no skipped weekdays)
    sorted_rivs = sorted(window_rivs, key=lambda x: x[0])
    changes = []
    for j in range(1, len(sorted_rivs)):
        d_prev, riv_prev = sorted_rivs[j - 1]
        d_curr, riv_curr = sorted_rivs[j]
        if is_consecutive_for_sigma(d_prev, d_curr):
            changes.append(riv_curr - riv_prev)

    n_pairs = len(changes)
    if n_pairs >= 2:
        sigma = float(np.std(changes, ddof=1))
    elif n_pairs == 1:
        # With only one change, stdev is undefined (ddof=1 → division by 0)
        # Use absolute change as a conservative estimate
        sigma = abs(changes[0])
    else:
        sigma = 0.0

    sigma_effective = max(sigma, SIGMA_FLOOR)
    floor_binding = sigma < SIGMA_FLOOR

    # Quality check
    if n_valid >= min_required:
        quality = "VALID"
    else:
        quality = "DEGRADED"

    return {
        "baseline_riv": baseline,
        "sigma_daily": sigma,
        "sigma_daily_effective": sigma_effective,
        "sigma_floor_binding": floor_binding,
        "baseline_quality": quality,
        "baseline_n_valid": n_valid,
        "baseline_n_required": min_required,
        "n_consecutive_pairs": n_pairs,
    }


# --- Label Computation ---

def compute_labels(
    riv_t: Optional[float],
    riv_t_plus_T: Optional[float],
    baseline_riv: Optional[float],
    sigma_daily_effective: Optional[float],
    T: int,
) -> dict:
    """Compute z-score and labels per §2.1.4–§2.1.5.

    Returns dict with:
        delta_iv, z_score, label_expansion, label_compression, label_stable
    """
    if (riv_t_plus_T is None or baseline_riv is None
            or sigma_daily_effective is None or sigma_daily_effective == 0):
        return {
            "delta_iv": None,
            "z_score": None,
            "label_expansion": None,
            "label_compression": None,
            "label_stable": None,
        }

    delta_iv = riv_t_plus_T - baseline_riv
    z_score = delta_iv / (sigma_daily_effective * math.sqrt(T))

    expansion = 1 if z_score >= Z_EXPAND else 0
    compression = 1 if z_score <= -Z_COMPRESS else 0
    stable = 1 if (expansion == 0 and compression == 0) else 0

    return {
        "delta_iv": delta_iv,
        "z_score": z_score,
        "label_expansion": expansion,
        "label_compression": compression,
        "label_stable": stable,
    }


# --- Main Pipeline ---

def run_labeling_pipeline(
    underlying: str,
    bucket: str = S3_BUCKET,
) -> dict:
    """End-to-end C2 labeling pipeline for one underlying.

    Steps:
        1. Load B6 expiry mapping + C1 RIV series
        2. Build RIV lookup indexed by (trade_date, reference_expiry)
        3. For each (trade_date, horizon): compute baseline, σ, forward, z, labels
        4. Validate
        5. Upload to S3
    """
    logger.info(f"{'='*60}")
    logger.info(f"C2 Labeling Pipeline: {underlying}")
    logger.info(f"{'='*60}")

    s3 = boto3.client("s3", region_name=S3_REGION)
    cal = NSETradingCalendar()

    # Step 1: Load data
    t0 = time.time()

    obj = s3.get_object(
        Bucket=bucket,
        Key=f"computed/expiry_mapping/{underlying}_expiry_mapping.parquet",
    )
    mapping = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()

    obj = s3.get_object(
        Bucket=bucket,
        Key=f"computed/riv_series/{underlying}_riv_daily.parquet",
    )
    riv_df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()

    logger.info(
        f"Loaded: {len(mapping)} mapping rows, {len(riv_df)} RIV rows "
        f"({time.time() - t0:.1f}s)"
    )

    # Step 2: Build RIV lookup
    # (trade_date, reference_expiry) → (riv, quality)
    riv_lookup: dict[tuple[date, date], tuple[Optional[float], str]] = {}
    riv_spot_lookup: dict[tuple[date, date], Optional[float]] = {}
    riv_atm_lookup: dict[tuple[date, date], Optional[int]] = {}
    riv_detail_lookup: dict[tuple[date, date], dict] = {}

    for _, r in riv_df.iterrows():
        td = r["trade_date"]
        re = r["reference_expiry"]
        if hasattr(td, "date"): td = td.date()
        if hasattr(re, "date"): re = re.date()
        key = (td, re)
        riv_val = float(r["riv"]) if pd.notna(r["riv"]) else None
        quality = str(r["riv_quality"])
        riv_lookup[key] = (riv_val, quality)
        riv_spot_lookup[key] = float(r["spot_price"]) if pd.notna(r.get("spot_price")) else None
        riv_atm_lookup[key] = int(r["atm_strike"]) if pd.notna(r.get("atm_strike")) else None
        riv_detail_lookup[key] = {
            "iv_call_atm": float(r["iv_call_atm"]) if pd.notna(r.get("iv_call_atm")) else None,
            "iv_put_atm": float(r["iv_put_atm"]) if pd.notna(r.get("iv_put_atm")) else None,
            "cp_divergence": float(r["cp_divergence"]) if pd.notna(r.get("cp_divergence")) else None,
        }

    # Build per-expiry sorted valid date lists (for baseline window filtering)
    expiry_valid_dates: dict[date, list[date]] = {}
    for (td, re), (riv_val, quality) in riv_lookup.items():
        if quality == "VALID" and riv_val is not None:
            expiry_valid_dates.setdefault(re, []).append(td)
    for re in expiry_valid_dates:
        expiry_valid_dates[re].sort()

    # Also build per-expiry ALL date lists (for any-quality lookups)
    expiry_all_dates: dict[date, list[date]] = {}
    for (td, re) in riv_lookup:
        expiry_all_dates.setdefault(re, []).append(td)
    for re in expiry_all_dates:
        expiry_all_dates[re].sort()

    logger.info(f"RIV lookup: {len(riv_lookup)} entries, "
                f"{len(expiry_valid_dates)} unique expiries with VALID data")

    # Step 3: Process each (trade_date, horizon)
    t1 = time.time()
    rows = []

    for horizon_name, hparams in HORIZONS.items():
        T = hparams["T"]
        L = hparams["L"]

        # Get mapping rows for this horizon
        h_mapping = mapping[mapping["horizon_name"] == horizon_name].copy()
        h_mapping = h_mapping.sort_values("trade_date")

        for _, mrow in h_mapping.iterrows():
            td = mrow["trade_date"]
            ref_exp = mrow["reference_expiry"]
            fwd_date = mrow["forward_date"]
            b6_usable = bool(mrow["observation_usable"])
            expiry_regime = str(mrow["expiry_regime"])
            is_monthly = bool(mrow["is_monthly_expiry"])

            if hasattr(td, "date"): td = td.date()
            if hasattr(ref_exp, "date"): ref_exp = ref_exp.date()
            if hasattr(fwd_date, "date"): fwd_date = fwd_date.date()

            # --- observation_available (C1 success) ---
            riv_key = (td, ref_exp)
            riv_entry = riv_lookup.get(riv_key)
            riv_t = riv_entry[0] if riv_entry else None
            riv_t_quality = riv_entry[1] if riv_entry else "NOT_FOUND"
            spot = riv_spot_lookup.get(riv_key)
            atm = riv_atm_lookup.get(riv_key)
            detail = riv_detail_lookup.get(riv_key, {})

            observation_available = (riv_t is not None)

            # --- Forward RIV lookup (locked expiry at forward_date) ---
            fwd_key = (fwd_date, ref_exp) if fwd_date else None
            fwd_entry = riv_lookup.get(fwd_key) if fwd_key else None
            riv_t_plus_T = fwd_entry[0] if fwd_entry else None
            riv_fwd_quality = fwd_entry[1] if fwd_entry else "NOT_AVAILABLE"

            # --- Baseline & σ_daily ---
            valid_dates = expiry_valid_dates.get(ref_exp, [])
            if observation_available and ref_exp is not None:
                bl = compute_baseline_and_sigma(
                    td, ref_exp, L, riv_lookup, valid_dates, cal,
                )
            else:
                bl = {
                    "baseline_riv": None, "sigma_daily": None,
                    "sigma_daily_effective": None, "sigma_floor_binding": False,
                    "baseline_quality": "MISSING", "baseline_n_valid": 0,
                    "baseline_n_required": math.ceil(L * MIN_DATA_FRACTION),
                    "n_consecutive_pairs": 0,
                }

            # --- Labels ---
            lab = compute_labels(
                riv_t, riv_t_plus_T,
                bl["baseline_riv"], bl["sigma_daily_effective"], T,
            )

            # --- Three-field observation status (R2 review) ---
            label_computable = (
                riv_t_plus_T is not None
                and bl["baseline_riv"] is not None
                and lab["z_score"] is not None
            )

            training_eligible = (
                b6_usable
                and observation_available
                and riv_t_quality == "VALID"
                and label_computable
                and bl["baseline_quality"] != "MISSING"
            )

            # Exclusion reason
            if training_eligible:
                exclusion_reason = "none"
            elif not b6_usable:
                exclusion_reason = "B6_NOT_USABLE"
            elif not observation_available:
                exclusion_reason = "NO_RIV_AT_T"
            elif riv_t_quality != "VALID":
                exclusion_reason = "RIV_T_DEGRADED"
            elif riv_t_plus_T is None:
                exclusion_reason = "NO_FORWARD_RIV"
            elif bl["baseline_quality"] == "MISSING":
                exclusion_reason = "NO_BASELINE"
            elif lab["z_score"] is None:
                exclusion_reason = "Z_SCORE_FAIL"
            else:
                exclusion_reason = "UNKNOWN"

            rows.append({
                "underlying": underlying,
                "trade_date": td,
                "horizon_name": horizon_name,
                "horizon_T": T,
                "reference_expiry": ref_exp,
                "forward_date": fwd_date,
                "spot_price": spot,
                "atm_strike": atm,
                "iv_call_atm": detail.get("iv_call_atm"),
                "iv_put_atm": detail.get("iv_put_atm"),
                "riv_t": riv_t,
                "riv_t_quality": riv_t_quality,
                "riv_t_plus_T": riv_t_plus_T,
                "riv_forward_quality": riv_fwd_quality,
                "baseline_riv": bl["baseline_riv"],
                "baseline_n_valid": bl["baseline_n_valid"],
                "baseline_n_required": bl["baseline_n_required"],
                "baseline_quality": bl["baseline_quality"],
                "sigma_daily": bl["sigma_daily"],
                "sigma_daily_effective": bl["sigma_daily_effective"],
                "sigma_floor_binding": bl["sigma_floor_binding"],
                "n_consecutive_pairs": bl["n_consecutive_pairs"],
                "delta_iv": lab["delta_iv"],
                "z_score": lab["z_score"],
                "label_expansion": lab["label_expansion"],
                "label_compression": lab["label_compression"],
                "label_stable": lab["label_stable"],
                "observation_available": observation_available,
                "label_computable": label_computable,
                "training_eligible": training_eligible,
                "exclusion_reason": exclusion_reason,
                "expiry_regime": expiry_regime,
                "is_monthly_expiry": is_monthly,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Built {len(df)} labeled rows ({time.time() - t1:.1f}s)")

    # Step 4: Validate
    validate_labels(underlying, df)

    # Step 5: Summary
    summary = generate_label_summary(underlying, df)

    # Step 6: Upload
    upload_labels_to_s3(underlying, df, summary, bucket)

    return summary


# --- Validation ---

def validate_labels(underlying: str, df: pd.DataFrame) -> None:
    """Post-build validation for C2."""
    logger.info(f"{underlying}: Running C2 validation...")

    total = len(df)

    # Check 1: Row count
    for hn in HORIZONS:
        h_rows = len(df[df["horizon_name"] == hn])
        logger.info(f"  [1] {hn}: {h_rows} rows")

    # Check 2: Training eligible rate
    eligible = int(df["training_eligible"].sum())
    elig_pct = 100.0 * eligible / total if total > 0 else 0
    logger.info(f"  [2] Training eligible: {eligible}/{total} ({elig_pct:.1f}%)")

    # Check 3: Exclusion breakdown
    excluded = df[~df["training_eligible"]]
    if len(excluded) > 0:
        reasons = excluded["exclusion_reason"].value_counts()
        logger.info(f"  [3] Exclusion breakdown:")
        for reason, count in reasons.items():
            logger.info(f"      {reason}: {count}")

    # Check 4: Per-horizon exclusion rate (§2.1.7 guardrail: <15%)
    for hn in HORIZONS:
        h_df = df[df["horizon_name"] == hn]
        h_total = len(h_df)
        h_elig = int(h_df["training_eligible"].sum())
        h_excl_pct = 100.0 * (h_total - h_elig) / h_total if h_total > 0 else 0
        status = "PASS" if h_excl_pct < 15 else "WARNING (>15%)"
        logger.info(f"  [4] {hn} exclusion rate: {h_excl_pct:.1f}% — {status}")

    # Check 5: Mutual exclusivity of labels
    labeled = df[df["label_expansion"].notna()]
    if len(labeled) > 0:
        label_sum = (
            labeled["label_expansion"]
            + labeled["label_compression"]
            + labeled["label_stable"]
        )
        violations = int((label_sum != 1).sum())
        logger.info(
            f"  [5] Label mutual exclusivity: "
            f"{'PASS' if violations == 0 else f'FATAL — {violations} violations'}"
        )

    # Check 6: Label distribution (for training-eligible only)
    elig_df = df[df["training_eligible"]]
    if len(elig_df) > 0:
        for hn in HORIZONS:
            h_elig = elig_df[elig_df["horizon_name"] == hn]
            n = len(h_elig)
            if n > 0:
                exp_pct = 100.0 * h_elig["label_expansion"].sum() / n
                comp_pct = 100.0 * h_elig["label_compression"].sum() / n
                stab_pct = 100.0 * h_elig["label_stable"].sum() / n
                logger.info(
                    f"  [6] {hn} labels (n={n}): "
                    f"expansion={exp_pct:.1f}%, compression={comp_pct:.1f}%, "
                    f"stable={stab_pct:.1f}%"
                )

    # Check 7: σ_floor binding rate
    if len(elig_df) > 0:
        binding = int(elig_df["sigma_floor_binding"].sum())
        bind_pct = 100.0 * binding / len(elig_df)
        logger.info(f"  [7] σ_floor binding rate: {binding}/{len(elig_df)} ({bind_pct:.1f}%)")

    # Check 8: Baseline quality distribution
    if len(elig_df) > 0:
        bl_valid = int((elig_df["baseline_quality"] == "VALID").sum())
        bl_deg = int((elig_df["baseline_quality"] == "DEGRADED").sum())
        logger.info(
            f"  [8] Baseline quality (eligible): "
            f"VALID={bl_valid}, DEGRADED={bl_deg}"
        )

    # Check 9: z-score range
    z_vals = elig_df["z_score"].dropna()
    if len(z_vals) > 0:
        in_range = int(((z_vals >= -4) & (z_vals <= 4)).sum())
        pct = 100.0 * in_range / len(z_vals)
        logger.info(
            f"  [9] z-score within [-4, +4]: {in_range}/{len(z_vals)} ({pct:.1f}%)"
        )
        logger.info(
            f"  [9] z-score stats: min={z_vals.min():.3f} "
            f"median={z_vals.median():.3f} max={z_vals.max():.3f}"
        )

    logger.info(f"{underlying}: C2 validation complete.")


# --- Summary ---

def generate_label_summary(underlying: str, df: pd.DataFrame) -> dict:
    """Generate sidecar summary JSON."""
    elig = df[df["training_eligible"]]
    z_vals = elig["z_score"].dropna()

    summary = {
        "underlying": underlying,
        "total_observations": len(df),
        "training_eligible": int(df["training_eligible"].sum()),
        "observation_status": {
            "available": int(df["observation_available"].sum()),
            "label_computable": int(df["label_computable"].sum()),
            "training_eligible": int(df["training_eligible"].sum()),
        },
        "exclusion_breakdown": (
            df[~df["training_eligible"]]["exclusion_reason"]
            .value_counts().to_dict()
        ),
        "per_horizon": {},
        "parameters": {
            "L_short": HORIZONS["horizon_short"]["L"],
            "L_medium": HORIZONS["horizon_medium"]["L"],
            "min_data_fraction": MIN_DATA_FRACTION,
            "sigma_floor": SIGMA_FLOOR,
            "z_expand": Z_EXPAND,
            "z_compress": Z_COMPRESS,
        },
        "z_score_stats": {
            "count": int(z_vals.count()),
            "min": round(float(z_vals.min()), 4) if len(z_vals) > 0 else None,
            "median": round(float(z_vals.median()), 4) if len(z_vals) > 0 else None,
            "max": round(float(z_vals.max()), 4) if len(z_vals) > 0 else None,
        },
        "spec_reference": "Oracle-V §2.1.3-§2.1.6, Data Contract §3.3-§3.5",
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    for hn in HORIZONS:
        h_df = df[df["horizon_name"] == hn]
        h_elig = h_df[h_df["training_eligible"]]
        n = len(h_elig)
        summary["per_horizon"][hn] = {
            "total_rows": len(h_df),
            "training_eligible": n,
            "exclusion_rate_pct": round(100 * (len(h_df) - n) / len(h_df), 1) if len(h_df) > 0 else 0,
            "label_distribution": {
                "expansion": int(h_elig["label_expansion"].sum()) if n > 0 else 0,
                "compression": int(h_elig["label_compression"].sum()) if n > 0 else 0,
                "stable": int(h_elig["label_stable"].sum()) if n > 0 else 0,
                "expansion_pct": round(100 * h_elig["label_expansion"].sum() / n, 1) if n > 0 else 0,
                "compression_pct": round(100 * h_elig["label_compression"].sum() / n, 1) if n > 0 else 0,
            },
            "baseline_quality": {
                "VALID": int((h_elig["baseline_quality"] == "VALID").sum()) if n > 0 else 0,
                "DEGRADED": int((h_elig["baseline_quality"] == "DEGRADED").sum()) if n > 0 else 0,
            },
            "sigma_floor_binding_count": int(h_elig["sigma_floor_binding"].sum()) if n > 0 else 0,
        }

    return summary


# --- S3 Upload ---

def upload_labels_to_s3(
    underlying: str,
    df: pd.DataFrame,
    summary: dict,
    bucket: str = S3_BUCKET,
) -> None:
    """Upload labeled observations to S3."""
    s3 = boto3.client("s3", region_name=S3_REGION)

    parquet_key = f"computed/labeled_observations/{underlying}_labeled_obs.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=parquet_key, Body=buf.getvalue())
    logger.info(f"Uploaded: s3://{bucket}/{parquet_key}")

    json_key = f"computed/labeled_observations/{underlying}_labeled_obs_summary.json"
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
        summary = run_labeling_pipeline(ul)
        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print(f"{ul} -- C2 Labeling Complete ({elapsed:.1f}s)")
        print(f"{'='*60}")
        print(json.dumps(summary, indent=2, default=str))
        print()
