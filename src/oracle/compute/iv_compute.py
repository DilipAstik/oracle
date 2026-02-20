"""
IV Computation Pipeline - per-day IV surface computation and batch runner.

Module: src/oracle/compute/iv_compute.py
Phase: B5 - IV Computation Service
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from oracle.compute.black_scholes import (
    BSParams,
    bs_price,
    implied_volatility,
    intrinsic_value,
    IV_STATUS_CONVERGED,
    IV_STATUS_EXPIRY_DAY,
    IV_STATUS_EXTREME_VALUE,
    IV_STATUS_MISSING_FUTURES,
    IV_STATUS_MISSING_RFR,
    IV_STATUS_OUT_OF_MONEYNESS,
    IV_STATUS_PRICE_TOO_LOW,
)
from oracle.compute.rate_loader import (
    compute_dividend_yield,
    get_rfr_for_date,
    get_rfr_version,
    load_risk_free_rates,
)

logger = logging.getLogger(__name__)
np.seterr(all="raise")

DEFAULT_MONEYNESS_BOUNDS = (0.75, 1.25)
DEFAULT_MIN_SETTLEMENT_PRICE = 0.05
DEFAULT_MIN_DTE = 1
DEFAULT_IV_RANGE = (0.02, 2.0)


def _process_single_option(row, trade_date, underlying, rfr, q,
                           moneyness_bounds, min_settlement_price, min_dte,
                           rfr_is_stale, div_status):
    expiry_date = row["expiry_date"]
    if isinstance(expiry_date, pd.Timestamp):
        expiry_date = expiry_date.date()

    strike = float(row["strike"])
    option_type = str(row["option_type"])
    settlement_price = float(row["settlement_price"])
    underlying_price = float(row["underlying_price"])
    volume = int(row.get("volume", 0))
    open_interest = int(row.get("open_interest", 0))

    calendar_dte = (expiry_date - trade_date).days
    T = calendar_dte / 365.0
    moneyness = strike / underlying_price if underlying_price > 0 else float("nan")

    result = {
        "trade_date": trade_date,
        "underlying": underlying,
        "expiry_date": expiry_date,
        "strike": strike,
        "option_type": option_type,
        "settlement_price": settlement_price,
        "underlying_price": underlying_price,
        "iv": float("nan"),
        "iv_status": "",
        "time_to_expiry_years": T,
        "calendar_days_to_expiry": calendar_dte,
        "moneyness": moneyness,
        "risk_free_rate": rfr,
        "dividend_yield": q,
        "rfr_version_id": get_rfr_version(),
        "intrinsic_value_approx": float("nan"),
        "time_value": float("nan"),
        "bs_price": float("nan"),
        "price_error": float("nan"),
        "volume": volume,
        "open_interest": open_interest,
        "traded": volume > 0,
        "solver_iterations": 0,
    }

    if calendar_dte < min_dte:
        result["iv_status"] = IV_STATUS_EXPIRY_DAY
        return result
    if not (moneyness_bounds[0] <= moneyness <= moneyness_bounds[1]):
        result["iv_status"] = IV_STATUS_OUT_OF_MONEYNESS
        return result
    if settlement_price < min_settlement_price:
        result["iv_status"] = IV_STATUS_PRICE_TOO_LOW
        return result
    if settlement_price <= 0:
        result["iv_status"] = IV_STATUS_PRICE_TOO_LOW
        return result

    try:
        params = BSParams(
            S=np.float64(underlying_price), K=np.float64(strike),
            T=np.float64(T), r=np.float64(rfr), q=np.float64(q),
        )
        iv_approx = intrinsic_value(params, option_type)
        result["intrinsic_value_approx"] = iv_approx
        result["time_value"] = settlement_price - iv_approx
    except (ValueError, FloatingPointError) as e:
        result["iv_status"] = "SOLVER_FAILED"
        return result

    try:
        iv, status, iterations = implied_volatility(
            params=params, option_type=option_type,
            market_price=np.float64(settlement_price),
        )
    except Exception:
        result["iv_status"] = "SOLVER_FAILED"
        return result

    result["iv_status"] = status
    result["solver_iterations"] = iterations

    if iv is not None:
        result["iv"] = iv
        try:
            bs_p = bs_price(params, option_type, iv)
            result["bs_price"] = bs_p
            result["price_error"] = abs(bs_p - settlement_price)
        except (FloatingPointError, OverflowError):
            pass

    if div_status == "MISSING":
        result["iv_status"] = IV_STATUS_MISSING_FUTURES

    return result


def _compute_atm_summary(iv_df, trade_date, underlying):
    if len(iv_df) == 0:
        return {"atm_iv_call": None, "atm_iv_put": None,
                "atm_iv_midpoint": None, "atm_strike": None}
    converged = iv_df[iv_df["iv_status"] == IV_STATUS_CONVERGED].copy()
    if len(converged) == 0:
        return {"atm_iv_call": None, "atm_iv_put": None,
                "atm_iv_midpoint": None, "atm_strike": None}

    underlying_price = converged["underlying_price"].iloc[0]
    nearest_expiry = converged["expiry_date"].min()
    near_exp = converged[converged["expiry_date"] == nearest_expiry]
    if len(near_exp) == 0:
        return {"atm_iv_call": None, "atm_iv_put": None,
                "atm_iv_midpoint": None, "atm_strike": None}

    strikes = near_exp["strike"].unique()
    diffs = np.abs(strikes - underlying_price)
    min_diff = diffs.min()
    atm_candidates = strikes[diffs == min_diff]
    atm_strike = float(min(atm_candidates))

    atm_records = near_exp[near_exp["strike"] == atm_strike]
    atm_call = atm_records[atm_records["option_type"] == "CE"]
    atm_put = atm_records[atm_records["option_type"] == "PE"]

    atm_iv_call = float(atm_call["iv"].iloc[0]) if len(atm_call) > 0 else None
    atm_iv_put = float(atm_put["iv"].iloc[0]) if len(atm_put) > 0 else None
    atm_iv_midpoint = None
    if atm_iv_call is not None and atm_iv_put is not None:
        atm_iv_midpoint = (atm_iv_call + atm_iv_put) / 2.0

    return {"atm_iv_call": atm_iv_call, "atm_iv_put": atm_iv_put,
            "atm_iv_midpoint": atm_iv_midpoint, "atm_strike": atm_strike}


def _count_zero_volume_atm(iv_df):
    if len(iv_df) == 0 or "volume" not in iv_df.columns:
        return 0
    converged = iv_df[iv_df["iv_status"] == IV_STATUS_CONVERGED]
    if len(converged) == 0:
        return 0
    underlying_price = converged["underlying_price"].iloc[0]
    atm_range = underlying_price * 0.005
    atm_vicinity = iv_df[
        (iv_df["strike"] >= underlying_price - atm_range)
        & (iv_df["strike"] <= underlying_price + atm_range)
    ]
    return int((atm_vicinity["volume"] == 0).sum())


def compute_iv_surface_for_day(options_df, futures_df, rfr,
                               rfr_is_stale=False,
                               moneyness_bounds=DEFAULT_MONEYNESS_BOUNDS,
                               min_settlement_price=DEFAULT_MIN_SETTLEMENT_PRICE,
                               min_dte=DEFAULT_MIN_DTE):
    if options_df is None or len(options_df) == 0:
        return pd.DataFrame(), {"error": "No options data"}

    trade_date = options_df["trade_date"].iloc[0]
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.date()
    underlying = options_df["underlying"].iloc[0]

    start_time = time.time()

    q_implied, futures_expiry_used, div_status = compute_dividend_yield(
        futures_df, underlying, trade_date, rfr
    )
    if q_implied is None:
        q_implied = 0.01
        logger.warning("No dividend yield for %s on %s - using default q=0.01",
                        underlying, trade_date)

    results = []
    status_counts = {
        "total": 0, "converged": 0, "not_converged": 0, "solver_failed": 0,
        "extreme_value": 0, "below_intrinsic": 0, "expiry_day": 0,
        "price_too_low": 0, "out_of_moneyness": 0, "missing_futures": 0,
        "missing_rfr": 0,
    }

    for _, row in options_df.iterrows():
        status_counts["total"] += 1
        record = _process_single_option(
            row=row, trade_date=trade_date, underlying=underlying,
            rfr=rfr, q=q_implied, moneyness_bounds=moneyness_bounds,
            min_settlement_price=min_settlement_price, min_dte=min_dte,
            rfr_is_stale=rfr_is_stale, div_status=div_status,
        )
        results.append(record)
        iv_status = record["iv_status"]
        status_key = iv_status.lower()
        if status_key in status_counts:
            status_counts[status_key] += 1

    iv_df = pd.DataFrame(results)

    float_cols = [
        "iv", "settlement_price", "underlying_price", "time_to_expiry_years",
        "moneyness", "risk_free_rate", "dividend_yield", "intrinsic_value_approx",
        "time_value", "bs_price", "price_error",
    ]
    for col in float_cols:
        if col in iv_df.columns:
            iv_df[col] = iv_df[col].astype("float64")

    atm_summary = _compute_atm_summary(iv_df, trade_date, underlying)

    processing_time = time.time() - start_time
    summary = {
        "trade_date": str(trade_date),
        "underlying": underlying,
        "total_options": status_counts["total"],
        "iv_computed": status_counts["total"] - (
            status_counts["expiry_day"] + status_counts["price_too_low"]
            + status_counts["out_of_moneyness"]
        ),
        **{k: v for k, v in status_counts.items() if k != "total"},
        "risk_free_rate": rfr,
        "rfr_is_stale": rfr_is_stale,
        "rfr_version_id": get_rfr_version(),
        "dividend_yield": q_implied,
        "dividend_yield_status": div_status,
        "futures_tenor_used": str(futures_expiry_used) if futures_expiry_used else None,
        **atm_summary,
        "zero_volume_count": int((iv_df["volume"] == 0).sum()) if "volume" in iv_df.columns else 0,
        "zero_volume_atm_count": _count_zero_volume_atm(iv_df),
        "num_expiries": iv_df["expiry_date"].nunique() if len(iv_df) > 0 else 0,
        "processing_time_seconds": round(processing_time, 2),
    }

    return iv_df, summary


# --- S3 I/O Helpers ---

def _read_parquet_from_s3(s3_client, bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning("Failed to read s3://%s/%s: %s", bucket, key, e)
        return None


def _write_parquet_to_s3(s3_client, bucket, key, df):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def _write_json_to_s3(s3_client, bucket, key, data):
    body = json.dumps(data, indent=2, default=str)
    s3_client.put_object(
        Bucket=bucket, Key=key, Body=body.encode("utf-8"),
        ContentType="application/json",
    )


def run_iv_pipeline(s3_client, bucket, underlying, start_date, end_date,
                    rfr_csv_path=None, moneyness_bounds=DEFAULT_MONEYNESS_BOUNDS):
    logger.info("Starting IV pipeline: %s, %s to %s", underlying, start_date, end_date)

    rfr_df = load_risk_free_rates(rfr_csv_path)

    current = start_date
    days_processed = 0
    days_skipped = 0
    total_converged = 0
    total_options = 0

    while current <= end_date:
        year = current.year
        month = current.month
        date_str = current.strftime("%Y%m%d")

        options_key = (
            f"canonical/option_chains/underlying={underlying}/"
            f"year={year}/month={month:02d}/{underlying}_{date_str}.parquet"
        )
        futures_key = (
            f"canonical/futures/underlying={underlying}/"
            f"year={year}/month={month:02d}/{underlying}_FUT_{date_str}.parquet"
        )

        options_df = _read_parquet_from_s3(s3_client, bucket, options_key)
        if options_df is None or len(options_df) == 0:
            current += timedelta(days=1)
            days_skipped += 1
            continue

        futures_df = _read_parquet_from_s3(s3_client, bucket, futures_key)

        try:
            rfr, rfr_is_stale = get_rfr_for_date(rfr_df, current)
        except ValueError:
            logger.error("No RFR for %s - skipping", current)
            current += timedelta(days=1)
            days_skipped += 1
            continue

        iv_df, summary = compute_iv_surface_for_day(
            options_df=options_df, futures_df=futures_df,
            rfr=rfr, rfr_is_stale=rfr_is_stale,
            moneyness_bounds=moneyness_bounds,
        )

        if len(iv_df) > 0:
            iv_key = (
                f"computed/iv_surfaces/underlying={underlying}/"
                f"year={year}/month={month:02d}/{underlying}_iv_{date_str}.parquet"
            )
            _write_parquet_to_s3(s3_client, bucket, iv_key, iv_df)

            summary_key = (
                f"computed/iv_surfaces/underlying={underlying}/"
                f"year={year}/month={month:02d}/{underlying}_iv_{date_str}_summary.json"
            )
            _write_json_to_s3(s3_client, bucket, summary_key, summary)

            days_processed += 1
            total_converged += summary.get("converged", 0)
            total_options += summary.get("total_options", 0)

            logger.info(
                "%s %s: %d/%d converged (%.1f%%), ATM IV=%.2f%%, q=%.4f, %.1fs",
                underlying, current,
                summary.get("converged", 0), summary.get("total_options", 0),
                100.0 * summary.get("converged", 0) / max(summary.get("total_options", 1), 1),
                (summary.get("atm_iv_midpoint") or 0) * 100,
                summary.get("dividend_yield", 0),
                summary.get("processing_time_seconds", 0),
            )

        current += timedelta(days=1)

    convergence_rate = total_converged / max(total_options, 1) * 100

    aggregate = {
        "underlying": underlying,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "days_processed": days_processed,
        "days_skipped": days_skipped,
        "total_options": total_options,
        "total_converged": total_converged,
        "overall_convergence_rate_pct": round(convergence_rate, 2),
        "rfr_version_id": get_rfr_version(),
    }

    logger.info(
        "IV pipeline complete: %s - %d days, %d/%d converged (%.1f%%)",
        underlying, days_processed, total_converged, total_options, convergence_rate,
    )

    return aggregate
