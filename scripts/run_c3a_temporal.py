"""
C3a Runner — Temporal Feature Computation (Full Batch)
"""
import json
import logging
from datetime import date, timedelta
from io import BytesIO

import boto3
import pandas as pd

from oracle.calendar.trading_calendar import NSETradingCalendar
from oracle.compute.features.temporal import compute_temporal_features
from oracle.compute.features import HOLIDAY_HORIZON_MAX, HISTORICAL_TIME_BUCKET

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_PREFIX = "computed"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("c3a_runner")
s3 = boto3.client("s3")


def read_parquet_s3(key):
    logger.info("  Reading s3://%s/%s", S3_BUCKET, key)
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(resp["Body"].read()))


def write_parquet_s3(df, key):
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    logger.info("  Wrote s3://%s/%s (%d rows)", S3_BUCKET, key, len(df))


def write_json_s3(data, key):
    body = json.dumps(data, indent=2, default=str)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body.encode("utf-8"))
    logger.info("  Wrote s3://%s/%s", S3_BUCKET, key)


def main():
    logger.info("=" * 70)
    logger.info("C3a: Temporal Feature Computation — Full Batch")
    logger.info("=" * 70)

    cal = NSETradingCalendar()
    all_summaries = {}

    for ul in UNDERLYINGS:
        logger.info("-" * 50)
        logger.info("Processing %s", ul)

        # Load C2 labeled observations
        labeled_key = f"{S3_PREFIX}/labeled_observations/{ul}_labeled_obs.parquet"
        labeled_df = read_parquet_s3(labeled_key)
        logger.info("  Loaded %d labeled observations", len(labeled_df))

        # Extract unique trade dates (temporal features are horizon-independent)
        trade_dates_unique = sorted(set(
            d.date() if hasattr(d, "date") else d
            for d in labeled_df["trade_date"].unique()
        ))
        logger.info("  %d unique trade dates", len(trade_dates_unique))

        # Compute temporal features
        temporal_df = compute_temporal_features(ul, trade_dates_unique, cal)

        # ── Validation ──
        logger.info("  Running validation checks...")

        n = len(temporal_df)
        assert n == len(trade_dates_unique), \
            f"Row count mismatch: {n} vs {len(trade_dates_unique)}"

        feature_cols = ["day_of_week", "time_of_day_bucket", "days_to_current_expiry",
                        "is_monthly_expiry_week", "days_to_next_holiday"]
        for col in feature_cols:
            n_null = temporal_df[col].isna().sum()
            assert n_null == 0, f"NULL in {col}: {n_null}"

        assert temporal_df["day_of_week"].between(0, 4).all()
        assert (temporal_df["time_of_day_bucket"] == HISTORICAL_TIME_BUCKET).all()
        assert (temporal_df["days_to_current_expiry"] >= 0).all()
        assert temporal_df["is_monthly_expiry_week"].isin([0, 1]).all()
        assert temporal_df["days_to_next_holiday"].between(0, HOLIDAY_HORIZON_MAX).all()

        # NIFTY dte should be small (weekly), BANKNIFTY larger (monthly)
        dte_max = temporal_df["days_to_current_expiry"].max()
        dte_mean = temporal_df["days_to_current_expiry"].mean()
        logger.info("  OV-F-503 days_to_current_expiry: max=%d, mean=%.1f", dte_max, dte_mean)

        # Day-of-week distribution
        dow = temporal_df["day_of_week"].value_counts().sort_index().to_dict()
        logger.info("  OV-F-501 day_of_week dist: %s", dow)

        # Monthly expiry week percentage
        pct_mew = 100 * temporal_df["is_monthly_expiry_week"].mean()
        logger.info("  OV-F-504 is_monthly_expiry_week: %.1f%%", pct_mew)

        # Holiday proximity
        hol_mean = temporal_df["days_to_next_holiday"].mean()
        hol_at_cap = (temporal_df["days_to_next_holiday"] == HOLIDAY_HORIZON_MAX).sum()
        logger.info("  OV-F-505 days_to_next_holiday: mean=%.1f, at_cap=%d/%d",
                     hol_mean, hol_at_cap, n)

        logger.info("  ✓ All validation checks passed for %s", ul)

        # ── Save to S3 ──
        output_key = f"{S3_PREFIX}/features/{ul}_temporal_features.parquet"
        write_parquet_s3(temporal_df, output_key)

        # ── Summary ──
        summary = {
            "underlying": ul,
            "phase": "C3a",
            "family": "5_temporal",
            "n_trade_dates": len(trade_dates_unique),
            "n_output_rows": n,
            "features": {
                "OV_F_501_day_of_week": {"distribution": dow},
                "OV_F_502_time_of_day_bucket": {"value": HISTORICAL_TIME_BUCKET},
                "OV_F_503_days_to_current_expiry": {
                    "min": int(temporal_df["days_to_current_expiry"].min()),
                    "max": int(dte_max),
                    "mean": round(float(dte_mean), 2),
                },
                "OV_F_504_is_monthly_expiry_week": {
                    "pct": round(pct_mew, 1),
                    "count_1": int(temporal_df["is_monthly_expiry_week"].sum()),
                },
                "OV_F_505_days_to_next_holiday": {
                    "mean": round(float(hol_mean), 2),
                    "at_cap": int(hol_at_cap),
                },
            },
        }
        all_summaries[ul] = summary
        write_json_s3(summary, f"{S3_PREFIX}/features/{ul}_temporal_features_summary.json")

    # Final
    logger.info("=" * 70)
    logger.info("C3a COMPLETE")
    for ul in UNDERLYINGS:
        logger.info("  %s: %d rows", ul, all_summaries[ul]["n_output_rows"])
    logger.info("Output: s3://%s/%s/features/", S3_BUCKET, S3_PREFIX)
    logger.info("Next: C3b (Family 4 — Microstructure features)")


if __name__ == "__main__":
    main()
