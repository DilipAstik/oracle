"""
C3b Runner — Microstructure Feature Computation (Full Batch)
"""
import json
import logging
from io import BytesIO

import boto3
import numpy as np
import pandas as pd

from oracle.calendar.trading_calendar import NSETradingCalendar
from oracle.compute.features.microstructure import compute_microstructure_features

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_PREFIX = "computed"
CANONICAL_PREFIX = "canonical/option_chains"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("c3b_runner")
s3 = boto3.client("s3")


def read_parquet_s3(key):
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


def load_all_canonical_options(underlying):
    prefix = f"{CANONICAL_PREFIX}/underlying={underlying}/"
    logger.info("  Listing canonical options at s3://%s/%s", S3_BUCKET, prefix)

    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])

    logger.info("  Found %d parquet files for %s", len(keys), underlying)
    if not keys:
        raise RuntimeError(f"No canonical option files found for {underlying}")

    dfs = []
    for i, key in enumerate(sorted(keys)):
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(resp["Body"].read()))
        dfs.append(df)
        if (i + 1) % 100 == 0:
            logger.info("    Loaded %d/%d files...", i + 1, len(keys))

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("  Loaded %d total option records for %s", len(combined), underlying)
    return combined


def main():
    logger.info("=" * 70)
    logger.info("C3b: Microstructure Feature Computation — Full Batch")
    logger.info("=" * 70)

    cal = NSETradingCalendar()
    all_summaries = {}

    for ul in UNDERLYINGS:
        logger.info("-" * 50)
        logger.info("Processing %s", ul)

        labeled_key = f"{S3_PREFIX}/labeled_observations/{ul}_labeled_obs.parquet"
        logger.info("  Reading labeled observations...")
        labeled_df = read_parquet_s3(labeled_key)
        logger.info("  Loaded %d labeled observations", len(labeled_df))

        options_df = load_all_canonical_options(ul)

        micro_df = compute_microstructure_features(
            underlying=ul,
            labeled_obs=labeled_df,
            options_df=options_df,
            calendar=cal,
        )

        # Validation
        logger.info("  Running validation checks...")

        dup_check = micro_df.duplicated(subset=["underlying", "trade_date", "reference_expiry"])
        assert not dup_check.any(), f"Duplicate keys: {dup_check.sum()}"

        assert (micro_df["fno_ban_heavyweight_flag"] == 0).all()

        for col, label in [("pcr_volume", "F-401"), ("pcr_oi", "F-402")]:
            valid = micro_df[col].dropna()
            if len(valid) > 0:
                n_extreme = ((valid > 5.0) | (valid < 0.2)).sum()
                if n_extreme > 0:
                    logger.warning("  %s %s: %d extreme values out of %d",
                                   label, col, n_extreme, len(valid))

        valid_z = micro_df["volume_zscore_session"].dropna()
        if len(valid_z) > 0:
            n_extreme_z = ((valid_z > 4) | (valid_z < -4)).sum()
            logger.info("  F-404 z-score extremes (|z|>4): %d (%.1f%%)",
                        n_extreme_z, 100 * n_extreme_z / len(valid_z))

        logger.info("  ✓ All validation checks passed for %s", ul)

        output_key = f"{S3_PREFIX}/features/{ul}_microstructure_features.parquet"
        write_parquet_s3(micro_df, output_key)

        n = len(micro_df)
        summary = {
            "underlying": ul,
            "phase": "C3b",
            "family": "4_microstructure",
            "n_output_rows": n,
            "n_option_records": len(options_df),
            "features": {
                "OV_F_401_pcr_volume": {
                    "valid": int(micro_df["pcr_volume"].notna().sum()),
                    "null": int(micro_df["pcr_volume"].isna().sum()),
                    "median": round(float(micro_df["pcr_volume"].median()), 3)
                    if micro_df["pcr_volume"].notna().any() else None,
                },
                "OV_F_402_pcr_oi": {
                    "valid": int(micro_df["pcr_oi"].notna().sum()),
                    "null": int(micro_df["pcr_oi"].isna().sum()),
                    "median": round(float(micro_df["pcr_oi"].median()), 3)
                    if micro_df["pcr_oi"].notna().any() else None,
                },
                "OV_F_403_oi_change_net_1d": {
                    "valid": int(micro_df["oi_change_net_1d"].notna().sum()),
                    "null": int(micro_df["oi_change_net_1d"].isna().sum()),
                },
                "OV_F_404_volume_zscore": {
                    "valid": int(micro_df["volume_zscore_session"].notna().sum()),
                    "null": int(micro_df["volume_zscore_session"].isna().sum()),
                },
                "OV_F_405_fno_ban": {"status": "UNAVAILABLE", "all_zero": True},
            },
        }
        all_summaries[ul] = summary
        write_json_s3(summary, f"{S3_PREFIX}/features/{ul}_microstructure_features_summary.json")

        del options_df
        logger.info("  Memory released for %s", ul)

    logger.info("=" * 70)
    logger.info("C3b COMPLETE")
    for ul in UNDERLYINGS:
        s = all_summaries[ul]
        logger.info("  %s: %d rows from %d option records",
                     ul, s["n_output_rows"], s["n_option_records"])
    logger.info("Output: s3://%s/%s/features/", S3_BUCKET, S3_PREFIX)
    logger.info("Next: C3c (Family 1 — IV Surface features)")


if __name__ == "__main__":
    main()
