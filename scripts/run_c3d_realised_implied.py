"""
C3d Runner — Realised-Implied Spread Feature Computation
"""
import json, logging
from io import BytesIO
import boto3, pandas as pd
from oracle.compute.features.realised_implied import compute_realised_implied_features

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("c3d_runner")
s3 = boto3.client("s3")

def read_parquet_s3(key):
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(resp["Body"].read()))

def write_parquet_s3(df, key):
    buf = BytesIO(); df.to_parquet(buf, index=False, engine="pyarrow"); buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    logger.info("  Wrote s3://%s/%s (%d rows)", S3_BUCKET, key, len(df))

def write_json_s3(data, key):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data, indent=2, default=str).encode())
    logger.info("  Wrote s3://%s/%s", S3_BUCKET, key)


def main():
    logger.info("=" * 70)
    logger.info("C3d: Realised-Implied Spread — Full Batch")
    logger.info("=" * 70)

    all_summaries = {}
    for ul in UNDERLYINGS:
        logger.info("-" * 50)
        logger.info("Processing %s", ul)

        labeled_df = read_parquet_s3(f"computed/labeled_observations/{ul}_labeled_obs.parquet")
        logger.info("  Loaded %d labeled observations", len(labeled_df))

        riv_df = read_parquet_s3(f"computed/riv_series/{ul}_riv_daily.parquet")
        logger.info("  Loaded %d RIV daily rows", len(riv_df))

        ri_df = compute_realised_implied_features(
            underlying=ul, labeled_obs=labeled_df, riv_df=riv_df,
        )

        # Validation
        logger.info("  Running validation checks...")
        dup = ri_df.duplicated(subset=["underlying","trade_date","reference_expiry"])
        assert not dup.any(), f"Duplicates: {dup.sum()}"

        # Spreads should typically be in [-0.30, +0.30] (annualised)
        for col in ["iv_rv_spread_5d", "iv_rv_spread_10d"]:
            valid = ri_df[col].dropna()
            if len(valid) > 0:
                n_extreme = ((valid > 0.30) | (valid < -0.30)).sum()
                if n_extreme > 0:
                    logger.warning("  %s: %d extreme values (|spread| > 0.30)", col, n_extreme)

        logger.info("  ✓ All validation checks passed for %s", ul)

        output_key = f"computed/features/{ul}_realised_implied_features.parquet"
        write_parquet_s3(ri_df, output_key)

        n = len(ri_df)
        summary = {
            "underlying": ul, "phase": "C3d", "family": "2_realised_implied", "n_output_rows": n,
            "features": {
                "OV_F_201": {"valid": int(ri_df["iv_rv_spread_5d"].notna().sum()),
                             "null": int(ri_df["iv_rv_spread_5d"].isna().sum()),
                             "median": round(float(ri_df["iv_rv_spread_5d"].median()), 4)
                             if ri_df["iv_rv_spread_5d"].notna().any() else None},
                "OV_F_202": {"valid": int(ri_df["iv_rv_spread_10d"].notna().sum()),
                             "null": int(ri_df["iv_rv_spread_10d"].isna().sum()),
                             "median": round(float(ri_df["iv_rv_spread_10d"].median()), 4)
                             if ri_df["iv_rv_spread_10d"].notna().any() else None},
            },
        }
        all_summaries[ul] = summary
        write_json_s3(summary, f"computed/features/{ul}_realised_implied_features_summary.json")

    logger.info("=" * 70)
    logger.info("C3d COMPLETE")
    for ul in UNDERLYINGS:
        s = all_summaries[ul]
        logger.info("  %s: %d rows", ul, s["n_output_rows"])
    logger.info("Next: C3e (Feature Assembly — join all families into training table)")


if __name__ == "__main__":
    main()
