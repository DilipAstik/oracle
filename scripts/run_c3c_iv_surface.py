"""
C3c Runner — IV Surface Feature Computation (Full Batch)
"""
import json, logging
from io import BytesIO
import boto3, pandas as pd
from oracle.compute.features.iv_surface import compute_iv_surface_features

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("c3c_runner")
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

def load_all_iv_surfaces(underlying):
    prefix = f"computed/iv_surfaces/underlying={underlying}/"
    logger.info("  Listing IV surfaces at s3://%s/%s", S3_BUCKET, prefix)
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    logger.info("  Found %d IV surface files for %s", len(keys), underlying)
    dfs = []
    for i, key in enumerate(sorted(keys)):
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        dfs.append(pd.read_parquet(BytesIO(resp["Body"].read())))
        if (i + 1) % 100 == 0:
            logger.info("    Loaded %d/%d files...", i + 1, len(keys))
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("  Loaded %d total IV surface records for %s", len(combined), underlying)
    return combined


def main():
    logger.info("=" * 70)
    logger.info("C3c: IV Surface Feature Computation — Full Batch")
    logger.info("=" * 70)

    all_summaries = {}
    for ul in UNDERLYINGS:
        logger.info("-" * 50)
        logger.info("Processing %s", ul)

        labeled_df = read_parquet_s3(f"computed/labeled_observations/{ul}_labeled_obs.parquet")
        logger.info("  Loaded %d labeled observations", len(labeled_df))

        riv_df = read_parquet_s3(f"computed/riv_series/{ul}_riv_daily.parquet")
        logger.info("  Loaded %d RIV daily rows", len(riv_df))

        iv_surfaces_df = load_all_iv_surfaces(ul)

        iv_df = compute_iv_surface_features(
            underlying=ul,
            labeled_obs=labeled_df,
            riv_df=riv_df,
            iv_surfaces_df=iv_surfaces_df,
        )

        # Validation
        logger.info("  Running validation checks...")
        dup = iv_df.duplicated(subset=["underlying","trade_date","reference_expiry"])
        assert not dup.any(), f"Duplicates: {dup.sum()}"

        # RIV level should be in [0.02, 0.80] (annualised)
        valid_riv = iv_df["riv_level"].dropna()
        if len(valid_riv) > 0:
            assert valid_riv.between(0.02, 0.80).all(), \
                f"riv_level out of range: [{valid_riv.min():.4f}, {valid_riv.max():.4f}]"

        # Percentile should be [0, 1]
        valid_pct = iv_df["iv_percentile_rank"].dropna()
        if len(valid_pct) > 0:
            assert valid_pct.between(0, 1).all()

        # Skew should typically be positive (put premium)
        valid_skew = iv_df["iv_skew_25d"].dropna()
        if len(valid_skew) > 0:
            pct_positive = (valid_skew > 0).mean() * 100
            logger.info("  F-110 skew positive: %.1f%%", pct_positive)

        logger.info("  ✓ All validation checks passed for %s", ul)

        output_key = f"computed/features/{ul}_iv_surface_features.parquet"
        write_parquet_s3(iv_df, output_key)

        n = len(iv_df)
        summary = {
            "underlying": ul, "phase": "C3c", "family": "1_iv_surface", "n_output_rows": n,
            "features": {
                "OV_F_101": {"valid": int(iv_df["riv_level"].notna().sum()), "null": int(iv_df["riv_level"].isna().sum())},
                "OV_F_102": {"valid": int(iv_df["riv_change_1d"].notna().sum()), "null": int(iv_df["riv_change_1d"].isna().sum())},
                "OV_F_103": {"valid": int(iv_df["riv_change_3d"].notna().sum()), "null": int(iv_df["riv_change_3d"].isna().sum())},
                "OV_F_105": {"valid": int(iv_df["iv_percentile_rank"].notna().sum()), "null": int(iv_df["iv_percentile_rank"].isna().sum())},
                "OV_F_106": {"valid": int(iv_df["iv_term_structure_slope"].notna().sum()), "null": int(iv_df["iv_term_structure_slope"].isna().sum())},
                "OV_F_110": {"valid": int(iv_df["iv_skew_25d"].notna().sum()), "null": int(iv_df["iv_skew_25d"].isna().sum())},
                "OV_F_107_108_109": "deferred — India VIX not acquired",
            },
        }
        all_summaries[ul] = summary
        write_json_s3(summary, f"computed/features/{ul}_iv_surface_features_summary.json")

        del iv_surfaces_df
        logger.info("  Memory released for %s", ul)

    logger.info("=" * 70)
    logger.info("C3c COMPLETE")
    for ul in UNDERLYINGS:
        s = all_summaries[ul]
        logger.info("  %s: %d rows", ul, s["n_output_rows"])
    logger.info("Next: C3d (Family 2 — Realised-Implied Spread)")


if __name__ == "__main__":
    main()
