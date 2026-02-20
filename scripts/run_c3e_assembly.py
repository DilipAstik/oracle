"""
C3e Runner — Feature Assembly into Training Table
"""
import json, logging
from io import BytesIO
import boto3, pandas as pd
from oracle.compute.features.assembler import assemble_training_table

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("c3e_runner")
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
    logger.info("C3e: Feature Assembly — Training Table Construction")
    logger.info("=" * 70)

    all_summaries = {}
    for ul in UNDERLYINGS:
        logger.info("-" * 50)
        logger.info("Processing %s", ul)

        # Load all inputs
        labeled_df = read_parquet_s3(f"computed/labeled_observations/{ul}_labeled_obs.parquet")
        temporal_df = read_parquet_s3(f"computed/features/{ul}_temporal_features.parquet")
        micro_df = read_parquet_s3(f"computed/features/{ul}_microstructure_features.parquet")
        ivsf_df = read_parquet_s3(f"computed/features/{ul}_iv_surface_features.parquet")
        ri_df = read_parquet_s3(f"computed/features/{ul}_realised_implied_features.parquet")

        logger.info("  Loaded: labeled=%d, temporal=%d, micro=%d, iv_surface=%d, realised_implied=%d",
                     len(labeled_df), len(temporal_df), len(micro_df), len(ivsf_df), len(ri_df))

        # Assemble
        training_df = assemble_training_table(
            underlying=ul,
            labeled_obs=labeled_df,
            temporal_df=temporal_df,
            microstructure_df=micro_df,
            iv_surface_df=ivsf_df,
            realised_implied_df=ri_df,
        )

        # Validation
        logger.info("  Running validation checks...")
        assert len(training_df) == len(labeled_df), "Row count mismatch with C2"

        dup = training_df.duplicated(subset=["underlying","trade_date","horizon_name","reference_expiry"])
        assert not dup.any(), f"Duplicate keys: {dup.sum()}"

        # Check labels still present
        for lbl in ["label_expansion", "label_compression", "label_stable"]:
            assert lbl in training_df.columns, f"Missing label column: {lbl}"

        logger.info("  ✓ All validation checks passed for %s", ul)

        # Save
        output_key = f"computed/training/{ul}_training_features.parquet"
        write_parquet_s3(training_df, output_key)

        # Summary
        n = len(training_df)
        n_eligible = int(training_df["training_eligible"].sum()) if "training_eligible" in training_df.columns else None
        summary = {
            "underlying": ul, "phase": "C3e", "n_rows": n,
            "n_training_eligible": n_eligible,
            "n_columns": len(training_df.columns),
            "quality": {
                "mean_valid": round(float(training_df["n_features_valid"].mean()), 1),
                "mean_degraded": round(float(training_df["n_features_degraded"].mean()), 1),
                "mean_unavailable": round(float(training_df["n_features_unavailable"].mean()), 1),
                "mean_critical_valid": round(float(training_df["n_critical_valid"].mean()), 1),
            },
        }
        all_summaries[ul] = summary
        write_json_s3(summary, f"computed/training/{ul}_training_features_summary.json")

    logger.info("=" * 70)
    logger.info("C3e COMPLETE — TRAINING TABLES ASSEMBLED")
    logger.info("=" * 70)
    for ul in UNDERLYINGS:
        s = all_summaries[ul]
        logger.info("  %s: %d rows (%s training-eligible), %d columns",
                     ul, s["n_rows"], s["n_training_eligible"], s["n_columns"])
        logger.info("    Quality: %.1f valid, %.1f degraded, %.1f unavailable per row",
                     s["quality"]["mean_valid"], s["quality"]["mean_degraded"], s["quality"]["mean_unavailable"])
    logger.info("Output: s3://%s/computed/training/", S3_BUCKET)
    logger.info("")
    logger.info("C3 FEATURE ENGINEERING COMPLETE")
    logger.info("  Families computed: 5 (temporal), 4 (microstructure), 1 (iv_surface), 2 (realised_implied)")
    logger.info("  Deferred: Family 3 (events), India VIX (F-107/108/109), F&O ban (F-405)")
    logger.info("  Next: Model research phase")


if __name__ == "__main__":
    main()
