"""
Oracle-V Feature Assembly — C3e
================================
Joins all family outputs with C2 labeled observations into
the training-ready feature matrix.

Join logic:
    C2 labels (base)
    ├── LEFT JOIN Family 5 (temporal)     ON (underlying, trade_date)
    ├── LEFT JOIN Family 4 (microstructure) ON (underlying, trade_date, reference_expiry)
    ├── LEFT JOIN Family 1 (iv_surface)    ON (underlying, trade_date, reference_expiry)
    └── LEFT JOIN Family 2 (realised_impl) ON (underlying, trade_date, reference_expiry)

Output includes:
    - C2 identity + label columns (Y)
    - All feature columns (X)
    - Per-feature quality flags
    - Composite quality: n_features_valid, n_critical_valid, composite_quality

Created: 2026-02-20
"""
from __future__ import annotations
import logging
import pandas as pd
from oracle.compute.features import QUALITY_VALID, QUALITY_DEGRADED, QUALITY_UNAVAILABLE

logger = logging.getLogger(__name__)

# Feature columns by family (Data Contract §4.4 canonical names)
TEMPORAL_FEATURES = ["day_of_week", "time_of_day_bucket", "days_to_current_expiry",
                     "is_monthly_expiry_week", "days_to_next_holiday"]
MICROSTRUCTURE_FEATURES = ["pcr_volume", "pcr_oi", "oi_change_net_1d",
                           "volume_zscore_session", "fno_ban_heavyweight_flag"]
IV_SURFACE_FEATURES = ["riv_level", "riv_change_1d", "riv_change_3d",
                       "iv_percentile_rank", "iv_term_structure_slope", "iv_skew_25d",
                       "india_vix_level", "india_vix_percentile_rank", "india_vix_change"]
REALISED_IMPLIED_FEATURES = ["iv_rv_spread_5d", "iv_rv_spread_10d"]

ALL_FEATURES = TEMPORAL_FEATURES + IV_SURFACE_FEATURES + REALISED_IMPLIED_FEATURES + MICROSTRUCTURE_FEATURES

# CRITICAL features per Data Contract
CRITICAL_FEATURE_IDS = ["OV_F_101", "OV_F_102", "OV_F_105", "OV_F_201",
                        "OV_F_301", "OV_F_302", "OV_F_503"]
# Map ID -> quality column name
CRITICAL_QUALITY_COLS = {
    "OV_F_101": "quality_OV_F_101",
    "OV_F_102": "quality_OV_F_102",
    "OV_F_105": "quality_OV_F_105",
    "OV_F_201": "quality_OV_F_201",
    "OV_F_301": "quality_OV_F_301",  # deferred
    "OV_F_302": "quality_OV_F_302",  # deferred
    "OV_F_503": "quality_OV_F_503",
}


def _ensure_dates(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns and len(df) > 0 and hasattr(df[col].iloc[0], "date"):
            df[col] = df[col].apply(lambda x: x.date() if hasattr(x, "date") else x)
    return df


def assemble_training_table(
    underlying: str,
    labeled_obs: pd.DataFrame,
    temporal_df: pd.DataFrame,
    microstructure_df: pd.DataFrame,
    iv_surface_df: pd.DataFrame,
    realised_implied_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the complete training feature table.

    Returns DataFrame with C2 labels + all features + quality flags.
    """
    logger.info("Assembling training table for %s", underlying)
    logger.info("  C2 labeled obs: %d rows", len(labeled_obs))

    # Ensure date types
    base = _ensure_dates(labeled_obs, ["trade_date", "reference_expiry", "forward_date"])
    temp = _ensure_dates(temporal_df, ["trade_date"])
    micro = _ensure_dates(microstructure_df, ["trade_date", "reference_expiry"])
    ivsf = _ensure_dates(iv_surface_df, ["trade_date", "reference_expiry"])
    ri = _ensure_dates(realised_implied_df, ["trade_date", "reference_expiry"])

    # Drop 'underlying' from family dfs to avoid duplication on join
    for df in [temp, micro, ivsf, ri]:
        if "underlying" in df.columns:
            df.drop(columns=["underlying"], inplace=True)

    n_base = len(base)

    # JOIN 1: temporal (horizon-independent)
    result = base.merge(temp, on=["trade_date"], how="left", suffixes=("", "_temp"))
    assert len(result) == n_base, f"Temporal join changed row count: {len(result)} vs {n_base}"
    logger.info("  After temporal join: %d rows", len(result))

    # JOIN 2: microstructure
    result = result.merge(micro, on=["trade_date", "reference_expiry"], how="left", suffixes=("", "_micro"))
    assert len(result) == n_base, f"Microstructure join changed row count: {len(result)} vs {n_base}"
    logger.info("  After microstructure join: %d rows", len(result))

    # JOIN 3: IV surface
    result = result.merge(ivsf, on=["trade_date", "reference_expiry"], how="left", suffixes=("", "_iv"))
    assert len(result) == n_base, f"IV surface join changed row count: {len(result)} vs {n_base}"
    logger.info("  After IV surface join: %d rows", len(result))

    # JOIN 4: realised-implied
    result = result.merge(ri, on=["trade_date", "reference_expiry"], how="left", suffixes=("", "_ri"))
    assert len(result) == n_base, f"Realised-implied join changed row count: {len(result)} vs {n_base}"
    logger.info("  After realised-implied join: %d rows", len(result))

    # ── Composite quality metrics ──
    # Count valid features per row
    all_quality_cols = [c for c in result.columns if c.startswith("quality_OV_F_")]
    n_total_q = len(all_quality_cols)

    result["n_features_valid"] = result[all_quality_cols].apply(
        lambda row: (row == QUALITY_VALID).sum(), axis=1
    )
    result["n_features_degraded"] = result[all_quality_cols].apply(
        lambda row: (row == QUALITY_DEGRADED).sum(), axis=1
    )
    result["n_features_unavailable"] = result[all_quality_cols].apply(
        lambda row: (row == QUALITY_UNAVAILABLE).sum(), axis=1
    )

    # Critical features valid count
    available_critical = [c for c in CRITICAL_QUALITY_COLS.values() if c in result.columns]
    result["n_critical_valid"] = result[available_critical].apply(
        lambda row: (row == QUALITY_VALID).sum(), axis=1
    )
    result["n_critical_total"] = len(available_critical)

    # ── Logging ──
    n = len(result)
    n_features_present = sum(1 for f in ALL_FEATURES if f in result.columns)
    logger.info("  Assembly complete:")
    logger.info("    Total rows: %d", n)
    logger.info("    Feature columns present: %d / %d", n_features_present, len(ALL_FEATURES))
    logger.info("    Quality columns: %d", len(all_quality_cols))

    # Per-feature coverage
    logger.info("  Feature coverage (non-NULL):")
    for f in ALL_FEATURES:
        if f in result.columns:
            n_valid = result[f].notna().sum()
            pct = 100 * n_valid / n
            logger.info("    %-30s %d/%d (%.1f%%)", f, n_valid, n, pct)

    # Quality distribution
    logger.info("  Quality summary:")
    logger.info("    Mean features VALID per row: %.1f / %d",
                result["n_features_valid"].mean(), n_total_q)
    logger.info("    Mean features DEGRADED per row: %.1f",
                result["n_features_degraded"].mean())
    logger.info("    Mean features UNAVAILABLE per row: %.1f",
                result["n_features_unavailable"].mean())
    logger.info("    Mean critical VALID per row: %.1f / %d",
                result["n_critical_valid"].mean(), len(available_critical))

    # Training eligibility
    n_eligible = result["training_eligible"].sum() if "training_eligible" in result.columns else "N/A"
    logger.info("    Training-eligible rows: %s", n_eligible)

    return result
