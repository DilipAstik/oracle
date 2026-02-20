"""
Oracle-V Feature Engineering — Family 4: Options Microstructure Features
=========================================================================

Phase: C3b
Authority: Data Contract §4.4 (IDs/columns), Part III §5.2.4 (computation)
Features: 5 Core (0 CRITICAL, 5 STANDARD)

    OV-F-401  pcr_volume               STANDARD  Put/Call volume ratio (reference expiry only)
    OV-F-402  pcr_oi                   STANDARD  Put/Call OI ratio (reference + next expiry)
    OV-F-403  oi_change_net_1d         STANDARD  Net aggregate OI change (reference + next expiry)
    OV-F-404  volume_zscore_session    STANDARD  Total session volume z-score vs 20-day trailing
    OV-F-405  fno_ban_heavyweight_flag STANDARD  Placeholder: constant 0, UNAVAILABLE

Design decisions (from three-expert review):
    - F-401: reference expiry only (volume = directional flow, not roll structure)
    - F-402/F-403: reference + next expiry (OI = positioning across roll)
    - F-404: total volume across ALL expiries (activity detector, spans expiries)
    - F-405: constant 0, quality = UNAVAILABLE (source not yet acquired)

Created: 2026-02-20
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from oracle.compute.features import (
    QUALITY_DEGRADED,
    QUALITY_UNAVAILABLE,
    QUALITY_VALID,
    REASON_SOURCE_MISSING,
)

logger = logging.getLogger(__name__)

VOLUME_ZSCORE_LOOKBACK = 20
PCR_LOW_WARN = 0.2
PCR_HIGH_WARN = 5.0


def _find_next_expiry(reference_expiry, all_expiry_dates):
    idx = bisect_right(all_expiry_dates, reference_expiry)
    if idx < len(all_expiry_dates):
        return all_expiry_dates[idx]
    return None


def _pre_aggregate_options(options_df):
    agg = options_df.groupby(
        ["trade_date", "expiry_date", "option_type"],
        as_index=False,
    ).agg(
        total_volume=("volume", "sum"),
        total_oi=("open_interest", "sum"),
        total_oi_change=("oi_change", "sum"),
    )
    return agg


def _compute_daily_total_volume(agg_df):
    daily = agg_df.groupby("trade_date")["total_volume"].sum()
    return daily.sort_index()


def compute_pcr_volume(agg_df, trade_date, reference_expiry):
    mask = (agg_df["trade_date"] == trade_date) & (agg_df["expiry_date"] == reference_expiry)
    subset = agg_df.loc[mask]
    call_vol = subset.loc[subset["option_type"] == "CE", "total_volume"].sum()
    put_vol = subset.loc[subset["option_type"] == "PE", "total_volume"].sum()
    if call_vol == 0:
        return None, QUALITY_DEGRADED
    return float(put_vol / call_vol), QUALITY_VALID


def compute_pcr_oi(agg_df, trade_date, reference_expiry, next_expiry):
    expiries = [reference_expiry]
    if next_expiry is not None:
        expiries.append(next_expiry)
    mask = (agg_df["trade_date"] == trade_date) & (agg_df["expiry_date"].isin(expiries))
    subset = agg_df.loc[mask]
    call_oi = subset.loc[subset["option_type"] == "CE", "total_oi"].sum()
    put_oi = subset.loc[subset["option_type"] == "PE", "total_oi"].sum()
    if call_oi == 0:
        return None, QUALITY_DEGRADED
    quality = QUALITY_VALID if next_expiry is not None else QUALITY_DEGRADED
    return float(put_oi / call_oi), quality


def compute_oi_change_net(agg_df, trade_date, reference_expiry, next_expiry):
    expiries = [reference_expiry]
    if next_expiry is not None:
        expiries.append(next_expiry)
    mask = (agg_df["trade_date"] == trade_date) & (agg_df["expiry_date"].isin(expiries))
    subset = agg_df.loc[mask]
    if len(subset) == 0:
        return None, QUALITY_UNAVAILABLE
    net_change = float(subset["total_oi_change"].sum())
    quality = QUALITY_VALID if next_expiry is not None else QUALITY_DEGRADED
    return net_change, quality


def compute_volume_zscore_series(daily_volumes, lookback=VOLUME_ZSCORE_LOOKBACK):
    rolling_mean = daily_volumes.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = daily_volumes.rolling(window=lookback, min_periods=lookback).std()
    z_scores = (daily_volumes - rolling_mean) / rolling_std.replace(0, np.nan)
    return z_scores


def compute_microstructure_features(underlying, labeled_obs, options_df, calendar):
    logger.info("Computing Family 4 (Microstructure) features for %s", underlying)

    options_df = options_df.copy()
    if hasattr(options_df["trade_date"].iloc[0], "date"):
        options_df["trade_date"] = options_df["trade_date"].apply(
            lambda x: x.date() if hasattr(x, "date") else x
        )
    if hasattr(options_df["expiry_date"].iloc[0], "date"):
        options_df["expiry_date"] = options_df["expiry_date"].apply(
            lambda x: x.date() if hasattr(x, "date") else x
        )

    obs = labeled_obs[["trade_date", "reference_expiry"]].copy()
    for col in ["trade_date", "reference_expiry"]:
        if hasattr(obs[col].iloc[0], "date"):
            obs[col] = obs[col].apply(lambda x: x.date() if hasattr(x, "date") else x)

    pairs = obs.drop_duplicates().sort_values(["trade_date", "reference_expiry"])
    logger.info("  %d unique (trade_date, reference_expiry) pairs", len(pairs))

    all_expiry_dates = sorted(options_df["expiry_date"].unique())
    logger.info("  %d unique expiry dates in canonical options", len(all_expiry_dates))

    logger.info("  Pre-aggregating %d option records...", len(options_df))
    agg_df = _pre_aggregate_options(options_df)
    logger.info("  Aggregated to %d rows", len(agg_df))

    daily_volumes = _compute_daily_total_volume(agg_df)
    vol_zscores = compute_volume_zscore_series(daily_volumes)
    n_null_zscore = vol_zscores.isna().sum()
    logger.info("  F-404 volume z-scores: %d valid, %d NULL (insufficient lookback)",
                len(vol_zscores) - n_null_zscore, n_null_zscore)

    rows = []
    n_pcr_warn = 0

    for _, row in pairs.iterrows():
        td = row["trade_date"]
        ref_exp = row["reference_expiry"]
        next_exp = _find_next_expiry(ref_exp, all_expiry_dates)

        f401_val, f401_q = compute_pcr_volume(agg_df, td, ref_exp)
        f402_val, f402_q = compute_pcr_oi(agg_df, td, ref_exp, next_exp)
        f403_val, f403_q = compute_oi_change_net(agg_df, td, ref_exp, next_exp)

        f404_val = vol_zscores.get(td, np.nan)
        if pd.isna(f404_val):
            f404_val = None
            f404_q = QUALITY_UNAVAILABLE
        else:
            f404_val = float(f404_val)
            f404_q = QUALITY_VALID

        f405_val = 0
        f405_q = QUALITY_UNAVAILABLE

        for pcr_val in [f401_val, f402_val]:
            if pcr_val is not None and (pcr_val > PCR_HIGH_WARN or pcr_val < PCR_LOW_WARN):
                n_pcr_warn += 1

        rows.append({
            "underlying": underlying,
            "trade_date": td,
            "reference_expiry": ref_exp,
            "pcr_volume": f401_val,
            "pcr_oi": f402_val,
            "oi_change_net_1d": f403_val,
            "volume_zscore_session": f404_val,
            "fno_ban_heavyweight_flag": f405_val,
            "quality_OV_F_401": f401_q,
            "quality_OV_F_402": f402_q,
            "quality_OV_F_403": f403_q,
            "quality_OV_F_404": f404_q,
            "quality_OV_F_405": f405_q,
        })

    df = pd.DataFrame(rows)
    df["fno_ban_heavyweight_flag"] = df["fno_ban_heavyweight_flag"].astype(int)

    n = len(df)
    n_f401_null = df["pcr_volume"].isna().sum()
    n_f402_null = df["pcr_oi"].isna().sum()
    n_f403_null = df["oi_change_net_1d"].isna().sum()
    n_f404_null = df["volume_zscore_session"].isna().sum()

    logger.info("  Results:")
    logger.info("    Total rows: %d", n)
    logger.info("    F-401 pcr_volume: %d valid, %d NULL", n - n_f401_null, n_f401_null)
    if n - n_f401_null > 0:
        valid_pcr = df["pcr_volume"].dropna()
        logger.info("      range: [%.3f, %.3f], median: %.3f",
                     valid_pcr.min(), valid_pcr.max(), valid_pcr.median())
    logger.info("    F-402 pcr_oi: %d valid, %d NULL", n - n_f402_null, n_f402_null)
    if n - n_f402_null > 0:
        valid_oi = df["pcr_oi"].dropna()
        logger.info("      range: [%.3f, %.3f], median: %.3f",
                     valid_oi.min(), valid_oi.max(), valid_oi.median())
    logger.info("    F-403 oi_change_net_1d: %d valid, %d NULL", n - n_f403_null, n_f403_null)
    logger.info("    F-404 volume_zscore: %d valid, %d NULL", n - n_f404_null, n_f404_null)
    logger.info("    F-405 fno_ban: all 0 (UNAVAILABLE)")
    if n_pcr_warn > 0:
        logger.info("    PCR outliers (>%.1f or <%.1f): %d instances",
                     PCR_HIGH_WARN, PCR_LOW_WARN, n_pcr_warn)

    logger.info("  Family 4 complete: %d rows", n)
    return df
