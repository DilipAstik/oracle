"""
Oracle-V Feature Engineering — Family 2: Realised-Implied Spread
================================================================
Phase: C3d
Authority: Data Contract §4.4, Part III §5.2.2
Features: 2 Core (1 CRITICAL, 1 STANDARD)

    OV-F-201  iv_rv_spread_5d    CRITICAL  RIV minus 5-day realised vol
    OV-F-202  iv_rv_spread_10d   STANDARD  RIV minus 10-day realised vol

Realised vol = std(ln(Close_i / Close_{i-1})) × sqrt(252)
Uses consecutive trading-day pairs only (C2 Interpretation C rule).

Created: 2026-02-20
"""
from __future__ import annotations
import logging
from datetime import date
import numpy as np
import pandas as pd
from oracle.compute.features import QUALITY_DEGRADED, QUALITY_UNAVAILABLE, QUALITY_VALID

logger = logging.getLogger(__name__)


def _compute_realised_vol(
    spot_series: pd.Series,
    trade_date: date,
    window: int,
) -> tuple[float | None, str]:
    """Compute annualised realised volatility from log returns.

    Uses `window` consecutive trading-day close prices ending at trade_date.
    Needs window+1 prices to get `window` returns.

    Returns (rv, quality).
    """
    # Get prices up to and including trade_date
    prices = spot_series[spot_series.index <= trade_date]
    if len(prices) < window + 1:
        return None, QUALITY_UNAVAILABLE

    # Take last window+1 prices
    recent = prices.tail(window + 1)
    log_returns = np.log(recent / recent.shift(1)).dropna()

    if len(log_returns) < window:
        return None, QUALITY_UNAVAILABLE

    rv = float(log_returns.std() * np.sqrt(252))
    return rv, QUALITY_VALID


def compute_realised_implied_features(
    underlying: str,
    labeled_obs: pd.DataFrame,
    riv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Family 2 features for one underlying.

    Args:
        underlying: "NIFTY" or "BANKNIFTY"
        labeled_obs: C2 labeled observations.
        riv_df: C1 RIV daily series (has spot_price, trade_date, reference_expiry, riv).
    """
    logger.info("Computing Family 2 (Realised-Implied) features for %s", underlying)

    riv = riv_df.copy()
    for col in ["trade_date", "reference_expiry"]:
        if hasattr(riv[col].iloc[0], "date"):
            riv[col] = riv[col].apply(lambda x: x.date() if hasattr(x, "date") else x)

    # Build daily spot series (one per trade_date, use front expiry)
    riv_sorted = riv.sort_values(["trade_date", "reference_expiry"])
    daily_spot = riv_sorted.groupby("trade_date").first().reset_index()
    spot_series = daily_spot.set_index("trade_date")["spot_price"].sort_index()
    logger.info("  Daily spot series: %d dates", len(spot_series))

    # Build RIV lookup
    riv_lookup = {}
    for _, r in riv.iterrows():
        key = (r["trade_date"], r["reference_expiry"])
        riv_lookup[key] = {"riv": float(r["riv"]), "quality": str(r["riv_quality"])}

    # Extract unique pairs
    obs = labeled_obs[["trade_date", "reference_expiry"]].copy()
    for col in ["trade_date", "reference_expiry"]:
        if hasattr(obs[col].iloc[0], "date"):
            obs[col] = obs[col].apply(lambda x: x.date() if hasattr(x, "date") else x)
    pairs = obs.drop_duplicates().sort_values(["trade_date", "reference_expiry"])
    logger.info("  %d unique (trade_date, reference_expiry) pairs", len(pairs))

    # Pre-compute RV for all dates (avoids repeated windowing)
    all_dates = sorted(pairs["trade_date"].unique())
    rv5_cache = {}
    rv10_cache = {}
    for td in all_dates:
        rv5_cache[td] = _compute_realised_vol(spot_series, td, 5)
        rv10_cache[td] = _compute_realised_vol(spot_series, td, 10)

    # Compute per pair
    rows = []
    s = {"f201_v": 0, "f201_n": 0, "f202_v": 0, "f202_n": 0}

    for _, row in pairs.iterrows():
        td = row["trade_date"]
        ref_exp = row["reference_expiry"]

        riv_entry = riv_lookup.get((td, ref_exp))
        if riv_entry is None:
            f201_val, f201_q = None, QUALITY_UNAVAILABLE
            f202_val, f202_q = None, QUALITY_UNAVAILABLE
        else:
            current_riv = riv_entry["riv"]
            riv_q = QUALITY_VALID if riv_entry["quality"] == "VALID" else QUALITY_DEGRADED

            # F-201: RIV - RV5
            rv5, rv5_q = rv5_cache[td]
            if rv5 is not None:
                f201_val = current_riv - rv5
                f201_q = riv_q if rv5_q == QUALITY_VALID else QUALITY_DEGRADED
            else:
                f201_val, f201_q = None, QUALITY_UNAVAILABLE

            # F-202: RIV - RV10
            rv10, rv10_q = rv10_cache[td]
            if rv10 is not None:
                f202_val = current_riv - rv10
                f202_q = riv_q if rv10_q == QUALITY_VALID else QUALITY_DEGRADED
            else:
                f202_val, f202_q = None, QUALITY_UNAVAILABLE

        s["f201_v" if f201_val is not None else "f201_n"] += 1
        s["f202_v" if f202_val is not None else "f202_n"] += 1

        rows.append({
            "underlying": underlying,
            "trade_date": td,
            "reference_expiry": ref_exp,
            "iv_rv_spread_5d": f201_val,
            "iv_rv_spread_10d": f202_val,
            "quality_OV_F_201": f201_q,
            "quality_OV_F_202": f202_q,
        })

    df = pd.DataFrame(rows)
    n = len(df)

    logger.info("  Results (%d rows):", n)
    logger.info("    F-201 iv_rv_spread_5d: %d valid, %d NULL", s["f201_v"], s["f201_n"])
    logger.info("    F-202 iv_rv_spread_10d: %d valid, %d NULL", s["f202_v"], s["f202_n"])

    for col, label in [("iv_rv_spread_5d", "F-201"), ("iv_rv_spread_10d", "F-202")]:
        valid = df[col].dropna()
        if len(valid) > 0:
            logger.info("    %s: median=%.4f, min=%.4f, max=%.4f",
                         label, valid.median(), valid.min(), valid.max())
            pct_pos = (valid > 0).mean() * 100
            logger.info("    %s positive (IV > RV): %.1f%%", label, pct_pos)

    logger.info("  Family 2 complete: %d rows", n)
    return df
