"""
Oracle-V Feature Engineering — Family 1: IV Surface Features
=============================================================
Phase: C3c
Authority: Data Contract §4.4, Part III §5.2.1-§5.2.2
6 immediate + 3 deferred (VIX) features.
Created: 2026-02-20
"""
from __future__ import annotations
import logging
from bisect import bisect_right
from datetime import date
from typing import Any
import numpy as np
import pandas as pd
from oracle.compute.features import QUALITY_DEGRADED, QUALITY_UNAVAILABLE, QUALITY_VALID

logger = logging.getLogger(__name__)
PERCENTILE_LOOKBACK = 252
OTM_MONEYNESS = {"NIFTY": 0.05, "BANKNIFTY": 0.06}


def _worst_quality(*qualities):
    if QUALITY_UNAVAILABLE in qualities:
        return QUALITY_UNAVAILABLE
    if QUALITY_DEGRADED in qualities:
        return QUALITY_DEGRADED
    return QUALITY_VALID


def _build_daily_riv_series(riv_df):
    valid = riv_df[riv_df["riv_quality"] == "VALID"].copy()
    valid = valid.sort_values(["trade_date", "reference_expiry"])
    daily = valid.groupby("trade_date").first().reset_index()
    return daily.set_index("trade_date")["riv"].sort_index()


def _compute_percentile_rank(current_riv, daily_riv_series, trade_date, lookback=PERCENTILE_LOOKBACK):
    window = daily_riv_series[daily_riv_series.index < trade_date]
    if len(window) < lookback:
        return None, QUALITY_UNAVAILABLE
    window = window.tail(lookback)
    rank = float((window <= current_riv).sum() / len(window))
    return rank, QUALITY_VALID


def _find_atm_iv(iv_surface, expiry_date, spot):
    exp_df = iv_surface[
        (iv_surface["expiry_date"] == expiry_date) & (iv_surface["iv_status"] == "CONVERGED")
    ]
    if len(exp_df) == 0:
        return None, QUALITY_UNAVAILABLE
    strikes = exp_df["strike"].unique()
    nearest_strike = min(strikes, key=lambda s: abs(s - spot))
    ce = exp_df[(exp_df["strike"] == nearest_strike) & (exp_df["option_type"] == "CE")]
    pe = exp_df[(exp_df["strike"] == nearest_strike) & (exp_df["option_type"] == "PE")]
    if len(ce) == 0 or len(pe) == 0:
        return None, QUALITY_UNAVAILABLE
    atm_iv = (float(ce.iloc[0]["iv"]) + float(pe.iloc[0]["iv"])) / 2.0
    return atm_iv, QUALITY_VALID


def _find_otm_iv(iv_surface, expiry_date, spot, moneyness_pct):
    exp_df = iv_surface[
        (iv_surface["expiry_date"] == expiry_date) & (iv_surface["iv_status"] == "CONVERGED")
    ]
    if len(exp_df) == 0:
        return None, None, QUALITY_UNAVAILABLE
    target_put = spot * (1.0 - moneyness_pct)
    target_call = spot * (1.0 + moneyness_pct)
    puts = exp_df[(exp_df["option_type"] == "PE") & (exp_df["strike"] <= target_put)]
    if len(puts) == 0:
        return None, None, QUALITY_UNAVAILABLE
    nearest_put = puts.loc[(puts["strike"] - target_put).abs().idxmin()]
    calls = exp_df[(exp_df["option_type"] == "CE") & (exp_df["strike"] >= target_call)]
    if len(calls) == 0:
        return None, None, QUALITY_UNAVAILABLE
    nearest_call = calls.loc[(calls["strike"] - target_call).abs().idxmin()]
    return float(nearest_put["iv"]), float(nearest_call["iv"]), QUALITY_VALID


def _compute_riv_change(expiry_groups, ref_exp, trade_date, n_back):
    grp = expiry_groups.get(ref_exp)
    if grp is None:
        return None, QUALITY_UNAVAILABLE
    dates = grp["dates"]
    rivs = grp["rivs"]
    quals = grp["qualities"]
    try:
        idx = dates.index(trade_date)
    except ValueError:
        return None, QUALITY_UNAVAILABLE
    if idx < n_back:
        return None, QUALITY_UNAVAILABLE
    change = rivs[idx] - rivs[idx - n_back]
    quality = _worst_quality(quals[idx], quals[idx - n_back])
    return float(change), quality


def compute_iv_surface_features(underlying, labeled_obs, riv_df, iv_surfaces_df):
    logger.info("Computing Family 1 (IV Surface) features for %s", underlying)
    moneyness_pct = OTM_MONEYNESS.get(underlying, 0.05)

    riv = riv_df.copy()
    for col in ["trade_date", "reference_expiry"]:
        if hasattr(riv[col].iloc[0], "date"):
            riv[col] = riv[col].apply(lambda x: x.date() if hasattr(x, "date") else x)

    iv_surf = iv_surfaces_df.copy()
    for col in ["trade_date", "expiry_date"]:
        if col in iv_surf.columns and hasattr(iv_surf[col].iloc[0], "date"):
            iv_surf[col] = iv_surf[col].apply(lambda x: x.date() if hasattr(x, "date") else x)

    obs = labeled_obs[["trade_date", "reference_expiry"]].copy()
    for col in ["trade_date", "reference_expiry"]:
        if hasattr(obs[col].iloc[0], "date"):
            obs[col] = obs[col].apply(lambda x: x.date() if hasattr(x, "date") else x)
    pairs = obs.drop_duplicates().sort_values(["trade_date", "reference_expiry"])
    logger.info("  %d unique (trade_date, reference_expiry) pairs", len(pairs))

    daily_riv = _build_daily_riv_series(riv)
    logger.info("  Daily RIV series: %d dates (VALID, front expiry)", len(daily_riv))

    riv_lookup = {}
    for _, r in riv.iterrows():
        key = (r["trade_date"], r["reference_expiry"])
        riv_lookup[key] = {"riv": float(r["riv"]), "quality": str(r["riv_quality"]), "spot_price": float(r["spot_price"])}
    logger.info("  RIV lookup: %d entries", len(riv_lookup))

    expiry_groups = {}
    for exp, grp in riv.groupby("reference_expiry"):
        sg = grp.sort_values("trade_date")
        expiry_groups[exp] = {"dates": sg["trade_date"].tolist(), "rivs": sg["riv"].tolist(), "qualities": sg["riv_quality"].tolist()}

    all_expiry_dates = sorted(iv_surf["expiry_date"].unique())
    logger.info("  %d unique expiry dates in IV surfaces", len(all_expiry_dates))

    iv_by_date = {td: grp for td, grp in iv_surf.groupby("trade_date")}
    logger.info("  IV surfaces indexed: %d dates", len(iv_by_date))

    rows = []
    stats = {k: 0 for k in ["f101_v","f101_n","f102_v","f102_n","f103_v","f103_n","f105_v","f105_n","f106_v","f106_n","f110_v","f110_n"]}

    for _, row in pairs.iterrows():
        td = row["trade_date"]
        ref_exp = row["reference_expiry"]

        # F-101
        riv_entry = riv_lookup.get((td, ref_exp))
        if riv_entry:
            f101_val = riv_entry["riv"]
            f101_q = QUALITY_VALID if riv_entry["quality"] == "VALID" else QUALITY_DEGRADED
            spot = riv_entry["spot_price"]
            stats["f101_v"] += 1
        else:
            f101_val = None; f101_q = QUALITY_UNAVAILABLE; spot = None
            stats["f101_n"] += 1

        # F-102
        f102_val, f102_q = _compute_riv_change(expiry_groups, ref_exp, td, 1)
        stats["f102_v" if f102_val is not None else "f102_n"] += 1

        # F-103
        f103_val, f103_q = _compute_riv_change(expiry_groups, ref_exp, td, 3)
        stats["f103_v" if f103_val is not None else "f103_n"] += 1

        # F-105
        if f101_val is not None:
            f105_val, f105_q = _compute_percentile_rank(f101_val, daily_riv, td)
        else:
            f105_val, f105_q = None, QUALITY_UNAVAILABLE
        stats["f105_v" if f105_val is not None else "f105_n"] += 1

        # F-106
        f106_val, f106_q = None, QUALITY_UNAVAILABLE
        if spot is not None and td in iv_by_date:
            day_iv = iv_by_date[td]
            idx = bisect_right(all_expiry_dates, ref_exp)
            next_exp = all_expiry_dates[idx] if idx < len(all_expiry_dates) else None
            if next_exp is not None:
                atm_curr, qc = _find_atm_iv(day_iv, ref_exp, spot)
                atm_next, qn = _find_atm_iv(day_iv, next_exp, spot)
                if atm_curr is not None and atm_next is not None:
                    f106_val = atm_next - atm_curr
                    f106_q = _worst_quality(qc, qn)
        stats["f106_v" if f106_val is not None else "f106_n"] += 1

        # F-110
        f110_val, f110_q = None, QUALITY_UNAVAILABLE
        if spot is not None and td in iv_by_date:
            day_iv = iv_by_date[td]
            put_iv, call_iv, sq = _find_otm_iv(day_iv, ref_exp, spot, moneyness_pct)
            if put_iv is not None and call_iv is not None:
                f110_val = put_iv - call_iv
                f110_q = sq
        stats["f110_v" if f110_val is not None else "f110_n"] += 1

        rows.append({
            "underlying": underlying, "trade_date": td, "reference_expiry": ref_exp,
            "riv_level": f101_val, "riv_change_1d": f102_val, "riv_change_3d": f103_val,
            "iv_percentile_rank": f105_val, "iv_term_structure_slope": f106_val, "iv_skew_25d": f110_val,
            "india_vix_level": None, "india_vix_percentile_rank": None, "india_vix_change": None,
            "quality_OV_F_101": f101_q, "quality_OV_F_102": f102_q, "quality_OV_F_103": f103_q,
            "quality_OV_F_105": f105_q, "quality_OV_F_106": f106_q, "quality_OV_F_107": QUALITY_UNAVAILABLE,
            "quality_OV_F_108": QUALITY_UNAVAILABLE, "quality_OV_F_109": QUALITY_UNAVAILABLE, "quality_OV_F_110": f110_q,
        })

    df = pd.DataFrame(rows)
    n = len(df)
    logger.info("  Results (%d rows):", n)
    for fid, label in [("f101","F-101 riv_level"),("f102","F-102 riv_change_1d"),("f103","F-103 riv_change_3d"),
                        ("f105","F-105 iv_percentile_rank"),("f106","F-106 term_structure_slope"),("f110","F-110 iv_skew_25d")]:
        logger.info("    %s: %d valid, %d NULL", label, stats[f"{fid}_v"], stats[f"{fid}_n"])

    valid_pct = df["iv_percentile_rank"].dropna()
    if len(valid_pct) > 0:
        logger.info("    F-105 distribution: median=%.3f, min=%.3f, max=%.3f", valid_pct.median(), valid_pct.min(), valid_pct.max())

    valid_skew = df["iv_skew_25d"].dropna()
    if len(valid_skew) > 0:
        logger.info("    F-110 skew: median=%.4f, min=%.4f, max=%.4f", valid_skew.median(), valid_skew.min(), valid_skew.max())

    logger.info("  Family 1 complete: %d rows", n)
    return df
