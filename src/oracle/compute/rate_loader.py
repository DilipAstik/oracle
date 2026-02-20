"""
Risk-free rate and dividend yield input pipeline for IV computation.

Module: src/oracle/compute/rate_loader.py
Phase: B5 - IV Computation Service
Spec References:
    - TDR-ORACLE-008 2.4 (IV computation parameters)
    - B5 Design 2.3 (risk-free rate), 2.4 (dividend yield)
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RFR_VERSION = "v1"

_DEFAULT_RFR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rfr_history.csv"
)

_STALE_RATE_WARNING_DAYS = 90


def load_risk_free_rates(csv_path=None):
    if csv_path is None:
        csv_path = os.path.normpath(_DEFAULT_RFR_PATH)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Risk-free rate CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, comment="#", parse_dates=["effective_date"])
    df["effective_date"] = df["effective_date"].dt.date
    if "rate" not in df.columns:
        raise ValueError("rfr_history.csv must contain 'rate' column")
    if len(df) == 0:
        raise ValueError("rfr_history.csv is empty")
    if (df["rate"] < 0).any() or (df["rate"] > 0.20).any():
        raise ValueError("Rate values outside [0, 0.20] range")
    df = df.sort_values("effective_date").reset_index(drop=True)
    return df


def get_rfr_for_date(rfr_df, trade_date):
    mask = rfr_df["effective_date"] <= trade_date
    if not mask.any():
        raise ValueError(
            f"No risk-free rate available for {trade_date}. "
            f"Earliest rate is {rfr_df['effective_date'].iloc[0]}"
        )
    row = rfr_df.loc[mask].iloc[-1]
    rate = float(row["rate"])
    last_known_date = rfr_df["effective_date"].iloc[-1]
    days_past = (trade_date - last_known_date).days
    is_stale = days_past > _STALE_RATE_WARNING_DAYS
    if is_stale:
        logger.warning(
            "Risk-free rate may be stale: trade_date=%s is %d days past "
            "last known rate change on %s (rate=%.4f). "
            "Update rfr_history.csv if new RBI decisions have occurred.",
            trade_date, days_past, last_known_date, rate,
        )
    return rate, is_stale


def get_rfr_version():
    return RFR_VERSION


_MIN_FUTURES_DTE = 3


def compute_dividend_yield(futures_df, underlying, trade_date, rfr, min_dte=_MIN_FUTURES_DTE):
    if futures_df is None or len(futures_df) == 0:
        return None, None, "MISSING"
    df = futures_df[futures_df["underlying"] == underlying].copy()
    if len(df) == 0:
        return None, None, "MISSING"
    df["dte"] = df["expiry_date"].apply(
        lambda x: (x - trade_date).days if isinstance(x, date) else None
    )
    df = df.dropna(subset=["dte"])
    df["dte"] = df["dte"].astype(int)
    qualifying = df[df["dte"] >= min_dte].sort_values("dte")
    status = "OK"
    if len(qualifying) == 0:
        any_positive = df[df["dte"] > 0].sort_values("dte")
        if len(any_positive) > 0:
            qualifying = any_positive
            status = "NEXT_MONTH"
        else:
            return None, None, "MISSING"
    row = qualifying.iloc[0]
    F = float(row["settlement_price"])
    S = float(row["underlying_price"])
    T_futures = float(row["dte"]) / 365.0
    if S <= 0 or F <= 0 or T_futures <= 0:
        return None, None, "MISSING"
    try:
        q_implied = rfr - np.float64(np.log(F / S)) / np.float64(T_futures)
        q_implied = float(q_implied)
    except (FloatingPointError, ZeroDivisionError, ValueError):
        return None, None, "MISSING"
    if q_implied < -0.05 or q_implied > 0.10:
        logger.warning(
            "Extreme dividend yield: q=%.4f for %s on %s (F=%.2f, S=%.2f, T=%.4f). "
            "Using clamped value.",
            q_implied, underlying, trade_date, F, S, T_futures,
        )
        q_implied = max(-0.05, min(q_implied, 0.10))
    futures_expiry = row["expiry_date"]
    if isinstance(futures_expiry, pd.Timestamp):
        futures_expiry = futures_expiry.date()
    return q_implied, futures_expiry, status
