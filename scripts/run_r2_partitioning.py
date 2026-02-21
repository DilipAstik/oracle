#!/usr/bin/env python3
"""
Oracle-V Model Research Phase — R2: Data Partitioning Design
=============================================================

Specification References:
  - Governance §10.2: "Partitioning scheme must be documented before training
    begins and may not be altered after training starts."
  - Part IV §7.2.2: Walk-forward mandatory, random splits prohibited.
    Purge gap = max(T) = 5 trading days.
  - Part IV §7.2.2: min_walkforward_folds = 5 (Tier 2, bounds [3, 20])

Design Principles (per Expert Review):
  - MECHANICAL expanding walk-forward — no manual regime-aware placement
  - Percentage-based boundaries on trading day index
  - Regime analysis is a POST-EVALUATION diagnostic, not a design input
  - Feature preprocessing must be refit inside each fold

Usage:
  cd /home/ssm-user/oracle
  source .venv/bin/activate
  export PYTHONPATH=/home/ssm-user/oracle/src
  python scripts/run_r2_partitioning.py

Output:
  research/partitioning_scheme.md   — Human-readable documentation
  research/partitioning.json        — Machine-readable fold boundaries
"""

import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
TRAINING_PREFIX = "computed/training"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

# Partitioning parameters
TEST_HOLDOUT_PCT = 0.15          # Final 15% reserved for single-shot test
PURGE_GAP_DAYS = 5               # max(T) = 5 trading days
N_FOLDS = 6                      # Above minimum 5, within Tier 2 [3, 20]

# Expanding walk-forward: train ends at these % of non-test data
# Validation windows are 15% each, starting after purge
FOLD_TRAIN_END_PCTS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.82]
FOLD_VAL_SIZE_PCT = 0.15         # Each validation window is ~15% of non-test

# Dropped features (from R1 findings)
DROPPED_FEATURES = ["time_of_day_bucket", "iv_percentile_rank"]
MODELLING_FEATURES = [
    "day_of_week", "days_to_current_expiry", "is_monthly_expiry_week",
    "days_to_next_holiday",
    "riv_level", "riv_change_1d", "riv_change_3d",
    "iv_term_structure_slope", "iv_skew_25d",
    "iv_rv_spread_5d", "iv_rv_spread_10d",
    "pcr_volume", "pcr_oi", "oi_change_net_1d", "volume_zscore_session",
]

# Output
BASE_DIR = Path("research")
SCHEME_MD = BASE_DIR / "partitioning_scheme.md"
SCHEME_JSON = BASE_DIR / "partitioning.json"


def load_trade_dates():
    """Load unique sorted trade dates from training data."""
    s3_path = f"s3://{S3_BUCKET}/{TRAINING_PREFIX}/NIFTY_training_features.parquet"
    print(f"Loading trade dates from {s3_path} ...")
    df = pd.read_parquet(s3_path, columns=["trade_date", "horizon_name", "training_eligible"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    all_dates = sorted(df["trade_date"].unique())
    eligible_dates = sorted(
        df[df["training_eligible"] == True]["trade_date"].unique()
    )

    print(f"  Total unique trade dates: {len(all_dates)}")
    print(f"  Dates with eligible observations: {len(eligible_dates)}")
    print(f"  Date range: {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}")

    return all_dates, eligible_dates, df


def compute_partitions(all_dates):
    """Compute mechanical expanding walk-forward partitions."""
    n_total = len(all_dates)

    # Test hold-out: final 15%
    test_start_idx = int(n_total * (1 - TEST_HOLDOUT_PCT))
    test_dates = all_dates[test_start_idx:]
    non_test_dates = all_dates[:test_start_idx]
    n_non_test = len(non_test_dates)

    print(f"\n--- Partition Design ---")
    print(f"Total trading days: {n_total}")
    print(f"Non-test pool: {n_non_test} days "
          f"({non_test_dates[0].strftime('%Y-%m-%d')} to {non_test_dates[-1].strftime('%Y-%m-%d')})")
    print(f"Test hold-out: {len(test_dates)} days "
          f"({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})")
    print(f"Purge gap: {PURGE_GAP_DAYS} trading days")
    print(f"Folds: {N_FOLDS}")

    # Build folds
    folds = []
    for i, train_end_pct in enumerate(FOLD_TRAIN_END_PCTS):
        fold_num = i + 1

        # Training: days 0 to train_end_idx (inclusive)
        train_end_idx = int(n_non_test * train_end_pct) - 1
        train_start_idx = 0

        # Purge gap: skip PURGE_GAP_DAYS after train end
        val_start_idx = train_end_idx + 1 + PURGE_GAP_DAYS

        # Validation: FOLD_VAL_SIZE_PCT of non-test, or until end of non-test
        val_size = max(int(n_non_test * FOLD_VAL_SIZE_PCT), 20)
        val_end_idx = min(val_start_idx + val_size - 1, n_non_test - 1)

        # Bounds check
        if val_start_idx >= n_non_test:
            print(f"  ⚠️ Fold {fold_num}: validation starts beyond non-test pool — skipping")
            continue
        if val_end_idx <= val_start_idx:
            print(f"  ⚠️ Fold {fold_num}: validation window too small — skipping")
            continue

        train_dates_range = non_test_dates[train_start_idx:train_end_idx + 1]
        purge_dates_range = non_test_dates[train_end_idx + 1:val_start_idx]
        val_dates_range = non_test_dates[val_start_idx:val_end_idx + 1]

        fold = {
            "fold": fold_num,
            "train_start": str(train_dates_range[0].strftime('%Y-%m-%d')),
            "train_end": str(train_dates_range[-1].strftime('%Y-%m-%d')),
            "train_days": len(train_dates_range),
            "purge_start": str(purge_dates_range[0].strftime('%Y-%m-%d')) if len(purge_dates_range) > 0 else "N/A",
            "purge_end": str(purge_dates_range[-1].strftime('%Y-%m-%d')) if len(purge_dates_range) > 0 else "N/A",
            "purge_days": len(purge_dates_range),
            "val_start": str(val_dates_range[0].strftime('%Y-%m-%d')),
            "val_end": str(val_dates_range[-1].strftime('%Y-%m-%d')),
            "val_days": len(val_dates_range),
        }
        folds.append(fold)

        print(f"  Fold {fold_num}: train {fold['train_start']}→{fold['train_end']} ({fold['train_days']}d), "
              f"purge {fold['purge_days']}d, "
              f"val {fold['val_start']}→{fold['val_end']} ({fold['val_days']}d)")

    test_partition = {
        "test_start": str(test_dates[0].strftime('%Y-%m-%d')),
        "test_end": str(test_dates[-1].strftime('%Y-%m-%d')),
        "test_days": len(test_dates),
    }

    print(f"\n  Test: {test_partition['test_start']}→{test_partition['test_end']} "
          f"({test_partition['test_days']}d)")

    return folds, test_partition


def validate_partitions(folds, test_partition, all_dates):
    """Validate partition integrity."""
    print(f"\n--- Partition Validation ---")
    issues = []

    # Check 1: No overlap between train and val in any fold
    for fold in folds:
        if fold["train_end"] >= fold["val_start"]:
            issues.append(f"Fold {fold['fold']}: train/val overlap!")

    # Check 2: Purge gap is at least PURGE_GAP_DAYS
    for fold in folds:
        if fold["purge_days"] < PURGE_GAP_DAYS:
            issues.append(f"Fold {fold['fold']}: purge gap {fold['purge_days']}d < {PURGE_GAP_DAYS}d required")

    # Check 3: No validation fold overlaps with test set
    for fold in folds:
        if fold["val_end"] >= test_partition["test_start"]:
            issues.append(f"Fold {fold['fold']}: validation extends into test set!")

    # Check 4: Expanding window — each fold's train includes all prior folds' train
    for i in range(1, len(folds)):
        if folds[i]["train_start"] != folds[0]["train_start"]:
            issues.append(f"Fold {folds[i]['fold']}: not expanding — train_start differs from fold 1")

    # Check 5: Chronological ordering — val windows advance
    for i in range(1, len(folds)):
        if folds[i]["val_start"] <= folds[i-1]["val_start"]:
            issues.append(f"Fold {folds[i]['fold']}: val_start not advancing vs fold {folds[i-1]['fold']}")

    # Check 6: Minimum fold count
    if len(folds) < 5:
        issues.append(f"Only {len(folds)} folds — minimum is 5 per §7.2.2")

    if issues:
        print("  ❌ VALIDATION FAILED:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ All validations passed:")
        print(f"    - No train/val overlap in any fold")
        print(f"    - Purge gap ≥ {PURGE_GAP_DAYS}d in all folds")
        print(f"    - No validation extends into test set")
        print(f"    - Expanding window property holds")
        print(f"    - Validation windows advance chronologically")
        print(f"    - Fold count ({len(folds)}) ≥ 5 minimum")

    return len(issues) == 0


def count_eligible_per_partition(folds, test_partition, df):
    """Count training-eligible observations in each partition."""
    print(f"\n--- Eligible Observation Counts ---")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    eligible = df[df["training_eligible"] == True].copy()

    counts = []
    for fold in folds:
        train_mask = (eligible["trade_date"] >= fold["train_start"]) & \
                     (eligible["trade_date"] <= fold["train_end"])
        val_mask = (eligible["trade_date"] >= fold["val_start"]) & \
                   (eligible["trade_date"] <= fold["val_end"])

        n_train = train_mask.sum()
        n_val = val_mask.sum()

        # Per-horizon breakdown
        for horizon in sorted(eligible["horizon_name"].unique()):
            h_train = (train_mask & (eligible["horizon_name"] == horizon)).sum()
            h_val = (val_mask & (eligible["horizon_name"] == horizon)).sum()

        fold["n_train_eligible"] = int(n_train)
        fold["n_val_eligible"] = int(n_val)

        print(f"  Fold {fold['fold']}: train={n_train} eligible, val={n_val} eligible")
        counts.append((n_train, n_val))

    # Test set
    test_mask = (eligible["trade_date"] >= test_partition["test_start"]) & \
                (eligible["trade_date"] <= test_partition["test_end"])
    n_test = test_mask.sum()
    test_partition["n_test_eligible"] = int(n_test)
    print(f"  Test: {n_test} eligible observations")

    return counts


def save_json(folds, test_partition):
    """Save machine-readable partition scheme."""
    scheme = {
        "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "spec_references": [
            "Governance §10.2 (partitioning documented before training)",
            "Part IV §7.2.2 (walk-forward, purge gap = max(T) = 5)",
            "Part IV §7.2.2 (min_walkforward_folds = 5)",
        ],
        "design_principles": [
            "Mechanical expanding walk-forward — no manual regime-aware placement",
            "Percentage-based boundaries on trading day index",
            "Feature preprocessing refit inside each fold",
            "Regime analysis is post-evaluation diagnostic only",
        ],
        "parameters": {
            "n_folds": N_FOLDS,
            "purge_gap_days": PURGE_GAP_DAYS,
            "test_holdout_pct": TEST_HOLDOUT_PCT,
            "fold_train_end_pcts": FOLD_TRAIN_END_PCTS,
            "fold_val_size_pct": FOLD_VAL_SIZE_PCT,
        },
        "dropped_features": DROPPED_FEATURES,
        "dropped_features_rationale": {
            "time_of_day_bucket": "Constant value (19), zero variance, infinite VIF — structurally constant in v1",
            "iv_percentile_rank": "49% UNAVAILABLE + r=0.92 with riv_level (§5.4.1 threshold exceeded) — Specification-Constrained Omission",
        },
        "modelling_features": MODELLING_FEATURES,
        "n_modelling_features": len(MODELLING_FEATURES),
        "folds": folds,
        "test": test_partition,
    }

    SCHEME_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEME_JSON, 'w') as f:
        json.dump(scheme, f, indent=2)
    print(f"\nSaved: {SCHEME_JSON}")


def save_markdown(folds, test_partition):
    """Save human-readable partition documentation."""
    lines = [
        "# Oracle-V Model Research — R2: Partitioning Scheme",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Status:** FROZEN — must not be altered after training begins (Governance §10.2)",
        "",
        "---",
        "",
        "## Specification Compliance",
        "",
        "| Requirement | Spec Reference | This Scheme |",
        "|---|---|---|",
        f"| Walk-forward mandatory | §7.2.2 | ✅ Expanding walk-forward, {N_FOLDS} folds |",
        f"| Random splits prohibited | §7.2.2 | ✅ Chronological only |",
        f"| Purge gap = max(T) | §7.2.2 | ✅ {PURGE_GAP_DAYS} trading days |",
        f"| min_walkforward_folds ≥ 5 | §7.2.2 | ✅ {N_FOLDS} folds |",
        "| Test set held out completely | Governance §10.2 | ✅ Final 15%, single-shot |",
        "| Documented before training | Governance §10.2 | ✅ This document |",
        "",
        "## Design Principles",
        "",
        "1. **Mechanical expanding walk-forward** — no manual regime-aware fold placement",
        "2. **Percentage-based boundaries** on trading day index — fully reproducible",
        "3. **Feature preprocessing refit** inside each fold — no global statistics leakage",
        "4. **Regime analysis** is a post-evaluation diagnostic, not a partition design input",
        "",
        "## Feature Set (R1 Decisions)",
        "",
        f"**Modelling features: {len(MODELLING_FEATURES)}**",
        "",
    ]

    for feat in MODELLING_FEATURES:
        lines.append(f"- `{feat}`")

    lines.extend([
        "",
        "**Dropped features (Specification-Constrained Omission):**",
        "",
        "- `time_of_day_bucket` — Constant value (19), zero variance, infinite VIF",
        "- `iv_percentile_rank` — 49% UNAVAILABLE + r=0.92 with `riv_level` (§5.4.1)",
        "",
        "## Walk-Forward Fold Structure",
        "",
        "| Fold | Train Period | Train Days | Purge | Val Period | Val Days | Train Eligible | Val Eligible |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for fold in folds:
        lines.append(
            f"| {fold['fold']} | {fold['train_start']} → {fold['train_end']} | "
            f"{fold['train_days']} | {fold['purge_days']}d | "
            f"{fold['val_start']} → {fold['val_end']} | {fold['val_days']} | "
            f"{fold.get('n_train_eligible', 'N/A')} | {fold.get('n_val_eligible', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Test Hold-Out",
        "",
        f"| Period | {test_partition['test_start']} → {test_partition['test_end']} |",
        "|---|---|",
        f"| Trading days | {test_partition['test_days']} |",
        f"| Eligible observations | {test_partition.get('n_test_eligible', 'N/A')} |",
        f"| Usage | Single-shot evaluation (R6) — NEVER touched before then |",
        "",
        "## Purge Gap Rationale",
        "",
        "The purge gap of 5 trading days equals max(T) = 5 (the longest prediction horizon).",
        "This prevents any label computed from data overlapping the validation period from",
        "entering training. A label at time t requires data at t+T to resolve — training data",
        "within T days of the validation boundary could contain labels computed from data that",
        "overlaps with the validation period.",
        "",
        "## Pre-Training Checklist",
        "",
        "Before any model training begins, verify:",
        "",
        "- [ ] This document has been committed to Git",
        "- [ ] `partitioning.json` matches this document",
        "- [ ] No training code accesses test set dates",
        "- [ ] Feature preprocessing pipeline refits per fold",
        "- [ ] Class weighting: START UNWEIGHTED (add only if recall collapses)",
        "",
        "---",
        f"*Document frozen at generation time. Any modification requires formal amendment.*",
    ])

    SCHEME_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEME_MD, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {SCHEME_MD}")


def main():
    print("=" * 70)
    print("Oracle-V Model Research — R2: Partitioning Design")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_dates, eligible_dates, df = load_trade_dates()
    folds, test_partition = compute_partitions(all_dates)

    valid = validate_partitions(folds, test_partition, all_dates)
    if not valid:
        print("\n❌ ABORT: Partition validation failed. Fix issues before proceeding.")
        return

    count_eligible_per_partition(folds, test_partition, df)

    save_json(folds, test_partition)
    save_markdown(folds, test_partition)

    print(f"\n{'=' * 70}")
    print("R2 COMPLETE — Partitioning scheme frozen.")
    print(f"{'=' * 70}")
    print(f"\nNext steps:")
    print(f"  1. Review: cat research/partitioning_scheme.md")
    print(f"  2. Commit: git add research/ scripts/run_r2_partitioning.py")
    print(f"     git commit -m 'R2: Partitioning scheme frozen — 6-fold walk-forward'")
    print(f"  3. Proceed to R3 (Baseline Construction)")


if __name__ == "__main__":
    main()
