#!/usr/bin/env python3
"""
Oracle-V Model Research Phase — R3: Baseline Construction
==========================================================

Specification References:
  - Governance §10.2, exit criterion 7: Model must outperform all baselines
  - Part IV §7.2.5: Three mandatory baselines + RIV momentum heuristic
  - Part IV §7.2.5: Outperformance margin ≥ 0.02 Brier (Tier 2)

Baselines:
  1. Naive base rate     — constant = training-set class frequency
  2. Persistence         — predict same outcome as most recently resolved label
  3. GARCH(1,1)          — fit on log-returns of RIV, empirical CDF → probability
  4. RIV momentum        — riv_change_3d binned → empirical probability (falsification test)

Metrics per baseline per fold:
  - Brier score, ECE (10-bin), AUC-ROC, Log-loss

Usage:
  cd /home/ssm-user/oracle
  source .venv/bin/activate
  export PYTHONPATH=/home/ssm-user/oracle/src
  python scripts/run_r3_baselines.py

Output:
  research/baselines/baseline_results.json
  research/baselines/baseline_results.md
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
TRAINING_PREFIX = "computed/training"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]
TARGETS = ["label_expansion", "label_compression"]
PARTITION_FILE = Path("research/partitioning.json")
OUTPUT_DIR = Path("research/baselines")

# GARCH settings
GARCH_N_BINS = 5  # Quantile bins for variance → probability mapping


def load_data():
    """Load training data and partitioning scheme."""
    # Partitioning
    with open(PARTITION_FILE) as f:
        partition = json.load(f)
    print(f"Loaded partitioning: {len(partition['folds'])} folds + test hold-out")

    # Training data
    data = {}
    for ul in UNDERLYINGS:
        s3_path = f"s3://{S3_BUCKET}/{TRAINING_PREFIX}/{ul}_training_features.parquet"
        df = pd.read_parquet(s3_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        data[ul] = df
        print(f"  {ul}: {len(df)} rows")

    return data, partition


# ===========================================================================
# Metrics
# ===========================================================================
def brier_score(y_true, y_prob):
    """Brier score: mean squared error of probability forecasts."""
    return np.mean((y_prob - y_true) ** 2)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: weighted average of per-bin calibration error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true) if len(y_true) > 0 else 0.0


def compute_metrics(y_true, y_prob, label=""):
    """Compute all evaluation metrics."""
    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)

    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    metrics = {
        "brier": float(brier_score(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }

    # AUC-ROC — requires both classes present
    try:
        if len(np.unique(y_true)) > 1:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["auroc"] = float('nan')
    except Exception:
        metrics["auroc"] = float('nan')

    # Log-loss
    try:
        metrics["logloss"] = float(log_loss(y_true, y_prob_clipped))
    except Exception:
        metrics["logloss"] = float('nan')

    return metrics


# ===========================================================================
# Baseline 1: Naive Base Rate
# ===========================================================================
def baseline_naive(train_labels, val_labels):
    """Predict constant = training base rate."""
    base_rate = train_labels.mean()
    preds = np.full(len(val_labels), base_rate)
    return preds, {"base_rate": float(base_rate)}


# ===========================================================================
# Baseline 2: Persistence
# ===========================================================================
def baseline_persistence(train_df, val_df, target, horizon):
    """Predict same outcome as most recently resolved label."""
    # Sort by date
    combined = pd.concat([train_df, val_df]).sort_values("trade_date")
    combined = combined[combined["horizon_name"] == horizon].copy()

    # Shift label forward: prediction at t = label at t-1
    combined["pred_persistence"] = combined[target].shift(1)

    # Extract validation predictions
    val_dates = set(val_df["trade_date"].unique())
    val_preds = combined[combined["trade_date"].isin(val_dates)]["pred_persistence"]

    # Handle NaN (first observation) — use training base rate
    base_rate = train_df[train_df["horizon_name"] == horizon][target].mean()
    val_preds = val_preds.fillna(base_rate)

    return val_preds.values, {"fallback_rate": float(base_rate)}


# ===========================================================================
# Baseline 3: GARCH(1,1)
# ===========================================================================
def baseline_garch(train_df, val_df, target, horizon, ul_name):
    """GARCH(1,1) on log-returns of RIV → empirical probability mapping."""
    from arch import arch_model

    # Get daily RIV series (deduplicate across horizons)
    all_df = pd.concat([train_df, val_df]).sort_values("trade_date")
    daily_riv = all_df.groupby("trade_date")["riv_level"].first().sort_index()

    # Log-returns of RIV
    log_returns = np.log(daily_riv / daily_riv.shift(1)).dropna()

    # Split into train/val periods
    train_end = pd.Timestamp(train_df["trade_date"].max())
    train_returns = log_returns[log_returns.index <= train_end]

    if len(train_returns) < 50:
        # Insufficient data — fall back to naive
        base_rate = train_df[train_df["horizon_name"] == horizon][target].mean()
        val_subset = val_df[val_df["horizon_name"] == horizon]
        return np.full(len(val_subset), base_rate), {"status": "FALLBACK_INSUFFICIENT_DATA"}

    # Fit GARCH(1,1) — scale returns to percentage for numerical stability
    scaled_returns = train_returns * 100
    try:
        am = arch_model(scaled_returns, vol='Garch', p=1, q=1, mean='Constant',
                        rescale=False)
        res = am.fit(disp='off', show_warning=False)
        garch_fitted = True
    except Exception as e:
        garch_fitted = False

    if not garch_fitted:
        base_rate = train_df[train_df["horizon_name"] == horizon][target].mean()
        val_subset = val_df[val_df["horizon_name"] == horizon]
        return np.full(len(val_subset), base_rate), {"status": f"FALLBACK_GARCH_FAILED"}

    # Get conditional variances for all dates (train + val)
    all_scaled = log_returns * 100
    try:
        am_full = arch_model(all_scaled, vol='Garch', p=1, q=1, mean='Constant',
                             rescale=False)
        res_full = am_full.fit(disp='off', show_warning=False,
                               starting_values=res.params.values)
        cond_var = res_full.conditional_volatility ** 2
    except Exception:
        # Fall back: use training fit and apply to full series
        cond_var = res.conditional_volatility ** 2
        # Extend with last known variance for val dates
        last_var = cond_var.iloc[-1]
        val_dates_missing = [d for d in log_returns.index if d not in cond_var.index]
        for d in val_dates_missing:
            cond_var[d] = last_var

    # Create variance → probability mapping from training data
    train_horizon = train_df[train_df["horizon_name"] == horizon].copy()
    train_horizon = train_horizon.set_index("trade_date")

    # Align conditional variance with training observations
    train_vars = []
    train_labels = []
    for date in train_horizon.index:
        if date in cond_var.index:
            train_vars.append(cond_var[date])
            train_labels.append(train_horizon.loc[date, target])

    if len(train_vars) < 20:
        base_rate = train_df[train_df["horizon_name"] == horizon][target].mean()
        val_subset = val_df[val_df["horizon_name"] == horizon]
        return np.full(len(val_subset), base_rate), {"status": "FALLBACK_ALIGNMENT"}

    train_vars = np.array(train_vars)
    train_labels = np.array(train_labels)

    # Bin by variance quantiles
    try:
        bin_edges = np.percentile(train_vars, np.linspace(0, 100, GARCH_N_BINS + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_indices = np.digitize(train_vars, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, GARCH_N_BINS - 1)

        bin_probs = {}
        for b in range(GARCH_N_BINS):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_probs[b] = float(train_labels[mask].mean())
            else:
                bin_probs[b] = float(train_labels.mean())
    except Exception:
        base_rate = float(train_labels.mean())
        bin_probs = {b: base_rate for b in range(GARCH_N_BINS)}
        bin_edges = np.array([-np.inf] + [0] * (GARCH_N_BINS - 1) + [np.inf])

    # Apply to validation
    val_horizon = val_df[val_df["horizon_name"] == horizon].copy()
    val_preds = []
    for _, row in val_horizon.iterrows():
        date = row["trade_date"]
        if date in cond_var.index:
            v = cond_var[date]
            b = np.clip(np.digitize(v, bin_edges) - 1, 0, GARCH_N_BINS - 1)
            val_preds.append(bin_probs.get(b, float(train_labels.mean())))
        else:
            val_preds.append(float(train_labels.mean()))

    info = {
        "status": "OK",
        "garch_params": {k: float(v) for k, v in res.params.items()},
        "n_bins": GARCH_N_BINS,
        "bin_probs": {str(k): v for k, v in bin_probs.items()},
    }

    return np.array(val_preds), info


# ===========================================================================
# Baseline 4: RIV Momentum Heuristic
# ===========================================================================
def baseline_riv_momentum(train_df, val_df, target, horizon):
    """Bin riv_change_3d → empirical P(target) from training data."""
    train_h = train_df[train_df["horizon_name"] == horizon].copy()
    val_h = val_df[val_df["horizon_name"] == horizon].copy()

    feat = "riv_change_3d"
    base_rate = float(train_h[target].mean())

    # Training: bin riv_change_3d into quantiles
    train_valid = train_h.dropna(subset=[feat])
    if len(train_valid) < 30:
        return np.full(len(val_h), base_rate), {"status": "FALLBACK_INSUFFICIENT"}

    try:
        n_bins = 5
        bin_edges = np.percentile(train_valid[feat].values,
                                  np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        train_valid = train_valid.copy()
        train_valid["bin"] = np.clip(
            np.digitize(train_valid[feat].values, bin_edges) - 1, 0, n_bins - 1
        )

        bin_probs = {}
        for b in range(n_bins):
            mask = train_valid["bin"] == b
            if mask.sum() > 0:
                bin_probs[b] = float(train_valid.loc[mask, target].mean())
            else:
                bin_probs[b] = base_rate
    except Exception:
        return np.full(len(val_h), base_rate), {"status": "FALLBACK_BINNING_FAILED"}

    # Apply to validation
    val_preds = []
    for _, row in val_h.iterrows():
        v = row[feat]
        if pd.isna(v):
            val_preds.append(base_rate)
        else:
            b = np.clip(np.digitize(v, bin_edges) - 1, 0, n_bins - 1)
            val_preds.append(bin_probs.get(b, base_rate))

    info = {
        "status": "OK",
        "feature": feat,
        "n_bins": n_bins,
        "bin_probs": {str(k): round(v, 4) for k, v in bin_probs.items()},
        "bin_edges": [round(float(e), 6) if np.isfinite(e) else str(e)
                      for e in bin_edges],
    }

    return np.array(val_preds), info


# ===========================================================================
# Main Evaluation Loop
# ===========================================================================
def run_baselines(data, partition):
    """Run all 4 baselines across all folds, underlyings, horizons, targets."""
    results = {}

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()
        horizons = sorted(eligible["horizon_name"].unique())

        results[ul] = {}

        for target in TARGETS:
            target_short = target.replace("label_", "")
            results[ul][target_short] = {}

            for horizon in horizons:
                results[ul][target_short][horizon] = {"folds": []}

                fold_metrics = {
                    "naive": [], "persistence": [], "garch": [], "riv_momentum": []
                }

                for fold_cfg in partition["folds"]:
                    fold_num = fold_cfg["fold"]

                    # Split data
                    train_mask = (eligible["trade_date"] >= fold_cfg["train_start"]) & \
                                 (eligible["trade_date"] <= fold_cfg["train_end"])
                    val_mask = (eligible["trade_date"] >= fold_cfg["val_start"]) & \
                               (eligible["trade_date"] <= fold_cfg["val_end"])

                    train = eligible[train_mask].copy()
                    val = eligible[val_mask].copy()

                    train_h = train[train["horizon_name"] == horizon]
                    val_h = val[val["horizon_name"] == horizon]

                    if len(val_h) < 5 or val_h[target].sum() < 1:
                        print(f"  ⚠️ {ul}/{target_short}/{horizon}/fold{fold_num}: "
                              f"insufficient val data — skipping")
                        continue

                    y_true = val_h[target].values.astype(float)
                    fold_result = {"fold": fold_num, "n_val": len(y_true),
                                   "n_positive": int(y_true.sum())}

                    # --- Baseline 1: Naive ---
                    preds_naive, info_naive = baseline_naive(train_h[target], y_true)
                    m_naive = compute_metrics(y_true, preds_naive)
                    fold_result["naive"] = m_naive
                    fold_metrics["naive"].append(m_naive)

                    # --- Baseline 2: Persistence ---
                    preds_pers, info_pers = baseline_persistence(
                        train, val, target, horizon)
                    # Align with val_h (same horizon)
                    val_h_sorted = val_h.sort_values("trade_date")
                    pers_aligned = []
                    combined_sorted = pd.concat([train, val]).sort_values("trade_date")
                    combined_h = combined_sorted[combined_sorted["horizon_name"] == horizon]
                    combined_h = combined_h.copy()
                    combined_h["pred_pers"] = combined_h[target].shift(1)
                    base_rate_pers = train_h[target].mean()
                    combined_h["pred_pers"] = combined_h["pred_pers"].fillna(base_rate_pers)
                    val_dates_set = set(val_h["trade_date"].values)
                    pers_preds_final = combined_h[
                        combined_h["trade_date"].isin(val_dates_set)
                    ]["pred_pers"].values

                    if len(pers_preds_final) == len(y_true):
                        m_pers = compute_metrics(y_true, pers_preds_final)
                    else:
                        # Fallback if alignment fails
                        m_pers = compute_metrics(y_true, np.full(len(y_true), base_rate_pers))
                    fold_result["persistence"] = m_pers
                    fold_metrics["persistence"].append(m_pers)

                    # --- Baseline 3: GARCH ---
                    preds_garch, info_garch = baseline_garch(
                        train, val, target, horizon, ul)
                    if len(preds_garch) == len(y_true):
                        m_garch = compute_metrics(y_true, preds_garch)
                    else:
                        m_garch = compute_metrics(y_true,
                                                  np.full(len(y_true), train_h[target].mean()))
                    fold_result["garch"] = m_garch
                    fold_result["garch_info"] = info_garch.get("status", "OK")
                    fold_metrics["garch"].append(m_garch)

                    # --- Baseline 4: RIV Momentum ---
                    preds_mom, info_mom = baseline_riv_momentum(
                        train, val, target, horizon)
                    if len(preds_mom) == len(y_true):
                        m_mom = compute_metrics(y_true, preds_mom)
                    else:
                        m_mom = compute_metrics(y_true,
                                                np.full(len(y_true), train_h[target].mean()))
                    fold_result["riv_momentum"] = m_mom
                    fold_metrics["riv_momentum"].append(m_mom)

                    results[ul][target_short][horizon]["folds"].append(fold_result)

                    print(f"  {ul}/{target_short}/{horizon}/fold{fold_num}: "
                          f"Naive={m_naive['brier']:.4f} Pers={m_pers['brier']:.4f} "
                          f"GARCH={m_garch['brier']:.4f} Mom={m_mom['brier']:.4f}")

                # Aggregate across folds
                agg = {}
                for baseline_name in ["naive", "persistence", "garch", "riv_momentum"]:
                    if fold_metrics[baseline_name]:
                        agg[baseline_name] = {}
                        for metric in ["brier", "ece", "auroc", "logloss"]:
                            vals = [m[metric] for m in fold_metrics[baseline_name]
                                    if not np.isnan(m[metric])]
                            if vals:
                                agg[baseline_name][metric] = {
                                    "mean": float(np.mean(vals)),
                                    "std": float(np.std(vals)),
                                    "min": float(np.min(vals)),
                                    "max": float(np.max(vals)),
                                }

                results[ul][target_short][horizon]["aggregate"] = agg

    return results


def save_results(results):
    """Save results as JSON and markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = OUTPUT_DIR / "baseline_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Markdown report
    md_lines = [
        "# Oracle-V Model Research — R3: Baseline Results",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Purpose:** Establish performance floor that ML model must beat (§7.2.5)",
        "",
        "## Outperformance Requirement",
        "",
        "ML model Brier score must be at least **0.02 below** each baseline's Brier score",
        "on validation data, with statistical significance at 5% (Diebold-Mariano test).",
        "",
        "**Primary comparator:** Persistence baseline (given high label autocorrelation from R1j).",
        "",
        "---",
    ]

    for ul in UNDERLYINGS:
        md_lines.append(f"\n## {ul}\n")

        for target_short in ["expansion", "compression"]:
            md_lines.append(f"\n### {target_short.title()}\n")

            for horizon in sorted(results[ul][target_short].keys()):
                if horizon == "aggregate":
                    continue
                agg = results[ul][target_short][horizon].get("aggregate", {})
                folds = results[ul][target_short][horizon].get("folds", [])

                if not agg:
                    continue

                md_lines.append(f"\n**{horizon}** ({len(folds)} folds):\n")
                md_lines.append("| Baseline | Brier (mean±std) | ECE | AUC-ROC | Log-Loss |")
                md_lines.append("|---|---|---|---|---|")

                for bl_name in ["naive", "persistence", "garch", "riv_momentum"]:
                    if bl_name in agg:
                        bl = agg[bl_name]
                        brier = bl.get("brier", {})
                        ece = bl.get("ece", {})
                        auroc = bl.get("auroc", {})
                        ll = bl.get("logloss", {})

                        md_lines.append(
                            f"| {bl_name.replace('_', ' ').title()} | "
                            f"{brier.get('mean', 0):.4f} ± {brier.get('std', 0):.4f} | "
                            f"{ece.get('mean', 0):.4f} | "
                            f"{auroc.get('mean', 0):.4f} | "
                            f"{ll.get('mean', 0):.4f} |"
                        )

                # Best baseline (lowest Brier)
                best_bl = None
                best_brier = float('inf')
                for bl_name in ["naive", "persistence", "garch", "riv_momentum"]:
                    if bl_name in agg and "brier" in agg[bl_name]:
                        b = agg[bl_name]["brier"]["mean"]
                        if b < best_brier:
                            best_brier = b
                            best_bl = bl_name

                if best_bl:
                    target_brier = best_brier - 0.02
                    md_lines.append(f"\n**Best baseline:** {best_bl} "
                                    f"(Brier {best_brier:.4f})")
                    md_lines.append(f"**ML target:** Brier ≤ {target_brier:.4f} "
                                    f"(best - 0.02 margin)")

    # Per-fold detail table
    md_lines.extend([
        "\n---",
        "\n## Per-Fold Detail (Brier Scores)\n",
    ])

    for ul in UNDERLYINGS:
        for target_short in ["expansion", "compression"]:
            for horizon in sorted(results[ul][target_short].keys()):
                folds = results[ul][target_short][horizon].get("folds", [])
                if not folds:
                    continue

                md_lines.append(f"\n**{ul} / {target_short} / {horizon}:**\n")
                md_lines.append("| Fold | n_val | Positive | Naive | Persistence | GARCH | RIV Momentum |")
                md_lines.append("|---|---|---|---|---|---|---|")

                for fold in folds:
                    md_lines.append(
                        f"| {fold['fold']} | {fold['n_val']} | {fold['n_positive']} | "
                        f"{fold.get('naive', {}).get('brier', 'N/A'):.4f} | "
                        f"{fold.get('persistence', {}).get('brier', 'N/A'):.4f} | "
                        f"{fold.get('garch', {}).get('brier', 'N/A'):.4f} | "
                        f"{fold.get('riv_momentum', {}).get('brier', 'N/A'):.4f} |"
                    )

    md_lines.extend([
        "\n---",
        "\n## Interpretation Guide",
        "",
        "- **Naive baseline** sets the floor — any model must beat random prediction",
        "- **Persistence baseline** is the primary hurdle (Expert Review recommendation)",
        "  due to high label autocorrelation (R1j: lag-1 up to 0.573)",
        "- **GARCH baseline** captures volatility clustering — beating GARCH demonstrates",
        "  ML adds value beyond conditional variance dynamics",
        "- **RIV momentum** is a falsification test — if ML cannot beat a hand-crafted",
        "  threshold rule on `riv_change_3d`, Oracle-V adds no value",
        "",
        f"*Frozen at generation time.*",
    ])

    md_path = OUTPUT_DIR / "baseline_results.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")


def print_summary(results):
    """Print compact summary to console."""
    print(f"\n{'='*70}")
    print("BASELINE SUMMARY (Mean Brier Score Across Folds)")
    print(f"{'='*70}")
    print(f"{'UL':<12} {'Target':<14} {'Horizon':<18} {'Naive':>8} {'Persist':>8} "
          f"{'GARCH':>8} {'RIV Mom':>8} {'Best':>10} {'ML Target':>10}")
    print("-" * 110)

    for ul in UNDERLYINGS:
        for target_short in ["expansion", "compression"]:
            for horizon in sorted(results[ul][target_short].keys()):
                agg = results[ul][target_short][horizon].get("aggregate", {})
                if not agg:
                    continue

                vals = {}
                for bl in ["naive", "persistence", "garch", "riv_momentum"]:
                    if bl in agg and "brier" in agg[bl]:
                        vals[bl] = agg[bl]["brier"]["mean"]

                if not vals:
                    continue

                best_name = min(vals, key=vals.get)
                best_val = vals[best_name]
                ml_target = best_val - 0.02

                print(f"{ul:<12} {target_short:<14} {horizon:<18} "
                      f"{vals.get('naive', float('nan')):>8.4f} "
                      f"{vals.get('persistence', float('nan')):>8.4f} "
                      f"{vals.get('garch', float('nan')):>8.4f} "
                      f"{vals.get('riv_momentum', float('nan')):>8.4f} "
                      f"{best_name:>10} "
                      f"{ml_target:>10.4f}")


def main():
    print("=" * 70)
    print("Oracle-V Model Research — R3: Baseline Construction")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data, partition = load_data()

    print(f"\nRunning 4 baselines × {len(partition['folds'])} folds × "
          f"{len(UNDERLYINGS)} UL × 2 horizons × 2 targets ...\n")

    results = run_baselines(data, partition)
    save_results(results)
    print_summary(results)

    print(f"\n{'='*70}")
    print("R3 COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Review: cat research/baselines/baseline_results.md")
    print(f"  2. Commit: git add research/baselines/ scripts/run_r3_baselines.py")
    print(f"     git commit -m 'R3: Baseline construction — 4 baselines across 6 folds'")
    print(f"  3. Proceed to R4 (Architecture Selection TDR)")


if __name__ == "__main__":
    main()
