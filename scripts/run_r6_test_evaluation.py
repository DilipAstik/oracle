#!/usr/bin/env python3
"""
Oracle-V Model Research Phase — R6: Test Set Evaluation (Single Shot)
=====================================================================

This script evaluates the top 4 model combinations on the frozen test hold-out.

CRITICAL: R6_test_criteria.md MUST be committed to Git before running this script.

Methodology:
  1. Train on ALL non-test data using R5 best hyperparameters (last fold)
  2. Fit isotonic calibration on last fold's validation window
  3. Predict on test set — SINGLE SHOT, no iteration
  4. Compare against baselines recomputed on test set
  5. Report pass/fail per pre-declared criteria

Usage:
  cd /home/ssm-user/oracle
  source .venv/bin/activate
  export PYTHONPATH=/home/ssm-user/oracle/src
  python scripts/run_r6_test_evaluation.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline

import xgboost as xgb

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
TRAINING_PREFIX = "computed/training"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

PARTITION_FILE = Path("research/partitioning.json")
R5_RESULTS_FILE = Path("research/training/training_results.json")
OUTPUT_DIR = Path("research/test_evaluation")
FIG_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42

MODELLING_FEATURES = [
    "day_of_week", "days_to_current_expiry", "is_monthly_expiry_week",
    "days_to_next_holiday",
    "riv_level", "riv_change_1d", "riv_change_3d",
    "iv_term_structure_slope", "iv_skew_25d",
    "iv_rv_spread_5d", "iv_rv_spread_10d",
    "pcr_volume", "pcr_oi", "oi_change_net_1d", "volume_zscore_session",
]

# The 4 combinations to evaluate (Tier 1 + Tier 2 from R5)
TEST_COMBINATIONS = [
    {"ul": "BANKNIFTY", "target": "label_expansion", "horizon": "horizon_short",
     "tier": 1, "r5_brier": 0.1117, "r5_improvement": 0.0202},
    {"ul": "NIFTY", "target": "label_compression", "horizon": "horizon_medium",
     "tier": 2, "r5_brier": 0.1720, "r5_improvement": 0.0185},
    {"ul": "NIFTY", "target": "label_expansion", "horizon": "horizon_short",
     "tier": 2, "r5_brier": 0.1286, "r5_improvement": 0.0112},
    {"ul": "NIFTY", "target": "label_expansion", "horizon": "horizon_medium",
     "tier": 2, "r5_brier": 0.1307, "r5_improvement": 0.0099},
]

# Success thresholds (from R6_test_criteria.md)
BRIER_CEILING = 0.22
ECE_CEILING = 0.13
AUROC_FLOOR = 0.58
BASELINE_IMPROVEMENT_THRESHOLD = 0.02


# ===========================================================================
# Metrics
# ===========================================================================
def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / len(y_true)) if len(y_true) > 0 else 0.0

def compute_metrics(y_true, y_prob):
    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    metrics = {"brier": brier_score(y_true, y_prob),
               "ece": expected_calibration_error(y_true, y_prob)}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    except:
        metrics["auroc"] = float('nan')
    try:
        metrics["logloss"] = float(log_loss(y_true, y_prob_clipped))
    except:
        metrics["logloss"] = float('nan')
    return metrics

def bootstrap_brier_ci(y_true, y_prob, n_bootstrap=2000, ci=0.95):
    np.random.seed(RANDOM_SEED)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        scores.append(brier_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


# ===========================================================================
# Baselines on Test Set
# ===========================================================================
def compute_test_baselines(train_all, test_h, target):
    """Compute all 4 baselines on the test set."""
    train_h_target = train_all[target]
    base_rate = float(train_h_target.mean())

    baselines = {}

    # 1. Naive
    naive_preds = np.full(len(test_h), base_rate)
    baselines["naive"] = compute_metrics(test_h[target].values, naive_preds)
    baselines["naive"]["preds"] = naive_preds.tolist()

    # 2. Persistence
    combined = pd.concat([train_all, test_h]).sort_values("trade_date")
    combined["pred_pers"] = combined[target].shift(1).fillna(base_rate)
    test_dates = set(test_h["trade_date"].values)
    pers_preds = combined[combined["trade_date"].isin(test_dates)]["pred_pers"].values
    if len(pers_preds) == len(test_h):
        baselines["persistence"] = compute_metrics(test_h[target].values, pers_preds)
    else:
        baselines["persistence"] = compute_metrics(test_h[target].values, naive_preds)
    baselines["persistence"]["preds"] = pers_preds.tolist() if len(pers_preds) == len(test_h) else naive_preds.tolist()

    # 3. GARCH
    try:
        from arch import arch_model
        daily_riv = pd.concat([train_all, test_h]).groupby("trade_date")["riv_level"].first().sort_index()
        log_returns = np.log(daily_riv / daily_riv.shift(1)).dropna() * 100
        train_end = train_all["trade_date"].max()
        train_ret = log_returns[log_returns.index <= train_end]

        am = arch_model(train_ret, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
        res = am.fit(disp='off', show_warning=False)

        am_full = arch_model(log_returns, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
        res_full = am_full.fit(disp='off', show_warning=False, starting_values=res.params.values)
        cond_var = res_full.conditional_volatility ** 2

        # Variance → probability mapping from training
        train_vars, train_labels = [], []
        for _, row in train_all.iterrows():
            d = row["trade_date"]
            if d in cond_var.index:
                train_vars.append(cond_var[d])
                train_labels.append(row[target])

        train_vars = np.array(train_vars)
        train_labels = np.array(train_labels)
        n_bins = 5
        bin_edges = np.percentile(train_vars, np.linspace(0, 100, n_bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        bin_indices = np.clip(np.digitize(train_vars, bin_edges) - 1, 0, n_bins - 1)
        bin_probs = {}
        for b in range(n_bins):
            mask = bin_indices == b
            bin_probs[b] = float(train_labels[mask].mean()) if mask.sum() > 0 else base_rate

        garch_preds = []
        for _, row in test_h.iterrows():
            d = row["trade_date"]
            if d in cond_var.index:
                v = cond_var[d]
                b = np.clip(np.digitize(v, bin_edges) - 1, 0, n_bins - 1)
                garch_preds.append(bin_probs.get(b, base_rate))
            else:
                garch_preds.append(base_rate)
        garch_preds = np.array(garch_preds)
        baselines["garch"] = compute_metrics(test_h[target].values, garch_preds)
        baselines["garch"]["preds"] = garch_preds.tolist()
    except Exception as e:
        baselines["garch"] = compute_metrics(test_h[target].values, naive_preds)
        baselines["garch"]["status"] = f"FALLBACK: {str(e)}"

    # 4. RIV Momentum
    feat = "riv_change_3d"
    train_valid = train_all.dropna(subset=[feat])
    if len(train_valid) >= 30:
        n_bins = 5
        bin_edges = np.percentile(train_valid[feat].values, np.linspace(0, 100, n_bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        bin_indices = np.clip(np.digitize(train_valid[feat].values, bin_edges) - 1, 0, n_bins - 1)
        bin_probs = {}
        for b in range(n_bins):
            mask = bin_indices == b
            bin_probs[b] = float(train_valid.loc[mask.values, target].mean()) if mask.sum() > 0 else base_rate

        mom_preds = []
        for _, row in test_h.iterrows():
            v = row[feat]
            if pd.isna(v):
                mom_preds.append(base_rate)
            else:
                b = np.clip(np.digitize(v, bin_edges) - 1, 0, n_bins - 1)
                mom_preds.append(bin_probs.get(b, base_rate))
        mom_preds = np.array(mom_preds)
    else:
        mom_preds = naive_preds
    baselines["riv_momentum"] = compute_metrics(test_h[target].values, mom_preds)
    baselines["riv_momentum"]["preds"] = mom_preds.tolist()

    # Best baseline
    best_name = min(baselines, key=lambda k: baselines[k]["brier"])
    baselines["best"] = {"name": best_name, "brier": baselines[best_name]["brier"]}

    return baselines


# ===========================================================================
# Model Training (Full Non-Test Data)
# ===========================================================================
def get_best_params(r5_results, ul, target_short, horizon):
    """Extract best hyperparameters from R5 last fold."""
    try:
        folds = r5_results[ul][target_short][horizon]["folds"]
        last_fold = folds[-1]
        return last_fold["xgboost"]["best_params"]
    except (KeyError, IndexError):
        return None


def train_and_predict(train_all_h, test_h, target, best_params, train_last_fold_h, val_last_fold_h):
    """Train XGBoost + LR on full training data, predict test set."""
    results = {}

    X_train_full = train_all_h[MODELLING_FEATURES].values
    y_train_full = train_all_h[target].values.astype(int)
    X_test = test_h[MODELLING_FEATURES].values
    y_test = test_h[target].values.astype(int)

    # --- XGBoost ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    }
    if best_params:
        xgb_params.update(best_params)

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_full, y_train_full)

    raw_probs_test = xgb_model.predict_proba(X_test)[:, 1]

    # Isotonic calibration: fit on last fold's validation predictions
    X_val_last = val_last_fold_h[MODELLING_FEATURES].values
    y_val_last = val_last_fold_h[target].values.astype(int)
    raw_probs_val = xgb_model.predict_proba(X_val_last)[:, 1]

    try:
        ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        ir.fit(raw_probs_val, y_val_last)
        cal_probs_test = ir.transform(raw_probs_test)
    except Exception:
        cal_probs_test = raw_probs_test

    # Train metrics (sanity check — should be low)
    raw_probs_train = xgb_model.predict_proba(X_train_full)[:, 1]
    train_brier = brier_score(y_train_full, raw_probs_train)

    # Test metrics
    test_metrics_raw = compute_metrics(y_test, raw_probs_test)
    test_metrics_cal = compute_metrics(y_test, cal_probs_test)
    ci_low, ci_high = bootstrap_brier_ci(y_test, cal_probs_test)

    results["xgboost"] = {
        "train_brier": train_brier,
        "test_raw": test_metrics_raw,
        "test_calibrated": test_metrics_cal,
        "bootstrap_ci_95": [ci_low, ci_high],
        "params_used": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in xgb_params.items() if k not in
                        ["objective", "eval_metric", "use_label_encoder", "random_state", "verbosity"]},
        "cal_probs": cal_probs_test.tolist(),
        "raw_probs": raw_probs_test.tolist(),
    }

    # --- Logistic Regression ---
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(C=1.0, max_iter=1000,
                                        random_state=RANDOM_SEED, solver='lbfgs')),
    ])
    lr_pipeline.fit(X_train_full, y_train_full)
    lr_probs_test = lr_pipeline.predict_proba(X_test)[:, 1]

    try:
        lr_probs_val = lr_pipeline.predict_proba(X_val_last)[:, 1]
        ir_lr = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        ir_lr.fit(lr_probs_val, y_val_last)
        lr_cal_test = ir_lr.transform(lr_probs_test)
    except Exception:
        lr_cal_test = lr_probs_test

    results["logistic"] = {
        "test_raw": compute_metrics(y_test, lr_probs_test),
        "test_calibrated": compute_metrics(y_test, lr_cal_test),
    }

    return results


# ===========================================================================
# Visualisation
# ===========================================================================
def plot_results(combo_results):
    """Generate comparison plots."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    xgb_briers = []
    bl_briers = []
    thresholds = []

    for combo in combo_results:
        label = f"{combo['ul']}\n{combo['target_short']}\n{combo['horizon']}"
        labels.append(label)
        xgb_briers.append(combo["xgboost"]["test_calibrated"]["brier"])
        bl_briers.append(combo["baselines"]["best"]["brier"])
        thresholds.append(combo["baselines"]["best"]["brier"] - BASELINE_IMPROVEMENT_THRESHOLD)

    x = np.arange(len(labels))
    width = 0.3

    ax.bar(x - width, bl_briers, width, label='Best Baseline', color='lightcoral')
    ax.bar(x, xgb_briers, width, label='XGBoost (calibrated)', color='steelblue')
    ax.bar(x + width, thresholds, width, label='Target (BL - 0.02)', color='lightgreen',
           linestyle='--', edgecolor='green', linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("R6: Test Set — Model vs Baseline")
    ax.legend()

    # Add pass/fail annotations
    for i, combo in enumerate(combo_results):
        improvement = combo["improvement"]
        passed = combo["pass_baseline"]
        color = 'green' if passed else 'red'
        symbol = '✓' if passed else '✗'
        ax.annotate(f"{symbol} {improvement:+.4f}",
                    xy=(i, xgb_briers[i]), xytext=(0, -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "R6_test_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Calibration plot per combo
    for combo in combo_results:
        fig, ax = plt.subplots(figsize=(6, 6))
        probs = np.array(combo["xgboost"]["cal_probs"])
        y_true = np.array(combo["y_test"])

        n_bins = 8
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means_pred = []
        bin_means_true = []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means_pred.append(probs[mask].mean())
                bin_means_true.append(y_true[mask].mean())

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.scatter(bin_means_pred, bin_means_true, s=80, c='steelblue', zorder=5)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(f"Calibration: {combo['ul']} {combo['target_short']} {combo['horizon']}")
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        fname = f"R6_calibration_{combo['ul']}_{combo['target_short']}_{combo['horizon']}.png"
        fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)


# ===========================================================================
# Report
# ===========================================================================
def save_report(combo_results):
    """Save JSON and markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON (strip non-serializable)
    json_data = []
    for combo in combo_results:
        entry = {k: v for k, v in combo.items() if k not in ["y_test"]}
        json_data.append(entry)

    with open(OUTPUT_DIR / "test_results.json", 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

    # Markdown
    lines = [
        "# Oracle-V Model Research — R6: Test Set Evaluation Results",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Type:** Single-shot hold-out evaluation (Oct 2025 – Feb 2026)",
        "**Criteria:** Frozen in R6_test_criteria.md before evaluation",
        "",
        "---",
        "",
        "## Results Summary\n",
        "| # | UL | Target | Horizon | Tier | Test Brier | 95% CI | Best BL (Brier) | Improvement | Pass? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for i, combo in enumerate(combo_results, 1):
        ci = combo["xgboost"]["bootstrap_ci_95"]
        lines.append(
            f"| {i} | {combo['ul']} | {combo['target_short']} | {combo['horizon']} | "
            f"{combo['tier']} | {combo['xgboost']['test_calibrated']['brier']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{combo['baselines']['best']['name']} ({combo['baselines']['best']['brier']:.4f}) | "
            f"{combo['improvement']:+.4f} | "
            f"{'✅ PASS' if combo['pass_baseline'] else '❌ FAIL'} |"
        )

    # Detailed metrics
    lines.extend([
        "",
        "## Detailed Metrics\n",
        "| # | Brier | ECE | AUC-ROC | Log-Loss | Brier ≤ 0.22 | ECE ≤ 0.13 | AUC ≥ 0.58 |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for i, combo in enumerate(combo_results, 1):
        m = combo["xgboost"]["test_calibrated"]
        lines.append(
            f"| {i} | {m['brier']:.4f} | {m['ece']:.4f} | {m['auroc']:.4f} | "
            f"{m['logloss']:.4f} | "
            f"{'✅' if m['brier'] <= BRIER_CEILING else '❌'} | "
            f"{'✅' if m['ece'] <= ECE_CEILING else '❌'} | "
            f"{'✅' if m['auroc'] >= AUROC_FLOOR else '❌'} |"
        )

    # XGBoost vs LR on test
    lines.extend([
        "",
        "## XGBoost vs Logistic Regression (Test Set)\n",
        "| # | XGB Brier | LR Brier | Winner |",
        "|---|---|---|---|",
    ])

    for i, combo in enumerate(combo_results, 1):
        xgb_b = combo["xgboost"]["test_calibrated"]["brier"]
        lr_b = combo["logistic"]["test_calibrated"]["brier"]
        winner = "XGBoost" if xgb_b < lr_b else "Logistic" if lr_b < xgb_b else "Tie"
        lines.append(f"| {i} | {xgb_b:.4f} | {lr_b:.4f} | {winner} |")

    # Baseline detail
    lines.extend([
        "",
        "## Test Set Baseline Detail\n",
        "| # | Naive | Persistence | GARCH | RIV Momentum | Best |",
        "|---|---|---|---|---|---|",
    ])

    for i, combo in enumerate(combo_results, 1):
        bl = combo["baselines"]
        lines.append(
            f"| {i} | {bl['naive']['brier']:.4f} | {bl['persistence']['brier']:.4f} | "
            f"{bl['garch']['brier']:.4f} | {bl['riv_momentum']['brier']:.4f} | "
            f"{bl['best']['name']} |"
        )

    # Walk-forward vs Test comparison
    lines.extend([
        "",
        "## Walk-Forward (R5) vs Test (R6) Comparison\n",
        "| # | R5 Brier | R6 Brier | Δ | Interpretation |",
        "|---|---|---|---|---|",
    ])

    for i, combo in enumerate(combo_results, 1):
        r5_b = combo["r5_brier"]
        r6_b = combo["xgboost"]["test_calibrated"]["brier"]
        delta = r6_b - r5_b
        if abs(delta) < 0.02:
            interp = "Consistent"
        elif delta > 0:
            interp = "Degraded (expected some)"
        else:
            interp = "Improved (check for luck)"
        lines.append(f"| {i} | {r5_b:.4f} | {r6_b:.4f} | {delta:+.4f} | {interp} |")

    # Final verdict
    n_pass = sum(1 for c in combo_results if c["pass_baseline"])
    lines.extend([
        "",
        "---",
        "",
        "## Final Verdict",
        "",
        f"**{n_pass} of {len(combo_results)} combinations pass the 0.02 improvement threshold.**",
        "",
    ])

    passing = [c for c in combo_results if c["pass_baseline"]]
    if passing:
        lines.append("### Production Candidates\n")
        for c in passing:
            lines.append(f"- **{c['ul']} {c['target_short']} {c['horizon']}** — "
                         f"Brier {c['xgboost']['test_calibrated']['brier']:.4f}, "
                         f"improvement {c['improvement']:+.4f}")

    failing = [c for c in combo_results if not c["pass_baseline"]]
    if failing:
        lines.append("\n### Deferred to v2 (Below Threshold)\n")
        for c in failing:
            lines.append(f"- {c['ul']} {c['target_short']} {c['horizon']} — "
                         f"improvement {c['improvement']:+.4f}")

    lines.append(f"\n*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                 f"Results are final — no iteration permitted.*")

    with open(OUTPUT_DIR / "test_results.md", 'w') as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {OUTPUT_DIR / 'test_results.json'}")
    print(f"Saved: {OUTPUT_DIR / 'test_results.md'}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("Oracle-V Model Research — R6: Test Set Evaluation (Single Shot)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Verify criteria document exists
    criteria_file = Path("research/R6_test_criteria.md")
    if not criteria_file.exists():
        print("❌ ABORT: R6_test_criteria.md not found. Must be committed before test evaluation.")
        return
    print("✅ R6_test_criteria.md found — criteria frozen")

    # Load partition and R5 results
    with open(PARTITION_FILE) as f:
        partition = json.load(f)
    with open(R5_RESULTS_FILE) as f:
        r5_results = json.load(f)

    test_start = partition["test"]["test_start"]
    test_end = partition["test"]["test_end"]
    print(f"Test period: {test_start} → {test_end}")

    # Load data
    data = {}
    for ul in UNDERLYINGS:
        s3_path = f"s3://{S3_BUCKET}/{TRAINING_PREFIX}/{ul}_training_features.parquet"
        df = pd.read_parquet(s3_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        data[ul] = df[df["training_eligible"] == True].copy()
        print(f"  {ul}: {len(data[ul])} eligible rows")

    # Last fold config (for calibration window)
    last_fold = partition["folds"][-1]

    # Process each combination
    combo_results = []

    for combo in TEST_COMBINATIONS:
        ul = combo["ul"]
        target = combo["target"]
        target_short = target.replace("label_", "")
        horizon = combo["horizon"]

        print(f"\n--- {ul} / {target_short} / {horizon} (Tier {combo['tier']}) ---")

        eligible = data[ul]

        # Split: ALL non-test for training, test for evaluation
        train_all = eligible[eligible["trade_date"] < test_start].copy()
        test_set = eligible[(eligible["trade_date"] >= test_start) &
                            (eligible["trade_date"] <= test_end)].copy()

        train_all_h = train_all[train_all["horizon_name"] == horizon]
        test_h = test_set[test_set["horizon_name"] == horizon]

        print(f"  Training: {len(train_all_h)} obs, Test: {len(test_h)} obs "
              f"(positive: {test_h[target].sum()})")

        if len(test_h) < 5 or test_h[target].sum() < 1:
            print(f"  ⚠️ SKIPPED — insufficient test data")
            continue

        # Last fold val window (for calibration fitting)
        val_last = eligible[(eligible["trade_date"] >= last_fold["val_start"]) &
                            (eligible["trade_date"] <= last_fold["val_end"])]
        val_last_h = val_last[val_last["horizon_name"] == horizon]

        # Get R5 best params
        best_params = get_best_params(r5_results, ul, target_short, horizon)
        if best_params:
            print(f"  Using R5 params: {best_params}")
        else:
            print(f"  ⚠️ No R5 params found — using defaults")

        # Compute baselines on test set
        baselines = compute_test_baselines(train_all_h, test_h, target)
        print(f"  Baselines — Naive: {baselines['naive']['brier']:.4f}, "
              f"Persist: {baselines['persistence']['brier']:.4f}, "
              f"GARCH: {baselines['garch']['brier']:.4f}, "
              f"RIV Mom: {baselines['riv_momentum']['brier']:.4f}")
        print(f"  Best baseline: {baselines['best']['name']} ({baselines['best']['brier']:.4f})")

        # Train and predict
        model_results = train_and_predict(
            train_all_h, test_h, target, best_params,
            train_all_h, val_last_h  # Full training + last fold val for calibration
        )

        # Compute improvement
        xgb_brier = model_results["xgboost"]["test_calibrated"]["brier"]
        improvement = baselines["best"]["brier"] - xgb_brier
        pass_baseline = improvement >= BASELINE_IMPROVEMENT_THRESHOLD

        ci = model_results["xgboost"]["bootstrap_ci_95"]
        print(f"  XGBoost (cal): Brier={xgb_brier:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Improvement: {improvement:+.4f} {'✅ PASS' if pass_baseline else '❌ FAIL'}")

        combo_results.append({
            "ul": ul,
            "target": target,
            "target_short": target_short,
            "horizon": horizon,
            "tier": combo["tier"],
            "r5_brier": combo["r5_brier"],
            "r5_improvement": combo["r5_improvement"],
            "xgboost": model_results["xgboost"],
            "logistic": model_results["logistic"],
            "baselines": baselines,
            "improvement": float(improvement),
            "pass_baseline": pass_baseline,
            "n_test": len(test_h),
            "n_positive_test": int(test_h[target].sum()),
            "y_test": test_h[target].values.tolist(),
        })

    # Save and visualise
    save_report(combo_results)
    plot_results(combo_results)

    # Final summary
    print(f"\n{'='*70}")
    print("R6 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'UL':<12} {'Target':<14} {'Horizon':<18} {'Test Brier':>11} "
          f"{'Best BL':>10} {'Improve':>9} {'Pass':>6}")
    print("-" * 80)

    for combo in combo_results:
        print(f"{combo['ul']:<12} {combo['target_short']:<14} {combo['horizon']:<18} "
              f"{combo['xgboost']['test_calibrated']['brier']:>11.4f} "
              f"{combo['baselines']['best']['brier']:>10.4f} "
              f"{combo['improvement']:>+9.4f} "
              f"{'✅' if combo['pass_baseline'] else '❌':>6}")

    n_pass = sum(1 for c in combo_results if c["pass_baseline"])
    print(f"\n{n_pass} of {len(combo_results)} pass the 0.02 threshold.")

    print(f"\n{'='*70}")
    print("R6 COMPLETE — Results are final.")
    print(f"{'='*70}")
    print(f"\nCommit: git add research/test_evaluation/ research/R6_test_criteria.md")
    print(f"        git commit -m 'R6: Test evaluation — single shot, results final'")


if __name__ == "__main__":
    main()
