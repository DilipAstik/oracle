#!/usr/bin/env python3
"""
Oracle-V Model Research Phase — R1: Exploratory Data Analysis
=============================================================

Specification References:
  - Part IV §7.1.1 (Research Stage requirements)
  - Governance §10.2 (Research exit criteria)
  - OV-R-5 (Feature importance documented)
  - OV-R-6 (Multicollinearity assessment)
  - OV-R-7 (Expiry day transition handling)

Sub-tasks:
  R1a: Label distribution (base rates per underlying, horizon)
  R1b: Feature distributions & outliers
  R1c: Feature-label correlation + mutual information
  R1d: Multicollinearity (pairwise correlation + VIF)
  R1e: Temporal structure + regime break plots
  R1f: Quality & exclusion rate analysis (§2.1.7 guardrail: <15%)
  R1g: Rolling base rate stability (60-day window)
  R1h: Label noise estimation (near-threshold z-scores)
  R1i: Information decay check (rolling logistic model)        [SHOULD]
  R1j: Label temporal autocorrelation                          [SHOULD]

Usage:
  cd /home/ssm-user/oracle
  source .venv/bin/activate
  export PYTHONPATH=/home/ssm-user/oracle/src
  python scripts/run_r1_eda.py

Output:
  research/eda/figures/           — All plots (PNG)
  research/eda/R1_eda_report.md   — Summary report with findings
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for EC2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
TRAINING_PREFIX = "computed/training"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]

# Features available for modelling (17 usable, excluding 4 always-UNAVAILABLE)
USABLE_FEATURES = [
    # Family 5 — Temporal
    "day_of_week", "time_of_day_bucket", "days_to_current_expiry",
    "is_monthly_expiry_week", "days_to_next_holiday",
    # Family 1 — IV Surface (immediate only)
    "riv_level", "riv_change_1d", "riv_change_3d",
    "iv_percentile_rank", "iv_term_structure_slope", "iv_skew_25d",
    # Family 2 — Realised-Implied Spread
    "iv_rv_spread_5d", "iv_rv_spread_10d",
    # Family 4 — Microstructure
    "pcr_volume", "pcr_oi", "oi_change_net_1d", "volume_zscore_session",
]

# Deferred features (always UNAVAILABLE — excluded from modelling)
DEFERRED_FEATURES = [
    "india_vix_level", "india_vix_percentile_rank", "india_vix_change",
    "fno_ban_heavyweight_flag",
]

# Label columns
LABEL_COLS = ["label_expansion", "label_compression", "label_stable"]

# Quality column prefix
QUALITY_PREFIX = "quality_OV_F_"

# Output directories
BASE_DIR = Path("research/eda")
FIG_DIR = BASE_DIR / "figures"
REPORT_PATH = BASE_DIR / "R1_eda_report.md"


def setup_output_dirs():
    """Create output directories."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created: {BASE_DIR}")


def load_training_data():
    """Load training Parquet files from S3."""
    data = {}
    for ul in UNDERLYINGS:
        s3_path = f"s3://{S3_BUCKET}/{TRAINING_PREFIX}/{ul}_training_features.parquet"
        print(f"Loading {s3_path} ...")
        df = pd.read_parquet(s3_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        data[ul] = df
        print(f"  {ul}: {len(df)} rows, {len(df.columns)} columns")
    return data


# ===========================================================================
# R1a: Label Distribution Analysis
# ===========================================================================
def r1a_label_distribution(data, report_lines):
    """Examine class balance across underlyings and horizons."""
    print("\n" + "="*70)
    print("R1a: Label Distribution Analysis")
    print("="*70)

    report_lines.append("\n## R1a: Label Distribution Analysis\n")
    report_lines.append("**Purpose:** Verify class balance; compute naive baseline rates.\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("R1a: Label Distribution by Underlying × Horizon", fontsize=14, fontweight='bold')

    summary_rows = []

    for i, ul in enumerate(UNDERLYINGS):
        df = data[ul]
        for j, horizon in enumerate(sorted(df["horizon_name"].unique())):
            ax = axes[i][j]
            subset = df[df["horizon_name"] == horizon]
            eligible = subset[subset["training_eligible"] == True]

            n_total = len(subset)
            n_eligible = len(eligible)

            if n_eligible == 0:
                ax.text(0.5, 0.5, "No eligible data", ha='center', va='center')
                continue

            # Compute base rates
            expansion_rate = eligible["label_expansion"].mean()
            compression_rate = eligible["label_compression"].mean()
            stable_rate = eligible["label_stable"].mean()

            rates = [expansion_rate, compression_rate, stable_rate]
            labels_text = [
                f"Expansion\n{expansion_rate:.1%}",
                f"Compression\n{compression_rate:.1%}",
                f"Stable\n{stable_rate:.1%}",
            ]
            colors = ['#e74c3c', '#3498db', '#95a5a6']

            bars = ax.bar(labels_text, rates, color=colors, edgecolor='white', linewidth=1.5)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Proportion")
            ax.set_title(f"{ul} — {horizon} (n={n_eligible})")
            ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Uniform')

            # Add count annotations
            for bar, rate in zip(bars, rates):
                count = int(rate * n_eligible)
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"n={count}", ha='center', va='bottom', fontsize=9)

            summary_rows.append({
                "Underlying": ul, "Horizon": horizon,
                "N_Total": n_total, "N_Eligible": n_eligible,
                "Expansion%": f"{expansion_rate:.1%}",
                "Compression%": f"{compression_rate:.1%}",
                "Stable%": f"{stable_rate:.1%}",
            })

            print(f"  {ul}/{horizon}: n={n_eligible}, "
                  f"Expansion={expansion_rate:.1%}, Compression={compression_rate:.1%}, "
                  f"Stable={stable_rate:.1%}")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "R1a_label_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Report table
    report_lines.append("| Underlying | Horizon | N Total | N Eligible | Expansion | Compression | Stable |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for row in summary_rows:
        report_lines.append(
            f"| {row['Underlying']} | {row['Horizon']} | {row['N_Total']} | "
            f"{row['N_Eligible']} | {row['Expansion%']} | {row['Compression%']} | {row['Stable%']} |"
        )
    report_lines.append("")

    return summary_rows


# ===========================================================================
# R1b: Feature Distribution Profiling
# ===========================================================================
def r1b_feature_distributions(data, report_lines):
    """Profile distributions, outliers, and missing patterns for usable features."""
    print("\n" + "="*70)
    print("R1b: Feature Distribution Profiling")
    print("="*70)

    report_lines.append("\n## R1b: Feature Distribution Profiling\n")

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()

        report_lines.append(f"\n### {ul} (n={len(eligible)} eligible rows)\n")

        # Compute summary statistics for usable features
        stats_rows = []
        for feat in USABLE_FEATURES:
            if feat not in eligible.columns:
                continue
            col = eligible[feat]
            # Get quality column
            # Map feature name to quality column — need feature ID mapping
            q_col = _find_quality_column(feat, eligible.columns)

            n_valid = col.notna().sum()
            n_missing = col.isna().sum()
            pct_available = n_valid / len(eligible) * 100

            # Quality breakdown (if quality column exists)
            q_valid = q_degraded = q_unavailable = 0
            if q_col and q_col in eligible.columns:
                q_counts = eligible[q_col].value_counts()
                q_valid = q_counts.get("VALID", 0)
                q_degraded = q_counts.get("DEGRADED", 0)
                q_unavailable = q_counts.get("UNAVAILABLE", 0)

            row = {
                "Feature": feat,
                "Available%": f"{pct_available:.1f}%",
                "Mean": f"{col.mean():.4f}" if n_valid > 0 else "N/A",
                "Std": f"{col.std():.4f}" if n_valid > 0 else "N/A",
                "Min": f"{col.min():.4f}" if n_valid > 0 else "N/A",
                "Max": f"{col.max():.4f}" if n_valid > 0 else "N/A",
                "Skew": f"{col.skew():.2f}" if n_valid > 10 else "N/A",
                "VALID": q_valid, "DEGRADED": q_degraded, "UNAVAILABLE": q_unavailable,
            }
            stats_rows.append(row)

            if n_valid > 0:
                print(f"  {ul}/{feat}: mean={col.mean():.4f}, std={col.std():.4f}, "
                      f"avail={pct_available:.1f}%, skew={col.skew():.2f}")

        # Write summary table
        report_lines.append("| Feature | Available% | Mean | Std | Min | Max | Skew | VALID | DEGRADED | UNAVAIL |")
        report_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in stats_rows:
            report_lines.append(
                f"| {row['Feature']} | {row['Available%']} | {row['Mean']} | "
                f"{row['Std']} | {row['Min']} | {row['Max']} | {row['Skew']} | "
                f"{row['VALID']} | {row['DEGRADED']} | {row['UNAVAILABLE']} |"
            )
        report_lines.append("")

        # Plot distributions for continuous features
        continuous_features = [f for f in USABLE_FEATURES
                               if f in eligible.columns
                               and eligible[f].dtype in ['float64', 'float32']
                               and eligible[f].notna().sum() > 10]

        n_feats = len(continuous_features)
        if n_feats > 0:
            ncols = 4
            nrows = (n_feats + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
            axes = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])
            fig.suptitle(f"R1b: Feature Distributions — {ul} (eligible only)", fontsize=13, fontweight='bold')

            for idx, feat in enumerate(continuous_features):
                ax = axes[idx]
                vals = eligible[feat].dropna()
                ax.hist(vals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
                ax.set_title(feat, fontsize=9)
                ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=1, alpha=0.7)
                # Mark 1st/99th percentile
                p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
                ax.axvline(p1, color='orange', linestyle=':', linewidth=1)
                ax.axvline(p99, color='orange', linestyle=':', linewidth=1)

            # Hide unused axes
            for idx in range(n_feats, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            fig.savefig(FIG_DIR / f"R1b_feature_distributions_{ul}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)


def _find_quality_column(feature_name, columns):
    """Map feature name to its quality column."""
    # Feature name → Feature ID mapping
    FEAT_TO_ID = {
        "riv_level": "101", "riv_change_1d": "102", "riv_change_3d": "103",
        "iv_percentile_rank": "105", "iv_term_structure_slope": "106",
        "india_vix_level": "107", "india_vix_percentile_rank": "108",
        "india_vix_change": "109", "iv_skew_25d": "110",
        "iv_rv_spread_5d": "201", "iv_rv_spread_10d": "202",
        "pcr_volume": "401", "pcr_oi": "402", "oi_change_net_1d": "403",
        "volume_zscore_session": "404", "fno_ban_heavyweight_flag": "405",
        "day_of_week": "501", "time_of_day_bucket": "502",
        "days_to_current_expiry": "503", "is_monthly_expiry_week": "504",
        "days_to_next_holiday": "505",
    }
    fid = FEAT_TO_ID.get(feature_name)
    if fid:
        q_col = f"{QUALITY_PREFIX}{fid}"
        if q_col in columns:
            return q_col
    return None


# ===========================================================================
# R1c: Feature-Label Correlation + Mutual Information
# ===========================================================================
def r1c_feature_label_correlation(data, report_lines):
    """Point-biserial correlation + mutual information for each feature vs labels."""
    print("\n" + "="*70)
    print("R1c: Feature-Label Correlation + Mutual Information")
    print("="*70)

    report_lines.append("\n## R1c: Feature-Label Correlation + Mutual Information\n")
    report_lines.append("**Methods:** Point-biserial correlation (linear), Mutual Information (non-linear).\n")

    try:
        from sklearn.feature_selection import mutual_info_classif
        has_sklearn = True
    except ImportError:
        print("  WARNING: scikit-learn not installed. Skipping mutual information.")
        has_sklearn = False

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()

        report_lines.append(f"\n### {ul}\n")

        for target_label in ["label_expansion", "label_compression"]:
            report_lines.append(f"\n**Target: {target_label}**\n")
            report_lines.append("| Feature | Corr (r) | p-value | MI Score | Signal |")
            report_lines.append("|---|---|---|---|---|")

            corr_data = []

            for feat in USABLE_FEATURES:
                if feat not in eligible.columns:
                    continue
                mask = eligible[feat].notna() & eligible[target_label].notna()
                x = eligible.loc[mask, feat].values
                y = eligible.loc[mask, target_label].values

                if len(x) < 30:
                    continue

                # Point-biserial correlation
                try:
                    r, p = scipy_stats.pointbiserialr(y, x)
                except Exception:
                    r, p = 0.0, 1.0

                # Mutual information
                mi = 0.0
                if has_sklearn and len(x) > 50:
                    try:
                        mi = mutual_info_classif(
                            x.reshape(-1, 1), y.astype(int),
                            discrete_features=False, random_state=42, n_neighbors=5
                        )[0]
                    except Exception:
                        mi = 0.0

                # Signal classification
                signal = "🟢 Strong" if (abs(r) > 0.15 or mi > 0.05) else \
                         "🟡 Moderate" if (abs(r) > 0.08 or mi > 0.02) else \
                         "⚪ Weak"

                corr_data.append((feat, r, p, mi, signal))
                report_lines.append(
                    f"| {feat} | {r:.4f} | {p:.4f} | {mi:.4f} | {signal} |"
                )

            report_lines.append("")

            # Plot correlation bar chart
            if corr_data:
                corr_data_sorted = sorted(corr_data, key=lambda x: abs(x[1]), reverse=True)
                feats = [c[0] for c in corr_data_sorted]
                corrs = [c[1] for c in corr_data_sorted]
                mis = [c[3] for c in corr_data_sorted]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"R1c: Feature-Label Relationships — {ul} / {target_label}",
                             fontsize=13, fontweight='bold')

                colors = ['#e74c3c' if c > 0 else '#3498db' for c in corrs]
                ax1.barh(range(len(feats)), corrs, color=colors, edgecolor='white')
                ax1.set_yticks(range(len(feats)))
                ax1.set_yticklabels(feats, fontsize=8)
                ax1.set_xlabel("Point-Biserial Correlation")
                ax1.set_title("Linear Correlation")
                ax1.axvline(0, color='black', linewidth=0.5)
                ax1.invert_yaxis()

                ax2.barh(range(len(feats)), mis, color='#27ae60', edgecolor='white')
                ax2.set_yticks(range(len(feats)))
                ax2.set_yticklabels(feats, fontsize=8)
                ax2.set_xlabel("Mutual Information (bits)")
                ax2.set_title("Non-Linear Relationship (MI)")
                ax2.invert_yaxis()

                plt.tight_layout()
                label_short = target_label.replace("label_", "")
                fig.savefig(FIG_DIR / f"R1c_correlation_{ul}_{label_short}.png",
                            dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"  {ul}/{target_label}: top features by |r|: "
                  f"{', '.join(c[0] + f'({c[1]:.3f})' for c in sorted(corr_data, key=lambda x: abs(x[1]), reverse=True)[:5])}")


# ===========================================================================
# R1d: Multicollinearity (Pairwise Correlation + VIF)
# ===========================================================================
def r1d_multicollinearity(data, report_lines):
    """Pairwise correlation matrix + VIF for usable features."""
    print("\n" + "="*70)
    print("R1d: Multicollinearity Assessment (OV-R-6)")
    print("="*70)

    report_lines.append("\n## R1d: Multicollinearity Assessment (OV-R-6)\n")
    report_lines.append("**Spec Reference:** §5.4.1 — Flag pairs with |correlation| > 0.85.\n")

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()

        # Get available continuous features
        avail_feats = [f for f in USABLE_FEATURES if f in eligible.columns
                       and eligible[f].notna().sum() > 50]

        feat_matrix = eligible[avail_feats].dropna()

        report_lines.append(f"\n### {ul} (n={len(feat_matrix)} complete rows)\n")

        # Pairwise correlation
        corr_matrix = feat_matrix.corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(avail_feats)))
        ax.set_yticks(range(len(avail_feats)))
        ax.set_xticklabels(avail_feats, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(avail_feats, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"R1d: Feature Correlation Matrix — {ul}", fontsize=13, fontweight='bold')

        # Annotate high correlations
        for i in range(len(avail_feats)):
            for j in range(len(avail_feats)):
                if i != j and abs(corr_matrix.iloc[i, j]) > 0.5:
                    ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                            ha='center', va='center', fontsize=6,
                            color='white' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'black')

        plt.tight_layout()
        fig.savefig(FIG_DIR / f"R1d_correlation_matrix_{ul}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Flag high-correlation pairs (|r| > 0.85)
        flagged = []
        for i in range(len(avail_feats)):
            for j in range(i + 1, len(avail_feats)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.85:
                    flagged.append((avail_feats[i], avail_feats[j], r))

        if flagged:
            report_lines.append("**⚠️ Flagged pairs (|r| > 0.85, per §5.4.1):**\n")
            report_lines.append("| Feature A | Feature B | Correlation |")
            report_lines.append("|---|---|---|")
            for a, b, r in flagged:
                report_lines.append(f"| {a} | {b} | {r:.4f} |")
                print(f"  ⚠️ {ul}: {a} ↔ {b} = {r:.4f}")
        else:
            report_lines.append("**✅ No pairs exceed |r| > 0.85 threshold.**\n")
            print(f"  ✅ {ul}: No pairs exceed 0.85 threshold")

        # Also flag moderate pairs (|r| > 0.70) for awareness
        moderate = [(a, b, r) for i in range(len(avail_feats))
                    for j in range(i+1, len(avail_feats))
                    for a, b, r in [(avail_feats[i], avail_feats[j], corr_matrix.iloc[i, j])]
                    if 0.70 < abs(r) <= 0.85]
        if moderate:
            report_lines.append("\n**Notable pairs (0.70 < |r| ≤ 0.85):**\n")
            for a, b, r in moderate:
                report_lines.append(f"- {a} ↔ {b} = {r:.4f}")

        # VIF computation
        report_lines.append(f"\n**Variance Inflation Factor (VIF):**\n")
        try:
            from numpy.linalg import LinAlgError
            # Standardise features for VIF
            X = feat_matrix.values
            X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
            X_with_const = np.column_stack([np.ones(len(X_std)), X_std])

            vif_values = []
            for k in range(1, X_with_const.shape[1]):  # skip constant
                others = [c for c in range(X_with_const.shape[1]) if c != k]
                try:
                    r_sq = 1 - np.var(
                        X_with_const[:, k] - X_with_const[:, others] @
                        np.linalg.lstsq(X_with_const[:, others], X_with_const[:, k], rcond=None)[0]
                    ) / np.var(X_with_const[:, k])
                    vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
                except Exception:
                    vif = float('nan')
                vif_values.append(vif)

            report_lines.append("| Feature | VIF | Flag |")
            report_lines.append("|---|---|---|")
            for feat, vif in zip(avail_feats, vif_values):
                flag = "⚠️ HIGH" if vif > 10 else ("🟡" if vif > 5 else "✅")
                report_lines.append(f"| {feat} | {vif:.2f} | {flag} |")
                if vif > 5:
                    print(f"  VIF {ul}/{feat}: {vif:.2f} {'⚠️ HIGH' if vif > 10 else '🟡 Moderate'}")

        except Exception as e:
            report_lines.append(f"VIF computation failed: {e}\n")
            print(f"  VIF computation failed: {e}")

        report_lines.append("")


# ===========================================================================
# R1e: Temporal Structure + Regime Break Plots
# ===========================================================================
def r1e_temporal_structure(data, report_lines):
    """Plot features and labels over time; identify structural breaks."""
    print("\n" + "="*70)
    print("R1e: Temporal Structure & Regime Breaks (OV-R-7)")
    print("="*70)

    report_lines.append("\n## R1e: Temporal Structure & Regime Breaks (OV-R-7)\n")
    report_lines.append("**Known break:** BANKNIFTY weekly→monthly expiry, November 2024.\n")

    # Key features to plot over time
    temporal_features = [
        "riv_level", "riv_change_1d", "iv_skew_25d",
        "iv_rv_spread_5d", "pcr_volume", "volume_zscore_session",
    ]

    for ul in UNDERLYINGS:
        df = data[ul]
        # Use reference_expiry horizon (e.g., horizon_name for T=2)
        # Aggregate to daily for temporal plots
        daily = df.groupby("trade_date").first().reset_index()

        fig, axes = plt.subplots(len(temporal_features) + 1, 1,
                                 figsize=(16, 3 * (len(temporal_features) + 1)),
                                 sharex=True)
        fig.suptitle(f"R1e: Temporal Structure — {ul}", fontsize=14, fontweight='bold')

        # Plot RIV level with regime annotation
        ax = axes[0]
        if "riv_level" in daily.columns:
            ax.plot(daily["trade_date"], daily["riv_level"], linewidth=0.8, color='steelblue')
            ax.set_ylabel("RIV Level")
            ax.set_title("RIV Level (proxy for IV regime)")
            # Mark BANKNIFTY break
            if ul == "BANKNIFTY":
                break_date = pd.Timestamp("2024-11-01")
                ax.axvline(break_date, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(break_date, ax.get_ylim()[1] * 0.9, " Weekly→Monthly",
                        color='red', fontsize=8, va='top')

        # Plot other features
        for idx, feat in enumerate(temporal_features):
            ax = axes[idx + 1] if idx + 1 < len(axes) else axes[-1]
            if feat in daily.columns:
                vals = daily[feat].dropna()
                dates = daily.loc[vals.index, "trade_date"]
                ax.plot(dates, vals, linewidth=0.7, color='steelblue', alpha=0.8)
                ax.set_ylabel(feat, fontsize=8)
                # Add 20-day rolling mean
                if len(vals) > 20:
                    rolling = vals.rolling(20, min_periods=10).mean()
                    ax.plot(dates, rolling, linewidth=1.5, color='red', alpha=0.6)

                if ul == "BANKNIFTY":
                    ax.axvline(pd.Timestamp("2024-11-01"), color='red',
                               linestyle='--', linewidth=1, alpha=0.5)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"R1e_temporal_structure_{ul}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Daily eligible observation count
        daily_eligible = df[df["training_eligible"] == True].groupby("trade_date").size()
        report_lines.append(f"\n### {ul} — Daily eligible observations\n")
        report_lines.append(f"- Mean per day: {daily_eligible.mean():.1f}")
        report_lines.append(f"- Min per day: {daily_eligible.min()}")
        report_lines.append(f"- Max per day: {daily_eligible.max()}")
        report_lines.append(f"- Days with 0 eligible: {(daily_eligible == 0).sum()}\n")

        print(f"  {ul}: daily eligible mean={daily_eligible.mean():.1f}, "
              f"min={daily_eligible.min()}, max={daily_eligible.max()}")


# ===========================================================================
# R1f: Quality & Exclusion Rate Analysis
# ===========================================================================
def r1f_quality_exclusion(data, report_lines):
    """Check exclusion rates against §2.1.7 guardrail (<15%)."""
    print("\n" + "="*70)
    print("R1f: Quality & Exclusion Rate Analysis (§2.1.7 guardrail)")
    print("="*70)

    report_lines.append("\n## R1f: Quality & Exclusion Rate Analysis\n")
    report_lines.append("**Guardrail:** §2.1.7 — exclusion rate must be <15% per horizon.\n")

    for ul in UNDERLYINGS:
        df = data[ul]
        report_lines.append(f"\n### {ul}\n")

        for horizon in sorted(df["horizon_name"].unique()):
            subset = df[df["horizon_name"] == horizon]
            n_total = len(subset)
            n_eligible = subset["training_eligible"].sum()
            n_excluded = n_total - n_eligible
            excl_rate = n_excluded / n_total * 100

            flag = "🔴 EXCEEDS 15%" if excl_rate > 15 else "✅ OK"

            report_lines.append(f"**{horizon}:** {n_excluded}/{n_total} excluded "
                                f"({excl_rate:.1f}%) — {flag}")

            # Breakdown by exclusion reason
            if "exclusion_reason" in subset.columns:
                reasons = subset[subset["training_eligible"] == False]["exclusion_reason"].value_counts()
                if len(reasons) > 0:
                    report_lines.append("")
                    for reason, count in reasons.items():
                        report_lines.append(f"  - {reason}: {count}")

            print(f"  {ul}/{horizon}: excluded {excl_rate:.1f}% — {flag}")

        # Quality over time: % DEGRADED/UNAVAILABLE per feature per month
        eligible = df[df["training_eligible"] == True].copy()
        eligible["month"] = eligible["trade_date"].dt.to_period("M")

        quality_cols = [c for c in eligible.columns if c.startswith(QUALITY_PREFIX)]
        if quality_cols:
            fig, ax = plt.subplots(figsize=(14, 6))
            monthly_unavail = []
            for month in sorted(eligible["month"].unique()):
                month_data = eligible[eligible["month"] == month]
                for qc in quality_cols:
                    unavail_rate = (month_data[qc] == "UNAVAILABLE").mean()
                    monthly_unavail.append({
                        "month": str(month), "quality_col": qc,
                        "unavail_rate": unavail_rate
                    })

            unavail_df = pd.DataFrame(monthly_unavail)
            pivot = unavail_df.pivot_table(index="month", columns="quality_col",
                                           values="unavail_rate", aggfunc="mean")
            # Only plot features with some variation
            varying = pivot.columns[pivot.std() > 0.01]
            if len(varying) > 0:
                pivot[varying].plot(ax=ax, linewidth=1)
                ax.set_ylabel("UNAVAILABLE Rate")
                ax.set_title(f"R1f: Feature UNAVAILABLE Rate Over Time — {ul}")
                ax.legend(fontsize=6, ncol=3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                fig.savefig(FIG_DIR / f"R1f_quality_over_time_{ul}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        report_lines.append("")


# ===========================================================================
# R1g: Rolling Base Rate Stability
# ===========================================================================
def r1g_rolling_base_rate(data, report_lines):
    """Rolling 60-day base rate of expansion/compression."""
    print("\n" + "="*70)
    print("R1g: Rolling Base Rate Stability")
    print("="*70)

    report_lines.append("\n## R1g: Rolling Base Rate Stability\n")
    report_lines.append("**Method:** 60-trading-day rolling mean of expansion/compression rates.\n")
    report_lines.append("**Red flag:** >10–15% swing between regime periods.\n")

    for ul in UNDERLYINGS:
        df = data[ul]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"R1g: Rolling Base Rate (60-day) — {ul}", fontsize=13, fontweight='bold')

        for j, horizon in enumerate(sorted(df["horizon_name"].unique())):
            eligible = df[(df["horizon_name"] == horizon) & (df["training_eligible"] == True)].copy()
            eligible = eligible.sort_values("trade_date")

            ax = axes[j]
            for label, color, name in [
                ("label_expansion", '#e74c3c', "Expansion"),
                ("label_compression", '#3498db', "Compression"),
            ]:
                rolling = eligible.set_index("trade_date")[label].rolling(60, min_periods=20).mean()
                ax.plot(rolling.index, rolling.values, linewidth=1.5, color=color, label=name)
                # Overall rate
                overall = eligible[label].mean()
                ax.axhline(overall, color=color, linestyle='--', linewidth=0.8, alpha=0.5)

            ax.set_ylabel("Base Rate")
            ax.set_title(f"{horizon}")
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylim(0, 0.5)

            if ul == "BANKNIFTY":
                ax.axvline(pd.Timestamp("2024-11-01"), color='red',
                           linestyle='--', linewidth=1.5, alpha=0.5)
                ax.text(pd.Timestamp("2024-11-01"), 0.45, " BN break",
                        color='red', fontsize=8)

            # Compute max swing
            for label, name in [("label_expansion", "Expansion"), ("label_compression", "Compression")]:
                rolling_vals = eligible.set_index("trade_date")[label].rolling(60, min_periods=20).mean().dropna()
                if len(rolling_vals) > 0:
                    swing = rolling_vals.max() - rolling_vals.min()
                    report_lines.append(f"- {ul}/{horizon}/{name}: "
                                        f"rolling range = {rolling_vals.min():.3f}–{rolling_vals.max():.3f}, "
                                        f"swing = {swing:.3f} "
                                        f"{'⚠️' if swing > 0.15 else '✅'}")
                    print(f"  {ul}/{horizon}/{name}: swing={swing:.3f}")

        plt.tight_layout()
        fig.savefig(FIG_DIR / f"R1g_rolling_base_rate_{ul}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    report_lines.append("")


# ===========================================================================
# R1h: Label Noise Estimation
# ===========================================================================
def r1h_label_noise(data, report_lines):
    """Estimate label noise by counting observations near z-score thresholds."""
    print("\n" + "="*70)
    print("R1h: Label Noise Estimation (z-score threshold proximity)")
    print("="*70)

    report_lines.append("\n## R1h: Label Noise Estimation\n")
    report_lines.append("**Method:** Count observations where |z_score| is in the 'ambiguous zone' "
                        "near the classification threshold.\n")
    report_lines.append("**Implication:** High density near threshold → structural ceiling on accuracy.\n")

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()

        if "z_score" not in eligible.columns:
            report_lines.append(f"\n### {ul}: z_score column not found — skipping.\n")
            continue

        report_lines.append(f"\n### {ul}\n")

        for horizon in sorted(eligible["horizon_name"].unique()):
            subset = eligible[eligible["horizon_name"] == horizon]
            z = subset["z_score"].dropna()

            if len(z) == 0:
                continue

            # The threshold is typically at |z| = 1.5 (expansion/compression boundary)
            # Count observations in ambiguous bands
            bands = [
                (1.0, 1.3, "Safe below threshold"),
                (1.3, 1.5, "Near threshold (below)"),
                (1.5, 1.7, "Near threshold (above)"),
                (1.7, 2.0, "Safe above threshold"),
            ]

            abs_z = z.abs()
            report_lines.append(f"\n**{horizon}** (n={len(z)}):\n")
            report_lines.append("| |z| Band | Count | % | Interpretation |")
            report_lines.append("|---|---|---|---|")

            total_ambiguous = 0
            for low, high, interp in bands:
                count = ((abs_z >= low) & (abs_z < high)).sum()
                pct = count / len(z) * 100
                if 1.3 <= low <= 1.5 or 1.5 <= low <= 1.7:
                    total_ambiguous += count
                report_lines.append(f"| {low}–{high} | {count} | {pct:.1f}% | {interp} |")

            ambig_pct = total_ambiguous / len(z) * 100
            report_lines.append(f"\n**Ambiguous zone (1.3 ≤ |z| < 1.7):** "
                                f"{total_ambiguous} observations ({ambig_pct:.1f}%) — "
                                f"{'⚠️ High noise' if ambig_pct > 30 else '✅ Acceptable'}")

            print(f"  {ul}/{horizon}: {ambig_pct:.1f}% in ambiguous zone "
                  f"{'⚠️' if ambig_pct > 30 else '✅'}")

            # Plot z-score distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(z, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
            # Mark threshold zones
            for threshold in [-1.5, 1.5]:
                ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5)
            ax.axvspan(-1.7, -1.3, alpha=0.15, color='orange', label='Ambiguous zone')
            ax.axvspan(1.3, 1.7, alpha=0.15, color='orange')
            ax.set_xlabel("z-score")
            ax.set_ylabel("Count")
            ax.set_title(f"R1h: z-score Distribution — {ul} / {horizon}")
            ax.legend()
            plt.tight_layout()
            fig.savefig(FIG_DIR / f"R1h_zscore_dist_{ul}_{horizon}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    report_lines.append("")


# ===========================================================================
# R1i: Information Decay Check (SHOULD — rolling logistic model)
# ===========================================================================
def r1i_information_decay(data, report_lines):
    """Train simple logistic on first 12 months, evaluate on next 6, roll forward."""
    print("\n" + "="*70)
    print("R1i: Information Decay Check [SHOULD]")
    print("="*70)

    report_lines.append("\n## R1i: Information Decay Check\n")
    report_lines.append("**Method:** Train logistic regression on first 12 months, "
                        "evaluate on next 6 months, roll forward.\n")
    report_lines.append("**Red flag:** AUC collapses after first roll → signal is regime-fragile.\n")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
    except ImportError:
        report_lines.append("**SKIPPED:** scikit-learn not installed.\n")
        print("  SKIPPED: scikit-learn not installed")
        return

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()
        eligible = eligible.sort_values("trade_date")

        report_lines.append(f"\n### {ul}\n")

        # Use first horizon only for simplicity
        first_horizon = sorted(eligible["horizon_name"].unique())[0]
        subset = eligible[eligible["horizon_name"] == first_horizon].copy()

        # Prepare features (drop NaN rows)
        avail_feats = [f for f in USABLE_FEATURES if f in subset.columns]
        complete = subset.dropna(subset=avail_feats)

        if len(complete) < 100:
            report_lines.append("Insufficient data for information decay check.\n")
            continue

        dates = complete["trade_date"]
        min_date = dates.min()
        max_date = dates.max()

        # Define rolling windows: 12 months train, 6 months test, roll by 3 months
        results = []
        train_months = 12
        test_months = 6
        roll_months = 3

        for target in ["label_expansion", "label_compression"]:
            window_start = min_date
            window_results = []

            while True:
                train_end = window_start + pd.DateOffset(months=train_months)
                test_end = train_end + pd.DateOffset(months=test_months)

                train_mask = (dates >= window_start) & (dates < train_end)
                test_mask = (dates >= train_end) & (dates < test_end)

                X_train = complete.loc[train_mask, avail_feats].values
                y_train = complete.loc[train_mask, target].values
                X_test = complete.loc[test_mask, avail_feats].values
                y_test = complete.loc[test_mask, target].values

                if len(X_train) < 50 or len(X_test) < 20:
                    break
                if y_train.sum() < 5 or y_test.sum() < 3:
                    window_start += pd.DateOffset(months=roll_months)
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                lr.fit(X_train_s, y_train)

                try:
                    y_prob = lr.predict_proba(X_test_s)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                except Exception:
                    auc = 0.5

                window_results.append({
                    "train_period": f"{window_start.strftime('%Y-%m')} → {train_end.strftime('%Y-%m')}",
                    "test_period": f"{train_end.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}",
                    "n_train": len(X_train), "n_test": len(X_test),
                    "auc": auc,
                })

                window_start += pd.DateOffset(months=roll_months)

            if window_results:
                report_lines.append(f"\n**Target: {target}** ({first_horizon}):\n")
                report_lines.append("| Train Period | Test Period | n_train | n_test | AUC |")
                report_lines.append("|---|---|---|---|---|")
                for wr in window_results:
                    report_lines.append(
                        f"| {wr['train_period']} | {wr['test_period']} | "
                        f"{wr['n_train']} | {wr['n_test']} | {wr['auc']:.3f} |"
                    )

                aucs = [wr['auc'] for wr in window_results]
                decay = aucs[0] - aucs[-1] if len(aucs) > 1 else 0
                report_lines.append(f"\nAUC decay (first→last): {decay:.3f} "
                                    f"{'⚠️ Signal decays' if decay > 0.05 else '✅ Stable'}")
                print(f"  {ul}/{target}: AUC sequence = {[f'{a:.3f}' for a in aucs]}, "
                      f"decay={decay:.3f}")

    report_lines.append("")


# ===========================================================================
# R1j: Label Temporal Autocorrelation
# ===========================================================================
def r1j_label_autocorrelation(data, report_lines):
    """Compute autocorrelation of labels to check for temporal dependence."""
    print("\n" + "="*70)
    print("R1j: Label Temporal Autocorrelation [SHOULD]")
    print("="*70)

    report_lines.append("\n## R1j: Label Temporal Autocorrelation\n")
    report_lines.append("**Purpose:** Check if consecutive labels are quasi-duplicates "
                        "(inflates apparent walk-forward performance).\n")

    for ul in UNDERLYINGS:
        df = data[ul]
        eligible = df[df["training_eligible"] == True].copy()

        report_lines.append(f"\n### {ul}\n")

        for horizon in sorted(eligible["horizon_name"].unique()):
            subset = eligible[eligible["horizon_name"] == horizon].sort_values("trade_date")

            for target in ["label_expansion", "label_compression"]:
                series = subset[target].values
                if len(series) < 50:
                    continue

                # Compute autocorrelation at lags 1–10
                acfs = []
                for lag in range(1, 11):
                    if lag < len(series):
                        acf = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        acfs.append((lag, acf))

                if acfs:
                    report_lines.append(f"\n**{horizon} / {target}:**")
                    lag1 = acfs[0][1]
                    lag5 = acfs[4][1] if len(acfs) > 4 else 0
                    report_lines.append(f"  Lag-1 autocorr: {lag1:.3f}, "
                                        f"Lag-5: {lag5:.3f} "
                                        f"{'⚠️ High persistence' if lag1 > 0.3 else '✅'}")

                    print(f"  {ul}/{horizon}/{target}: lag-1={lag1:.3f}, lag-5={lag5:.3f}")

        # Plot autocorrelation for expansion labels
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"R1j: Label Autocorrelation — {ul}", fontsize=13, fontweight='bold')

        for j, horizon in enumerate(sorted(eligible["horizon_name"].unique())):
            if j >= 2:
                break
            subset = eligible[eligible["horizon_name"] == horizon].sort_values("trade_date")
            ax = axes[j]
            for target, color, name in [
                ("label_expansion", '#e74c3c', "Expansion"),
                ("label_compression", '#3498db', "Compression"),
            ]:
                series = subset[target].values
                lags = range(1, min(21, len(series)))
                acfs = [np.corrcoef(series[:-l], series[l:])[0, 1] for l in lags]
                ax.bar([l - 0.15 if name == "Expansion" else l + 0.15 for l in lags],
                       acfs, width=0.3, color=color, alpha=0.7, label=name)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axhline(2/np.sqrt(len(series)), color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(-2/np.sqrt(len(series)), color='gray', linestyle='--', linewidth=0.8)
            ax.set_xlabel("Lag (trading days)")
            ax.set_ylabel("Autocorrelation")
            ax.set_title(horizon)
            ax.legend()

        plt.tight_layout()
        fig.savefig(FIG_DIR / f"R1j_label_autocorrelation_{ul}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    report_lines.append("")


# ===========================================================================
# Report Generation
# ===========================================================================
def generate_report(report_lines):
    """Write the consolidated R1 EDA report."""
    header = [
        "# Oracle-V Model Research — R1: Exploratory Data Analysis Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Phase:** Model Research, Sub-phase R1",
        f"**Spec References:** Part IV §7.1.1, Governance §10.2, OV-R-5/6/7\n",
        "---",
    ]

    full_report = "\n".join(header + report_lines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(full_report)

    print(f"\n{'='*70}")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Figures saved: {FIG_DIR}/")
    print(f"{'='*70}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("="*70)
    print("Oracle-V Model Research — R1: Exploratory Data Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    setup_output_dirs()
    data = load_training_data()

    report_lines = []

    # MUST tasks
    r1a_label_distribution(data, report_lines)
    r1b_feature_distributions(data, report_lines)
    r1c_feature_label_correlation(data, report_lines)
    r1d_multicollinearity(data, report_lines)
    r1e_temporal_structure(data, report_lines)
    r1f_quality_exclusion(data, report_lines)
    r1g_rolling_base_rate(data, report_lines)
    r1h_label_noise(data, report_lines)

    # SHOULD tasks
    r1i_information_decay(data, report_lines)
    r1j_label_autocorrelation(data, report_lines)

    # Generate report
    report_lines.append("\n---\n")
    report_lines.append("## Deferred Features (Excluded from Modelling)\n")
    report_lines.append("The following features are always UNAVAILABLE and are "
                        "excluded from all R1 analyses and subsequent modelling:\n")
    for feat in DEFERRED_FEATURES:
        report_lines.append(f"- `{feat}`")
    report_lines.append("\nThese will be included when their data sources are acquired "
                        "(India VIX download, F&O ban list integration).\n")

    generate_report(report_lines)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
