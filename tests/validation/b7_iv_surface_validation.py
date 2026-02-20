"""
B7 End-to-End Validation — IV Surface Quality Report

Phase: B7 — IV Surface Validation
Purpose: Formal quality sign-off on B5 IV computation output.

Checks performed:
    1. Convergence analysis (overall + per-day distribution)
    2. ATM IV range and call-put divergence
    3. Theoretical settlement monitoring (zero-volume)
    4. Dividend yield sanity
    5. Price round-trip error + put-call parity + surface smoothness
    6. Metadata consistency (RFR version, staleness)

Usage:
    python tests/validation/b7_iv_surface_validation.py

Output: Prints full validation report to stdout.

Spec References:
    - Data Contract §3.5 (RIV computation, call-put divergence <= 2.0 ppts)
    - Data Contract §3.7 (IV extraction convention)
    - TDR-ORACLE-008 §2.4 (IV computation parameters)
    - B5 Design Document (convergence target > 90% of eligible)
"""

import sys
sys.path.insert(0, "src")

import boto3, io, json, math
import pandas as pd
import numpy as np

S3_BUCKET = "oracle-data-lake-644701781379-ap-south-1"
S3_REGION = "ap-south-1"
UNDERLYINGS = ["NIFTY", "BANKNIFTY"]


def load_iv_parquet(s3, bucket, underlying, date_str):
    y, m = date_str[:4], date_str[4:6]
    key = (f"computed/iv_surfaces/underlying={underlying}/"
           f"year={y}/month={m}/{underlying}_iv_{date_str}.parquet")
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except:
        return None


def run_validation():
    s3 = boto3.client("s3", region_name=S3_REGION)

    print("=" * 70)
    print("B7 VALIDATION — ORACLE IV SURFACE END-TO-END QUALITY REPORT")
    print("=" * 70)

    for underlying in UNDERLYINGS:
        print(f"\n{'='*70}")
        print(f"  {underlying}")
        print(f"{'='*70}")

        # Gather all summaries
        paginator = s3.get_paginator("list_objects_v2")
        summary_keys = []
        for page in paginator.paginate(Bucket=S3_BUCKET,
                Prefix=f"computed/iv_surfaces/underlying={underlying}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    summary_keys.append(obj["Key"])

        summaries = []
        for sk in sorted(summary_keys):
            resp = s3.get_object(Bucket=S3_BUCKET, Key=sk)
            summaries.append(json.loads(resp["Body"].read()))

        # --- 1. Convergence ---
        print(f"\n--- 1. CONVERGENCE ANALYSIS ({len(summaries)} days) ---")
        total_opts = sum(s["total_options"] for s in summaries)
        total_conv = sum(s["converged"] for s in summaries)
        total_expday = sum(s.get("expiry_day", 0) for s in summaries)
        total_oom = sum(s.get("out_of_moneyness", 0) for s in summaries)
        total_pricelo = sum(s.get("price_too_low", 0) for s in summaries)
        total_below = sum(s.get("below_intrinsic", 0) for s in summaries)
        total_failed = sum(s.get("solver_failed", 0) for s in summaries)
        total_notconv = sum(s.get("not_converged", 0) for s in summaries)
        total_extreme = sum(s.get("extreme_value", 0) for s in summaries)

        iv_eligible = total_opts - total_expday - total_oom - total_pricelo
        eligible_rate = total_conv / max(iv_eligible, 1) * 100

        print(f"  Total options:           {total_opts:>10,}")
        print(f"  Pre-filtered out:")
        print(f"    Expiry-day:            {total_expday:>10,}")
        print(f"    Out-of-moneyness:      {total_oom:>10,}")
        print(f"    Price too low:         {total_pricelo:>10,}")
        print(f"  IV-eligible:             {iv_eligible:>10,}")
        print(f"  Converged:               {total_conv:>10,} ({eligible_rate:.1f}% of eligible)")
        print(f"  Below intrinsic:         {total_below:>10,}")
        print(f"  Solver failed:           {total_failed:>10,}")
        print(f"  Not converged:           {total_notconv:>10,}")
        print(f"  Extreme value:           {total_extreme:>10,}")

        daily_rates = []
        for s in summaries:
            elig = (s["total_options"] - s.get("expiry_day", 0)
                    - s.get("out_of_moneyness", 0) - s.get("price_too_low", 0))
            if elig > 0:
                daily_rates.append(s["converged"] / elig * 100)
        daily_rates.sort()
        print(f"\n  Per-day eligible convergence rate:")
        print(f"    Min:    {daily_rates[0]:.1f}%")
        print(f"    P5:     {daily_rates[int(len(daily_rates)*0.05)]:.1f}%")
        print(f"    P25:    {daily_rates[int(len(daily_rates)*0.25)]:.1f}%")
        print(f"    Median: {daily_rates[len(daily_rates)//2]:.1f}%")
        print(f"    P75:    {daily_rates[int(len(daily_rates)*0.75)]:.1f}%")
        print(f"    P95:    {daily_rates[int(len(daily_rates)*0.95)]:.1f}%")
        print(f"    Max:    {daily_rates[-1]:.1f}%")
        low_days = [r for r in daily_rates if r < 85]
        print(f"    Days below 85%: {len(low_days)}/{len(daily_rates)}")

        # --- 2. ATM IV ---
        print(f"\n--- 2. ATM IV ANALYSIS ---")
        atm_mids = [(s["trade_date"], s["atm_iv_midpoint"] * 100)
                    for s in summaries if s.get("atm_iv_midpoint") is not None]
        if atm_mids:
            ivs = sorted([x[1] for x in atm_mids])
            print(f"  Days with ATM IV:  {len(atm_mids)}/{len(summaries)}")
            print(f"  Min:    {ivs[0]:.2f}%")
            print(f"  P5:     {ivs[int(len(ivs)*0.05)]:.2f}%")
            print(f"  P25:    {ivs[int(len(ivs)*0.25)]:.2f}%")
            print(f"  Median: {ivs[len(ivs)//2]:.2f}%")
            print(f"  P75:    {ivs[int(len(ivs)*0.75)]:.2f}%")
            print(f"  P95:    {ivs[int(len(ivs)*0.95)]:.2f}%")
            print(f"  Max:    {ivs[-1]:.2f}%")

        cp_divs = []
        for s in summaries:
            c, p = s.get("atm_iv_call"), s.get("atm_iv_put")
            if c is not None and p is not None:
                cp_divs.append(abs(c - p) * 100)
        if cp_divs:
            cp_divs.sort()
            violations = [d for d in cp_divs if d > 2.0]
            print(f"\n  ATM call-put IV divergence (spec: <= 2.0 ppts):")
            print(f"    Median: {cp_divs[len(cp_divs)//2]:.2f} ppts")
            print(f"    P95:    {cp_divs[int(len(cp_divs)*0.95)]:.2f} ppts")
            print(f"    Max:    {cp_divs[-1]:.2f} ppts")
            print(f"    Days > 2.0 ppts: {len(violations)}/{len(cp_divs)}")

        # --- 3. Theoretical settlements ---
        print(f"\n--- 3. THEORETICAL SETTLEMENT MONITORING ---")
        total_zvol = sum(s.get("zero_volume_count", 0) for s in summaries)
        total_zvol_atm = sum(s.get("zero_volume_atm_count", 0) for s in summaries)
        print(f"  Total zero-volume records:     {total_zvol:,}")
        print(f"  Total zero-volume ATM records: {total_zvol_atm:,}")
        print(f"  Zero-vol as % of total:        {total_zvol/max(total_opts,1)*100:.1f}%")

        # --- 4. Dividend yield ---
        print(f"\n--- 4. DIVIDEND YIELD ANALYSIS ---")
        divs = sorted([s["dividend_yield"] * 100 for s in summaries
                       if s.get("dividend_yield") is not None])
        div_missing = sum(1 for s in summaries
                         if s.get("dividend_yield_status") == "MISSING")
        print(f"  Range:   {divs[0]:.3f}% - {divs[-1]:.3f}%")
        print(f"  Median:  {divs[len(divs)//2]:.3f}%")
        print(f"  Days with MISSING status: {div_missing}")

        # --- 5. Detailed sampled checks ---
        print(f"\n--- 5. PUT-CALL PARITY & PRICE ERROR (sampled 10 days) ---")
        sample_indices = [int(i * len(summary_keys) / 10) for i in range(10)]
        parity_violations = 0
        parity_checks = 0
        max_price_err = 0.0
        total_price_checks = 0
        spike_count = 0

        for idx in sample_indices:
            sk = sorted(summary_keys)[idx]
            fname = sk.split("/")[-1]
            date_str = fname.replace(f"{underlying}_iv_", "").replace("_summary.json", "")
            df = load_iv_parquet(s3, S3_BUCKET, underlying, date_str)
            if df is None or len(df) == 0:
                continue
            conv = df[df["iv_status"] == "CONVERGED"].copy()
            if len(conv) == 0:
                continue

            errs = conv["price_error"].dropna()
            if len(errs) > 0:
                max_price_err = max(max_price_err, errs.max())
                total_price_checks += len(errs)

            traded = conv[conv["traded"] == True]
            for expiry in traded["expiry_date"].unique():
                exp_df = traded[traded["expiry_date"] == expiry]
                calls = exp_df[exp_df["option_type"] == "CE"].set_index("strike")
                puts = exp_df[exp_df["option_type"] == "PE"].set_index("strike")
                common = calls.index.intersection(puts.index)
                for k in common:
                    c_row, p_row = calls.loc[k], puts.loc[k]
                    S, K = c_row["underlying_price"], k
                    T = c_row["time_to_expiry_years"]
                    r, q = c_row["risk_free_rate"], c_row["dividend_yield"]
                    if T <= 0:
                        continue
                    lhs = c_row["settlement_price"] - p_row["settlement_price"]
                    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
                    parity_err = abs(lhs - rhs)
                    tol = max(1.0, 0.005 * S)
                    parity_checks += 1
                    if parity_err > tol:
                        parity_violations += 1

            for expiry in conv["expiry_date"].unique():
                for otype in ["CE", "PE"]:
                    subset = conv[(conv["expiry_date"] == expiry) &
                                 (conv["option_type"] == otype)].sort_values("strike")
                    if len(subset) < 2:
                        continue
                    ivv = subset["iv"].values
                    for i in range(1, len(ivv)):
                        if not (np.isnan(ivv[i]) or np.isnan(ivv[i-1])):
                            if abs(ivv[i] - ivv[i-1]) > 0.15:
                                spike_count += 1

        print(f"  Price round-trip error:")
        print(f"    Max across sampled days:   {max_price_err:.6f}")
        print(f"    Records checked:           {total_price_checks:,}")
        print(f"  Put-call parity (tol = max(1.0, 0.5% of spot)):")
        print(f"    Pairs checked:             {parity_checks:,}")
        print(f"    Violations:                {parity_violations}")
        print(f"    Violation rate:            {parity_violations/max(parity_checks,1)*100:.2f}%")
        print(f"  Surface smoothness (>15 ppts between adjacent strikes):")
        print(f"    Spikes detected:           {spike_count}")

        # --- 6. Metadata ---
        print(f"\n--- 6. METADATA CONSISTENCY ---")
        rfr_versions = set(s.get("rfr_version_id") for s in summaries)
        stale_days = sum(1 for s in summaries if s.get("rfr_is_stale"))
        print(f"  RFR versions used:    {rfr_versions}")
        print(f"  Stale-rate days:      {stale_days}")

    print(f"\n{'='*70}")
    print("B7 VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_validation()
