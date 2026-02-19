# TDR-ORACLE-008: Historical Data Storage Architecture

| Field | Value |
|---|---|
| **TDR ID** | TDR-ORACLE-008 |
| **Title** | Historical Data Storage Architecture |
| **Status** | APPROVED |
| **Date** | 2026-02-19 |
| **Author** | Dilip Astik |
| **Supersedes** | Partially supersedes TDR-ORACLE-003 (IV Source Convention) |
| **Affects** | Architecture §5.6, §8.1, §8.5; Data Contract §3.7, §4.1; Governance §17.3 |

---

## 1. Context

Oracle-V requires 5 years of historical option chain data (Jan 2021 – Feb 2026) to train volatility prediction models. A Kite Connect API assessment conducted on 2026-02-19 established the following hard constraints:

**Kite Connect Historical API limitations:**
- Provides OHLCV candles only (no IV, no Greeks, no bid-ask spread)
- Requires `instrument_token`, which is only available for **currently listed** instruments
- Expired option contracts are inaccessible — tokens are invalidated post-expiry
- Per-instrument queries only — no chain-snapshot capability
- OI is available via `oi=1` parameter, but only for active instruments

**Conclusion:** Kite Connect cannot serve as Oracle-V's historical backfill source. NSE Bhavcopy (daily F&O settlement data) is the correct historical anchor — it provides authoritative exchange-defined settlement prices, open interest, and volume for all strikes and expiries, including expired contracts, going back well beyond 5 years.

This decision also invalidates Data Contract §3.7's default assumption that IVs would be sourced from broker-provided feeds (Kite Connect). IV must instead be computed internally from settlement prices using a controlled Black-Scholes implementation.

---

## 2. Decision

### 2.1 Two-Layer Storage Architecture

Oracle adopts a hybrid storage pattern:

| Layer | Technology | Content | Access Pattern |
|---|---|---|---|
| **Data Lake (raw)** | S3 | NSE Bhavcopy archives, parsed canonical datasets, IV computation outputs | Write-once, read-for-processing |
| **Operational Store** | DynamoDB | Feature Store, Prediction Log, Governance, Model Registry, all §8.1 tables | High-frequency reads (inference), transactional writes via ODAL |

**Architectural principle:** Raw historical data and intermediate computation artifacts belong in S3. DynamoDB stores only Oracle's operational state — computed features, predictions, governance records, and audit trails.

### 2.2 S3 Bucket Design

**Bucket name:** `oracle-data-lake-{account_id}-ap-south-1`

**Bucket configuration:**
- Region: ap-south-1 (Mumbai) — co-located with DynamoDB and EC2
- Versioning: Enabled (write-once semantics, supports audit/rollback)
- Encryption: SSE-S3 (AES-256, server-side)
- Public access: Blocked (all four Block Public Access settings enabled)
- Lifecycle: No automatic deletion — all data retained until explicit governance decision
- Tags: `Framework=Oracle`, `Environment=Development`, `Component=DataLake`

**Prefix structure:**
```
oracle-data-lake-{account_id}-ap-south-1/
├── raw/
│   ├── bhavcopy/
│   │   ├── fo/                          # F&O Bhavcopy CSVs
│   │   │   ├── year=2021/
│   │   │   │   ├── month=01/
│   │   │   │   │   ├── fo_bhavcopy_20210104.csv.gz
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── index/                       # Index Bhavcopy (NIFTY/BANKNIFTY spot)
│   │   │   └── year=YYYY/month=MM/...
│   │   └── vix/                         # India VIX historical
│   │       └── year=YYYY/month=MM/...
│   └── ingestion_manifest/              # What was downloaded, when, checksum
│       └── manifest_YYYYMMDD.json
├── canonical/
│   ├── option_chains/                   # Parsed, normalised option chain data
│   │   └── underlying=NIFTY/
│   │       └── year=2021/
│   │           └── month=01/
│   │               └── NIFTY_20210104.parquet
│   └── futures/                         # Parsed futures data
│       └── underlying=NIFTY/...
├── computed/
│   ├── iv_surfaces/                     # Computed IV per strike/expiry/day
│   │   └── underlying=NIFTY/
│   │       └── year=2021/...
│   └── realised_vol/                    # Realised volatility series
│       └── underlying=NIFTY/...
├── datasets/
│   └── oracle_v/                        # Assembled training datasets
│       └── version=v001/
│           ├── dataset_metadata.json
│           ├── train.parquet
│           └── validation.parquet
└── archive/                             # DynamoDB archival per §8.5
    └── ...
```

### 2.3 Oracle_Raw_Market_Data Table — Repurposed as Metadata Index

The `Oracle_Raw_Market_Data` DynamoDB table (already created per Architecture §8.1) is repurposed as a **metadata index** for S3 objects — not a store for individual option records.

| Attribute | Description | Example |
|---|---|---|
| `partition_key` (PK) | `{source}#{underlying}` | `BHAVCOPY_FO#NIFTY` |
| `sort_key` (SK) | `{date}` (ISO 8601) | `2025-03-15` |
| `s3_uri` | Full S3 path to raw file | `s3://oracle-data-lake-.../raw/bhavcopy/fo/year=2025/month=03/fo_bhavcopy_20250315.csv.gz` |
| `s3_canonical_uri` | Path to parsed canonical file | `s3://...canonical/option_chains/...` |
| `file_checksum` | SHA-256 of the raw file | `a1b2c3...` |
| `record_count` | Number of records in the raw file | `4523` |
| `nifty_strikes` | Count of NIFTY option strikes | `198` |
| `banknifty_strikes` | Count of BANKNIFTY option strikes | `142` |
| `ingestion_timestamp` | When the file was ingested | `2026-02-19T14:00:00Z` |
| `processing_status` | `RAW` → `PARSED` → `IV_COMPUTED` → `FEATURES_READY` | `IV_COMPUTED` |
| `quality_flags` | Data quality issues detected | `[]` |

### 2.4 IV Source Convention — Supersedes TDR-ORACLE-003

**Original assumption (Data Contract §3.7):** IVs sourced from broker-provided feed (Kite Connect).

**Revised decision:** Oracle computes IV internally from settlement prices using controlled Black-Scholes.

**Rationale:**
1. Kite Connect historical API does not provide IV
2. NSE Bhavcopy provides settlement prices, not IV
3. Oracle-V's append-only observational philosophy requires deterministic, reproducible IV
4. NIFTY and BANKNIFTY options are European-style — Black-Scholes is appropriate
5. Oracle controls the methodology, avoiding vendor lock-in

**IV computation parameters:**

| Parameter | v1 Default | Source |
|---|---|---|
| Pricing model | Black-Scholes (European) | Standard for NIFTY/BANKNIFTY |
| Risk-free rate | RBI 91-day T-bill rate | Weekly RBI auction results |
| Dividend yield | Implied from futures basis | `(F - S) / S` annualised |
| Time to expiry | Calendar days / 365 | Standard annualisation |

### 2.5 Historical Backfill Source and Timeframe

| Parameter | Decision |
|---|---|
| **Primary source** | NSE F&O Bhavcopy (daily settlement data) |
| **Supplementary sources** | NSE Index Bhavcopy (spot), India VIX historical, Event Calendar |
| **Target timeframe** | 5 years (January 2021 – February 2026) |
| **Minimum viable** | 3 years (January 2023 – February 2026) |
| **Validation approach** | 1 year proof-of-concept (2025) first, then extend |
| **Granularity** | Daily EOD (settlement prices) |
| **Label alignment** | EOD for v1; intraday deferred to v2 |

### 2.6 Forward Collection — Deferred to Phase C

Live Kite Connect quote-based snapshots deferred until Oracle-V enters live inference.

---

## 3. ODAL Governance for S3 Writes

| Guarantee | DynamoDB (via ODAL) | S3 (new pattern) |
|---|---|---|
| Write-once / immutable | Conditional PutItem | S3 versioning + no-delete bucket policy |
| Audit trail | ODAL audit log | Manifest in S3 + metadata record in DynamoDB via ODAL |
| Hash integrity | Per-record hash chain | File-level SHA-256 in metadata index |
| Lineage | operation_context | Manifest: source URL, download timestamp, checksum |
| Access control | IAM scoped to Oracle_ | IAM scoped to oracle-data-lake bucket |

---

## 4. Specification Amendment Log

| Document | Section | Change | Type |
|---|---|---|---|
| Data Contract v0.1 | §3.7 | IV source: broker-provided → internally computed Black-Scholes | Override |
| Data Contract v0.1 | §6 #2 | TDR-ORACLE-003 decision revised | Amendment |
| Architecture v0.2 | §5.6 | Oracle_Raw_Market_Data = metadata index for S3 | Refinement |
| Architecture v0.2 | §8.5 | S3 role expanded to include primary raw data storage | Extension |

---

## 5. Decision Approval

| Role | Name | Date | Decision |
|---|---|---|---|
| Framework Operator | Dilip Astik | 2026-02-19 | **APPROVED** |
| Consultant Review | Claude (Anthropic) | 2026-02-19 | RECOMMENDED |
