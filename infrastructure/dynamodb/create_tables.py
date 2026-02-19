"""Create all Oracle DynamoDB tables per Architecture §8.1.

This script is idempotent — it skips tables that already exist.
All tables use PAY_PER_REQUEST billing and are tagged Framework=Oracle.

Key schema sources:
    - Architecture §8.1: Table Taxonomy + Logical Key Structure
    - Governance §4.5: Oracle_Governance_Config schema
    - Data Contract §4.1: Feature Store grain (underlying, ref_timestamp, ref_expiry)

Run: cd /home/ssm-user/oracle && source .venv/bin/activate
     python infrastructure/dynamodb/create_tables.py
"""

import boto3
import sys
from botocore.exceptions import ClientError

REGION = "ap-south-1"
dynamodb = boto3.client("dynamodb", region_name=REGION)

# Standard tags for all Oracle tables
ORACLE_TAGS = [
    {"Key": "Framework", "Value": "Oracle"},
    {"Key": "Environment", "Value": "Development"},
]


def tag(component: str, classification: str) -> list[dict]:
    """Build tag list with component and classification."""
    return ORACLE_TAGS + [
        {"Key": "Component", "Value": component},
        {"Key": "Classification", "Value": classification},
    ]


# --- Table Definitions ---
# Each tuple: (TableName, KeySchema, AttributeDefinitions, Tags)
# KeySchema: HASH = partition key, RANGE = sort key

TABLES = [
    # ============================================================
    # PRODUCTION DOMAIN — Critical Tables (§8.2)
    # ============================================================

    # Prediction Log — Hash-chained, Append-only (§8.1, §8.4)
    # PK: product (ORACLE_V, ORACLE_R, etc.) — per-product hash chains
    # SK: prediction_id (embeds timestamp: YYYYMMDDTHHMMSS_uuid)
    (
        "Oracle_Prediction_Log",
        [
            {"AttributeName": "product", "KeyType": "HASH"},
            {"AttributeName": "prediction_id", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "product", "AttributeType": "S"},
            {"AttributeName": "prediction_id", "AttributeType": "S"},
        ],
        tag("PredictionPipeline", "Critical"),
    ),

    # Trade Log — Hash-chained, Append-only (§8.1, §8.4)
    # PK: atlas_product (ATLAS_V, ATLAS_R, etc.)
    # SK: trade_id (embeds timestamp)
    (
        "Oracle_Trade_Log",
        [
            {"AttributeName": "atlas_product", "KeyType": "HASH"},
            {"AttributeName": "trade_id", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "atlas_product", "AttributeType": "S"},
            {"AttributeName": "trade_id", "AttributeType": "S"},
        ],
        tag("Execution", "Critical"),
    ),

    # Governance Config — Mutable, Audited (§4.5, §8.1)
    # PK: parameter_category, SK: parameter_name (explicit from Governance §4.5)
    (
        "Oracle_Governance_Config",
        [
            {"AttributeName": "parameter_category", "KeyType": "HASH"},
            {"AttributeName": "parameter_name", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "parameter_category", "AttributeType": "S"},
            {"AttributeName": "parameter_name", "AttributeType": "S"},
        ],
        tag("Governance", "Critical"),
    ),

    # Config Audit — Hash-chained, Append-only (§8.1, §15.5)
    # PK: config_key (parameter_category#parameter_name)
    # SK: change_timestamp (ISO 8601, enables temporal ordering for hash chain)
    (
        "Oracle_Config_Audit",
        [
            {"AttributeName": "config_key", "KeyType": "HASH"},
            {"AttributeName": "change_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "config_key", "AttributeType": "S"},
            {"AttributeName": "change_timestamp", "AttributeType": "S"},
        ],
        tag("Governance", "Critical"),
    ),

    # Governance Events — Hash-chained, Append-only (§8.1, §8.4)
    # PK: event_type_partition (e.g., "2026-02" monthly partition)
    # SK: event_id (embeds timestamp)
    (
        "Oracle_Governance_Events",
        [
            {"AttributeName": "event_type_partition", "KeyType": "HASH"},
            {"AttributeName": "event_id", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "event_type_partition", "AttributeType": "S"},
            {"AttributeName": "event_id", "AttributeType": "S"},
        ],
        tag("Governance", "Critical"),
    ),

    # Model Registry — Mutable, Audited (§8.1, §9.2)
    # PK: model_id (e.g., "ORACLE_V_NIFTY_2D")
    # SK: version_timestamp (ISO 8601 — lifecycle state changes append new SK entries)
    (
        "Oracle_Model_Registry",
        [
            {"AttributeName": "model_id", "KeyType": "HASH"},
            {"AttributeName": "version_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "model_id", "AttributeType": "S"},
            {"AttributeName": "version_timestamp", "AttributeType": "S"},
        ],
        tag("ModelLifecycle", "Critical"),
    ),

    # Portfolio State — Mutable, Audited (§8.1)
    # PK: portfolio_id (e.g., "ATLAS_V", "ORACLE_AGGREGATE")
    # SK: snapshot_timestamp (ISO 8601)
    (
        "Oracle_Portfolio_State",
        [
            {"AttributeName": "portfolio_id", "KeyType": "HASH"},
            {"AttributeName": "snapshot_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "portfolio_id", "AttributeType": "S"},
            {"AttributeName": "snapshot_timestamp", "AttributeType": "S"},
        ],
        tag("Portfolio", "Critical"),
    ),

    # ============================================================
    # PRODUCTION DOMAIN — High Criticality Tables (§8.2)
    # ============================================================

    # Raw Market Data — Append-only, not hash-chained (§8.1)
    # PK: underlying#data_date (e.g., "NIFTY#2026-03-01") — daily partitions
    # SK: timestamp#strike#expiry (composite for unique identification)
    (
        "Oracle_Raw_Market_Data",
        [
            {"AttributeName": "partition_key", "KeyType": "HASH"},
            {"AttributeName": "sort_key", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "partition_key", "AttributeType": "S"},
            {"AttributeName": "sort_key", "AttributeType": "S"},
        ],
        tag("Ingestion", "High"),
    ),

    # Feature Store — Append-only, not hash-chained (§8.1)
    # Grain: (underlying, reference_timestamp, reference_expiry) per Data Contract §4.1
    # PK: underlying#reference_expiry (e.g., "NIFTY#2026-03-31")
    # SK: reference_timestamp (ISO 8601 — enables time-range queries per expiry)
    (
        "Oracle_Feature_Store",
        [
            {"AttributeName": "underlying_expiry", "KeyType": "HASH"},
            {"AttributeName": "reference_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "underlying_expiry", "AttributeType": "S"},
            {"AttributeName": "reference_timestamp", "AttributeType": "S"},
        ],
        tag("FeatureStore", "High"),
    ),

    # Feature Registry — Append-only, not hash-chained (§8.1, §6.2)
    # PK: feature_id (e.g., "OV-F-101")
    # SK: registry_version (version of the registration entry)
    (
        "Oracle_Feature_Registry",
        [
            {"AttributeName": "feature_id", "KeyType": "HASH"},
            {"AttributeName": "registry_version", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "feature_id", "AttributeType": "S"},
            {"AttributeName": "registry_version", "AttributeType": "S"},
        ],
        tag("FeatureStore", "High"),
    ),

    # Pipeline State — Mutable, Audited (§8.1)
    # PK: prediction_id (active prediction being tracked through gates)
    # SK: gate_name (which gate this entry represents)
    (
        "Oracle_Pipeline_State",
        [
            {"AttributeName": "prediction_id", "KeyType": "HASH"},
            {"AttributeName": "gate_name", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "prediction_id", "AttributeType": "S"},
            {"AttributeName": "gate_name", "AttributeType": "S"},
        ],
        tag("Pipeline", "High"),
    ),

    # ============================================================
    # PRODUCTION DOMAIN — Medium Criticality Tables (§8.2)
    # ============================================================

    # Ingestion Rejection Log — Append-only, not hash-chained (§8.1)
    # PK: source#date (e.g., "KITE_CONNECT#2026-03-01")
    # SK: rejection_timestamp (ISO 8601)
    (
        "Oracle_Ingestion_Rejection_Log",
        [
            {"AttributeName": "source_date", "KeyType": "HASH"},
            {"AttributeName": "rejection_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "source_date", "AttributeType": "S"},
            {"AttributeName": "rejection_timestamp", "AttributeType": "S"},
        ],
        tag("Ingestion", "Medium"),
    ),

    # Feature Computation Log — Append-only, not hash-chained (§8.1)
    # PK: underlying#date (e.g., "NIFTY#2026-03-01")
    # SK: computation_timestamp (ISO 8601)
    (
        "Oracle_Feature_Computation_Log",
        [
            {"AttributeName": "underlying_date", "KeyType": "HASH"},
            {"AttributeName": "computation_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "underlying_date", "AttributeType": "S"},
            {"AttributeName": "computation_timestamp", "AttributeType": "S"},
        ],
        tag("FeatureStore", "Medium"),
    ),

    # ============================================================
    # RESEARCH DOMAIN (§8.1)
    # ============================================================

    # Training Datasets — Research domain (§8.1)
    # PK: dataset_id (e.g., "ORACLE_V_NIFTY_2D_v1")
    # SK: version_timestamp
    (
        "Oracle_Training_Datasets",
        [
            {"AttributeName": "dataset_id", "KeyType": "HASH"},
            {"AttributeName": "version_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "dataset_id", "AttributeType": "S"},
            {"AttributeName": "version_timestamp", "AttributeType": "S"},
        ],
        tag("Research", "Medium"),
    ),

    # Experiment Log — Research domain (§8.1)
    # PK: experiment_id
    # SK: run_timestamp
    (
        "Oracle_Experiment_Log",
        [
            {"AttributeName": "experiment_id", "KeyType": "HASH"},
            {"AttributeName": "run_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "experiment_id", "AttributeType": "S"},
            {"AttributeName": "run_timestamp", "AttributeType": "S"},
        ],
        tag("Research", "Medium"),
    ),

    # Research Feature Store — Research domain (§8.1)
    # Same grain as production but isolated for experimentation
    (
        "Oracle_Research_Feature_Store",
        [
            {"AttributeName": "underlying_expiry", "KeyType": "HASH"},
            {"AttributeName": "reference_timestamp", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "underlying_expiry", "AttributeType": "S"},
            {"AttributeName": "reference_timestamp", "AttributeType": "S"},
        ],
        tag("Research", "Medium"),
    ),

    # Research Model Artifacts — Research domain (§8.1)
    # PK: model_id, SK: artifact_version
    (
        "Oracle_Research_Model_Artifacts",
        [
            {"AttributeName": "model_id", "KeyType": "HASH"},
            {"AttributeName": "artifact_version", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "model_id", "AttributeType": "S"},
            {"AttributeName": "artifact_version", "AttributeType": "S"},
        ],
        tag("Research", "Medium"),
    ),

    # ============================================================
    # EVENT CALENDAR (Data Contract §4.6, continuation prompt Priority 3)
    # ============================================================

    # Event Calendar — Reference data for event context features
    # PK: event_type (e.g., "RBI_MPC", "FOMC", "BUDGET")
    # SK: event_id (format: YYYYMMDD_event_type_version per Data Contract)
    (
        "Oracle_Event_Calendar",
        [
            {"AttributeName": "event_type", "KeyType": "HASH"},
            {"AttributeName": "event_id", "KeyType": "RANGE"},
        ],
        [
            {"AttributeName": "event_type", "AttributeType": "S"},
            {"AttributeName": "event_id", "AttributeType": "S"},
        ],
        tag("EventCalendar", "High"),
    ),
]


def create_table(table_name: str, key_schema: list, attr_defs: list, tags: list) -> str:
    """Create a DynamoDB table if it doesn't exist. Returns status string."""
    try:
        dynamodb.describe_table(TableName=table_name)
        return "EXISTS"
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    dynamodb.create_table(
        TableName=table_name,
        KeySchema=key_schema,
        AttributeDefinitions=attr_defs,
        BillingMode="PAY_PER_REQUEST",
        Tags=tags,
    )
    waiter = dynamodb.get_waiter("table_exists")
    waiter.wait(TableName=table_name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})
    return "CREATED"


def main() -> None:
    print(f"Creating {len(TABLES)} Oracle DynamoDB tables in {REGION}...")
    print(f"{'Table':<45} {'Status':<10}")
    print("-" * 55)

    created = 0
    existed = 0
    for table_name, key_schema, attr_defs, tags in TABLES:
        status = create_table(table_name, key_schema, attr_defs, tags)
        print(f"{table_name:<45} {status}")
        if status == "CREATED":
            created += 1
        else:
            existed += 1

    print("-" * 55)
    print(f"Done. Created: {created}, Already existed: {existed}, Total: {created + existed}")

    # Verify: list all Oracle_ tables
    paginator = dynamodb.get_paginator("list_tables")
    oracle_tables = []
    for page in paginator.paginate():
        oracle_tables.extend(t for t in page["TableNames"] if t.startswith("Oracle_"))
    print(f"\nAll Oracle_ tables in DynamoDB: {len(oracle_tables)}")
    for t in sorted(oracle_tables):
        print(f"  {t}")


if __name__ == "__main__":
    main()
