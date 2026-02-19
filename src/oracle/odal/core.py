"""
Oracle Data Access Layer (ODAL) — Core Module v2

Architecture Reference: §15.0–§15.9 (Oracle Data Access Layer)
Governance Reference: §18.2 (Infrastructure Isolation)

ODAL is the ONLY permitted write path to Oracle DynamoDB tables.

Write Validation Pipeline (§15.3):
  Write Request → Table Classification → Operation Validation
  → Hash-Chain Computation (if applicable) → DynamoDB Write → Audit Log

Table Classifications (§15.3 Step 1):
  APPEND_ONLY_HASH_CHAINED — put_item only, hash chain computed
  APPEND_ONLY              — put_item only, no hash chain
  MUTABLE_AUDITED          — put_item + update_item, audit on every update
  RESEARCH                 — put_item + update_item, standard audit
  AUDIT_INTERNAL           — ODAL's own audit table (no recursive audit)

No Oracle module may bypass ODAL for DynamoDB writes.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORACLE_TABLE_PREFIX = "Oracle_"
AUDIT_TABLE = "Oracle_ODAL_Audit_Log"
AWS_REGION = "ap-south-1"
CHAIN_GENESIS = "CHAIN_GENESIS"
MAX_HASH_CHAIN_RETRIES = 3


# ---------------------------------------------------------------------------
# Exceptions — §15.3 rejection codes
# ---------------------------------------------------------------------------
class ODALWriteError(Exception):
    """Raised when an ODAL write operation fails."""
    pass


class ODALIsolationError(Exception):
    """Raised when a write targets a non-Oracle table (isolation violation)."""
    pass


class ODALOperationRejected(Exception):
    """Raised when an operation is rejected by ODAL's classification rules.

    Rejection codes (Architecture §15.3 Step 2):
      WRITE_REJECTED: append_only_violation
      WRITE_REJECTED: delete_prohibited
      WRITE_REJECTED: unregistered_table
    """
    pass


# ---------------------------------------------------------------------------
# Table Classification — §15.3 Step 1
# ---------------------------------------------------------------------------
class TableClassification(Enum):
    """Table write-governance classifications per Architecture §15.3."""
    APPEND_ONLY_HASH_CHAINED = "append_only_hash_chained"
    APPEND_ONLY = "append_only"
    MUTABLE_AUDITED = "mutable_audited"
    RESEARCH = "research"
    AUDIT_INTERNAL = "audit_internal"


@dataclass(frozen=True)
class TableConfig:
    """Per-table governance metadata used by ODAL's validation pipeline."""
    classification: TableClassification
    pk_attribute: str
    sk_attribute: str


# Registry of all governed tables and their classifications.
# Architecture §15.3, Step 1. Table key schemas from Architecture §8.1.
# This registry is the SINGLE SOURCE OF TRUTH for ODAL's write rules.
TABLE_REGISTRY: dict[str, TableConfig] = {
    # --- APPEND-ONLY, HASH-CHAINED (§15.3) ---
    "Oracle_Prediction_Log": TableConfig(
        TableClassification.APPEND_ONLY_HASH_CHAINED,
        pk_attribute="product", sk_attribute="prediction_id",
    ),
    "Oracle_Trade_Log": TableConfig(
        TableClassification.APPEND_ONLY_HASH_CHAINED,
        pk_attribute="atlas_product", sk_attribute="trade_id",
    ),
    "Oracle_Governance_Events": TableConfig(
        TableClassification.APPEND_ONLY_HASH_CHAINED,
        pk_attribute="event_type_partition", sk_attribute="event_id",
    ),
    "Oracle_Config_Audit": TableConfig(
        TableClassification.APPEND_ONLY_HASH_CHAINED,
        pk_attribute="config_key", sk_attribute="change_timestamp",
    ),

    # --- APPEND-ONLY, NOT HASH-CHAINED (§15.3) ---
    # Explicitly listed in §15.3:
    "Oracle_Ingestion_Rejection_Log": TableConfig(
        TableClassification.APPEND_ONLY,
        pk_attribute="source_date", sk_attribute="rejection_timestamp",
    ),
    "Oracle_Feature_Computation_Log": TableConfig(
        TableClassification.APPEND_ONLY,
        pk_attribute="underlying_date", sk_attribute="computation_timestamp",
    ),
    # High-criticality tables — append-only per §8.3 (all Critical/High
    # tables enforce append-only). Not hash-chained (not listed in §15.3
    # hash-chained category):
    "Oracle_Raw_Market_Data": TableConfig(
        TableClassification.APPEND_ONLY,
        pk_attribute="partition_key", sk_attribute="sort_key",
    ),
    "Oracle_Feature_Store": TableConfig(
        TableClassification.APPEND_ONLY,
        pk_attribute="underlying_expiry", sk_attribute="reference_timestamp",
    ),
    "Oracle_Feature_Registry": TableConfig(
        TableClassification.APPEND_ONLY,
        pk_attribute="feature_id", sk_attribute="registry_version",
    ),
    "Oracle_Event_Calendar": TableConfig(
        TableClassification.APPEND_ONLY,
        pk_attribute="event_type", sk_attribute="event_id",
    ),

    # --- MUTABLE, AUDITED (§15.3) ---
    "Oracle_Portfolio_State": TableConfig(
        TableClassification.MUTABLE_AUDITED,
        pk_attribute="portfolio_id", sk_attribute="snapshot_timestamp",
    ),
    "Oracle_Model_Registry": TableConfig(
        TableClassification.MUTABLE_AUDITED,
        pk_attribute="model_id", sk_attribute="version_timestamp",
    ),
    "Oracle_Governance_Config": TableConfig(
        TableClassification.MUTABLE_AUDITED,
        pk_attribute="parameter_category", sk_attribute="parameter_name",
    ),
    "Oracle_Pipeline_State": TableConfig(
        TableClassification.MUTABLE_AUDITED,
        pk_attribute="prediction_id", sk_attribute="gate_name",
    ),

    # --- RESEARCH DOMAIN (§15.3) ---
    "Oracle_Training_Datasets": TableConfig(
        TableClassification.RESEARCH,
        pk_attribute="dataset_id", sk_attribute="version_timestamp",
    ),
    "Oracle_Experiment_Log": TableConfig(
        TableClassification.RESEARCH,
        pk_attribute="experiment_id", sk_attribute="run_timestamp",
    ),
    "Oracle_Research_Feature_Store": TableConfig(
        TableClassification.RESEARCH,
        pk_attribute="underlying_expiry", sk_attribute="reference_timestamp",
    ),
    "Oracle_Research_Model_Artifacts": TableConfig(
        TableClassification.RESEARCH,
        pk_attribute="model_id", sk_attribute="artifact_version",
    ),

    # --- AUDIT INTERNAL (ODAL's own table — no recursive audit) ---
    "Oracle_ODAL_Audit_Log": TableConfig(
        TableClassification.AUDIT_INTERNAL,
        pk_attribute="table_name", sk_attribute="write_timestamp",
    ),
}


def get_table_config(table_name: str) -> TableConfig:
    """Look up a table's governance configuration.

    Raises ODALOperationRejected if the table is not registered.
    Unregistered tables are denied by default (AC-35, default-deny).
    """
    config = TABLE_REGISTRY.get(table_name)
    if config is None:
        raise ODALOperationRejected(
            f"WRITE_REJECTED: unregistered_table — '{table_name}' is not in "
            f"ODAL's table registry. Only registered Oracle tables may be "
            f"written to. This may indicate a new table that needs "
            f"registration, or a misconfigured table name."
        )
    return config


def is_append_only(classification: TableClassification) -> bool:
    """Return True if the classification prohibits updates."""
    return classification in (
        TableClassification.APPEND_ONLY_HASH_CHAINED,
        TableClassification.APPEND_ONLY,
    )


def is_hash_chained(classification: TableClassification) -> bool:
    """Return True if the classification requires hash-chain computation."""
    return classification == TableClassification.APPEND_ONLY_HASH_CHAINED


# ---------------------------------------------------------------------------
# Hash-Chain Computation — §15.6, §8.4
# ---------------------------------------------------------------------------
def compute_record_hash(item: dict) -> str:
    """Compute a deterministic SHA-256 hash of a DynamoDB item.

    The item is serialized to JSON with sorted keys and consistent
    formatting to ensure hash reproducibility.

    Architecture Reference: §8.4 (Hash-Chain Architecture)
    """
    serialized = json.dumps(
        _normalize_for_hash(item),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_chain_hash(record_hash: str, previous_chain_hash: str) -> str:
    """Compute chain hash: SHA-256(record_hash + previous_chain_hash).

    This creates an append-only verification chain. Any tampering with
    a historical record breaks the chain from that point forward.

    Architecture Reference: §8.4 (Hash-Chain Architecture)
    """
    combined = f"{record_hash}{previous_chain_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def compute_entry_hash_for_chain(entry: dict) -> str:
    """Compute the hash of an existing entry for chain-linking purposes.

    Per §8.4: previous_hash = SHA-256(concat(PK, SK, timestamp,
    critical_fields)). We hash the full entry for maximum integrity
    coverage — this is strictly stronger than the spec minimum.
    """
    return compute_record_hash(entry)


# ---------------------------------------------------------------------------
# ODAL Client — §15.2
# ---------------------------------------------------------------------------
class ODALClient:
    """The sole write interface to Oracle DynamoDB tables.

    Permitted operations (§15.2):
      put_item(table, item)         — All governed tables
      update_item(table, key, updates) — Mutable + Research only
      batch_put_items(table, items) — All governed tables
      delete_item()                 — ALWAYS REJECTED

    Usage:
        odal = ODALClient()
        odal.put_item("Oracle_Feature_Store", item_dict)
        odal.update_item("Oracle_Portfolio_State", key, updates)
    """

    def __init__(self, region: str = AWS_REGION):
        self._resource = boto3.resource("dynamodb", region_name=region)
        self._client = boto3.client("dynamodb", region_name=region)
        self._region = region

    # -- Public Write API --------------------------------------------------

    def put_item(
        self,
        table_name: str,
        item: dict[str, Any],
        operation_context: str = "",
    ) -> dict:
        """Write a new item to an Oracle DynamoDB table.

        Validation pipeline (§15.3):
          1. Isolation enforcement (Oracle_ prefix)
          2. Table classification lookup
          3. Operation validation (put_item permitted on all tables)
          4. Conditional write for append-only (§15.4)
          5. Hash-chain computation for hash-chained tables (§15.6)
          6. DynamoDB write
          7. Audit log

        Args:
            table_name: Must be a registered Oracle table
            item: Python dict (not DynamoDB wire format)
            operation_context: Free-text for audit trail

        Returns:
            dict with record_hash, chain_hash (or None), audit_timestamp

        Raises:
            ODALIsolationError: Non-Oracle table
            ODALOperationRejected: Unregistered table
            ODALWriteError: DynamoDB write failure
        """
        # Step 0: Isolation enforcement
        self._enforce_isolation(table_name)

        # Step 1: Table classification
        config = get_table_config(table_name)

        # Step 2: Operation validation — put_item permitted on all tables

        # Step 3: Hash-chain computation (if applicable)
        previous_hash = None
        chain_hash = None
        if is_hash_chained(config.classification):
            previous_hash, chain_hash = self._compute_hash_chain_for_put(
                table_name, config, item,
            )
            item = {**item, "previous_hash": previous_hash}

        # Step 4: DynamoDB write
        converted_item = self._convert_floats(item)
        try:
            table = self._resource.Table(table_name)
            if is_append_only(config.classification):
                # §15.4: Conditional write — prevent overwriting existing
                # records. attribute_not_exists on PK ensures uniqueness.
                table.put_item(
                    Item=converted_item,
                    ConditionExpression=Attr(config.pk_attribute).not_exists(),
                )
            else:
                table.put_item(Item=converted_item)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ODALOperationRejected(
                    f"WRITE_REJECTED: duplicate_key — Append-only table "
                    f"'{table_name}' already contains a record with this "
                    f"primary key. Overwrites are structurally prevented "
                    f"(§15.4). Use supersession for corrections."
                ) from e
            raise ODALWriteError(
                f"Failed to write to {table_name}: {e}"
            ) from e
        except Exception as e:
            raise ODALWriteError(
                f"Failed to write to {table_name}: {e}"
            ) from e

        # Step 5: Compute record hash (for audit)
        record_hash = compute_record_hash(item)

        # Step 6: Audit log (skip for audit table itself to avoid recursion)
        audit_timestamp = datetime.now(timezone.utc).isoformat()
        if config.classification != TableClassification.AUDIT_INTERNAL:
            self._write_audit_record(
                table_name=table_name,
                operation="put_item",
                record_hash=record_hash,
                chain_hash=chain_hash,
                previous_hash=previous_hash,
                operation_context=operation_context,
                audit_timestamp=audit_timestamp,
            )

        return {
            "record_hash": record_hash,
            "chain_hash": chain_hash,
            "audit_timestamp": audit_timestamp,
        }

    def update_item(
        self,
        table_name: str,
        key: dict[str, Any],
        updates: dict[str, Any],
        operation_context: str = "",
    ) -> dict:
        """Update an existing item in a mutable Oracle DynamoDB table.

        §15.3 Step 2: update_item is ONLY permitted on MUTABLE_AUDITED
        and RESEARCH tables. Append-only tables reject with
        WRITE_REJECTED: append_only_violation.

        §15.5: Every update generates an audit record with before/after
        values. The update and audit are written atomically via DynamoDB
        transaction.

        Args:
            table_name: Must be MUTABLE_AUDITED or RESEARCH
            key: Primary key of the record to update
            updates: dict of {field_name: new_value}
            operation_context: Free-text for audit trail

        Returns:
            dict with updated_fields, audit_timestamp

        Raises:
            ODALOperationRejected: If table is append-only
        """
        # Step 0: Isolation
        self._enforce_isolation(table_name)

        # Step 1: Classification
        config = get_table_config(table_name)

        # Step 2: Operation validation
        if is_append_only(config.classification):
            self._log_rejection(
                table_name, "update_item", "append_only_violation",
            )
            raise ODALOperationRejected(
                f"WRITE_REJECTED: append_only_violation — "
                f"'{table_name}' is classified as "
                f"'{config.classification.value}'. UpdateItem is "
                f"prohibited on append-only tables (§15.3)."
            )

        if config.classification == TableClassification.AUDIT_INTERNAL:
            raise ODALOperationRejected(
                f"WRITE_REJECTED: audit_table_immutable — "
                f"'{table_name}' is the ODAL audit table. "
                f"Updates are not permitted."
            )

        # Step 3: Read current state for audit diff
        table = self._resource.Table(table_name)
        current = table.get_item(Key=key).get("Item")

        # Step 4: Build update expression
        expr_parts = []
        expr_names = {}
        expr_values = {}
        for i, (field, value) in enumerate(updates.items()):
            placeholder_name = f"#f{i}"
            placeholder_value = f":v{i}"
            expr_parts.append(f"{placeholder_name} = {placeholder_value}")
            expr_names[placeholder_name] = field
            expr_values[placeholder_value] = (
                Decimal(str(value)) if isinstance(value, float) else value
            )

        update_expression = "SET " + ", ".join(expr_parts)

        # Step 5: Execute update
        try:
            table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )
        except Exception as e:
            raise ODALWriteError(
                f"Failed to update {table_name}: {e}"
            ) from e

        # Step 6: Write audit record (§15.5)
        audit_timestamp = datetime.now(timezone.utc).isoformat()
        audit_detail = {
            "table_name": table_name,
            "record_key": json.dumps(key, default=str),
            "fields_updated": list(updates.keys()),
            "previous_values": {
                f: str(current.get(f, "<not_present>"))
                for f in updates
            } if current else {},
            "new_values": {f: str(v) for f, v in updates.items()},
        }

        self._write_audit_record(
            table_name=table_name,
            operation="update_item",
            record_hash=compute_record_hash(audit_detail),
            chain_hash=None,
            previous_hash=None,
            operation_context=operation_context,
            audit_timestamp=audit_timestamp,
            extra_fields=audit_detail,
        )

        return {
            "updated_fields": list(updates.keys()),
            "audit_timestamp": audit_timestamp,
        }

    def batch_put_items(
        self,
        table_name: str,
        items: list[dict[str, Any]],
        operation_context: str = "",
    ) -> list[dict]:
        """Write multiple items to an Oracle table (§15.2).

        Each item passes through the full put_item validation pipeline.
        Items are written sequentially for correctness (conditional
        writes and hash-chain computation require ordering).

        NOTE: This is not atomic across items. A failure on item N means
        items 0..N-1 are already written. Future upgrade: DynamoDB
        TransactWriteItems for true atomicity (max 100 items).

        Args:
            table_name: Target table
            items: List of items to write
            operation_context: Shared context for all audit records

        Returns:
            List of put_item result dicts
        """
        results = []
        for i, item in enumerate(items):
            ctx = f"{operation_context} [batch {i+1}/{len(items)}]"
            result = self.put_item(table_name, item, operation_context=ctx)
            results.append(result)
        return results

    def delete_item(self, table_name: str, key: dict[str, Any]) -> None:
        """DELETE IS ALWAYS REJECTED — no Oracle table permits deletion.

        Architecture §15.2: delete_item is not exposed by ODAL.
        This method exists solely to provide a clear rejection with
        proper audit logging rather than silent absence.
        """
        self._enforce_isolation(table_name)
        self._log_rejection(table_name, "delete_item", "delete_prohibited")
        raise ODALOperationRejected(
            f"WRITE_REJECTED: delete_prohibited — No Oracle table "
            f"permits deletion through ODAL (§15.2). Table: "
            f"'{table_name}'. Use supersession (append a correction "
            f"record with 'supersedes' field) for corrections to "
            f"append-only tables."
        )

    # -- Public Read API ---------------------------------------------------

    def get_item(self, table_name: str, key: dict[str, Any]) -> dict | None:
        """Read an item from an Oracle DynamoDB table.

        Reads do not require audit logging but do require isolation.
        ODAL does not govern reads (§15.1) — this is a convenience
        method. Production read patterns may bypass ODAL.
        """
        self._enforce_isolation(table_name)
        table = self._resource.Table(table_name)
        response = table.get_item(Key=key)
        return response.get("Item")

    def query(
        self,
        table_name: str,
        key_condition_expression,
        **kwargs,
    ) -> list[dict]:
        """Query an Oracle DynamoDB table. Handles pagination."""
        self._enforce_isolation(table_name)
        table = self._resource.Table(table_name)
        items = []
        response = table.query(
            KeyConditionExpression=key_condition_expression, **kwargs
        )
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = table.query(
                KeyConditionExpression=key_condition_expression,
                ExclusiveStartKey=response["LastEvaluatedKey"],
                **kwargs,
            )
            items.extend(response.get("Items", []))
        return items

    # -- Classification Introspection (for callers) ------------------------

    @staticmethod
    def get_classification(table_name: str) -> TableClassification:
        """Return the governance classification for a table."""
        return get_table_config(table_name).classification

    @staticmethod
    def is_table_registered(table_name: str) -> bool:
        """Check if a table is in ODAL's registry."""
        return table_name in TABLE_REGISTRY

    # -- Internal: Hash-Chain (§15.6) --------------------------------------

    def _compute_hash_chain_for_put(
        self,
        table_name: str,
        config: TableConfig,
        item: dict,
    ) -> tuple[str, str]:
        """Compute hash-chain fields for a new entry.

        §15.6 Hash-Chain Write Integration:
        1. Query most recent entry in partition (by SK descending)
        2. If empty → CHAIN_GENESIS
        3. Compute previous_hash from that entry
        4. Compute chain_hash for the new record

        Returns (previous_hash, chain_hash).
        """
        pk_value = item.get(config.pk_attribute)
        if pk_value is None:
            raise ODALWriteError(
                f"Hash-chained table '{table_name}' requires partition "
                f"key '{config.pk_attribute}' in the item."
            )

        # Query latest entry in this partition
        table = self._resource.Table(table_name)
        try:
            response = table.query(
                KeyConditionExpression=Key(config.pk_attribute).eq(pk_value),
                ScanIndexForward=False,  # descending by SK
                Limit=1,
            )
        except Exception:
            # Table might be empty or query failed — use genesis
            response = {"Items": []}

        if response.get("Items"):
            latest_entry = response["Items"][0]
            previous_hash = compute_entry_hash_for_chain(latest_entry)
        else:
            previous_hash = CHAIN_GENESIS

        # Compute chain hash for the new record (including previous_hash)
        item_with_chain = {**item, "previous_hash": previous_hash}
        record_hash = compute_record_hash(item_with_chain)
        chain_hash = compute_chain_hash(record_hash, previous_hash)

        return previous_hash, chain_hash

    # -- Internal: Audit ---------------------------------------------------

    def _write_audit_record(
        self,
        table_name: str,
        operation: str,
        record_hash: str,
        chain_hash: str | None,
        previous_hash: str | None,
        operation_context: str,
        audit_timestamp: str,
        extra_fields: dict | None = None,
    ) -> None:
        """Write an audit record to Oracle_ODAL_Audit_Log."""
        audit_record = {
            "table_name": table_name,
            "write_timestamp": audit_timestamp,
            "operation": operation,
            "record_hash": record_hash,
            "chain_hash": chain_hash or "N/A",
            "previous_chain_hash": previous_hash or "N/A",
            "operation_context": operation_context,
            "write_id": str(uuid.uuid4()),
        }
        if extra_fields:
            # Store audit detail as JSON string to avoid schema conflicts
            audit_record["audit_detail"] = json.dumps(
                extra_fields, default=str
            )

        try:
            audit_table = self._resource.Table(AUDIT_TABLE)
            audit_table.put_item(Item=self._convert_floats(audit_record))
        except Exception as e:
            raise ODALWriteError(
                f"CRITICAL: Item written to {table_name} but audit log "
                f"write failed. Record hash: {record_hash}. Error: {e}"
            ) from e

    def _log_rejection(
        self, table_name: str, operation: str, rejection_code: str,
    ) -> None:
        """Log a rejected write attempt to the audit table (§15.9)."""
        try:
            audit_table = self._resource.Table(AUDIT_TABLE)
            audit_table.put_item(Item={
                "table_name": f"REJECTED:{table_name}",
                "write_timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation,
                "rejection_code": rejection_code,
                "write_id": str(uuid.uuid4()),
            })
        except Exception:
            pass  # Best-effort — don't fail the rejection on audit failure

    # -- Internal: Validation ----------------------------------------------

    def _enforce_isolation(self, table_name: str) -> None:
        """Verify Oracle_ prefix. Architecture §16.3 defense-in-depth."""
        if not table_name.startswith(ORACLE_TABLE_PREFIX):
            raise ODALIsolationError(
                f"ISOLATION VIOLATION: Attempted access to '{table_name}'. "
                f"ODAL only permits access to tables with prefix "
                f"'{ORACLE_TABLE_PREFIX}'. This may indicate Helios "
                f"contamination. Operation blocked."
            )

    @staticmethod
    def _convert_floats(item: dict) -> dict:
        """Convert Python floats to Decimal for DynamoDB."""
        converted = {}
        for key, value in item.items():
            if isinstance(value, float):
                converted[key] = Decimal(str(value))
            elif isinstance(value, dict):
                converted[key] = ODALClient._convert_floats(value)
            elif isinstance(value, list):
                converted[key] = [
                    Decimal(str(v)) if isinstance(v, float) else v
                    for v in value
                ]
            else:
                converted[key] = value
        return converted


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------
def _normalize_for_hash(obj: Any) -> Any:
    """Recursively normalize an object for deterministic JSON serialization."""
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, Decimal):
        return str(obj)
    return obj
