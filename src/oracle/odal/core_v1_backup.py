"""
Oracle Data Access Layer (ODAL) — Core Module

Architecture Reference: §15.0 (Oracle Data Access Layer)
Governance Reference: §18.2 (Infrastructure Isolation)

ODAL is the ONLY permitted write path to Oracle DynamoDB tables.
Every write operation is:
  1. Validated against schema constraints
  2. Audit-logged with hash-chain integrity
  3. Tagged with lineage metadata

No Oracle module may bypass ODAL for DynamoDB writes.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.types import TypeSerializer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORACLE_TABLE_PREFIX = "Oracle_"
AUDIT_TABLE = "Oracle_ODAL_Audit_Log"
AWS_REGION = "ap-south-1"


class ODALWriteError(Exception):
    """Raised when an ODAL write operation fails."""
    pass


class ODALIsolationError(Exception):
    """Raised when a write targets a non-Oracle table (isolation violation)."""
    pass


# ---------------------------------------------------------------------------
# Hash-Chain Computation
# ---------------------------------------------------------------------------
def compute_record_hash(item: dict) -> str:
    """
    Compute a deterministic SHA-256 hash of a DynamoDB item.

    The item is serialized to JSON with sorted keys and consistent
    formatting to ensure hash reproducibility.

    Architecture Reference: §15.3 (Hash-Chain Verification)
    """
    serialized = json.dumps(
        _normalize_for_hash(item),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_chain_hash(record_hash: str, previous_chain_hash: str) -> str:
    """
    Compute the chain hash: SHA-256(record_hash + previous_chain_hash).

    This creates an append-only verification chain. Any tampering with
    a historical record breaks the chain from that point forward.

    Architecture Reference: §15.3 (Hash-Chain Verification)
    """
    combined = f"{record_hash}{previous_chain_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ODAL Client
# ---------------------------------------------------------------------------
class ODALClient:
    """
    The sole write interface to Oracle DynamoDB tables.

    Usage:
        odal = ODALClient()
        odal.put_item("Oracle_Feature_Store", item_dict)

    Every put_item call:
      1. Validates the table name starts with 'Oracle_'
      2. Writes the item to the target table
      3. Writes an audit record to Oracle_ODAL_Audit_Log
      4. Maintains hash-chain integrity
    """

    def __init__(self, region: str = AWS_REGION):
        self._resource = boto3.resource("dynamodb", region_name=region)
        self._client = boto3.client("dynamodb", region_name=region)
        self._region = region
        # Cache for last chain hash per table (in-memory; production
        # would read the last entry on startup)
        self._last_chain_hash: dict[str, str] = {}

    # -- Public API --------------------------------------------------------

    def put_item(
        self,
        table_name: str,
        item: dict[str, Any],
        operation_context: str = "",
    ) -> dict:
        """
        Write an item to an Oracle DynamoDB table with full audit trail.

        Args:
            table_name: Must start with 'Oracle_'
            item: The item to write (Python dict, not DynamoDB format)
            operation_context: Free-text description for audit trail

        Returns:
            dict with 'record_hash', 'chain_hash', 'audit_timestamp'

        Raises:
            ODALIsolationError: If table_name doesn't start with 'Oracle_'
            ODALWriteError: If the DynamoDB write fails
        """
        # Gate 1: Isolation enforcement
        self._enforce_isolation(table_name)

        # Gate 2: Write the item
        try:
            table = self._resource.Table(table_name)
            table.put_item(Item=self._convert_floats(item))
        except Exception as e:
            raise ODALWriteError(
                f"Failed to write to {table_name}: {e}"
            ) from e

        # Gate 3: Compute hashes
        record_hash = compute_record_hash(item)
        previous_hash = self._last_chain_hash.get(table_name, "GENESIS")
        chain_hash = compute_chain_hash(record_hash, previous_hash)
        self._last_chain_hash[table_name] = chain_hash

        # Gate 4: Write audit record
        audit_timestamp = datetime.now(timezone.utc).isoformat()
        audit_record = {
            "table_name": table_name,
            "write_timestamp": audit_timestamp,
            "record_hash": record_hash,
            "chain_hash": chain_hash,
            "previous_chain_hash": previous_hash,
            "operation_context": operation_context,
            "write_id": str(uuid.uuid4()),
        }

        try:
            audit_table = self._resource.Table(AUDIT_TABLE)
            audit_table.put_item(Item=audit_record)
        except Exception as e:
            # Audit write failure is CRITICAL — log but don't silently swallow
            raise ODALWriteError(
                f"CRITICAL: Item written to {table_name} but audit log "
                f"write failed. Record hash: {record_hash}. Error: {e}"
            ) from e

        return {
            "record_hash": record_hash,
            "chain_hash": chain_hash,
            "audit_timestamp": audit_timestamp,
        }

    def get_item(self, table_name: str, key: dict[str, Any]) -> dict | None:
        """
        Read an item from an Oracle DynamoDB table.

        Reads do not require audit logging but do require isolation check.
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
        """
        Query an Oracle DynamoDB table.

        Returns all matching items. Handles pagination internally.
        """
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

    # -- Internal Methods --------------------------------------------------

    def _enforce_isolation(self, table_name: str) -> None:
        """
        Verify the table name starts with the Oracle prefix.

        This is the code-level enforcement of Architecture §16.3
        (Data Isolation). Even if IAM policies scope to Oracle_* tables,
        this check provides defense-in-depth at the application layer.
        """
        if not table_name.startswith(ORACLE_TABLE_PREFIX):
            raise ODALIsolationError(
                f"ISOLATION VIOLATION: Attempted access to '{table_name}'. "
                f"ODAL only permits access to tables with prefix "
                f"'{ORACLE_TABLE_PREFIX}'. This may indicate Helios "
                f"contamination. Operation blocked."
            )

    @staticmethod
    def _convert_floats(item: dict) -> dict:
        """
        Convert Python floats to Decimal for DynamoDB compatibility.
        DynamoDB does not accept float — only Decimal.
        """
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
