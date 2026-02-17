"""
ODAL Specification Compliance Tests

Tests verify compliance with:
  - Architecture §15.0 (ODAL requirements)
  - Architecture §16.3 (Data Isolation)
  - Governance §18.2 (Infrastructure Isolation)

These are NOT unit tests — they verify the live system behaves
per specification. They hit real DynamoDB tables.
"""

import pytest
from oracle.odal.core import (
    ODALClient,
    ODALIsolationError,
    ODALWriteError,
    compute_record_hash,
    compute_chain_hash,
)


@pytest.fixture
def odal():
    return ODALClient(region="ap-south-1")


# -----------------------------------------------------------------------
# §16.3 — Data Isolation: ODAL must reject non-Oracle table access
# -----------------------------------------------------------------------

class TestIsolationEnforcement:
    """Architecture §16.3, DATA-ISO-001 through DATA-ISO-009"""

    def test_rejects_helios_table_write(self, odal):
        """ODAL must refuse to write to any non-Oracle-prefixed table."""
        with pytest.raises(ODALIsolationError, match="ISOLATION VIOLATION"):
            odal.put_item("Helios_Some_Table", {"id": "test"})

    def test_rejects_helios_table_read(self, odal):
        """ODAL must refuse to read from any non-Oracle-prefixed table."""
        with pytest.raises(ODALIsolationError, match="ISOLATION VIOLATION"):
            odal.get_item("Helios_Positions", {"id": "test"})

    def test_rejects_unprefixed_table(self, odal):
        """ODAL must reject tables without any prefix."""
        with pytest.raises(ODALIsolationError):
            odal.put_item("SomeRandomTable", {"id": "test"})

    def test_accepts_oracle_prefix(self, odal):
        """ODAL must accept tables with Oracle_ prefix."""
        # This will fail on actual write (table doesn't exist)
        # but should NOT raise ODALIsolationError
        with pytest.raises(ODALWriteError):  # write fails, but isolation passes
            odal.put_item("Oracle_NonExistent_Table", {"id": "test"})


# -----------------------------------------------------------------------
# §15.3 — Hash-Chain Integrity
# -----------------------------------------------------------------------

class TestHashChainIntegrity:
    """Architecture §15.3"""

    def test_record_hash_deterministic(self):
        """Same input must always produce the same hash."""
        item = {"key": "value", "number": 42}
        h1 = compute_record_hash(item)
        h2 = compute_record_hash(item)
        assert h1 == h2, "Record hash must be deterministic"

    def test_record_hash_order_independent(self):
        """Dict key order must not affect the hash."""
        item_a = {"z_key": "last", "a_key": "first"}
        item_b = {"a_key": "first", "z_key": "last"}
        assert compute_record_hash(item_a) == compute_record_hash(item_b)

    def test_record_hash_changes_on_modification(self):
        """Any change to the item must change the hash."""
        item_original = {"key": "value"}
        item_modified = {"key": "value_changed"}
        assert compute_record_hash(item_original) != compute_record_hash(item_modified)

    def test_chain_hash_links(self):
        """Chain hash must incorporate the previous chain hash."""
        record_hash = compute_record_hash({"test": "data"})
        chain_1 = compute_chain_hash(record_hash, "GENESIS")
        chain_2 = compute_chain_hash(record_hash, chain_1)
        # Same record but different position in chain → different chain hash
        assert chain_1 != chain_2

    def test_genesis_chain(self):
        """First record in a table uses 'GENESIS' as previous hash."""
        record_hash = compute_record_hash({"first": "record"})
        chain = compute_chain_hash(record_hash, "GENESIS")
        assert len(chain) == 64  # SHA-256 hex digest


# -----------------------------------------------------------------------
# §15.0 — ODAL Audit Trail (Live DynamoDB)
# -----------------------------------------------------------------------

class TestAuditTrail:
    """Architecture §15.0 — requires live DynamoDB access"""

    def test_write_produces_audit_record(self, odal):
        """Every ODAL write must produce a corresponding audit entry."""
        import time

        test_item = {
            "table_name": "ODAL_COMPLIANCE_TEST",
            "write_timestamp": f"TEST_{int(time.time())}",
            "test_data": "spec_compliance_verification",
        }

        result = odal.put_item(
            "Oracle_ODAL_Audit_Log",  # write to audit table itself as test
            test_item,
            operation_context="spec_compliance_test",
        )

        # Verify result contains required fields
        assert "record_hash" in result
        assert "chain_hash" in result
        assert "audit_timestamp" in result
        assert len(result["record_hash"]) == 64
        assert len(result["chain_hash"]) == 64

    def test_audit_record_is_readable(self, odal):
        """Audit records written by ODAL must be retrievable."""
        import time

        timestamp = f"READ_TEST_{int(time.time())}"
        test_item = {
            "table_name": "ODAL_READ_TEST",
            "write_timestamp": timestamp,
        }

        odal.put_item(
            "Oracle_ODAL_Audit_Log",
            test_item,
            operation_context="read_test",
        )

        # Read back
        retrieved = odal.get_item(
            "Oracle_ODAL_Audit_Log",
            {"table_name": "ODAL_READ_TEST", "write_timestamp": timestamp},
        )
        assert retrieved is not None
        assert retrieved["table_name"] == "ODAL_READ_TEST"
