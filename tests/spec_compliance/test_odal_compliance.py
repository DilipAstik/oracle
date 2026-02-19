"""
ODAL v2 Specification Compliance Tests

Tests verify compliance with:
  - Architecture §15.0–§15.9 (ODAL)
  - Architecture §16.3 (Data Isolation)
  - Architecture §8.3 (Append-Only Enforcement)
  - Architecture §8.4 (Hash-Chain Architecture)
  - Governance §18.2 (Infrastructure Isolation)

These are NOT unit tests — they verify the live system behaves
per specification. Hash-chain and append-only tests hit real DynamoDB.
"""

import time
import pytest
from oracle.odal.core import (
    ODALClient,
    ODALIsolationError,
    ODALOperationRejected,
    ODALWriteError,
    TableClassification,
    TABLE_REGISTRY,
    CHAIN_GENESIS,
    compute_record_hash,
    compute_chain_hash,
    get_table_config,
    is_append_only,
    is_hash_chained,
)


@pytest.fixture
def odal():
    return ODALClient(region="ap-south-1")


# =====================================================================
# §15.3 Step 1 — Table Classification Registry
# =====================================================================

class TestTableClassificationRegistry:
    """Every Oracle table must be registered with correct classification."""

    def test_all_19_tables_registered(self):
        """All 19 Oracle DynamoDB tables must be in the registry."""
        assert len(TABLE_REGISTRY) == 19

    def test_hash_chained_tables(self):
        """§15.3: 4 tables are append-only hash-chained."""
        hc_tables = [
            name for name, cfg in TABLE_REGISTRY.items()
            if cfg.classification == TableClassification.APPEND_ONLY_HASH_CHAINED
        ]
        assert set(hc_tables) == {
            "Oracle_Prediction_Log",
            "Oracle_Trade_Log",
            "Oracle_Governance_Events",
            "Oracle_Config_Audit",
        }

    def test_append_only_tables(self):
        """§15.3 + §8.3: Append-only non-hash-chained tables."""
        ao_tables = [
            name for name, cfg in TABLE_REGISTRY.items()
            if cfg.classification == TableClassification.APPEND_ONLY
        ]
        expected = {
            "Oracle_Ingestion_Rejection_Log",
            "Oracle_Feature_Computation_Log",
            "Oracle_Raw_Market_Data",
            "Oracle_Feature_Store",
            "Oracle_Feature_Registry",
            "Oracle_Event_Calendar",
        }
        assert set(ao_tables) == expected

    def test_mutable_audited_tables(self):
        """§15.3: 4 tables are mutable with audit."""
        mut_tables = [
            name for name, cfg in TABLE_REGISTRY.items()
            if cfg.classification == TableClassification.MUTABLE_AUDITED
        ]
        assert set(mut_tables) == {
            "Oracle_Portfolio_State",
            "Oracle_Model_Registry",
            "Oracle_Governance_Config",
            "Oracle_Pipeline_State",
        }

    def test_research_tables(self):
        """§15.3: 4 tables are research domain."""
        res_tables = [
            name for name, cfg in TABLE_REGISTRY.items()
            if cfg.classification == TableClassification.RESEARCH
        ]
        assert set(res_tables) == {
            "Oracle_Training_Datasets",
            "Oracle_Experiment_Log",
            "Oracle_Research_Feature_Store",
            "Oracle_Research_Model_Artifacts",
        }

    def test_audit_internal_table(self):
        """ODAL_Audit_Log is classified as AUDIT_INTERNAL."""
        cfg = get_table_config("Oracle_ODAL_Audit_Log")
        assert cfg.classification == TableClassification.AUDIT_INTERNAL

    def test_unregistered_table_rejected(self):
        """§15.3 AC-35: Unregistered tables are denied by default."""
        with pytest.raises(ODALOperationRejected, match="unregistered_table"):
            get_table_config("Oracle_Nonexistent_Table")


# =====================================================================
# §16.3 — Data Isolation
# =====================================================================

class TestIsolationEnforcement:
    """ODAL must reject non-Oracle table access."""

    def test_rejects_helios_table_write(self, odal):
        with pytest.raises(ODALIsolationError, match="ISOLATION VIOLATION"):
            odal.put_item("Helios_Some_Table", {"id": "test"})

    def test_rejects_helios_table_read(self, odal):
        with pytest.raises(ODALIsolationError, match="ISOLATION VIOLATION"):
            odal.get_item("Helios_Positions", {"id": "test"})

    def test_rejects_unprefixed_table(self, odal):
        with pytest.raises(ODALIsolationError):
            odal.put_item("SomeRandomTable", {"id": "test"})

    def test_accepts_oracle_prefix(self, odal):
        """Isolation passes but unregistered table is rejected at Step 1."""
        with pytest.raises(ODALOperationRejected, match="unregistered_table"):
            odal.put_item("Oracle_Fake_Unregistered", {"id": "test"})


# =====================================================================
# §15.3 Step 2 — Operation Validation
# =====================================================================

class TestOperationValidation:
    """Operation must be permitted for the table's classification."""

    # -- update_item on append-only → REJECTED --

    def test_update_rejected_on_prediction_log(self, odal):
        """§15.3: update_item on hash-chained append-only → rejected."""
        with pytest.raises(ODALOperationRejected, match="append_only_violation"):
            odal.update_item(
                "Oracle_Prediction_Log",
                {"product": "TEST", "prediction_id": "TEST"},
                {"field": "value"},
            )

    def test_update_rejected_on_feature_store(self, odal):
        """§8.3: Feature Store is append-only (High criticality)."""
        with pytest.raises(ODALOperationRejected, match="append_only_violation"):
            odal.update_item(
                "Oracle_Feature_Store",
                {"underlying_expiry": "TEST", "reference_timestamp": "TEST"},
                {"field": "value"},
            )

    def test_update_rejected_on_trade_log(self, odal):
        with pytest.raises(ODALOperationRejected, match="append_only_violation"):
            odal.update_item(
                "Oracle_Trade_Log",
                {"atlas_product": "T", "trade_id": "T"},
                {"f": "v"},
            )

    def test_update_rejected_on_raw_market_data(self, odal):
        with pytest.raises(ODALOperationRejected, match="append_only_violation"):
            odal.update_item(
                "Oracle_Raw_Market_Data",
                {"partition_key": "T", "sort_key": "T"},
                {"f": "v"},
            )

    def test_update_rejected_on_all_append_only_tables(self, odal):
        """Comprehensive: every append-only table rejects update_item."""
        for name, cfg in TABLE_REGISTRY.items():
            if is_append_only(cfg.classification):
                with pytest.raises(ODALOperationRejected):
                    odal.update_item(name, {"k": "v"}, {"f": "v"})

    # -- delete_item → ALWAYS REJECTED --

    def test_delete_rejected_on_any_table(self, odal):
        """§15.2: delete_item is never permitted on any Oracle table."""
        for name in TABLE_REGISTRY:
            with pytest.raises(ODALOperationRejected, match="delete_prohibited"):
                odal.delete_item(name, {"k": "v"})

    def test_delete_rejected_on_mutable_table(self, odal):
        """Even mutable tables do not permit deletion."""
        with pytest.raises(ODALOperationRejected, match="delete_prohibited"):
            odal.delete_item(
                "Oracle_Portfolio_State",
                {"portfolio_id": "T", "snapshot_timestamp": "T"},
            )

    def test_delete_rejected_on_research_table(self, odal):
        with pytest.raises(ODALOperationRejected, match="delete_prohibited"):
            odal.delete_item(
                "Oracle_Experiment_Log",
                {"experiment_id": "T", "run_timestamp": "T"},
            )

    # -- update_item on mutable → PERMITTED (tested in live DynamoDB) --

    def test_update_permitted_classification_check(self):
        """Mutable tables allow update_item per classification rules."""
        for name, cfg in TABLE_REGISTRY.items():
            if cfg.classification == TableClassification.MUTABLE_AUDITED:
                assert not is_append_only(cfg.classification)

    def test_update_permitted_research_classification(self):
        """Research tables allow update_item per classification rules."""
        for name, cfg in TABLE_REGISTRY.items():
            if cfg.classification == TableClassification.RESEARCH:
                assert not is_append_only(cfg.classification)


# =====================================================================
# §8.4, §15.6 — Hash-Chain Integrity
# =====================================================================

class TestHashChainIntegrity:
    """Hash-chain computation must be deterministic and chain-linked."""

    def test_record_hash_deterministic(self):
        item = {"key": "value", "number": 42}
        assert compute_record_hash(item) == compute_record_hash(item)

    def test_record_hash_order_independent(self):
        item_a = {"z_key": "last", "a_key": "first"}
        item_b = {"a_key": "first", "z_key": "last"}
        assert compute_record_hash(item_a) == compute_record_hash(item_b)

    def test_record_hash_changes_on_modification(self):
        assert (
            compute_record_hash({"key": "value"})
            != compute_record_hash({"key": "changed"})
        )

    def test_chain_hash_links(self):
        record_hash = compute_record_hash({"test": "data"})
        chain_1 = compute_chain_hash(record_hash, CHAIN_GENESIS)
        chain_2 = compute_chain_hash(record_hash, chain_1)
        assert chain_1 != chain_2  # Same record, different chain position

    def test_genesis_sentinel(self):
        """§8.4: First entry uses CHAIN_GENESIS sentinel."""
        assert CHAIN_GENESIS == "CHAIN_GENESIS"
        record_hash = compute_record_hash({"first": "record"})
        chain = compute_chain_hash(record_hash, CHAIN_GENESIS)
        assert len(chain) == 64  # SHA-256 hex digest

    def test_hash_chained_classification_helper(self):
        """is_hash_chained returns True only for APPEND_ONLY_HASH_CHAINED."""
        assert is_hash_chained(TableClassification.APPEND_ONLY_HASH_CHAINED)
        assert not is_hash_chained(TableClassification.APPEND_ONLY)
        assert not is_hash_chained(TableClassification.MUTABLE_AUDITED)
        assert not is_hash_chained(TableClassification.RESEARCH)


# =====================================================================
# §15.0 — Audit Trail (Live DynamoDB)
# =====================================================================

class TestAuditTrail:
    """Every ODAL write produces a corresponding audit entry."""

    def test_write_produces_audit_record(self, odal):
        test_item = {
            "table_name": "ODAL_V2_COMPLIANCE_TEST",
            "write_timestamp": f"TEST_{int(time.time())}",
            "test_data": "v2_spec_compliance",
        }
        result = odal.put_item(
            "Oracle_ODAL_Audit_Log",
            test_item,
            operation_context="v2_spec_compliance_test",
        )
        assert "record_hash" in result
        assert "audit_timestamp" in result
        assert len(result["record_hash"]) == 64

    def test_audit_record_is_readable(self, odal):
        timestamp = f"V2_READ_{int(time.time())}"
        test_item = {
            "table_name": "ODAL_V2_READ_TEST",
            "write_timestamp": timestamp,
        }
        odal.put_item(
            "Oracle_ODAL_Audit_Log", test_item,
            operation_context="v2_read_test",
        )
        retrieved = odal.get_item(
            "Oracle_ODAL_Audit_Log",
            {"table_name": "ODAL_V2_READ_TEST", "write_timestamp": timestamp},
        )
        assert retrieved is not None
        assert retrieved["table_name"] == "ODAL_V2_READ_TEST"


# =====================================================================
# §15.4 — Append-Only Enforcement (Live DynamoDB)
# =====================================================================

class TestAppendOnlyEnforcement:
    """Append-only tables must reject duplicate primary keys."""

    def test_duplicate_write_rejected_on_append_only(self, odal):
        """§15.4: Conditional write prevents overwriting existing records."""
        ts = f"DUP_TEST_{int(time.time())}"
        item = {
            "source_date": f"ODAL_TEST#{ts}",
            "rejection_timestamp": ts,
            "reason": "first_write",
        }
        # First write succeeds
        odal.put_item("Oracle_Ingestion_Rejection_Log", item)

        # Second write with same key must be rejected
        item_dup = {**item, "reason": "duplicate_attempt"}
        with pytest.raises(ODALOperationRejected, match="duplicate_key"):
            odal.put_item("Oracle_Ingestion_Rejection_Log", item_dup)

    def test_different_keys_permitted_on_append_only(self, odal):
        """Different primary keys are fine on append-only tables."""
        ts = str(int(time.time()))
        item1 = {
            "source_date": f"ODAL_MULTI_1#{ts}",
            "rejection_timestamp": f"{ts}_A",
            "reason": "test_1",
        }
        item2 = {
            "source_date": f"ODAL_MULTI_2#{ts}",
            "rejection_timestamp": f"{ts}_B",
            "reason": "test_2",
        }
        r1 = odal.put_item("Oracle_Ingestion_Rejection_Log", item1)
        r2 = odal.put_item("Oracle_Ingestion_Rejection_Log", item2)
        assert r1["record_hash"] != r2["record_hash"]


# =====================================================================
# §15.5 — Mutable Table Update with Audit (Live DynamoDB)
# =====================================================================

class TestMutableTableUpdate:
    """Mutable tables permit updates with audit records."""

    def test_update_mutable_table(self, odal):
        """§15.5: update_item on mutable table succeeds with audit."""
        ts = f"MUT_TEST_{int(time.time())}"
        # First, put an initial record
        odal.put_item("Oracle_Governance_Config", {
            "parameter_category": f"ODAL_TEST_{ts}",
            "parameter_name": "test_param",
            "value": "initial",
        })
        # Now update it
        result = odal.update_item(
            "Oracle_Governance_Config",
            key={
                "parameter_category": f"ODAL_TEST_{ts}",
                "parameter_name": "test_param",
            },
            updates={"value": "updated"},
            operation_context="v2_mutable_test",
        )
        assert "updated_fields" in result
        assert "value" in result["updated_fields"]

        # Verify the update took effect
        item = odal.get_item("Oracle_Governance_Config", {
            "parameter_category": f"ODAL_TEST_{ts}",
            "parameter_name": "test_param",
        })
        assert item["value"] == "updated"


# =====================================================================
# §15.2 — batch_put_items
# =====================================================================

class TestBatchPutItems:
    """batch_put_items writes multiple records through the pipeline."""

    def test_batch_put_items(self, odal):
        ts = str(int(time.time()))
        items = [
            {
                "source_date": f"BATCH_{ts}",
                "rejection_timestamp": f"{ts}_batch_{i}",
                "reason": f"batch_item_{i}",
            }
            for i in range(3)
        ]
        results = odal.batch_put_items(
            "Oracle_Ingestion_Rejection_Log",
            items,
            operation_context="batch_test",
        )
        assert len(results) == 3
        assert all("record_hash" in r for r in results)


# =====================================================================
# Introspection API
# =====================================================================

class TestIntrospection:
    """Callers can query table classifications."""

    def test_get_classification(self, odal):
        assert (
            odal.get_classification("Oracle_Feature_Store")
            == TableClassification.APPEND_ONLY
        )
        assert (
            odal.get_classification("Oracle_Prediction_Log")
            == TableClassification.APPEND_ONLY_HASH_CHAINED
        )

    def test_is_table_registered(self, odal):
        assert odal.is_table_registered("Oracle_Feature_Store")
        assert not odal.is_table_registered("Oracle_Fake_Table")
