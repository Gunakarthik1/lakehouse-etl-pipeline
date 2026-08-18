"""
Tests for the Bronze ingestion layer.

Covers:
  - Schema presence validation (rejects records missing required fields)
  - Internal metadata fields added on ingestion
  - JSON Lines persistence and read-back
  - Stats tracking across multiple ingestions
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pipeline.bronze import BronzeLayer, REQUIRED_SCHEMA_FIELDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_bronze(tmp_path):
    """BronzeLayer backed by a temporary directory."""
    return BronzeLayer(bronze_dir=tmp_path / "bronze")


def _good_record(**overrides) -> dict:
    base = {
        "event_id": "evt_001",
        "user_id": "user_1001",
        "session_id": "sess_abc",
        "event_type": "purchase",
        "timestamp": "2024-06-15T10:00:00+00:00",
        "product_id": "prod_001",
        "product_name": "Widget",
        "category": "electronics",
        "price": 49.99,
        "quantity": 2,
        "country": "US",
        "device_type": "desktop",
        "referrer": "google.com",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestBronzeSchemaValidation:
    def test_accepts_complete_record(self, tmp_bronze):
        summary = tmp_bronze.ingest([_good_record()])
        assert summary["records_accepted"] == 1
        assert summary["records_rejected_schema"] == 0

    def test_rejects_record_missing_event_id(self, tmp_bronze):
        rec = _good_record()
        del rec["event_id"]
        summary = tmp_bronze.ingest([rec])
        assert summary["records_accepted"] == 0
        assert summary["records_rejected_schema"] == 1

    def test_rejects_record_missing_timestamp(self, tmp_bronze):
        rec = _good_record()
        del rec["timestamp"]
        summary = tmp_bronze.ingest([rec])
        assert summary["records_accepted"] == 0
        assert summary["records_rejected_schema"] == 1

    def test_accepts_record_with_null_values(self, tmp_bronze):
        """Bronze only checks field presence, not field values."""
        rec = _good_record(price=None, user_id=None)
        summary = tmp_bronze.ingest([rec])
        assert summary["records_accepted"] == 1

    def test_mixed_batch_partial_rejection(self, tmp_bronze):
        good = _good_record(event_id="good_1")
        bad = {"event_id": "bad_1"}  # missing most fields
        summary = tmp_bronze.ingest([good, bad])
        assert summary["records_accepted"] == 1
        assert summary["records_rejected_schema"] == 1

    def test_all_required_fields_must_be_present(self, tmp_bronze):
        for missing_field in REQUIRED_SCHEMA_FIELDS:
            rec = _good_record()
            del rec[missing_field]
            summary = tmp_bronze.ingest([rec])
            assert summary["records_rejected_schema"] == 1, (
                f"Expected rejection for missing field: {missing_field}"
            )


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

class TestBronzeMetadata:
    def test_adds_bronze_id(self, tmp_bronze):
        summary = tmp_bronze.ingest([_good_record()])
        df = tmp_bronze.read_bronze(summary["batch_id"])
        assert "_bronze_id" in df.columns
        assert df["_bronze_id"].notna().all()

    def test_adds_ingested_at(self, tmp_bronze):
        summary = tmp_bronze.ingest([_good_record()])
        df = tmp_bronze.read_bronze(summary["batch_id"])
        assert "_ingested_at" in df.columns
        assert df["_ingested_at"].notna().all()

    def test_adds_source_file(self, tmp_bronze):
        summary = tmp_bronze.ingest([_good_record()], source_label="test_source")
        df = tmp_bronze.read_bronze(summary["batch_id"])
        assert "_source_file" in df.columns
        assert "test_source" in df["_source_file"].iloc[0]

    def test_adds_batch_id_field(self, tmp_bronze):
        summary = tmp_bronze.ingest([_good_record()])
        df = tmp_bronze.read_bronze(summary["batch_id"])
        assert "_batch_id" in df.columns
        assert df["_batch_id"].iloc[0] == summary["batch_id"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestBronzePersistence:
    def test_read_returns_all_accepted_records(self, tmp_bronze):
        records = [_good_record(event_id=f"evt_{i}") for i in range(10)]
        summary = tmp_bronze.ingest(records)
        df = tmp_bronze.read_bronze(summary["batch_id"])
        assert len(df) == 10

    def test_raises_on_missing_batch(self, tmp_bronze):
        with pytest.raises(FileNotFoundError):
            tmp_bronze.read_bronze("bronze_nonexistent")

    def test_list_batches_empty_initially(self, tmp_bronze):
        assert tmp_bronze.list_batches() == []

    def test_list_batches_after_ingestion(self, tmp_bronze):
        tmp_bronze.ingest([_good_record(event_id="e1")])
        tmp_bronze.ingest([_good_record(event_id="e2")])
        batches = tmp_bronze.list_batches()
        assert len(batches) == 2

    def test_persisted_as_json_lines(self, tmp_bronze):
        summary = tmp_bronze.ingest([_good_record()])
        batch_file = Path(tmp_bronze._bronze_dir) / f"{summary['batch_id']}.jsonl"
        assert batch_file.exists()
        lines = batch_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_id"] == "evt_001"


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------

class TestBronzeStats:
    def test_records_ingested_accumulates(self, tmp_bronze):
        tmp_bronze.ingest([_good_record(event_id="e1"), _good_record(event_id="e2")])
        tmp_bronze.ingest([_good_record(event_id="e3")])
        assert tmp_bronze.records_ingested == 3

    def test_batches_processed_increments(self, tmp_bronze):
        tmp_bronze.ingest([_good_record(event_id="e1")])
        tmp_bronze.ingest([_good_record(event_id="e2")])
        assert tmp_bronze.batches_processed == 2

    def test_last_batch_id_updates(self, tmp_bronze):
        s1 = tmp_bronze.ingest([_good_record(event_id="e1")])
        s2 = tmp_bronze.ingest([_good_record(event_id="e2")])
        assert tmp_bronze.last_batch_id == s2["batch_id"]

    def test_stats_dict(self, tmp_bronze):
        tmp_bronze.ingest([_good_record()])
        s = tmp_bronze.stats()
        assert s["records_ingested"] == 1
        assert s["batches_processed"] == 1
        assert s["last_batch_id"] is not None
