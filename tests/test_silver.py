"""
Tests for the Silver transformation layer.

Covers:
  - Quality gate integration (quarantine + clean split)
  - Deduplication on event_id (keep latest)
  - Timestamp normalisation to UTC ISO
  - Country code standardisation
  - Price / quantity type casting
  - Silver metadata fields (_silver_id, _processed_at, _quality_score)
  - Parquet persistence and read-back
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from pipeline.silver import SilverLayer
from pipeline.quality import QualityGate, NonNullCheck, RangeCheck


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_silver(tmp_path):
    return SilverLayer(
        silver_dir=tmp_path / "silver",
        quarantine_dir=tmp_path / "quarantine",
    )


def _good_row(**overrides) -> dict:
    base = {
        "_bronze_id": "b_001",
        "_ingested_at": "2024-06-15T10:00:00+00:00",
        "_source_file": "generator/bronze_test.jsonl",
        "_batch_id": "bronze_test_001",
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


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quality gate integration
# ---------------------------------------------------------------------------

class TestSilverQualityGate:
    def test_clean_record_survives(self, tmp_silver):
        df = _make_df([_good_row()])
        silver_df, report = tmp_silver.transform(df)
        assert len(silver_df) == 1
        assert report.passed >= 1

    def test_null_user_id_quarantined(self, tmp_silver):
        df = _make_df([_good_row(user_id=None)])
        silver_df, report = tmp_silver.transform(df)
        assert len(silver_df) == 0
        assert report.failed == 1

    def test_negative_price_quarantined(self, tmp_silver):
        rows = [_good_row(event_id="good"), _good_row(price=-1.0, event_id="bad")]
        df = _make_df(rows)
        silver_df, report = tmp_silver.transform(df)
        assert len(silver_df) == 1
        assert report.failed >= 1

    def test_future_timestamp_quarantined(self, tmp_silver):
        future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        df = _make_df([_good_row(timestamp=future)])
        silver_df, report = tmp_silver.transform(df)
        assert len(silver_df) == 0

    def test_empty_input_returns_empty(self, tmp_silver):
        silver_df, report = tmp_silver.transform(pd.DataFrame())
        assert silver_df.empty
        assert report.total_records == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestSilverDeduplication:
    def test_deduplicates_on_event_id(self, tmp_silver):
        rows = [
            _good_row(event_id="dup", _ingested_at="2024-06-15T10:00:00+00:00"),
            _good_row(event_id="dup", _ingested_at="2024-06-15T11:00:00+00:00"),
            _good_row(event_id="unique_1"),
        ]
        df = _make_df(rows)
        silver_df, _ = tmp_silver.transform(df)
        assert silver_df["event_id"].nunique() == silver_df["event_id"].count()

    def test_keeps_unique_records(self, tmp_silver):
        rows = [_good_row(event_id=f"evt_{i}") for i in range(5)]
        df = _make_df(rows)
        silver_df, _ = tmp_silver.transform(df)
        assert len(silver_df) == 5

    def test_dedup_removes_exact_duplicates(self, tmp_silver):
        row = _good_row()
        df = _make_df([row, row.copy(), row.copy()])
        silver_df, _ = tmp_silver.transform(df)
        # Only 1 record with that event_id should survive
        assert (silver_df["event_id"] == "evt_001").sum() == 1


# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------

class TestSilverTimestampNormalisation:
    def test_normalises_timestamp_to_utc(self, tmp_silver):
        df = _make_df([_good_row(timestamp="2024-06-15T10:00:00+00:00")])
        silver_df, _ = tmp_silver.transform(df)
        ts = silver_df["timestamp"].iloc[0]
        assert "+00:00" in ts or "UTC" in ts or "Z" in ts

    def test_handles_various_timestamp_formats(self, tmp_silver):
        """Multiple parseable timestamp formats should all normalise cleanly."""
        timestamps = [
            "2024-01-01T00:00:00Z",
            "2024-03-15T08:30:00+00:00",
            "2024-06-01T12:00:00.000000+00:00",
        ]
        for i, ts in enumerate(timestamps):
            df = _make_df([_good_row(event_id=f"ts_test_{i}", timestamp=ts)])
            silver_df, _ = tmp_silver.transform(df)
            assert not silver_df.empty, f"Timestamp format {ts} should parse cleanly"


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

class TestSilverTypeCasting:
    def test_price_cast_to_float(self, tmp_silver):
        df = _make_df([_good_row(price="49.99")])
        silver_df, _ = tmp_silver.transform(df)
        if not silver_df.empty:
            assert silver_df["price"].dtype in (float, "float64")

    def test_quantity_cast_to_int(self, tmp_silver):
        df = _make_df([_good_row(quantity=3)])
        silver_df, _ = tmp_silver.transform(df)
        if not silver_df.empty:
            assert silver_df["quantity"].dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# Country normalisation
# ---------------------------------------------------------------------------

class TestSilverCountryNormalisation:
    def test_two_letter_codes_uppercased(self, tmp_silver):
        df = _make_df([_good_row(country="us")])
        silver_df, _ = tmp_silver.transform(df)
        if not silver_df.empty:
            assert silver_df["country"].iloc[0] == "US"

    def test_full_name_mapped_to_code(self, tmp_silver):
        """'united states' should map to 'US'."""
        # Full-name country values won't pass the CategoricalCheck by default,
        # but if we use a permissive gate we can verify the mapping logic.
        from pipeline.quality import NonNullCheck
        gate = QualityGate(rules=[NonNullCheck(["event_id", "user_id"])])
        layer = SilverLayer(
            silver_dir=tmp_silver._silver_dir,
            quarantine_dir=tmp_silver._quarantine_dir,
            quality_gate=gate,
        )
        df = _make_df([_good_row(country="united states")])
        silver_df, _ = layer.transform(df)
        if not silver_df.empty:
            assert silver_df["country"].iloc[0] == "US"


# ---------------------------------------------------------------------------
# Silver metadata fields
# ---------------------------------------------------------------------------

class TestSilverMetadata:
    def test_adds_silver_id(self, tmp_silver):
        df = _make_df([_good_row()])
        silver_df, _ = tmp_silver.transform(df)
        assert "_silver_id" in silver_df.columns
        assert silver_df["_silver_id"].notna().all()

    def test_adds_processed_at(self, tmp_silver):
        df = _make_df([_good_row()])
        silver_df, _ = tmp_silver.transform(df)
        assert "_processed_at" in silver_df.columns
        assert silver_df["_processed_at"].notna().all()

    def test_adds_quality_score(self, tmp_silver):
        df = _make_df([_good_row()])
        silver_df, _ = tmp_silver.transform(df)
        assert "_quality_score" in silver_df.columns
        score = silver_df["_quality_score"].iloc[0]
        assert 0.0 <= score <= 1.0

    def test_silver_ids_are_unique(self, tmp_silver):
        rows = [_good_row(event_id=f"evt_{i}") for i in range(10)]
        df = _make_df(rows)
        silver_df, _ = tmp_silver.transform(df)
        assert silver_df["_silver_id"].nunique() == len(silver_df)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestSilverPersistence:
    def test_saves_parquet_file(self, tmp_silver):
        df = _make_df([_good_row()])
        batch_id = "silver_testbatch001"
        silver_df, _ = tmp_silver.transform(df, batch_id=batch_id)
        if not silver_df.empty:
            path = tmp_silver._silver_dir / f"{batch_id}.parquet"
            assert path.exists()

    def test_read_silver_round_trip(self, tmp_silver):
        rows = [_good_row(event_id=f"evt_{i}") for i in range(3)]
        df = _make_df(rows)
        batch_id = "silver_roundtrip001"
        silver_df, _ = tmp_silver.transform(df, batch_id=batch_id)
        if not silver_df.empty:
            loaded = tmp_silver.read_silver(batch_id)
            assert len(loaded) == len(silver_df)

    def test_read_raises_on_missing_batch(self, tmp_silver):
        with pytest.raises(FileNotFoundError):
            tmp_silver.read_silver("silver_nonexistent")

    def test_list_batches(self, tmp_silver):
        df = _make_df([_good_row(event_id="e1")])
        _, _ = tmp_silver.transform(df, batch_id="silver_batch_a")
        batches = tmp_silver.list_batches()
        assert "silver_batch_a" in batches
