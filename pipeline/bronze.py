"""
Bronze Layer — Raw ingestion with schema presence enforcement.

Responsibilities:
- Accept raw records (list of dicts)
- Validate that required fields are present (not their values)
- Assign internal tracking fields: _bronze_id, _ingested_at, _source_file
- Persist records as JSON Lines to data/bronze/
- Track ingestion statistics
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Fields that must exist as keys in every record (value can be None/null)
REQUIRED_SCHEMA_FIELDS = {
    "event_id",
    "user_id",
    "session_id",
    "event_type",
    "timestamp",
    "product_id",
    "product_name",
    "category",
    "price",
    "quantity",
    "country",
    "device_type",
    "referrer",
}

BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = BASE_DIR / "data" / "bronze"


class SchemaValidationError(Exception):
    """Raised when a record is missing required schema fields."""


class BronzeLayer:
    """
    Raw ingestion layer.

    Validates that all required fields exist (schema presence check),
    enriches records with provenance metadata, and persists them as
    JSON Lines files partitioned by batch_id.
    """

    def __init__(self, bronze_dir: Path | None = None) -> None:
        self._bronze_dir = Path(bronze_dir) if bronze_dir else BRONZE_DIR
        self._bronze_dir.mkdir(parents=True, exist_ok=True)

        # Cumulative statistics
        self.records_ingested: int = 0
        self.batches_processed: int = 0
        self.last_batch_id: str | None = None
        self._schema_rejected: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        records: list[dict[str, Any]],
        source_label: str = "generator",
    ) -> dict[str, Any]:
        """
        Ingest a batch of raw records into the Bronze layer.

        Schema presence is enforced: records missing any required field
        are rejected before persistence. Value quality is NOT checked here
        — that is the Silver layer's responsibility.

        Args:
            records: List of raw event dicts.
            source_label: Human-readable label for the source system.

        Returns:
            Ingestion summary dict with batch_id and counts.
        """
        batch_id = f"bronze_{uuid.uuid4().hex[:12]}"
        ingested_at = datetime.now(timezone.utc).isoformat()
        source_file = f"{source_label}/{batch_id}.jsonl"

        valid_records: list[dict[str, Any]] = []
        rejected_count = 0

        for raw in records:
            missing = REQUIRED_SCHEMA_FIELDS - set(raw.keys())
            if missing:
                logger.debug(
                    "Schema rejection — batch %s missing fields: %s",
                    batch_id,
                    missing,
                )
                rejected_count += 1
                continue

            enriched = {
                **raw,
                "_bronze_id": str(uuid.uuid4()),
                "_ingested_at": ingested_at,
                "_source_file": source_file,
                "_batch_id": batch_id,
            }
            valid_records.append(enriched)

        # Persist to JSON Lines
        output_path = self._bronze_dir / f"{batch_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as fh:
            for rec in valid_records:
                fh.write(json.dumps(rec, default=str) + "\n")

        # Update stats
        self.records_ingested += len(valid_records)
        self.batches_processed += 1
        self.last_batch_id = batch_id
        self._schema_rejected += rejected_count

        summary = {
            "batch_id": batch_id,
            "records_accepted": len(valid_records),
            "records_rejected_schema": rejected_count,
            "total_input": len(records),
            "output_path": str(output_path),
            "ingested_at": ingested_at,
        }

        logger.info(
            "Bronze ingestion complete — batch=%s accepted=%d rejected=%d",
            batch_id,
            len(valid_records),
            rejected_count,
        )
        return summary

    def read_bronze(self, batch_id: str) -> pd.DataFrame:
        """
        Read a persisted Bronze batch back into a DataFrame.

        Args:
            batch_id: The batch_id returned by ingest().

        Returns:
            DataFrame with all records from the batch.

        Raises:
            FileNotFoundError: If the batch file does not exist.
        """
        path = self._bronze_dir / f"{batch_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Bronze batch not found: {batch_id} (looked in {path})"
            )

        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def list_batches(self) -> list[str]:
        """Return a list of all persisted batch IDs."""
        return [
            p.stem
            for p in sorted(self._bronze_dir.glob("bronze_*.jsonl"))
        ]

    def stats(self) -> dict[str, Any]:
        """Return cumulative ingestion statistics."""
        return {
            "records_ingested": self.records_ingested,
            "batches_processed": self.batches_processed,
            "last_batch_id": self.last_batch_id,
            "schema_rejected": self._schema_rejected,
        }
