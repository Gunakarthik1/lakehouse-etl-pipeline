"""
Silver Layer — Deduplication, normalization, transformation, and quality-gated
persistence.

Responsibilities:
1. Run quality gates → quarantine failures
2. Deduplicate on event_id (keep latest by _ingested_at)
3. Normalize timestamps to UTC ISO format
4. Standardize country codes (uppercase 2-letter ISO)
5. Cast price → float, quantity → int
6. Add _silver_id, _processed_at, _quality_score
7. Persist clean records as Parquet in data/silver/
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.quality import QualityGate, QualityReport

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = BASE_DIR / "data" / "silver"
QUARANTINE_DIR = BASE_DIR / "data" / "quarantine"

# Mapping of common country name variants → 2-letter ISO codes
COUNTRY_NORMALISATION: dict[str, str] = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "u.s.a.": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "canada": "CA",
    "australia": "AU",
    "india": "IN",
    "brazil": "BR",
    "brasil": "BR",
    "japan": "JP",
    "mexico": "MX",
    "méxico": "MX",
}


def _normalise_country(raw: Any) -> str | None:
    """
    Attempt to normalise a raw country value to a 2-letter ISO code.
    Returns the value uppercased if it already looks like a 2-letter code,
    otherwise performs a lookup. Returns None when the value cannot be mapped.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if len(s) == 2:
        return s.upper()
    return COUNTRY_NORMALISATION.get(s.lower())


def _compute_quality_score(row: pd.Series) -> float:
    """
    Compute a per-record quality score in [0, 1] based on field completeness
    and basic sanity. Used as a soft data confidence metric.
    """
    SCORED_FIELDS = [
        "event_id", "user_id", "session_id", "event_type", "timestamp",
        "product_id", "category", "price", "quantity", "country",
        "device_type", "referrer",
    ]
    score = sum(
        1
        for f in SCORED_FIELDS
        if f in row.index and row[f] is not None and str(row[f]).strip() != ""
    ) / len(SCORED_FIELDS)
    return round(score, 4)


class SilverLayer:
    """
    Silver transformation layer.

    Accepts a Bronze DataFrame, applies quality gates, deduplicates,
    normalises fields, and persists clean records as Parquet.
    """

    def __init__(
        self,
        silver_dir: Path | None = None,
        quarantine_dir: Path | None = None,
        quality_gate: QualityGate | None = None,
    ) -> None:
        self._silver_dir = Path(silver_dir) if silver_dir else SILVER_DIR
        self._quarantine_dir = Path(quarantine_dir) if quarantine_dir else QUARANTINE_DIR
        self._silver_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)

        self._quality_gate = quality_gate or QualityGate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        bronze_df: pd.DataFrame,
        batch_id: str | None = None,
    ) -> tuple[pd.DataFrame, QualityReport]:
        """
        Full Silver transformation pipeline.

        Args:
            bronze_df: Raw Bronze DataFrame.
            batch_id:  Optional identifier linking back to the Bronze batch.

        Returns:
            (silver_df, quality_report)
        """
        if bronze_df.empty:
            empty_report = QualityReport(
                total_records=0, passed=0, failed=0, failure_rate=0.0
            )
            return pd.DataFrame(), empty_report

        silver_batch_id = batch_id or f"silver_{uuid.uuid4().hex[:12]}"
        processed_at = datetime.now(timezone.utc).isoformat()

        df = bronze_df.copy()

        # --- Step 1: Quality gate ---
        clean_df, quarantine_df, report = self._quality_gate.validate(df)
        self._persist_quarantine(quarantine_df, silver_batch_id)

        if clean_df.empty:
            logger.warning(
                "Silver transform — no clean records after quality gate for batch %s",
                silver_batch_id,
            )
            return pd.DataFrame(), report

        # --- Step 2: Deduplicate on event_id (keep latest by _ingested_at) ---
        clean_df = self._deduplicate(clean_df)

        # --- Step 3: Normalise timestamps → UTC ISO-8601 ---
        clean_df = self._normalise_timestamps(clean_df)

        # --- Step 4: Standardise country codes ---
        clean_df["country"] = clean_df["country"].apply(_normalise_country)

        # --- Step 5: Type casting ---
        clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce").astype(float)
        clean_df["quantity"] = (
            pd.to_numeric(clean_df["quantity"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # --- Step 6: Flatten / clean string fields ---
        str_cols = ["event_type", "product_name", "category", "device_type", "referrer"]
        for col in str_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype(str).str.strip().str.lower()

        # Restore event_type to original casing (downstream groupby keys)
        # event_type values are already lowercased — keep consistent

        # --- Step 7: Add Silver metadata fields ---
        clean_df["_silver_id"] = [str(uuid.uuid4()) for _ in range(len(clean_df))]
        clean_df["_processed_at"] = processed_at
        clean_df["_silver_batch_id"] = silver_batch_id
        clean_df["_quality_score"] = clean_df.apply(_compute_quality_score, axis=1)

        # --- Step 8: Persist to Parquet ---
        output_path = self._silver_dir / f"{silver_batch_id}.parquet"
        clean_df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(
            "Silver transform complete — batch=%s records=%d quarantined=%d path=%s",
            silver_batch_id,
            len(clean_df),
            len(quarantine_df),
            output_path,
        )

        return clean_df, report

    def read_silver(self, batch_id: str) -> pd.DataFrame:
        """Read a persisted Silver batch."""
        path = self._silver_dir / f"{batch_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Silver batch not found: {path}")
        return pd.read_parquet(path, engine="pyarrow")

    def list_batches(self) -> list[str]:
        return [p.stem for p in sorted(self._silver_dir.glob("silver_*.parquet"))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate event_ids, keeping the record with the latest
        _ingested_at timestamp. Falls back to keeping the first occurrence
        if _ingested_at is absent.
        """
        if "event_id" not in df.columns:
            return df

        if "_ingested_at" in df.columns:
            df = df.sort_values("_ingested_at", ascending=False)

        before = len(df)
        df = df.drop_duplicates(subset=["event_id"], keep="first")
        after = len(df)

        if before != after:
            logger.debug("Deduplicated %d → %d records (removed %d dupes)", before, after, before - after)

        return df

    def _normalise_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp column to UTC ISO-8601 string."""
        if "timestamp" not in df.columns:
            return df

        parsed = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        return df

    def _persist_quarantine(
        self,
        quarantine_df: pd.DataFrame,
        batch_id: str,
    ) -> None:
        """Save quarantined records to the quarantine zone as Parquet.

        Quarantine records may contain mixed types (e.g. both floats and
        strings in a price column because the raw data was deliberately
        corrupted). We cast every column to string so PyArrow can always
        infer a clean schema, preserving the raw values for inspection.
        """
        if quarantine_df.empty:
            return
        path = self._quarantine_dir / f"{batch_id}_quarantine.parquet"
        # Cast object-dtype columns to str to avoid PyArrow schema conflicts
        safe_df = quarantine_df.copy()
        for col in safe_df.columns:
            if safe_df[col].dtype == object:
                safe_df[col] = safe_df[col].astype(str)
        safe_df.to_parquet(path, index=False, engine="pyarrow")
        logger.info(
            "Quarantined %d records → %s", len(quarantine_df), path
        )
