"""
Metadata Catalog — in-memory registry with JSON persistence.

Tracks every batch at every layer, stores schemas and record counts,
and maintains lineage links between Bronze → Silver → Gold.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "data" / "catalog.json"


class MetadataCatalog:
    """
    Thread-safe metadata catalog with JSON persistence.

    Each entry in the catalog represents one processed batch at one layer
    (bronze, silver, or gold). Lineage links connect related batches
    across layers.

    Usage::

        catalog = MetadataCatalog()
        catalog.register("bronze_abc123", "bronze", stats={...})
        catalog.link_lineage("bronze_abc123", "silver_def456", "bronze_to_silver")
        catalog.get_lineage("bronze_abc123")
    """

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._path = Path(catalog_path) if catalog_path else CATALOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # In-memory store: {batch_id: entry_dict}
        self._entries: dict[str, dict[str, Any]] = {}
        # Lineage graph: {batch_id: {relation: target_batch_id}}
        self._lineage: dict[str, dict[str, str]] = {}

        self._load()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        batch_id: str,
        layer: str,
        stats: dict[str, Any] | None = None,
        schema: dict[str, str] | None = None,
    ) -> None:
        """
        Register a batch at a given layer.

        Args:
            batch_id: Unique identifier for the batch.
            layer:    One of 'bronze', 'silver', 'gold'.
            stats:    Arbitrary statistics dict (record counts, quality metrics, etc.).
            schema:   Dict mapping column name → dtype string.
        """
        with self._lock:
            entry: dict[str, Any] = {
                "batch_id": batch_id,
                "layer": layer,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stats": stats or {},
                "schema": schema or {},
            }
            self._entries[batch_id] = entry
            if batch_id not in self._lineage:
                self._lineage[batch_id] = {}
            self._persist()
        logger.debug("Catalog — registered %s at layer=%s", batch_id, layer)

    def link_lineage(
        self,
        source_batch_id: str,
        target_batch_id: str,
        relation: str = "produces",
    ) -> None:
        """
        Record a lineage relationship between two batches.

        Args:
            source_batch_id: Upstream batch.
            target_batch_id: Downstream batch.
            relation:        Label for the relationship (e.g. 'bronze_to_silver').
        """
        with self._lock:
            if source_batch_id not in self._lineage:
                self._lineage[source_batch_id] = {}
            self._lineage[source_batch_id][relation] = target_batch_id

            if target_batch_id not in self._lineage:
                self._lineage[target_batch_id] = {}
            self._lineage[target_batch_id]["derived_from"] = source_batch_id
            self._persist()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_entry(self, batch_id: str) -> dict[str, Any] | None:
        """Return the catalog entry for a batch, or None if not found."""
        with self._lock:
            return self._entries.get(batch_id)

    def get_lineage(self, batch_id: str) -> dict[str, Any]:
        """
        Return a lineage trace for batch_id, walking both upstream
        (derived_from) and downstream (produces) links.

        Returns:
            Dict with keys: batch_id, layer, upstream, downstream,
            full_chain (ordered list of related batch entries).
        """
        with self._lock:
            entry = self._entries.get(batch_id, {})
            links = self._lineage.get(batch_id, {})

            upstream_id = links.get("derived_from")
            downstream_id = links.get("produces") or links.get("bronze_to_silver") or links.get("silver_to_gold")

            chain: list[dict[str, Any]] = []

            # Walk upstream
            current = upstream_id
            while current:
                node = self._entries.get(current, {"batch_id": current})
                chain.insert(0, node)
                current = self._lineage.get(current, {}).get("derived_from")

            # Current node
            chain.append(entry or {"batch_id": batch_id})

            # Walk downstream
            current = downstream_id
            while current:
                node = self._entries.get(current, {"batch_id": current})
                chain.append(node)
                next_links = self._lineage.get(current, {})
                current = next_links.get("produces") or next_links.get("silver_to_gold")

            return {
                "batch_id": batch_id,
                "layer": entry.get("layer"),
                "upstream": self._entries.get(upstream_id) if upstream_id else None,
                "downstream": self._entries.get(downstream_id) if downstream_id else None,
                "full_chain": chain,
            }

    def list_batches(self, layer: str | None = None) -> list[dict[str, Any]]:
        """
        List all catalog entries, optionally filtered by layer.

        Args:
            layer: If provided, only entries for this layer are returned.

        Returns:
            List of entry dicts sorted by created_at descending.
        """
        with self._lock:
            entries = list(self._entries.values())

        if layer:
            entries = [e for e in entries if e.get("layer") == layer]

        return sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)

    def all_entries(self) -> dict[str, Any]:
        """Return the full catalog as a serialisable dict."""
        with self._lock:
            return {
                "entries": list(self._entries.values()),
                "lineage": self._lineage,
            }

    def clear(self) -> None:
        """Wipe the catalog (useful for tests)."""
        with self._lock:
            self._entries.clear()
            self._lineage.clear()
            self._persist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write current state to JSON (caller must hold self._lock)."""
        try:
            payload = {
                "entries": self._entries,
                "lineage": self._lineage,
            }
            tmp_path = self._path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            tmp_path.replace(self._path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Catalog persistence failed: %s", exc)

    def _load(self) -> None:
        """Load catalog from JSON if it exists."""
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._entries = payload.get("entries", {})
            self._lineage = payload.get("lineage", {})
            logger.debug("Catalog loaded — %d entries", len(self._entries))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Catalog load failed (starting fresh): %s", exc)
