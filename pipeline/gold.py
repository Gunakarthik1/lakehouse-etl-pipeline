"""
Gold Layer — Business-level aggregations from Silver data.

Produces four analytical tables:
  1. revenue_by_category      — Daily revenue per product category
  2. user_cohorts             — Weekly new-user cohort retention (week 0 vs week 1)
  3. product_performance      — Top products by views → purchase conversion
  4. channel_mix              — Event counts and revenue by device × referrer
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data" / "gold"


class GoldLayer:
    """
    Gold aggregation layer.

    Reads a clean Silver DataFrame and produces four partitioned Parquet
    tables. Returns per-table summary statistics for monitoring.
    """

    def __init__(self, gold_dir: Path | None = None) -> None:
        self._gold_dir = Path(gold_dir) if gold_dir else GOLD_DIR
        self._gold_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, silver_df: pd.DataFrame) -> dict[str, Any]:
        """
        Run all four Gold aggregations against the supplied Silver DataFrame.

        Args:
            silver_df: Clean, normalised Silver DataFrame.

        Returns:
            Dict mapping table name → summary statistics dict.
        """
        if silver_df.empty:
            logger.warning("Gold aggregate called with empty Silver DataFrame.")
            return {}

        gold_run_id = f"gold_{uuid.uuid4().hex[:12]}"
        run_ts = datetime.now(timezone.utc).isoformat()

        df = self._prepare(silver_df)
        summaries: dict[str, Any] = {}

        summaries["revenue_by_category"] = self._revenue_by_category(df, gold_run_id)
        summaries["user_cohorts"] = self._user_cohorts(df, gold_run_id)
        summaries["product_performance"] = self._product_performance(df, gold_run_id)
        summaries["channel_mix"] = self._channel_mix(df, gold_run_id)

        summaries["_meta"] = {
            "gold_run_id": gold_run_id,
            "aggregated_at": run_ts,
            "input_records": len(silver_df),
        }

        logger.info(
            "Gold aggregation complete — run_id=%s input_records=%d",
            gold_run_id,
            len(silver_df),
        )
        return summaries

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast / parse fields needed across multiple aggregations."""
        df = df.copy()

        # Parse timestamp
        df["_ts"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
        df["_date"] = df["_ts"].dt.date
        # Strip tz before converting to Period to avoid pandas UserWarning,
        # then floor to week start in UTC.
        df["_week"] = (
            df["_ts"]
            .dt.tz_localize(None)
            .dt.to_period("W")
            .apply(lambda p: p.start_time if pd.notna(p) else pd.NaT)
        )

        # Ensure price / quantity are numeric
        df["price"] = pd.to_numeric(df.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        df["quantity"] = pd.to_numeric(df.get("quantity", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)
        df["_revenue"] = df["price"] * df["quantity"]

        # Normalise string categoricals
        for col in ("event_type", "category", "device_type", "referrer", "product_id", "product_name"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        return df

    # ------------------------------------------------------------------
    # Aggregation 1: Daily revenue by category
    # ------------------------------------------------------------------

    def _revenue_by_category(
        self, df: pd.DataFrame, run_id: str
    ) -> dict[str, Any]:
        purchase_df = df[df["event_type"] == "purchase"].copy()

        if purchase_df.empty:
            self._save_empty("revenue_by_category", run_id)
            return {"rows": 0, "total_revenue": 0.0, "categories": []}

        agg = (
            purchase_df
            .groupby(["_date", "category"], dropna=False)
            .agg(
                total_revenue=("_revenue", "sum"),
                order_count=("event_id", "count"),
                avg_order_value=("_revenue", "mean"),
                unique_users=("user_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"_date": "date"})
        )
        agg["gold_run_id"] = run_id

        path = self._gold_dir / "revenue_by_category.parquet"
        agg.to_parquet(path, index=False, engine="pyarrow")

        return {
            "rows": len(agg),
            "total_revenue": round(float(agg["total_revenue"].sum()), 2),
            "categories": sorted(agg["category"].dropna().unique().tolist()),
            "avg_order_value": round(float(agg["avg_order_value"].mean()), 2),
        }

    # ------------------------------------------------------------------
    # Aggregation 2: User cohort retention
    # ------------------------------------------------------------------

    def _user_cohorts(
        self, df: pd.DataFrame, run_id: str
    ) -> dict[str, Any]:
        user_first = (
            df.dropna(subset=["user_id", "_week"])
            .groupby("user_id")["_week"]
            .min()
            .rename("cohort_week")
            .reset_index()
        )

        merged = df.merge(user_first, on="user_id", how="left")
        merged["weeks_since_first"] = (
            (merged["_week"] - merged["cohort_week"]).dt.days / 7
        ).round().astype("Int64")

        cohort_activity = (
            merged.groupby(["cohort_week", "weeks_since_first"])["user_id"]
            .nunique()
            .rename("active_users")
            .reset_index()
        )

        # Cohort size = users active in week 0
        cohort_size = (
            cohort_activity[cohort_activity["weeks_since_first"] == 0]
            .set_index("cohort_week")["active_users"]
            .rename("cohort_size")
        )
        cohort_activity = cohort_activity.merge(
            cohort_size, on="cohort_week", how="left"
        )
        cohort_activity["retention_rate"] = (
            cohort_activity["active_users"] / cohort_activity["cohort_size"]
        ).round(4)

        cohort_activity["cohort_week"] = cohort_activity["cohort_week"].astype(str)
        cohort_activity["gold_run_id"] = run_id

        path = self._gold_dir / "user_cohorts.parquet"
        cohort_activity.to_parquet(path, index=False, engine="pyarrow")

        # Week-1 retention (average across cohorts)
        w1 = cohort_activity[cohort_activity["weeks_since_first"] == 1]
        avg_w1_retention = float(w1["retention_rate"].mean()) if not w1.empty else 0.0

        return {
            "rows": len(cohort_activity),
            "cohorts": int(cohort_size.shape[0]),
            "avg_week1_retention": round(avg_w1_retention, 4),
        }

    # ------------------------------------------------------------------
    # Aggregation 3: Product performance (views → purchase conversion)
    # ------------------------------------------------------------------

    def _product_performance(
        self, df: pd.DataFrame, run_id: str
    ) -> dict[str, Any]:
        views = (
            df[df["event_type"] == "page_view"]
            .groupby("product_id")["event_id"]
            .count()
            .rename("view_count")
        )
        carts = (
            df[df["event_type"] == "add_to_cart"]
            .groupby("product_id")["event_id"]
            .count()
            .rename("cart_count")
        )
        purchases = (
            df[df["event_type"] == "purchase"]
            .groupby("product_id")
            .agg(
                purchase_count=("event_id", "count"),
                total_revenue=("_revenue", "sum"),
            )
        )

        # Product name lookup (last seen)
        name_map = (
            df.dropna(subset=["product_id", "product_name"])
            .drop_duplicates("product_id")
            .set_index("product_id")["product_name"]
        )
        category_map = (
            df.dropna(subset=["product_id", "category"])
            .drop_duplicates("product_id")
            .set_index("product_id")["category"]
        )

        perf = (
            views.to_frame()
            .join(carts, how="outer")
            .join(purchases, how="outer")
            .fillna(0)
        )
        perf["product_name"] = perf.index.map(name_map)
        perf["category"] = perf.index.map(category_map)
        perf["view_to_purchase_rate"] = (
            perf["purchase_count"] / perf["view_count"].replace(0, float("nan"))
        ).round(4).fillna(0)
        perf["cart_to_purchase_rate"] = (
            perf["purchase_count"] / perf["cart_count"].replace(0, float("nan"))
        ).round(4).fillna(0)
        perf = perf.reset_index().rename(columns={"product_id": "product_id"})
        perf["gold_run_id"] = run_id

        path = self._gold_dir / "product_performance.parquet"
        perf.to_parquet(path, index=False, engine="pyarrow")

        top_product = (
            perf.sort_values("total_revenue", ascending=False).iloc[0]
            if not perf.empty
            else None
        )

        return {
            "rows": len(perf),
            "top_product_by_revenue": str(top_product["product_name"]) if top_product is not None else None,
            "avg_conversion_rate": round(float(perf["view_to_purchase_rate"].mean()), 4),
            "total_revenue": round(float(perf["total_revenue"].sum()), 2),
        }

    # ------------------------------------------------------------------
    # Aggregation 4: Device / channel mix
    # ------------------------------------------------------------------

    def _channel_mix(
        self, df: pd.DataFrame, run_id: str
    ) -> dict[str, Any]:
        mix = (
            df.groupby(["device_type", "referrer"], dropna=False)
            .agg(
                event_count=("event_id", "count"),
                unique_users=("user_id", "nunique"),
                revenue=("_revenue", "sum"),
            )
            .reset_index()
        )
        mix["gold_run_id"] = run_id

        path = self._gold_dir / "channel_mix.parquet"
        mix.to_parquet(path, index=False, engine="pyarrow")

        top_device = (
            mix.groupby("device_type")["event_count"].sum().idxmax()
            if not mix.empty
            else None
        )
        top_referrer = (
            mix.groupby("referrer")["event_count"].sum().idxmax()
            if not mix.empty
            else None
        )

        return {
            "rows": len(mix),
            "top_device": top_device,
            "top_referrer": top_referrer,
            "total_events": int(mix["event_count"].sum()),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_empty(self, table: str, run_id: str) -> None:
        path = self._gold_dir / f"{table}.parquet"
        pd.DataFrame().to_parquet(path, index=False, engine="pyarrow")

    def read_gold(self, table: str) -> pd.DataFrame:
        path = self._gold_dir / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Gold table not found: {table}")
        return pd.read_parquet(path, engine="pyarrow")

    def available_tables(self) -> list[str]:
        return [p.stem for p in self._gold_dir.glob("*.parquet")]
