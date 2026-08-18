"""
Pydantic schemas for the FastAPI control plane.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Layer(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RunPipelineRequest(BaseModel):
    batch_size: int = Field(default=1000, ge=10, le=50_000, description="Number of synthetic events to generate")
    corruption_rate: float = Field(default=0.08, ge=0.0, le=1.0, description="Fraction of deliberately corrupted records")
    label: str = Field(default="", max_length=100, description="Optional human label for this run")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class RuleResultSchema(BaseModel):
    rule_name: str
    column: str | list[str]
    passed: int
    failed: int
    failure_rate: float
    details: str


class QualityReportSchema(BaseModel):
    total_records: int
    passed: int
    failed: int
    failure_rate: float
    failure_breakdown_by_rule: list[RuleResultSchema]


class StageProgress(BaseModel):
    stage: str
    status: PipelineStatus
    records_in: int = 0
    records_out: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class PipelineRunResponse(BaseModel):
    run_id: str
    status: PipelineStatus
    label: str
    batch_size: int
    started_at: datetime
    completed_at: datetime | None = None
    stages: list[StageProgress] = Field(default_factory=list)
    bronze_batch_id: str | None = None
    silver_batch_id: str | None = None
    gold_run_id: str | None = None
    quality_report: QualityReportSchema | None = None
    error: str | None = None
    duration_seconds: float | None = None


class PipelineRunSummary(BaseModel):
    run_id: str
    status: PipelineStatus
    label: str
    started_at: datetime
    completed_at: datetime | None = None
    batch_size: int
    records_passed: int = 0
    failure_rate: float = 0.0
    duration_seconds: float | None = None


class QuarantineRecord(BaseModel):
    index: int
    event_id: str | None
    user_id: str | None
    event_type: str | None
    timestamp: str | None
    price: Any
    quantity: Any
    country: str | None
    quarantine_reasons: str


class QuarantineResponse(BaseModel):
    run_id: str
    total_quarantined: int
    sample: list[QuarantineRecord]


class GoldTableStat(BaseModel):
    table: str
    stats: dict[str, Any]


class GoldSummaryResponse(BaseModel):
    total_revenue: float
    unique_categories: int
    avg_order_value: float
    top_product: str | None
    top_device: str | None
    top_referrer: str | None
    tables: list[GoldTableStat]


class CatalogEntrySchema(BaseModel):
    batch_id: str
    layer: str
    created_at: str
    stats: dict[str, Any]
    schema_: dict[str, str] = Field(default_factory=dict, alias="schema")

    class Config:
        populate_by_name = True


class LineageResponse(BaseModel):
    batch_id: str
    layer: str | None
    upstream: dict[str, Any] | None
    downstream: dict[str, Any] | None
    full_chain: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
