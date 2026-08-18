"""
FastAPI Control Plane — Lakehouse ETL Pipeline.

Endpoints:
  POST  /api/pipeline/run              — Trigger a full Bronze→Silver→Gold run
  GET   /api/pipeline/{run_id}/status  — Status + stage progress
  GET   /api/pipeline/runs             — List all runs (most recent first)
  GET   /api/quality/{run_id}          — Quality report for a run
  GET   /api/gold/summary              — Aggregated Gold layer stats
  GET   /api/catalog                   — Full lineage catalog
  GET   /api/quarantine/{run_id}       — Sample of quarantined records + reasons
  GET   /api/health                    — Health check
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.models import (
    GoldSummaryResponse,
    GoldTableStat,
    HealthResponse,
    Layer,
    LineageResponse,
    PipelineRunResponse,
    PipelineRunSummary,
    PipelineStatus,
    QualityReportSchema,
    QuarantineRecord,
    QuarantineResponse,
    RunPipelineRequest,
    RuleResultSchema,
    StageProgress,
)
from pipeline.bronze import BronzeLayer
from pipeline.catalog import MetadataCatalog
from pipeline.generator import generate_batch
from pipeline.gold import GoldLayer
from pipeline.quality import QualityReport
from pipeline.silver import SilverLayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
QUARANTINE_DIR = BASE_DIR / "data" / "quarantine"

app = FastAPI(
    title="Lakehouse ETL Pipeline",
    description="Medallion architecture (Bronze→Silver→Gold) ETL control plane.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Lakehouse ETL Pipeline API — visit /docs"}


# ---------------------------------------------------------------------------
# Shared singletons (process-level state)
# ---------------------------------------------------------------------------

_bronze = BronzeLayer()
_silver = SilverLayer()
_gold = GoldLayer()
_catalog = MetadataCatalog()

# In-memory run registry {run_id: PipelineRunResponse}
_runs: dict[str, PipelineRunResponse] = {}

# Per-run quarantine data {run_id: quarantine_df}
_quarantine_store: dict[str, pd.DataFrame] = {}

# Per-run gold summaries {run_id: gold_summaries_dict}
_gold_summaries: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Background task — full pipeline execution
# ---------------------------------------------------------------------------

def _run_pipeline(run_id: str, request: RunPipelineRequest) -> None:
    """Execute Bronze→Silver→Gold pipeline in background thread."""
    run = _runs[run_id]
    run.status = PipelineStatus.RUNNING
    now = lambda: datetime.now(timezone.utc)  # noqa: E731

    stages: list[StageProgress] = [
        StageProgress(stage="bronze", status=PipelineStatus.PENDING),
        StageProgress(stage="silver", status=PipelineStatus.PENDING),
        StageProgress(stage="gold", status=PipelineStatus.PENDING),
    ]
    run.stages = stages

    try:
        # ---- Bronze ----
        stages[0].status = PipelineStatus.RUNNING
        stages[0].started_at = now()

        records = generate_batch(n=request.batch_size, corruption_rate=request.corruption_rate)
        bronze_summary = _bronze.ingest(records, source_label="api_trigger")
        bronze_batch_id = bronze_summary["batch_id"]
        bronze_df = _bronze.read_bronze(bronze_batch_id)

        _catalog.register(
            bronze_batch_id,
            "bronze",
            stats=bronze_summary,
            schema={c: str(bronze_df[c].dtype) for c in bronze_df.columns},
        )

        stages[0].status = PipelineStatus.COMPLETED
        stages[0].completed_at = now()
        stages[0].records_in = len(records)
        stages[0].records_out = bronze_summary["records_accepted"]
        run.bronze_batch_id = bronze_batch_id

        # ---- Silver ----
        stages[1].status = PipelineStatus.RUNNING
        stages[1].started_at = now()

        silver_batch_id = f"silver_{uuid.uuid4().hex[:12]}"
        silver_df, quality_report = _silver.transform(bronze_df, batch_id=silver_batch_id)

        if silver_df is None or silver_df.empty:
            silver_df = pd.DataFrame()

        _catalog.register(
            silver_batch_id,
            "silver",
            stats={
                "records": len(silver_df),
                "quality": quality_report.to_dict(),
            },
            schema={c: str(silver_df[c].dtype) for c in silver_df.columns} if not silver_df.empty else {},
        )
        _catalog.link_lineage(bronze_batch_id, silver_batch_id, "bronze_to_silver")

        # Store quarantined records for later retrieval
        quarantine_path = QUARANTINE_DIR / f"{silver_batch_id}_quarantine.parquet"
        if quarantine_path.exists():
            _quarantine_store[run_id] = pd.read_parquet(quarantine_path)

        stages[1].status = PipelineStatus.COMPLETED
        stages[1].completed_at = now()
        stages[1].records_in = bronze_summary["records_accepted"]
        stages[1].records_out = len(silver_df)
        run.silver_batch_id = silver_batch_id
        run.quality_report = _to_quality_schema(quality_report)

        # ---- Gold ----
        stages[2].status = PipelineStatus.RUNNING
        stages[2].started_at = now()

        if not silver_df.empty:
            gold_summaries = _gold.aggregate(silver_df)
        else:
            gold_summaries = {}

        gold_run_id = gold_summaries.get("_meta", {}).get("gold_run_id", f"gold_{uuid.uuid4().hex[:12]}")
        _gold_summaries[run_id] = gold_summaries

        _catalog.register(
            gold_run_id,
            "gold",
            stats={k: v for k, v in gold_summaries.items() if k != "_meta"},
        )
        _catalog.link_lineage(silver_batch_id, gold_run_id, "silver_to_gold")

        stages[2].status = PipelineStatus.COMPLETED
        stages[2].completed_at = now()
        stages[2].records_in = len(silver_df)
        stages[2].records_out = sum(
            v.get("rows", 0)
            for k, v in gold_summaries.items()
            if isinstance(v, dict) and k != "_meta"
        )
        run.gold_run_id = gold_run_id

        # Finalise run
        run.status = PipelineStatus.COMPLETED
        run.completed_at = now()
        run.duration_seconds = (run.completed_at - run.started_at).total_seconds()

        logger.info("Pipeline run %s completed successfully.", run_id)

    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline run %s failed: %s\n%s", run_id, exc, traceback.format_exc())
        for stage in stages:
            if stage.status == PipelineStatus.RUNNING:
                stage.status = PipelineStatus.FAILED
                stage.error = str(exc)
                stage.completed_at = now()
        run.status = PipelineStatus.FAILED
        run.completed_at = now()
        run.error = str(exc)
        run.duration_seconds = (run.completed_at - run.started_at).total_seconds()


def _to_quality_schema(report: QualityReport) -> QualityReportSchema:
    return QualityReportSchema(
        total_records=report.total_records,
        passed=report.passed,
        failed=report.failed,
        failure_rate=report.failure_rate,
        failure_breakdown_by_rule=[
            RuleResultSchema(
                rule_name=r.rule_name,
                column=r.column if isinstance(r.column, str) else str(r.column),
                passed=r.passed,
                failed=r.failed,
                failure_rate=r.failure_rate,
                details=r.details,
            )
            for r in report.failure_breakdown_by_rule
        ],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/api/pipeline/run", response_model=PipelineRunResponse, status_code=202)
async def run_pipeline(request: RunPipelineRequest, background_tasks: BackgroundTasks):
    """Trigger a full Bronze→Silver→Gold pipeline run asynchronously."""
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)

    run = PipelineRunResponse(
        run_id=run_id,
        status=PipelineStatus.PENDING,
        label=request.label or f"Run {len(_runs) + 1}",
        batch_size=request.batch_size,
        started_at=now,
    )
    _runs[run_id] = run
    background_tasks.add_task(_run_pipeline, run_id, request)
    return run


@app.get("/api/pipeline/runs", response_model=list[PipelineRunSummary])
async def list_runs():
    """List all pipeline runs, most recent first."""
    summaries = []
    for run in sorted(_runs.values(), key=lambda r: r.started_at, reverse=True):
        qr = run.quality_report
        summaries.append(
            PipelineRunSummary(
                run_id=run.run_id,
                status=run.status,
                label=run.label,
                started_at=run.started_at,
                completed_at=run.completed_at,
                batch_size=run.batch_size,
                records_passed=qr.passed if qr else 0,
                failure_rate=qr.failure_rate if qr else 0.0,
                duration_seconds=run.duration_seconds,
            )
        )
    return summaries[:50]


@app.get("/api/pipeline/{run_id}/status", response_model=PipelineRunResponse)
async def get_run_status(run_id: str):
    """Return full status and stage progress for a run."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.get("/api/quality/{run_id}", response_model=QualityReportSchema)
async def get_quality_report(run_id: str):
    """Return the quality report for a completed run."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if not run.quality_report:
        raise HTTPException(status_code=404, detail="Quality report not yet available for this run.")
    return run.quality_report


@app.get("/api/quarantine/{run_id}", response_model=QuarantineResponse)
async def get_quarantine(run_id: str, limit: int = 50):
    """Return a sample of quarantined records for a run."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    qdf = _quarantine_store.get(run_id)
    if qdf is None or qdf.empty:
        return QuarantineResponse(run_id=run_id, total_quarantined=0, sample=[])

    sample_df = qdf.head(limit)
    records = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        records.append(
            QuarantineRecord(
                index=i,
                event_id=str(row.get("event_id", "")) or None,
                user_id=str(row.get("user_id", "")) or None,
                event_type=str(row.get("event_type", "")) or None,
                timestamp=str(row.get("timestamp", "")) or None,
                price=row.get("price"),
                quantity=row.get("quantity"),
                country=str(row.get("country", "")) or None,
                quarantine_reasons=str(row.get("_quarantine_reasons", "")),
            )
        )

    return QuarantineResponse(
        run_id=run_id,
        total_quarantined=len(qdf),
        sample=records,
    )


@app.get("/api/gold/summary", response_model=GoldSummaryResponse)
async def gold_summary():
    """Return aggregated Gold layer statistics from the most recent run."""
    if not _gold_summaries:
        raise HTTPException(status_code=404, detail="No Gold data available yet. Run the pipeline first.")

    # Merge summaries from all runs (latest values win)
    merged: dict[str, Any] = {}
    for run_summaries in _gold_summaries.values():
        merged.update(run_summaries)

    revenue_stats = merged.get("revenue_by_category", {})
    product_stats = merged.get("product_performance", {})
    channel_stats = merged.get("channel_mix", {})

    total_revenue = float(revenue_stats.get("total_revenue", 0.0))
    avg_order_value = float(revenue_stats.get("avg_order_value", 0.0))
    categories = revenue_stats.get("categories", [])
    top_product = product_stats.get("top_product_by_revenue")
    top_device = channel_stats.get("top_device")
    top_referrer = channel_stats.get("top_referrer")

    tables = [
        GoldTableStat(table=k, stats=v)
        for k, v in merged.items()
        if isinstance(v, dict) and k != "_meta"
    ]

    return GoldSummaryResponse(
        total_revenue=round(total_revenue, 2),
        unique_categories=len(categories),
        avg_order_value=round(avg_order_value, 2),
        top_product=top_product,
        top_device=top_device,
        top_referrer=top_referrer,
        tables=tables,
    )


@app.get("/api/catalog")
async def get_catalog():
    """Return the full metadata catalog with lineage information."""
    return _catalog.all_entries()


@app.get("/api/catalog/lineage/{batch_id}", response_model=LineageResponse)
async def get_lineage(batch_id: str):
    """Return the lineage trace for a given batch ID."""
    lineage = _catalog.get_lineage(batch_id)
    return LineageResponse(**lineage)
