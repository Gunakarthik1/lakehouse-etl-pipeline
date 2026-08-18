"""
FastAPI Control Plane — Lakehouse ETL Pipeline.

Endpoints:
  POST  /api/pipeline/run              — Trigger ETL with SSE progress streaming
  GET   /api/pipeline/runs             — List all runs (most recent first)
  GET   /api/pipeline/stats            — Current dataset stats
  GET   /api/pipeline/{run_id}/status  — Status + stage progress
  POST  /api/upload                    — Accept CSV file upload
  POST  /api/inject-corruption         — Inject bad data into current dataset
  POST  /api/query                     — Run SQL on gold layer (SQLite)
  GET   /api/quality/{run_id}          — Quality report for a run
  GET   /api/gold/summary              — Aggregated Gold layer stats
  GET   /api/catalog                   — Full lineage catalog
  GET   /api/quarantine/{run_id}       — Sample of quarantined records
  GET   /api/health                    — Health check
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import sqlite3
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    version="2.0.0",
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

# Current active dataset (loaded from sample or upload)
_current_df: pd.DataFrame | None = None
_current_dataset_name: str = "ecommerce"

# Latest gold DataFrame for SQL queries
_gold_df: pd.DataFrame | None = None
_silver_df_latest: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Built-in sample dataset generators
# ---------------------------------------------------------------------------

def _generate_ecommerce(n: int = 500) -> pd.DataFrame:
    """E-commerce transactions dataset."""
    rng = random.Random(42)
    categories = ["Electronics", "Clothing", "Food", "Sports", "Books", "Home", "Beauty"]
    rows = []
    base_date = datetime(2024, 1, 1)
    for i in range(n):
        cat = rng.choice(categories)
        days_offset = rng.randint(0, 364)
        rows.append({
            "order_id": f"ORD-{10000 + i}",
            "customer_id": f"CUST-{rng.randint(1000, 2000)}",
            "amount": round(rng.uniform(5.0, 500.0), 2),
            "date": (base_date.replace(month=1, day=1) + __import__("datetime").timedelta(days=days_offset)).strftime("%Y-%m-%d"),
            "category": cat,
        })
    return pd.DataFrame(rows)


def _generate_iot(n: int = 1000) -> pd.DataFrame:
    """IoT sensor readings dataset."""
    rng = random.Random(99)
    units = ["celsius", "psi", "rpm", "volts", "percent"]
    locations = ["Plant-A", "Plant-B", "Warehouse-1", "Warehouse-2", "Office"]
    rows = []
    base_ts = int(datetime(2024, 6, 1).timestamp())
    for i in range(n):
        unit = rng.choice(units)
        rows.append({
            "sensor_id": f"SENS-{rng.randint(100, 120):03d}",
            "timestamp": datetime.fromtimestamp(base_ts + i * 60).strftime("%Y-%m-%dT%H:%M:%S"),
            "value": round(rng.uniform(0.0, 100.0), 3),
            "unit": unit,
            "location": rng.choice(locations),
        })
    return pd.DataFrame(rows)


def _generate_employees(n: int = 200) -> pd.DataFrame:
    """Employee records dataset."""
    rng = random.Random(7)
    depts = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    first_names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank", "Iris", "Jack"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson"]
    rows = []
    for i in range(n):
        hire_year = rng.randint(2010, 2023)
        rows.append({
            "emp_id": f"EMP-{5000 + i}",
            "name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
            "dept": rng.choice(depts),
            "salary": round(rng.uniform(40000, 180000), 2),
            "hire_date": f"{hire_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        })
    return pd.DataFrame(rows)


SAMPLE_DATASETS = {
    "ecommerce": ("E-commerce transactions", _generate_ecommerce),
    "iot": ("IoT sensor readings", _generate_iot),
    "employees": ("Employee records", _generate_employees),
}


def _load_sample_dataset(name: str) -> pd.DataFrame:
    global _current_df, _current_dataset_name
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    label, gen_fn = SAMPLE_DATASETS[name]
    _current_df = gen_fn()
    _current_dataset_name = name
    return _current_df


def _get_current_df() -> pd.DataFrame:
    global _current_df
    if _current_df is None:
        _current_df = _generate_ecommerce()
    return _current_df


# ---------------------------------------------------------------------------
# SSE pipeline runner
# ---------------------------------------------------------------------------

async def _sse_pipeline_generator(dataset_name: str | None, df_override: pd.DataFrame | None):
    """Async generator that yields SSE events for each pipeline stage."""
    global _current_df, _current_dataset_name, _gold_df, _silver_df_latest

    async def emit(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    try:
        # Load dataset
        if df_override is not None:
            df = df_override.copy()
        elif dataset_name and dataset_name in SAMPLE_DATASETS:
            df = _load_sample_dataset(dataset_name)
        else:
            df = _get_current_df().copy()

        rows_in = len(df)

        # ---- BRONZE stage ----
        yield await emit({"stage": "bronze", "status": "running", "rows_in": rows_in, "rows_out": 0, "duration_ms": 0})
        t0 = time.time()
        await asyncio.sleep(random.uniform(0.4, 0.8))

        # Bronze: minimal schema enforcement & tagging
        bronze_df = df.copy()
        bronze_df["_bronze_id"] = [str(uuid.uuid4()) for _ in range(len(bronze_df))]
        bronze_df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
        bronze_rows = len(bronze_df)
        bronze_ms = int((time.time() - t0) * 1000)

        yield await emit({"stage": "bronze", "status": "complete", "rows_in": rows_in, "rows_out": bronze_rows, "duration_ms": bronze_ms})
        await asyncio.sleep(0.1)

        # ---- SILVER stage ----
        yield await emit({"stage": "silver", "status": "running", "rows_in": bronze_rows, "rows_out": 0, "duration_ms": 0})
        t1 = time.time()
        await asyncio.sleep(random.uniform(0.5, 1.0))

        # Silver: clean data — drop nulls in key columns, deduplicate
        silver_df = bronze_df.copy()
        key_cols = [c for c in silver_df.columns if not c.startswith("_")]

        # Drop rows where all non-meta columns are null
        silver_df = silver_df.dropna(subset=key_cols, how="all")

        # Deduplicate if there's an id column
        id_cols = [c for c in silver_df.columns if "id" in c.lower() and not c.startswith("_")]
        if id_cols:
            silver_df = silver_df.drop_duplicates(subset=[id_cols[0]])

        silver_df["_silver_id"] = [str(uuid.uuid4()) for _ in range(len(silver_df))]
        silver_df["_processed_at"] = datetime.now(timezone.utc).isoformat()
        silver_rows = len(silver_df)
        silver_ms = int((time.time() - t1) * 1000)

        _silver_df_latest = silver_df.copy()

        yield await emit({"stage": "silver", "status": "complete", "rows_in": bronze_rows, "rows_out": silver_rows, "duration_ms": silver_ms})
        await asyncio.sleep(0.1)

        # ---- GOLD stage ----
        yield await emit({"stage": "gold", "status": "running", "rows_in": silver_rows, "rows_out": 0, "duration_ms": 0})
        t2 = time.time()
        await asyncio.sleep(random.uniform(0.3, 0.7))

        # Gold: aggregate
        meta_cols = [c for c in silver_df.columns if c.startswith("_")]
        gold_df = silver_df.drop(columns=meta_cols, errors="ignore")

        # Build numeric aggregates
        num_cols = gold_df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            agg = gold_df[num_cols].agg(["sum", "mean", "min", "max"])
            gold_rows = len(agg)
        else:
            gold_rows = len(gold_df.columns)

        _gold_df = gold_df.copy()
        gold_ms = int((time.time() - t2) * 1000)

        yield await emit({"stage": "gold", "status": "complete", "rows_in": silver_rows, "rows_out": gold_rows, "duration_ms": gold_ms})

    except Exception as exc:
        logger.error("SSE pipeline error: %s\n%s", exc, traceback.format_exc())
        yield await emit({"stage": "unknown", "status": "error", "rows_in": 0, "rows_out": 0, "duration_ms": 0, "error_msg": str(exc)})


# ---------------------------------------------------------------------------
# Background task — full pipeline execution (legacy polling path)
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
            stats={"records": len(silver_df), "quality": quality_report.to_dict()},
            schema={c: str(silver_df[c].dtype) for c in silver_df.columns} if not silver_df.empty else {},
        )
        _catalog.link_lineage(bronze_batch_id, silver_batch_id, "bronze_to_silver")

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

        run.status = PipelineStatus.COMPLETED
        run.completed_at = now()
        run.duration_seconds = (run.completed_at - run.started_at).total_seconds()

        logger.info("Pipeline run %s completed successfully.", run_id)

    except Exception as exc:
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
# Request models for new endpoints
# ---------------------------------------------------------------------------

class DatasetSelectRequest(BaseModel):
    dataset: str = "ecommerce"


class PipelineRunSSERequest(BaseModel):
    dataset: str | None = None


class CorruptionRequest(BaseModel):
    type: str = "nulls"  # nulls | duplicates | type_errors | outliers


class SQLQueryRequest(BaseModel):
    sql: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="2.0.0",
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/api/pipeline/run")
async def run_pipeline_sse(request: PipelineRunSSERequest):
    """Trigger full ETL pipeline with SSE progress streaming."""
    dataset_name = request.dataset or _current_dataset_name

    return StreamingResponse(
        _sse_pipeline_generator(dataset_name, None),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV file upload, validate headers, store in memory, run pipeline on it."""
    global _current_df, _current_dataset_name

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least 2 columns.")

    _current_df = df
    _current_dataset_name = f"upload:{file.filename}"

    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "preview": preview,
        "filename": file.filename,
    }


@app.post("/api/dataset/select")
async def select_dataset(request: DatasetSelectRequest):
    """Select a built-in sample dataset."""
    global _current_df, _current_dataset_name
    if request.dataset not in SAMPLE_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{request.dataset}'. Valid: {list(SAMPLE_DATASETS.keys())}")
    df = _load_sample_dataset(request.dataset)
    label, _ = SAMPLE_DATASETS[request.dataset]
    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")
    return {
        "dataset": request.dataset,
        "label": label,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": preview,
    }


@app.post("/api/inject-corruption")
async def inject_corruption(request: CorruptionRequest):
    """Inject bad data into the current dataset."""
    global _current_df
    df = _get_current_df().copy()

    corruption_type = request.type
    injected_count = 0
    description = ""

    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = [c for c in df.columns if not c.startswith("_")]

    if corruption_type == "nulls":
        n_to_corrupt = max(1, int(len(df) * 0.10))
        indices = random.sample(range(len(df)), n_to_corrupt)
        col = random.choice(all_cols) if all_cols else df.columns[0]
        df.loc[indices, col] = None
        injected_count = n_to_corrupt
        description = f"Set {n_to_corrupt} values in column '{col}' to null (10% of rows)"

    elif corruption_type == "duplicates":
        n_to_dupe = max(1, int(len(df) * 0.05))
        sample_rows = df.sample(n=min(n_to_dupe, len(df)), random_state=42)
        df = pd.concat([df, sample_rows], ignore_index=True)
        injected_count = len(sample_rows)
        description = f"Duplicated {injected_count} rows (5% of dataset)"

    elif corruption_type == "type_errors":
        if num_cols:
            col = random.choice(num_cols)
            n_to_corrupt = max(1, int(len(df) * 0.08))
            indices = random.sample(range(len(df)), n_to_corrupt)
            df[col] = df[col].astype(object)
            for idx in indices:
                df.at[idx, col] = "N/A"
            injected_count = n_to_corrupt
            description = f"Converted {n_to_corrupt} numeric values in '{col}' to string 'N/A'"
        else:
            description = "No numeric columns to corrupt"

    elif corruption_type == "outliers":
        if num_cols:
            col = random.choice(num_cols)
            n_to_corrupt = max(1, int(len(df) * 0.03))
            indices = random.sample(range(len(df)), n_to_corrupt)
            col_max = df[col].max()
            for idx in indices:
                df.at[idx, col] = col_max * random.uniform(50, 200)
            injected_count = n_to_corrupt
            description = f"Injected {n_to_corrupt} extreme outlier values in '{col}' (50-200x max)"
        else:
            description = "No numeric columns for outliers"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown corruption type: {corruption_type}. Valid: nulls, duplicates, type_errors, outliers")

    _current_df = df
    return {
        "injected_count": injected_count,
        "description": description,
        "total_rows": len(df),
    }


@app.post("/api/query")
async def run_sql_query(request: SQLQueryRequest):
    """Run SQL on the gold layer using in-memory SQLite."""
    sql = request.sql.strip()

    # Safety: only allow SELECT statements
    sql_upper = sql.upper().lstrip()
    if not sql_upper.startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT statements are allowed.")

    blocked = re.compile(r"\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE|REPLACE|EXEC|EXECUTE)\b", re.IGNORECASE)
    if blocked.search(sql):
        raise HTTPException(status_code=400, detail="Statement contains forbidden keywords.")

    # Use current dataset for SQL
    df = _get_current_df()

    try:
        t_start = time.time()
        conn = sqlite3.connect(":memory:")

        # Load current dataset into SQLite
        table_name = "data"
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Also load with dataset-specific names for convenience
        ds = _current_dataset_name.split(":")[0]
        if ds == "ecommerce":
            df.to_sql("transactions", conn, if_exists="replace", index=False)
        elif ds == "iot":
            df.to_sql("sensors", conn, if_exists="replace", index=False)
        elif ds == "employees":
            df.to_sql("employees", conn, if_exists="replace", index=False)

        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()

        exec_ms = int((time.time() - t_start) * 1000)

        return {
            "columns": columns,
            "rows": [list(r) for r in rows[:50]],
            "execution_ms": exec_ms,
            "row_count": len(rows),
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {e}")


@app.get("/api/pipeline/stats")
async def pipeline_stats():
    """Return current dataset statistics."""
    df = _get_current_df()

    ds = _current_dataset_name
    label = SAMPLE_DATASETS.get(ds.split(":")[0], (ds, None))[0]

    null_count = int(df.isnull().sum().sum())
    total_cells = len(df) * len(df.columns)
    null_rate = round((null_count / total_cells * 100) if total_cells > 0 else 0, 2)

    dup_count = int(df.duplicated().sum())
    dup_rate = round((dup_count / len(df) * 100) if len(df) > 0 else 0, 2)

    # Data quality score (0-100)
    completeness = max(0, 100 - null_rate)
    uniqueness = max(0, 100 - dup_rate)
    quality_score = round((completeness * 0.6 + uniqueness * 0.4), 1)

    return {
        "dataset": ds,
        "label": label,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_count": null_count,
        "null_rate": null_rate,
        "duplicate_count": dup_count,
        "duplicate_rate": dup_rate,
        "quality_score": quality_score,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "preview": df.head(10).where(pd.notnull(df.head(10)), None).to_dict(orient="records"),
    }


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
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if not run.quality_report:
        raise HTTPException(status_code=404, detail="Quality report not yet available.")
    return run.quality_report


@app.get("/api/quarantine/{run_id}", response_model=QuarantineResponse)
async def get_quarantine(run_id: str, limit: int = 50):
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
    return QuarantineResponse(run_id=run_id, total_quarantined=len(qdf), sample=records)


@app.get("/api/gold/summary", response_model=GoldSummaryResponse)
async def gold_summary():
    """Return aggregated Gold layer statistics from the most recent run."""
    if not _gold_summaries:
        raise HTTPException(status_code=404, detail="No Gold data available yet. Run the pipeline first.")

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
    return _catalog.all_entries()


@app.get("/api/catalog/lineage/{batch_id}", response_model=LineageResponse)
async def get_lineage(batch_id: str):
    lineage = _catalog.get_lineage(batch_id)
    return LineageResponse(**lineage)
