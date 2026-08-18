# Distributed Lakehouse ETL & Data Quality Pipeline

A production-grade Medallion architecture (Bronze → Silver → Gold) ETL pipeline with data quality gates, quarantine zones, and a FastAPI control plane with visual dashboard.

---

## Architecture

```
Raw Events (generator)
        │
        ▼
┌─────────────────┐
│   Bronze Layer  │  Schema presence check, JSON Lines persistence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Quality Gates  │  10 configurable rules → quarantine failures
└────────┬────────┘
         │ clean records
         ▼
┌─────────────────┐
│   Silver Layer  │  Dedup, normalise, cast, enrich → Parquet
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Gold Layer    │  4 analytical aggregations → partitioned Parquet
└─────────────────┘
```

## Quality Rules

| Rule | What it checks |
|---|---|
| `non_null_check` | event_id, user_id, event_type, timestamp must not be null |
| `range_check` (price) | price in [0.01, 100000] |
| `range_check` (quantity) | quantity in [1, 10000] |
| `type_check` (price) | price castable to float |
| `type_check` (quantity) | quantity castable to int |
| `date_range_check` | timestamp between 2020-01-01 and now (no future events) |
| `uniqueness_check` | event_id must be unique per batch |
| `categorical_check` (event_type) | one of page_view, add_to_cart, purchase, search, review |
| `categorical_check` (device_type) | one of desktop, mobile, tablet |
| `categorical_check` (country) | valid 2-letter ISO code |

## Gold Tables

| Table | Description |
|---|---|
| `revenue_by_category.parquet` | Daily revenue, order count, AOV by category |
| `user_cohorts.parquet` | Weekly cohort retention rates |
| `product_performance.parquet` | View → cart → purchase conversion by product |
| `channel_mix.parquet` | Event counts and revenue by device × referrer |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000` for the pipeline dashboard.
Open `http://localhost:8000/docs` for the interactive API documentation.

### 3. Run with Docker Compose

```bash
docker compose up --build
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/pipeline/run` | Trigger a full pipeline run |
| GET | `/api/pipeline/{run_id}/status` | Run status + stage progress |
| GET | `/api/pipeline/runs` | List all runs |
| GET | `/api/quality/{run_id}` | Quality report for a run |
| GET | `/api/gold/summary` | Aggregated Gold layer stats |
| GET | `/api/catalog` | Full metadata catalog + lineage |
| GET | `/api/quarantine/{run_id}` | Quarantined records + failure reasons |
| GET | `/api/health` | Health check |

### Example: trigger a run

```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 2000, "corruption_rate": 0.10, "label": "test run"}'
```

---

## Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Bronze schema validation (all 13 required fields)
- All 6 quality rule classes
- Quality gate integration (failure breakdown, quarantine reasons)
- Silver deduplication, timestamp normalisation, type casting, persistence
- Silver metadata field generation

---

## Project Structure

```
lakehouse-etl-pipeline/
├── pipeline/
│   ├── bronze.py        Bronze ingestion layer
│   ├── silver.py        Silver transformation layer
│   ├── gold.py          Gold aggregation layer
│   ├── quality.py       Data quality gates + quarantine
│   ├── catalog.py       Metadata catalog + lineage tracking
│   └── generator.py     Synthetic e-commerce event generator
├── api/
│   ├── main.py          FastAPI control plane (8 endpoints)
│   └── models.py        Pydantic request/response schemas
├── frontend/
│   └── index.html       Pipeline dashboard (vanilla JS, no frameworks)
├── tests/
│   ├── test_bronze.py
│   ├── test_silver.py
│   └── test_quality.py
├── data/                Created at runtime
│   ├── bronze/          JSON Lines raw ingestion files
│   ├── silver/          Parquet clean records
│   ├── gold/            Parquet analytical aggregations
│   └── quarantine/      Parquet rejected records
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
