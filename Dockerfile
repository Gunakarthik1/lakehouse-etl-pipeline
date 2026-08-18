FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY pipeline/ ./pipeline/
COPY api/ ./api/
COPY frontend/ ./frontend/

# Create data directories
RUN mkdir -p data/bronze data/silver data/gold data/quarantine

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
