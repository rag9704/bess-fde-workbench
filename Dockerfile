FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core_model/ core_model/
COPY ai_harness/ ai_harness/
COPY service/ service/
COPY data/april_2026_hourly.csv data/april_2026_hourly.csv
COPY main.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

# PORT is honoured so the same image runs on Cloud Run / App Runner.
CMD ["sh", "-c", "uvicorn service.api:app --host 0.0.0.0 --port ${PORT}"]
