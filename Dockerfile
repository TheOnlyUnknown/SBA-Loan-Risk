FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir $(grep -v '^xgboost' requirements.txt) && \
    pip install --no-cache-dir --no-deps "xgboost>=2.0"

RUN useradd --create-home appuser

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
