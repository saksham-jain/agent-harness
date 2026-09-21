FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_qdrant.py .

# Docs are mounted at /docs by docker compose
ENTRYPOINT ["python", "rag_qdrant.py"]
CMD ["/docs"]
