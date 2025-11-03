FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["hwr-serve", "--weights", "/app/artifacts/model_latest.pt", "--host", "0.0.0.0", "--port", "8000"]