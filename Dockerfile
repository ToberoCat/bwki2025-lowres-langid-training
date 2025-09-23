FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    libicu-dev \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY src/ /app/src/
COPY create_dataset_fasttext.py /app/
COPY pyproject.toml /app/
COPY train.py /app/
COPY uv.lock/ /app/
COPY tools/ /app/tools/

RUN uv sync

#  docker run --cpus="16" --memory="15g" \
  #  -v $(pwd)/src/bwki2025/data_processing/fasttext_pipeline:/app/src/bwki2025/data_processing/fasttext_pipeline \
  #  -v $(pwd)/create_dataset_fasttext.py:/app/create_dataset_fasttext.py \
  #  bwki2025_test_container \
  #  uv run create_dataset_fasttext.py
  # zum Runnen der docker file