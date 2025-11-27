# Use Python 3.11 slim image as base
FROM python:3.11-slim

WORKDIR /chatter

COPY pyproject.toml uv.lock server.py .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv
RUN uv venv create --python /usr/local/bin/python
RUN uv sync --locked --no-editable

CMD [".venv/bin/python", "server.py"] 