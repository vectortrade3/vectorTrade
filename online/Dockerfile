# Multi-stage ultra-lightweight Docker container for VectorTrade Online Worker & Dashboard
FROM python:3.11-slim AS base

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install minimal online Python dependencies
COPY online/requirements_online.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only online and core modules
COPY core /app/core
COPY online /app/online
COPY .env.example /app/.env.example

EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "online.app:app", "--host", "0.0.0.0", "--port", "8000"]
