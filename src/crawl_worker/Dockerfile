FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project configuration and readme
COPY pyproject.toml /app/
COPY README.md /app/

# Install dependencies
RUN uv pip install --system -e .

# Copy source code
COPY src /app/src

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install Playwright browser dependencies
RUN playwright install
RUN playwright install-deps

# Run the worker
CMD ["python", "-m", "src.crawl_worker.main"]
