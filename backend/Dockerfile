# Multi-stage Docker build for production optimization
# Stage 1: Build environment with all dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG ENVIRONMENT=production
ARG INSTALL_DEV=false

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create and set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies with optimization
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download required ML models and data
RUN python -c "import spacy; spacy.cli.download('en_core_web_md')" || \
    python -c "import spacy; spacy.cli.download('en_core_web_sm')" && \
    python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Stage 2: Production runtime environment
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/temp && \
    chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose application port
EXPOSE ${PORT}

# Start application with production server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# Development stage (optional)
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black isort flake8 mypy

# Set development environment
ENV ENVIRONMENT=development

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to application user
USER appuser

# Development server command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]