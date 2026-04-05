FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install -e ".[all]" --no-cache-dir

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 neuralnode && \
    chown -R neuralnode:neuralnode /app
USER neuralnode

# Expose port for API (if needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import neuralnode; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import neuralnode; print('NeuralNode v' + neuralnode.__version__ + ' ready')"]
