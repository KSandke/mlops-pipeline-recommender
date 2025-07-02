FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY Recommender/requirements.txt /workspace/Recommender/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r Recommender/requirements.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/Recommender/data/raw && \
    mkdir -p /workspace/Recommender/data/processed && \
    mkdir -p /workspace/Recommender/models

# Set environment variables
ENV PYTHONPATH=/workspace
ENV WANDB_MODE=offline

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "Recommender/src/predict.py"] 