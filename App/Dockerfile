FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files before installing requirements to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with GPU support (CUDA 12.8)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy rest of the application
COPY . .

# Expose port (optional but good practice)
EXPOSE 8000

# Default command
ENTRYPOINT ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

