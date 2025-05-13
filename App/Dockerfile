# Use a CUDA-compatible base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for Python 3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Run the script
CMD ["python", "app.py"]
