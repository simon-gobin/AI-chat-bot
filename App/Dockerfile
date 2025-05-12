FROM python:3.10-slim

# Optional: Use NVIDIA CUDA base image for GPU support
# FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git libgl1-mesa-glx

# Copy files
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "app.py"]