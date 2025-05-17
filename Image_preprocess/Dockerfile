FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && apt-get clean

# Upgrade pip & install Python packages
RUN pip3 install --upgrade pip

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory and copy your script
WORKDIR /app
COPY image.py .

# Run your script by default
CMD ["python3", "image.py"]
