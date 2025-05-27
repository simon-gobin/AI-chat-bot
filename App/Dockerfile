FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Install system packages and pip
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev git \
    && apt-get clean

# Upgrade pip and install PyTorch with --break-system-packages
RUN pip3 install --break-system-packages --upgrade pip && \
    pip3 install --break-system-packages torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install app dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

# Set working directory and copy code
WORKDIR /app
COPY . .

# Run your script
ENTRYPOINT ["python3", "app.py"]
