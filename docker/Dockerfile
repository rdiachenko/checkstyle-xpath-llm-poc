FROM --platform=linux/arm64 ubuntu:22.04

ENV HF_TOKEN=""

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-ca-certificates

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir \
    transformers \
    accelerate>=1.3.0 \
    huggingface_hub>=0.27.1 \
    tqdm \
    requests \
    urllib3>=2.3.0

WORKDIR /app

COPY inference.py .

RUN mkdir -p /models

CMD ["python3", "inference.py"]
