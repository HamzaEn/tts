FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install chatterbox-tts without deps to avoid PyTorch conflicts with base image
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Chatterbox deps (no torch/torchaudio — already in base image)
RUN pip install --no-cache-dir \
    conformer \
    s3tokenizer \
    librosa \
    resemble-perth \
    huggingface_hub \
    safetensors \
    transformers \
    diffusers \
    einops \
    soundfile \
    scipy \
    omegaconf \
    pyloudnorm \
    numpy \
    runpod

# Pre-bake model weights — cold start = VRAM load only
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); ChatterboxTTS.from_pretrained(device='cpu'); print('Done.')" \
    && find /root/.cache/huggingface -name "*.lock" -delete 2>/dev/null || true \
    && find /root/.cache/huggingface -name "tmp*"   -delete 2>/dev/null || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
