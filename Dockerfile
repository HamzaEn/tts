FROM runpod/base:0.6.2-cuda12.4.1

WORKDIR /app

# Only libsndfile — no git needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# torch only (no torchaudio — saves ~500 MB)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu124

# Runtime deps
RUN pip install --no-cache-dir chatterbox-tts soundfile runpod numpy

# Pre-bake model weights — cold start = VRAM load only, no HuggingFace download
# Clean HF cache temp files after download to save space
RUN python3 -c "\
from chatterbox.tts import ChatterboxTTS; \
ChatterboxTTS.from_pretrained(device='cpu'); \
print('Model weights cached.')" \
    && find /root/.cache/huggingface -name "*.lock" -delete \
    && find /root/.cache/huggingface -name "tmp*" -delete \
    && find /root/.cache/pip -type f -delete 2>/dev/null || true

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
