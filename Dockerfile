FROM runpod/base:0.6.2-cuda12.4.1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Install torch first from PyTorch index — bundled wheel, no nvidia_cudnn download
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Now chatterbox-tts sees torch already satisfied — skips it
RUN pip install --no-cache-dir chatterbox-tts soundfile runpod

COPY handler.py .

CMD ["python3", "-u", "handler.py"]