"""
RunPod Serverless Handler — Chatterbox TTS

Model is loaded at cold start (before runpod.serverless.start()),
so it's already in VRAM for every request on a warm worker.

Input:
{
  "input": {
    "texts":               ["text 1", "text 2", ...],   // list of chunks
    "reference_audio_url": "https://...",                // optional public WAV URL
    "reference_audio":     "<base64 WAV>",               // optional base64 WAV
    "exaggeration":        0.1,
    "cfg_weight":          0.1,
    "sample_rate":         24000
  }
}

Output:
{
  "audio_b64":   "<base64 WAV>",
  "sample_rate": 24000,
  "chunks_done": 5
}
"""

import os
import base64
import tempfile
import numpy as np
import soundfile as sf
import torch
import runpod

from chatterbox.tts import ChatterboxTTS

# ── Load model at cold start — reused on every warm request ───────────────────
print("[handler] Loading Chatterbox model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = ChatterboxTTS.from_pretrained(device=DEVICE)
print(f"[handler] Model ready on {DEVICE}")


# ── Handler ────────────────────────────────────────────────────────────────────

def handler(job):
    inp         = job["input"]
    # Accept "texts" (primary) or "chunks" (legacy alias)
    texts       = inp.get("texts") or inp.get("chunks") or []
    ref_b64     = inp.get("reference_audio")
    ref_url     = inp.get("reference_audio_url")
    exaggeration = max(0.1, float(inp.get("exaggeration", 0.1)))
    cfg_weight   = max(0.1, float(inp.get("cfg_weight",   0.1)))
    sample_rate  = int(inp.get("sample_rate", 24000))

    if not texts:
        return {"error": "No texts provided. Send a list via 'texts' key."}

    print(f"[handler] {len(texts)} chunks  exag={exaggeration}  cfg={cfg_weight}")

    # ── Resolve voice reference ────────────────────────────────────────────────
    ref_path = None
    try:
        if ref_b64:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(base64.b64decode(ref_b64))
            tmp.flush()
            ref_path = tmp.name
            print(f"[handler] Voice ref: base64 ({len(ref_b64)} chars)")
        elif ref_url:
            import urllib.request
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            urllib.request.urlretrieve(ref_url, tmp.name)
            ref_path = tmp.name
            print(f"[handler] Voice ref: URL downloaded → {tmp.name}")
    except Exception as e:
        print(f"[handler] WARNING: could not load voice ref: {e}")
        ref_path = None

    try:
        parts = []
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            print(f"[handler] [{i+1}/{len(texts)}] {text[:80]}{'...' if len(text) > 80 else ''}")
            wav = MODEL.generate(
                text,
                audio_prompt_path = ref_path,
                exaggeration      = exaggeration,
                cfg_weight        = cfg_weight,
            )
            arr = wav.squeeze().cpu().numpy().astype(np.float32)
            if len(arr):
                # 5 ms fade-in/out — kills click artifacts at splice points
                n = min(int(sample_rate * 0.005), len(arr) // 4)
                if n >= 2:
                    arr = arr.copy()
                    arr[:n]  *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                    arr[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                parts.append(arr)

        if not parts:
            return {"error": "No audio generated — all chunks were empty"}

        audio = np.concatenate(parts)
        peak  = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.891

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            sf.write(out.name, audio, sample_rate)
            wav_bytes = open(out.name, "rb").read()
            os.unlink(out.name)

        print(f"[handler] Done — {len(parts)} chunks, {len(wav_bytes)//1024} KB")
        return {
            "audio_b64":   base64.b64encode(wav_bytes).decode("utf-8"),
            "sample_rate": sample_rate,
            "chunks_done": len(parts),
        }

    finally:
        if ref_path:
            try: os.unlink(ref_path)
            except: pass


runpod.serverless.start({"handler": handler})
