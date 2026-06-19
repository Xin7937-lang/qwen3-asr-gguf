"""
Audio utilities for Qwen3-ASR GGUF Server
==========================================
Loads, resamples, and chunks audio into WAV bytes for llama-server.
"""
import io
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("qwen3-gguf")


def load_audio(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio from bytes and resample to target sample rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        data, sr = sf.read(tmp_path, dtype="float32")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        sr = target_sr

    return data, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    old_len = len(audio)
    new_len = int(old_len * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, old_len - 1, new_len),
        np.arange(old_len),
        audio,
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Convert float audio array to 16-bit mono WAV bytes."""
    audio = normalize_audio(audio)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def chunk_audio(
    audio: np.ndarray,
    sr: int = 16000,
    max_duration_s: float = 30.0,
    min_duration_s: float = 1.0,
) -> List[Tuple[int, float, float, bytes]]:
    """
    Split audio into chunks and return WAV bytes for each chunk.

    Returns:
        List of (chunk_idx, start_s, end_s, wav_bytes)
    """
    chunk_samples = int(max_duration_s * sr)
    min_samples = int(min_duration_s * sr)
    total_samples = len(audio)

    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        if end - start < min_samples:
            continue
        chunk_audio_arr = audio[start:end]
        wav_bytes = audio_to_wav_bytes(chunk_audio_arr, sr)
        chunks.append((
            len(chunks),
            round(start / sr, 2),
            round(end / sr, 2),
            wav_bytes,
        ))

    logger.info("Split %ds audio into %d chunks (%.1fs each)", int(total_samples / sr), len(chunks), max_duration_s)
    return chunks


def get_audio_info(audio_bytes: bytes) -> dict:
    """Get basic information about audio data."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        info = sf.info(tmp_path)
        return {
            "duration_s": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def validate_audio(audio_bytes: bytes, max_size_mb: int = 500) -> bool:
    """Validate audio file before processing."""
    size_mb = len(audio_bytes) / 1024 / 1024
    if size_mb > max_size_mb:
        logger.error("Audio too large: %.1fMB (max: %dMB)", size_mb, max_size_mb)
        return False

    if len(audio_bytes) == 0:
        logger.error("Audio file is empty")
        return False

    return True


def estimate_processing_time(duration_s: float, backend: str = "vulkan") -> float:
    """Estimate processing time based on audio duration and backend."""
    if backend == "vulkan":
        return duration_s * 0.1
    elif backend in ("metal", "cuda"):
        return duration_s * 0.08
    else:
        return duration_s * 0.4
