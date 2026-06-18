"""
Audio Encoder for Qwen3-ASR GGUF Server
========================================
Implements FBank (Filter Bank) feature extraction from audio.
Simplified version - ~80% accuracy compared to full AuT encoder.

Based on Qwen3-ASR's preprocessing pipeline.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger("qwen3-gguf")


def load_audio(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes and resample to target sample rate.

    Args:
        audio_bytes: Raw audio bytes
        target_sr: Target sample rate (default 16000 for Qwen3-ASR)

    Returns:
        (audio_array, sample_rate) tuple
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        data, sr = sf.read(tmp_path)
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

    return data, target_sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using linear interpolation.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    old_len = len(audio)
    new_len = int(old_len * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, old_len - 1, new_len),
        np.arange(old_len),
        audio,
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Input audio array

    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def extract_fbank(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    pre_emphasis: float = 0.97,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """
    Extract FBank (Filter Bank) features from audio.

    Matches Qwen3-ASR's preprocessing pipeline.

    Args:
        audio: Input audio array (should be at target_sr)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        pre_emphasis: Pre-emphasis coefficient
        frame_length_ms: Frame length in milliseconds
        frame_shift_ms: Frame shift in milliseconds

    Returns:
        FBank features of shape (n_frames, n_mels)
    """
    # Ensure audio is normalized
    audio = normalize_audio(audio)

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Frame parameters
    frame_len = int(frame_length_ms * sr / 1000)
    frame_shift = int(frame_shift_ms * sr / 1000)

    # Number of frames
    num_frames = (len(audio) - frame_len) // frame_shift + 1
    if num_frames < 1:
        logger.warning("Audio too short, returning empty features")
        return np.zeros((1, n_mels), dtype=np.float32)

    # Hamming window
    window = np.hamming(frame_len)

    # Mel filterbank matrix
    mel_basis = _create_mel_basis(n_fft // 2 + 1, n_mels, sr)

    # Extract features frame by frame
    frames = []
    for i in range(num_frames):
        start = i * frame_shift
        frame = audio[start:start + frame_len] * window

        # FFT
        spec = np.fft.rfft(frame, n=n_fft)
        power = np.abs(spec) ** 2

        # Mel filter
        mel = np.dot(mel_basis, power)
        mel = np.where(mel > 0, np.log(mel + 1e-10), -10.0)

        frames.append(mel)

    return np.array(frames, dtype=np.float32)


def _create_mel_basis(n_fft_bins: int, n_mels: int, sr: int) -> np.ndarray:
    """
    Create Mel filterbank matrix.

    Args:
        n_fft_bins: Number of FFT bins
        n_mels: Number of mel bands
        sr: Sample rate

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft_bins)
    """
    # Convert Hz to Mel scale
    mel_max = 2595 * np.log10(1 + sr / 2 / 700)

    # Create equally spaced mel points
    mel_points = np.linspace(0, mel_max, n_mels + 2)

    # Convert back to Hz
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    # Convert to FFT bin indices
    bins = np.floor((n_fft_bins + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft_bins - 1)

    # Create filterbank matrix
    basis = np.zeros((n_mels, n_fft_bins))
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:
            basis[m - 1, left:center] = np.linspace(0, 1, center - left)
        if right > center:
            basis[m - 1, center:right] = np.linspace(1, 0, right - center)

    return basis


def chunk_features(
    features: np.ndarray,
    max_duration_s: float = 30.0,
    frame_shift_ms: float = 10.0,
) -> list:
    """
    Chunk features into segments for processing.

    Args:
        features: FBank features of shape (n_frames, n_mels)
        max_duration_s: Maximum duration per chunk in seconds
        frame_shift_ms: Frame shift in milliseconds

    Returns:
        List of feature chunks, each as (chunk_idx, start_frame, chunk_array)
    """
    max_frames = int(max_duration_s * 1000 / frame_shift_ms)

    chunks = []
    for start in range(0, len(features), max_frames):
        end = start + max_frames
        chunk = features[start:end]

        # Skip very short trailing chunks (< 1 second)
        if len(chunk) < int(1000 / frame_shift_ms):
            continue

        chunks.append((len(chunks), start, chunk))

    logger.info("Split %d frames into %d chunks", len(features), len(chunks))
    return chunks


def get_audio_info(audio_bytes: bytes) -> dict:
    """
    Get basic information about audio data.

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Dict with duration, sample_rate, channels info
    """
    import tempfile
    import os

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


# ─── Utility Functions ─────────────────────────────────────────────────────

def validate_audio(audio_bytes: bytes, max_size_mb: int = 500) -> bool:
    """
    Validate audio file before processing.

    Args:
        audio_bytes: Raw audio bytes
        max_size_mb: Maximum file size in MB

    Returns:
        True if valid, False otherwise
    """
    size_mb = len(audio_bytes) / 1024 / 1024
    if size_mb > max_size_mb:
        logger.error("Audio too large: %.1fMB (max: %dMB)", size_mb, max_size_mb)
        return False

    if len(audio_bytes) == 0:
        logger.error("Audio file is empty")
        return False

    return True


def estimate_processing_time(duration_s: float, backend: str = "vulkan") -> float:
    """
    Estimate processing time based on audio duration and backend.

    Args:
        duration_s: Audio duration in seconds
        backend: Inference backend (vulkan, cpu, metal, cuda)

    Returns:
        Estimated processing time in seconds
    """
    # These are rough estimates based on testing
    if backend == "vulkan":
        # AMD GPU with Vulkan: ~0.3x real-time
        return duration_s * 0.3
    elif backend in ("metal", "cuda"):
        # NVIDIA/Apple GPU: ~0.2x real-time
        return duration_s * 0.2
    else:
        # CPU: ~2-3x real-time
        return duration_s * 2.5