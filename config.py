"""
Configuration for Qwen3-ASR GGUF Server (AMD GPU + Vulkan)
===========================================================
Based on llama-cpp-python with Vulkan acceleration.
Tuned for AMD GPU environments.
"""
import os
from pathlib import Path

# ─── Model ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("ASR_MODEL_DIR", str(Path(__file__).parent / "model")))
MODEL_FILE = os.getenv("ASR_MODEL_FILE", "Qwen3-ASR-0.6B-Q8_0.gguf")
MODEL_PATH = MODEL_DIR / MODEL_FILE

# Default HuggingFace model for download
HF_MODEL_REPO = os.getenv("ASR_HF_MODEL_REPO", "ggml-org/Qwen3-ASR-0.6B-GGUF")

# ─── Vulkan/GPU Settings ─────────────────────────────────────────────────────
# Enable Vulkan acceleration (requires Vulkan SDK and llama-cpp-python compiled with Vulkan)
ENABLE_VULKAN = os.getenv("ASR_ENABLE_VULKAN", "true").lower() == "true"

# Number of layers to offload to GPU (-1 = all layers)
N_GPU_LAYERS = int(os.getenv("ASR_N_GPU_LAYERS", "-1"))

# Vulkan backend selection (vulkan, cpu, metal, cuda)
LLAMA_BACKEND = os.getenv("ASR_LLAMA_BACKEND", "vulkan" if ENABLE_VULKAN else "cpu")

# ─── Llama.cpp Settings ──────────────────────────────────────────────────────
# Context window size (in tokens)
N_CTX = int(os.getenv("ASR_N_CTX", "4096"))

# Number of threads for CPU inference (0 = auto-detect)
N_THREADS = int(os.getenv("ASR_N_THREADS", "0"))

# ─── Audio Processing ────────────────────────────────────────────────────────
# Sample rate for audio processing
SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))

# FBank parameters
N_MELS = int(os.getenv("ASR_N_MELS", "80"))
N_FFT = int(os.getenv("ASR_N_FFT", "512"))

# Chunk duration in seconds for long audio
CHUNK_DURATION_S = int(os.getenv("ASR_CHUNK_DURATION_S", "30"))

# Pre-emphasis factor
PRE_EMPHASIS = float(os.getenv("ASR_PRE_EMPHASIS", "0.97"))

# ─── Server ────────────────────────────────────────────────────────────────
HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "8001"))
MAX_FILE_SIZE_MB = int(os.getenv("ASR_MAX_FILE_SIZE_MB", "500"))
MAX_CONCURRENCY = int(os.getenv("ASR_MAX_CONCURRENCY", "2"))

# ─── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.getenv("ASR_OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Debug ──────────────────────────────────────────────────────────────────
DEBUG = os.getenv("ASR_DEBUG", "false").lower() == "true"
VERBOSE_LLM = os.getenv("ASR_VERBOSE_LLM", "false").lower() == "true"

# ─── Helper Functions ────────────────────────────────────────────────────────

def get_vulkan_env_vars() -> dict:
    """
    Return Vulkan-related environment variables for debugging.

    These can be set to help llama-cpp-python find Vulkan libraries:
    - LLAMA_VULKAN_PATH: Path to Vulkan loader (vulkan.dll/libvulkan.so)
    - VK_LAYER_PATH: Path to Vulkan validation layers (optional)
    """
    return {
        "LLAMA_VULKAN_PATH": os.getenv("LLAMA_VULKAN_PATH"),
        "VK_LAYER_PATH": os.getenv("VK_LAYER_PATH"),
    }


def check_vulkan_available() -> bool:
    """
    Check if Vulkan environment is properly configured.

    Returns True if Vulkan SDK is detected.
    """
    import shutil
    return shutil.which("vulkaninfo") is not None


def get_model_info() -> dict:
    """
    Get information about the current model configuration.

    Returns:
        {
            "model_path": str,
            "model_exists": bool,
            "model_size_mb": float | None,
            "backend": str,
            "vulkan_enabled": bool,
            "n_gpu_layers": int,
            "n_ctx": int,
        }
    """
    model_exists = MODEL_PATH.exists()
    model_size_mb = None
    if model_exists:
        model_size_mb = MODEL_PATH.stat().st_size / 1024 / 1024

    return {
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "model_size_mb": round(model_size_mb, 2) if model_size_mb else None,
        "backend": LLAMA_BACKEND,
        "vulkan_enabled": ENABLE_VULKAN,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
    }


# ─── Print configuration on import (when DEBUG is true) ─────────────────────
if DEBUG:
    import sys
    print("=" * 60)
    print("  Qwen3-ASR GGUF Configuration")
    print("=" * 60)
    print(f"  Backend:      {LLAMA_BACKEND}")
    print(f"  Vulkan:       {ENABLE_VULKAN}")
    print(f"  GPU Layers:   {N_GPU_LAYERS}")
    print(f"  Context:      {N_CTX}")
    print(f"  Model Path:   {MODEL_PATH}")
    print(f"  Model Exists: {MODEL_PATH.exists()}")
    print("=" * 60, file=sys.stderr)