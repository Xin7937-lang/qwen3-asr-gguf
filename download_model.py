"""
Download Qwen3-ASR GGUF model from HuggingFace.

This script downloads the quantized GGUF model for llama-cpp-python inference.

Usage:
    python download_model.py

With proxy (if needed):
    set HTTP_PROXY=http://127.0.0.1:7897
    set HTTPS_PROXY=http://127.0.0.1:7897
    python download_model.py
"""
import os
import sys

# Try to download with huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface-hub...")
    os.system(f"{sys.executable} -m pip install huggingface-hub -q")
    from huggingface_hub import hf_hub_download

# ─── Configuration ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model file info
REPO_ID = os.getenv("ASR_HF_MODEL_REPO", "ggml-org/Qwen3-ASR-0.6B-GGUF")
FILENAME = "qwen3-asr-0.6b-Q4_K_M.gguf"
EXPECTED_SIZE_MB = 480  # ~480MB

# ─── Download ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Qwen3-ASR GGUF Model Download")
print("=" * 60)
print(f"  Repo:    {REPO_ID}")
print(f"  File:    {FILENAME}")
print(f"  Size:    ~{EXPECTED_SIZE_MB}MB")
print(f"  Save to: {MODEL_DIR}")
print("=" * 60)
print()

# Check if already exists
model_path = os.path.join(MODEL_DIR, FILENAME)
if os.path.exists(model_path):
    actual_size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  Model already exists: {model_path}")
    print(f"  Size: {round(actual_size_mb, 2)}MB")

    confirm = input("\n  Download again? (y/N): ")
    if confirm.lower() != 'y':
        print("  Aborted.")
        sys.exit(0)

# Check proxy settings
http_proxy = os.getenv("HTTP_PROXY")
https_proxy = os.getenv("HTTPS_PROXY")

if http_proxy or https_proxy:
    print("  Using proxy:")
    if http_proxy:
        print(f"    HTTP_PROXY={http_proxy}")
    if https_proxy:
        print(f"    HTTPS_PROXY={https_proxy}")
    print()

print("  Starting download...")
print()

try:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # Verify download
    if os.path.exists(path):
        actual_size_mb = os.path.getsize(path) / 1024 / 1024
        print()
        print("=" * 60)
        print("  ✓ Download Complete!")
        print("=" * 60)
        print(f"  Saved to: {path}")
        print(f"  Size: {round(actual_size_mb, 2)}MB")
        print("=" * 60)
        print()
        print("  You can now start the server:")
        print("    start.bat")
        print()
    else:
        print()
        print("  ✗ Download failed: file not found at expected location")
        sys.exit(1)

except KeyboardInterrupt:
    print()
    print("  Download cancelled by user")
    sys.exit(1)

except Exception as e:
    print()
    print("  ✗ Download failed:")
    print(f"    {e}")
    print()
    print("  Possible solutions:")
    print("    1. Check your internet connection")
    print("    2. Set proxy if behind a firewall:")
    print("       set HTTP_PROXY=http://127.0.0.1:7897")
    print("       set HTTPS_PROXY=http://127.0.0.1:7897")
    print("    3. Download manually:")
    print(f"       https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}")
    print(f"       Then save to: {model_path}")
    print()
    sys.exit(1)