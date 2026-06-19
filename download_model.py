"""
Download Qwen3-ASR GGUF model and mmproj from HuggingFace.

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

REPO_ID = os.getenv("ASR_HF_MODEL_REPO", "ggml-org/Qwen3-ASR-0.6B-GGUF")
MODEL_FILE = "Qwen3-ASR-0.6B-Q8_0.gguf"
MMPROJ_FILE = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"

FILES = [
    (MODEL_FILE, 768),
    (MMPROJ_FILE, 205),
]


def download_file(filename: str, expected_size_mb: int) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        actual_size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {filename} already exists ({round(actual_size_mb, 2)}MB)")
        return path

    print(f"  Downloading {filename} (~{expected_size_mb}MB)...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    actual_size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ {filename} saved ({round(actual_size_mb, 2)}MB)")
    return path


# ─── Download ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Qwen3-ASR GGUF Model Download")
print("=" * 60)
print(f"  Repo:    {REPO_ID}")
print(f"  Save to: {MODEL_DIR}")
print("=" * 60)
print()

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

try:
    for filename, expected_size_mb in FILES:
        download_file(filename, expected_size_mb)

    print()
    print("=" * 60)
    print("  ✓ All downloads complete!")
    print("=" * 60)
    print("  You can now start the server:")
    print("    start.bat")
    print("=" * 60)

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
    print(f"    3. Download manually from: https://huggingface.co/{REPO_ID}")
    print()
    sys.exit(1)
