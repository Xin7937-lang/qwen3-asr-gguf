# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

Qwen3-ASR GGUF speech-to-text server optimized for AMD GPU using llama-cpp-python with Vulkan acceleration. CPU fallback available when Vulkan unavailable. API-compatible with qwen3-asr-server (main project).

### Key Differences from qwen3-asr-server

| Aspect | qwen3-asr-gguf (this project) | qwen3-asr-server (main project) |
|--------|------------------------------|---------------------------------|
| Backend | llama-cpp-python | qwen-asr package |
| GPU Support | AMD (Vulkan) | NVIDIA (CUDA) |
| Model Format | GGUF quantized (~480MB) | Native format (~1.2GB) |
| Audio Encoder | Simplified FBank (numpy) | Built-in AuT (ONNX-based) |
| Accuracy | ~80% | 100% |
| Dependencies | llama-cpp-python, numpy, soundfile | torch, transformers, qwen-asr |

### Project Independence

This is a **completely independent project** from qwen3-asr-server:
- Separate git repository
- No shared code files
- Can run simultaneously (use different ports)
- Compatible API endpoints

## Quick Start

```batch
# Auto-detect Vulkan/GPU and start
start.bat

# Or manually
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

# Download model first if needed
python download_model.py

# Start server
python main.py
```

## Vulkan Mode (AMD GPU)

- Uses `llama_cpp.Llama` with `n_gpu_layers=-1` for full GPU offload
- Requires Vulkan SDK installed and llama-cpp-python compiled with Vulkan support
- Vulkan installation: https://vulkan.lunarg.com/
- Compile with Vulkan: `CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`
- Fallback to CPU if Vulkan unavailable (automatic in `main.py`)

### Vulkan Installation

```batch
# Install Vulkan SDK from: https://vulkan.lunarg.com/
# Verify installation:
vulkaninfo --summary

# Reinstall llama-cpp-python with Vulkan support:
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## CPU Mode (auto-fallback when Vulkan unavailable)

- Uses llama-cpp-python CPU backend
- Same model, all layers on CPU
- Slower but works without Vulkan SDK
- Chunked processing (30s default via `ASR_CHUNK_DURATION_S`)

## Code Architecture

### Key Components

| File | Role |
|------|------|
| `main.py` | **Primary entry point.** FastAPI app with Vulkan detection, model loading, `/v1/transcribe`, `/asr`, `/health`, `/` endpoints |
| `config.py` | All tunable parameters from env vars (`ASR_*` prefix). Vulkan, model path, server config, audio processing |
| `audio_encoder.py` | Audio preprocessing: `load_audio()`, `extract_fbank()`, `chunk_features()`, `validate_audio()` |
| `agent_client.py` | Python client for remote agents: `ASRClient` class with `transcribe()`, `transcribe_full()`, `wait_for_service()`, `get_backend_info()` |
| `download_model.py` | Model download from HuggingFace using huggingface-hub |

### Inference Path

The inference pipeline in `main.py`:

1. **Audio Loading** (`audio_encoder.load_audio()`): Load bytes → resample to 16kHz → convert to mono
2. **Feature Extraction** (`audio_encoder.extract_fbank()`): Pre-emphasis → framing → FFT → Mel filterbank → log
3. **Chunking** (`audio_encoder.chunk_features()`): Split into 30s chunks for processing
4. **LLM Inference** (`llama_cpp.Llama()`): Process each chunk with temperature=0.0 (greedy decoding)
5. **Segment Assembly**: Combine chunk results into full transcription

### API Surface

All services expose:
- `POST /v1/transcribe` — upload file + optional `language` + `word_timestamps` → returns `{text, language, segments, processing_time, duration_s, rtf}`
- `POST /asr` — simplified alias
- `GET /` — service status + backend info (vulkan_enabled, vulkan_detected, n_gpu_layers)
- `GET /health` — health check

### Language Handling

The server accepts both short codes (`zh`, `en`, `ja`) and full English names (`Chinese`, `English`, `Japanese`). Mapping lives in `main.py` (`SHORT_TO_FULL`, `_CANONICAL`). Normalized to Qwen3-ASR's 30-language full-name format before passing to model.

## Important Behaviors

- **Model loaded once globally** (`_llm` in `main.py`) — lazy on first request
- **Vulkan detection on startup**: Checks `vulkaninfo` availability via `config.check_vulkan_available()`
- **Automatic fallback**: If Vulkan unavailable, `ENABLE_VULKAN` is auto-disabled and CPU mode used
- **Model download**: Use `download_model.py` to download GGUF model from HuggingFace (`HaujetZhao/Qwen3-ASR-0.6B-GGUF`)
- **GGUF model format**: Quantized model (~480MB), requires llama-cpp-python to load

## Common Tasks

```batch
# Start with specific backend
set ASR_ENABLE_VULKAN=false && python main.py  # Force CPU
set ASR_ENABLE_VULKAN=true && python main.py   # Force Vulkan

# Adjust GPU layers (reduce VRAM usage)
set ASR_N_GPU_LAYERS=20 && python main.py

# Increase context window for longer audio
set ASR_N_CTX=8192 && python main.py

# Test the server
curl http://localhost:8000/
curl -X POST "http://localhost:8000/v1/transcribe" -F "file=@test.mp3" -F "language=Chinese"

# Use the agent client (run from agent machine, not the server)
python agent_client.py
```

## Environment Variables

All prefixed `ASR_*`. Key overrides:

| Var | Default | Notes |
|-----|---------|-------|
| `ASR_ENABLE_VULKAN` | true | Vulkan acceleration |
| `ASR_N_GPU_LAYERS` | -1 | GPU layers (-1 = all) |
| `ASR_N_CTX` | 4096 | Context window size |
| `ASR_N_THREADS` | 0 | CPU threads (0 = auto) |
| `ASR_LLAMA_BACKEND` | auto | Backend selection |
| `ASR_HOST` | 0.0.0.0 | Server host |
| `ASR_PORT` | 8000 | Server port |
| `ASR_MAX_FILE_SIZE_MB` | 500 | Max file size |
| `ASR_MAX_CONCURRENCY` | 2 | API concurrency limit |
| `ASR_CHUNK_DURATION_S` | 30 | Chunk duration in seconds |
| `ASR_HF_MODEL_REPO` | HaujetZhao/Qwen3-ASR-0.6B-GGUF | HuggingFace repo |

## Model Information

- **Model**: Qwen3-ASR-0.6B-GGUF (quantized, ~480MB)
- **Format**: GGUF (Q4_K_M quantization)
- **HuggingFace**: `HaujetZhao/Qwen3-ASR-0.6B-GGUF`
- **Package**: `llama-cpp-python` (PyPI) provides `Llama()` class
- **Languages**: 30 supported (same as Qwen3-ASR)

## Saving Outputs

Transcription results are automatically saved to `output/` directory as both `.txt` (plain text) and `.json` (full result) files, timestamped per request. This serves as crash recovery — agent can re-read from disk if the API response is lost.

## Vulkan Troubleshooting

### Vulkan SDK Not Found

```batch
# Install from: https://vulkan.lunarg.com/
# Verify:
vulkaninfo --summary

# Set environment variable if needed:
set VULKAN_SDK=C:\VulkanSDK\1.3.x
```

### llama-cpp-python Not Compiled with Vulkan

```batch
# Reinstall with Vulkan support
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### GPU Not Used

Check status endpoint:
```json
{
  "vulkan_enabled": true,
  "vulkan_detected": false,
  "optimizations": {
    "n_gpu_layers": -1
  }
}
```

If `vulkan_detected` is `false`, Vulkan SDK is not properly installed.

## Performance Characteristics

| Backend | 60s Audio | RTF | VRAM Usage |
|---------|-----------|-----|------------|
| Vulkan (AMD) | ~18s | 0.30 | ~2GB |
| CPU (4 cores) | ~150s | 2.50 | N/A |
| CPU (8 cores) | ~90s | 1.50 | N/A |

## Testing Without Model

The server will start even without the model, but transcription will fail. Status endpoint will show `model_loaded: false`. Download model with `python download_model.py`.

## Firewall Configuration

Port 8000 must be open for LAN agents — same as qwen3-asr-server:
```powershell
New-NetFirewallRule -DisplayName "Qwen3-ASR GGUF" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```