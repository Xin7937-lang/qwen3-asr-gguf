# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project Overview

Qwen3-ASR GGUF — speech-to-text server wrapping **llama.cpp's llama-server** as a child subprocess behind a FastAPI endpoint. Optimized for AMD GPU via Vulkan; CPU fallback available. API-compatible with `qwen3-asr-server` (separate project with torch/qwen-asr backend).

### Other Projects

| Project | Backend | GPU | Dependencies |
|---------|---------|-----|-------------|
| **qwen3-asr-gguf** (this repo) | llama.cpp llama-server | AMD (Vulkan) | requests, numpy, soundfile, fastapi |
| **qwen3-asr-server** | qwen-asr package | NVIDIA (CUDA) | torch, transformers, qwen-asr |

Both can run simultaneously (different ports). No shared code.

## Architecture

### File Roles

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, llama-server subprocess management, `/v1/transcribe` logic, result saving, lifespan |
| `config.py` | All tunable parameters from env vars (`ASR_*` prefix) |
| `audio_encoder.py` | Audio loading (bytes→16kHz mono float32), chunking into ≤30s WAV segments |
| `agent_client.py` | Python `ASRClient` for remote agents to call this server |
| `download_model.py` | Downloads GGUF model and mmproj from HuggingFace |
| `start.bat` | One-click launcher: creates venv, installs deps, checks Vulkan, starts server |

### Inference Pipeline (main.py)

```
HTTP POST /v1/transcribe
  → audio_encoder.load_audio(bytes)          # resample to 16kHz mono
  → audio_encoder.chunk_audio()              # split into ~30s WAV chunks
  → for each chunk in sequence:
      → _transcribe_chunk()                  # POST /v1/chat/completions to internal llama-server
        → payload: {"input_audio": base64 WAV}
        → parse: regex "language <lang><asr_text>text"
      → detect language from first chunk
  → aggregate segments, compute RTF
  → _save_result() to output/ dir
  → return TranscribeResponse
```

### Subprocess Lifecycle

- **Start**: FastAPI `lifespan` context manager calls `_start_llama_server()` → spawns `llama-server.exe` as `subprocess.Popen`, polls `/health` up to 60s until ready
- **Stop**: On shutdown, `_stop_llama_server()` → `terminate()` → `wait(10)` → `kill()` fallback
- **Health**: External `/health` endpoint calls `_wait_for_llama_server(timeout=2)` which pings internal llama-server's `/health`

### Concurrency Model

- `asyncio.Semaphore(config.MAX_CONCURRENCY)` (default 2) limits concurrent `_transcribe_llama_server()` calls
- Inside each call, audio chunks are processed **sequentially** (for loop), not in parallel — llama-server handles one multimodal request at a time per connection
- The semaphore prevents N simultaneous chunk-processors from overwhelming llama-server

### Language Handling

- Accepts short codes (`zh`, `en`, `ja`) and full names (`Chinese`, `English`, `Japanese`)
- `normalize_language()` in `main.py` maps via `SHORT_TO_FULL` dict (line 65) and `_CANONICAL` (line 63)
- 30 supported languages (see `LANGUAGE_NAMES` set, line 56)
- When omitted, first chunk's detected language is used

### Result Persistence

- Every transcription saves three files under `output/YYYYMMDD/`:
  - `{filename}_{timestamp}.txt` — plain text
  - `{filename}_{timestamp}.json` — full result with segments
  - `{filename}_{timestamp}.meta.json` — metadata with file paths
- `output/latest_index.json` keeps last 50 entries for easy lookup

## Key Commands

```batch
# ─── Setup ────────────────────────────────────────────────────────────
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

# ─── Model ────────────────────────────────────────────────────────────
python download_model.py                              # Download from HuggingFace
set HTTP_PROXY=http://127.0.0.1:7897 && python download_model.py  # Via proxy

# ─── Run (production) ─────────────────────────────────────────────────
start.bat                                              # Auto-detect everything
set ASR_ENABLE_VULKAN=true && python main.py           # Force Vulkan (AMD GPU)
set ASR_ENABLE_VULKAN=false && python main.py          # Force CPU

# ─── Run (tuning) ─────────────────────────────────────────────────────
set ASR_N_GPU_LAYERS=20 && python main.py              # Limit GPU layers for VRAM
set ASR_N_CTX=8192 && python main.py                   # Larger context window
set ASR_MAX_CONCURRENCY=4 && python main.py            # More concurrent requests
set ASR_CHUNK_DURATION_S=15 && python main.py          # Smaller chunks

# ─── Compile llama-server with Vulkan ─────────────────────────────────
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server

# ─── Test server ──────────────────────────────────────────────────────
curl http://localhost:8001/
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/v1/transcribe" -F "file=@test.mp3" -F "language=Chinese"

# ─── Check logs ───────────────────────────────────────────────────────
tail -f llama-server.log                               # llama-server stderr
# Server logs go to stdout (available in start.bat output)

# ─── Agent client (from remote machine) ──────────────────────────────
python agent_client.py
```

## Environment Variables

All `ASR_*` prefix. See `config.py` for the full list.

| Var | Default | Notes |
|-----|---------|-------|
| `ASR_ENABLE_VULKAN` | false | Enables Vulkan GPU acceleration |
| `ASR_N_GPU_LAYERS` | -1 | GPU layers (-1 = all) |
| `ASR_N_CTX` | 4096 | llama-server context window (tokens) |
| `ASR_CHUNK_DURATION_S` | 10 | Audio chunk duration in seconds (Qwen3-ASR max ~10s) |
| `ASR_CHUNK_TIMEOUT` | 120 | Per-chunk llama-server API timeout (seconds) |
| `ASR_CHUNK_RETRIES` | 2 | Number of retries per failed chunk |
| `ASR_CHUNK_RETRY_DELAY` | 2 | Delay between chunk retries (seconds) |
| `ASR_MAX_CHUNK_FAILURES` | 5 | Max consecutive chunk failures before abort |
| `ASR_CHUNKS_PER_RESTART` | 10 | Proactively restart llama-server every N chunks (0=disable) |
| `ASR_N_PREDICT` | 1024 | Max tokens per chunk response from llama-server |
| `ASR_CHUNK_OVERLAP_S` | 2.0 | Audio overlap between adjacent chunks (seconds) |
| `ASR_PORT` | 8001 | FastAPI public port |
| `ASR_LLAMA_SERVER_PORT` | 8080 | Internal llama-server port |
| `ASR_LLAMA_SERVER_BIN` | `llama.cpp/build/bin/Release/llama-server.exe` | Executable path |
| `ASR_MAX_CONCURRENCY` | 2 | Max concurrent transcription requests |
| `ASR_MAX_FILE_SIZE_MB` | 500 | Max uploaded file size |
| `ASR_DEBUG` | false | Enable verbose debug logging |
| `ASR_MODEL_DIR` | `model/` | Model directory override |

## Model

- **Model**: Qwen3-ASR-0.6B-GGUF (Q8_0 quantization)
- **Files**: `Qwen3-ASR-0.6B-Q8_0.gguf` (~768MB) + `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` (~205MB)
- **Source**: `ggml-org/Qwen3-ASR-0.6B-GGUF` on HuggingFace
- **mmproj required**: The audio projector file is mandatory for multimodal input to work
- Models are gitignored (`model/*.gguf`); always download via `download_model.py`

## Important Behaviors

- **llama-server is a child subprocess** — not a Python package. Uses `subprocess.Popen` with stdio piped. On failure, check both the FastAPI logs and `llama-server.log`.
- **Internal llama-server** listens on `127.0.0.1:8080`. The external FastAPI server listens on `0.0.0.0:8001`.
- **Vulkan detection** is purely a `shutil.which("vulkaninfo")` check — not a runtime GPU test. If `vulkaninfo` is on PATH but Vulkan actually fails, the server will report `vulkan_detected: true` but llama-server may error.
- **No warmup model** (`--no-warmup` flag), so the first request is slower.
- **No tests** exist in this repository. Manual testing via `curl` against the server.
- **Audio chunking** is sequential: long audio files are split into chunks and processed one by one, which means total processing time scales linearly with audio length (modulated by RTF).
- **Output directory** `output/` is gitignored.

## Performance

| Backend | 20s Audio | RTF |
|---------|-----------|-----|
| Vulkan (AMD iGPU) | ~2.5s | 0.13 |
| CPU (8 cores) | ~4s | 0.20 |

## Vulkan Troubleshooting

```batch
# Verify Vulkan SDK installation
vulkaninfo --summary

# Set SDK path if needed
set VULKAN_SDK=C:\VulkanSDK\1.4.350.0

# Check server status for GPU usage
curl http://localhost:8001/ | jq .
# Look for: vulkan_enabled, vulkan_detected
```

If `vulkan_detected` is `false`: Vulkan SDK not on PATH. If `true` but GPU not actually used: check `llama-server.log` for `Vulkan0` in startup messages.

## Firewall Configuration

```powershell
New-NetFirewallRule -DisplayName "Qwen3-ASR GGUF" -Direction Inbound -LocalPort 8001 -Protocol TCP -Action Allow
```
