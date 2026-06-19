"""
Qwen3-ASR GGUF Server (AMD GPU + Vulkan)
========================================
Speech-to-text server using llama.cpp's llama-server with GGUF model.
Supports AMD GPU acceleration via Vulkan API.

Features:
- AMD GPU acceleration via Vulkan
- CPU fallback when Vulkan unavailable
- FastAPI compatible with qwen3-asr-server
- Local result saving with recovery

API Endpoints:
- GET /         - Service status
- GET /health   - Health check
- POST /v1/transcribe - Transcribe audio file

Environment Variables (ASR_* prefix):
- ASR_ENABLE_VULKAN=true      - Enable Vulkan GPU offloading (default: true)
- ASR_N_GPU_LAYERS=-1         - GPU layers (-1 = all)
- ASR_N_CTX=4096              - llama-server context window
- ASR_HOST=0.0.0.0            - FastAPI host
- ASR_PORT=8001               - FastAPI port
- ASR_LLAMA_SERVER_PORT=8080  - Internal llama-server port
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import config
import audio_encoder

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qwen3-gguf")

# ─── Language mapping ────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
}
_CANONICAL = {n.lower(): n for n in LANGUAGE_NAMES}

SHORT_TO_FULL = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "en": "English", "en-us": "English", "en-gb": "English",
    "ja": "Japanese", "jp": "Japanese",
    "ko": "Korean", "kr": "Korean",
    "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "it": "Italian",
    "ru": "Russian", "ar": "Arabic",
    "th": "Thai", "vi": "Vietnamese",
    "tr": "Turkish", "hi": "Hindi",
    "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech",
    "ms": "Malay", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian",
    "tl": "Filipino", "fa": "Persian", "mk": "Macedonian",
}


def normalize_language(lang: str | None) -> str | None:
    """Normalize language input to Qwen3-ASR full English name."""
    if lang is None:
        return None
    if not isinstance(lang, str):
        return None
    key = lang.strip().lower()
    if key in SHORT_TO_FULL:
        return SHORT_TO_FULL[key]
    if key in _CANONICAL:
        return _CANONICAL[key]
    return lang


# ─── llama-server subprocess ─────────────────────────────────────────────
_llama_server_proc: Optional[subprocess.Popen] = None
_llama_server_log: Optional[object] = None  # file handle for stdout log


def _llama_server_url() -> str:
    return f"http://{config.LLAMA_SERVER_HOST}:{config.LLAMA_SERVER_PORT}"


def _start_llama_server() -> subprocess.Popen:
    """Start the llama-server subprocess."""
    global _llama_server_proc

    if _llama_server_proc is not None and _llama_server_proc.poll() is None:
        return _llama_server_proc

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {config.MODEL_PATH}. "
            f"Download from: https://huggingface.co/{config.HF_MODEL_REPO}"
        )
    if not config.MMPROJ_PATH.exists():
        raise FileNotFoundError(
            f"mmproj not found: {config.MMPROJ_PATH}. "
            f"Download from: https://huggingface.co/{config.HF_MODEL_REPO}"
        )

    ngl = config.N_GPU_LAYERS if config.ENABLE_VULKAN else 0
    cmd = [
        str(config.LLAMA_SERVER_BIN),
        "-m", str(config.MODEL_PATH),
        "--mmproj", str(config.MMPROJ_PATH),
        "--host", config.LLAMA_SERVER_HOST,
        "--port", str(config.LLAMA_SERVER_PORT),
        "-ngl", str(ngl),
        "-c", str(config.N_CTX),
        "--no-warmup",
    ]

    logger.info("Starting llama-server: %s", " ".join(cmd))
    t0 = time.time()
    # Write logs to file instead of PIPE to avoid pipe buffer deadlock
    # (Windows pipe buffer is ~4KB; filling it blocks the subprocess)
    global _llama_server_log
    _llama_server_log = open("llama-server.log", "a", encoding="utf-8")
    _llama_server_proc = subprocess.Popen(
        cmd,
        stdout=_llama_server_log,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Wait for server to become ready
    url = f"{_llama_server_url()}/health"
    for i in range(120):  # up to 60s
        if _llama_server_proc.poll() is not None:
            stdout, _ = _llama_server_proc.communicate()
            raise RuntimeError(f"llama-server exited early (code {_llama_server_proc.returncode}):\n{stdout[-2000:]}")
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        _llama_server_proc.terminate()
        raise RuntimeError("llama-server did not become ready within 60s")

    elapsed = time.time() - t0
    logger.info("llama-server ready on %s (started in %.1fs)", _llama_server_url(), elapsed)
    return _llama_server_proc


def _stop_llama_server():
    """Stop the llama-server subprocess."""
    global _llama_server_proc, _llama_server_log
    if _llama_server_proc is not None:
        logger.info("Stopping llama-server")
        try:
            _llama_server_proc.terminate()
            _llama_server_proc.wait(timeout=10)
        except Exception:
            try:
                _llama_server_proc.kill()
            except Exception:
                pass
        _llama_server_proc = None
    if _llama_server_log is not None:
        try:
            _llama_server_log.close()
        except Exception:
            pass
        _llama_server_log = None


def _wait_for_llama_server(timeout: float = 5.0) -> bool:
    """Check if llama-server is responsive (basic health check)."""
    url = f"{_llama_server_url()}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False




# ─── Transcription via llama-server ──────────────────────────────────────

_QWEN3_ASR_RE = re.compile(r"language\s+(?P<lang>\S+?)<asr_text>(?P<text>.*)", re.DOTALL)

# ─── Result cache for agent retry ───────────────────────────────────────
# Cache completed transcription results by request_id so agents can
# retrieve them if their initial request times out.
_result_cache: dict[str, dict] = {}
_RESULT_CACHE_MAX = 50


def _cache_put(request_id: str, result: dict):
    """Store a transcription result for later retrieval by the agent."""
    _result_cache[request_id] = result
    # Trim oldest entries when over capacity
    if len(_result_cache) > _RESULT_CACHE_MAX:
        for key in list(_result_cache.keys())[:-_RESULT_CACHE_MAX]:
            _result_cache.pop(key, None)


def _make_request_id(filename: str) -> str:
    """Generate a deterministic request_id from filename + timestamp."""
    raw = f"{filename}_{datetime.now().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _transcribe_chunk(
    chunk_idx: int,
    wav_bytes: bytes,
    start_s: float,
    end_s: float,
    timeout: int = 300,
) -> dict:
    """Send one audio chunk to llama-server and return a segment."""
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "n_predict": config.N_PREDICT,
        "stop": ["<|im_end|>"],
    }

    url = f"{_llama_server_url()}/v1/chat/completions"
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"llama-server error {r.status_code}: {r.text}")

    data = r.json()
    content = data["choices"][0]["message"]["content"].strip()

    # Parse "language Chinese<asr_text>..."
    match = _QWEN3_ASR_RE.match(content)
    if match:
        lang = match.group("lang").strip()
        text = match.group("text").strip()
    else:
        lang = None
        text = content

    # Clean up possible trailing special tokens
    text = text.replace("<|im_end|>", "").strip()

    return {
        "start": start_s,
        "end": end_s,
        "text": text,
        "language": lang,
    }


def _merge_segments(segments: list) -> str:
    """
    Merge segment texts, deduplicating overlapping content between adjacent chunks.

    With overlapping audio chunks, the same text may appear at the end of
    one segment and the start of the next. This function trims duplicates.
    """
    if not segments:
        return ""

    texts = [segments[0]["text"]]
    for i in range(1, len(segments)):
        prev = texts[-1]
        curr = segments[i]["text"]
        merged = _dedup_overlap(prev, curr)
        texts.append(merged)

    return "".join(texts).strip()


def _dedup_overlap(prev: str, curr: str, min_overlap: int = 5) -> str:
    """
    Trim overlapping prefix from `curr` if it matches the suffix of `prev`.

    Example:
        prev = "那我也没办法。只要"
        curr = "只要我们应该增加社会福利"
        → "我们应该增加社会福利"  (trim "只要")
    """
    # Try longest overlap first (up to 40 chars ≈ 2s of speech)
    max_check = min(len(prev), len(curr), 40)
    for overlap_len in range(max_check, min_overlap - 1, -1):
        suffix = prev[-overlap_len:]
        prefix = curr[:overlap_len]
        if suffix == prefix:
            return curr[overlap_len:]
    return curr


def _transcribe_llama_server(
    audio_bytes: bytes,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> dict:
    """Transcribe audio using llama-server backend."""
    if not audio_encoder.validate_audio(audio_bytes, config.MAX_FILE_SIZE_MB):
        raise HTTPException(413, f"File too large. Max: {config.MAX_FILE_SIZE_MB}MB")

    audio_info = audio_encoder.get_audio_info(audio_bytes)
    duration_s = audio_info["duration_s"]
    logger.info("Audio: %.1fs @ %dHz, %d channels", duration_s, audio_info["sample_rate"], audio_info["channels"])

    # Load and resample to 16kHz mono
    audio, sr = audio_encoder.load_audio(audio_bytes, target_sr=config.SAMPLE_RATE)

    # Chunk audio into ~30s WAV segments with overlap
    chunks = audio_encoder.chunk_audio(
        audio, sr=sr, max_duration_s=config.CHUNK_DURATION_S,
        overlap_s=config.CHUNK_OVERLAP_S,
    )

    segments = []
    detected_lang = normalize_language(language)
    t_start = time.time()
    consecutive_failures = 0
    max_consecutive_failures = getattr(config, "MAX_CHUNK_FAILURES", 5)
    chunk_timeout = getattr(config, "CHUNK_TIMEOUT", 120)

    for idx, start_s, end_s, wav_bytes in chunks:
        t_chunk = time.time()

        # ── Retry loop for each chunk ────────────────────────────────
        # Use full timeout on first attempt, shorter on retries
        segment = None
        retry_timeout = max(chunk_timeout // 2, 30)
        for attempt in range(config.CHUNK_RETRIES + 1):
            try:
                to = chunk_timeout if attempt == 0 else retry_timeout
                segment = _transcribe_chunk(idx, wav_bytes, start_s, end_s, timeout=to)
                break
            except Exception as e:
                logger.warning(
                    "Chunk %d/%d (%.1fs-%.1fs) attempt %d/%d failed: %s",
                    idx + 1, len(chunks), start_s, end_s,
                    attempt + 1, config.CHUNK_RETRIES + 1, e,
                )
                if attempt < config.CHUNK_RETRIES:
                    time.sleep(config.CHUNK_RETRY_DELAY)

        if segment is not None:
            text = segment["text"]
            if segment.get("language"):
                detected_lang = normalize_language(segment["language"]) or detected_lang
            consecutive_failures = 0
        else:
            logger.error("Chunk %d/%d failed after all retries", idx + 1, len(chunks))
            text = ""
            segment = {"start": start_s, "end": end_s, "text": "", "language": None}
            consecutive_failures += 1

        # ── Abort if too many consecutive failures ───────────────────
        if consecutive_failures >= max_consecutive_failures:
            logger.error(
                "%d consecutive chunk failures, aborting transcription",
                consecutive_failures,
            )
            break

        chunk_dur = time.time() - t_chunk
        if text:
            segments.append(segment)
            logger.info(
                "Chunk %d/%d (%.1fs-%.1fs) in %.1fs: %s",
                idx + 1, len(chunks), start_s, end_s, chunk_dur, text[:60]
            )

    total_time = time.time() - t_start
    full_text = _merge_segments(segments)
    rtf = total_time / duration_s if duration_s > 0 else 0

    logger.info(
        "Done: %.1fs audio -> %.1fs processing (RTF=%.2f, %d segments)",
        duration_s, total_time, rtf, len(segments),
    )

    return {
        "text": full_text,
        "language": detected_lang,
        "segments": segments if word_timestamps else [],
        "processing_time": round(total_time, 2),
        "duration_s": round(duration_s, 2),
        "rtf": round(rtf, 3),
    }


# ─── Save result locally ──────────────────────────────────────────────────

def _save_result(
    filename: str,
    text: str,
    result: dict,
    segments: List[dict],
    processing_time: float,
):
    """Save transcription result locally for recovery."""
    base = Path(filename).stem
    safe_name = "".join(c if c.isalnum() or c in " _-." else "_" for c in base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")

    out_dir = Path(config.OUTPUT_DIR) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{safe_name}_{timestamp}.txt"
    txt_path.write_text(text, encoding="utf-8")

    json_data = {
        **result,
        "segments": segments,
        "saved_at": datetime.now().isoformat(),
    }
    json_path = out_dir / f"{safe_name}_{timestamp}.json"
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_path = out_dir / f"{safe_name}_{timestamp}.meta.json"
    meta_data = {
        "id": f"{safe_name}_{timestamp}",
        "original_filename": filename,
        "file_size_mb": round(len(text.encode("utf-8")) / 1024 / 1024, 2),
        "duration_s": result.get("duration_s", 0),
        "language": result.get("language"),
        "processing_time": processing_time,
        "rtf": result.get("rtf", 0),
        "timestamp": datetime.now().isoformat(),
        "backend": config.LLAMA_BACKEND,
        "segments_count": len(segments),
        "files": {
            "txt": os.path.relpath(txt_path),
            "json": os.path.relpath(json_path),
            "meta": os.path.relpath(meta_path),
        },
    }
    meta_path.write_text(
        json.dumps(meta_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_path = Path(config.OUTPUT_DIR) / "latest_index.json"
    index = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            index = []

    index.insert(0, {
        "id": f"{safe_name}_{timestamp}",
        "original_filename": filename,
        "timestamp": datetime.now().isoformat(),
        "duration_s": meta_data["duration_s"],
        "language": meta_data["language"],
        "txt_file": os.path.relpath(txt_path),
    })

    index = index[:50]
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved transcription to:\n  %s\n  %s\n  %s",
                txt_path, json_path, meta_path)


# ─── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Service lifespan: start llama-server, stop on shutdown."""
    logger.info("=" * 60)
    logger.info("  Qwen3-ASR GGUF Server (v2.0)")
    logger.info("=" * 60)
    logger.info("Model:      %s", config.MODEL_PATH)
    logger.info("mmproj:     %s", config.MMPROJ_PATH)
    logger.info("Backend:    %s", config.LLAMA_BACKEND)
    logger.info("Vulkan:     %s", config.ENABLE_VULKAN)
    logger.info("GPU Layers: %d", config.N_GPU_LAYERS if config.ENABLE_VULKAN else 0)
    logger.info("Context:    %d", config.N_CTX)
    logger.info("API:        http://localhost:%d", config.PORT)
    logger.info("=" * 60)

    if config.ENABLE_VULKAN and not config.check_vulkan_available():
        logger.warning("Vulkan SDK not found. CPU-only mode will be used.")

    try:
        _start_llama_server()
    except FileNotFoundError as e:
        logger.error("Model/mmproj not found: %s", e)
        logger.error("Download using: python download_model.py")
        raise
    except Exception as e:
        logger.error("Failed to start llama-server: %s", e)
        raise

    yield
    _stop_llama_server()
    logger.info("Shutting down...")


# ─── FastAPI App ─────────────────────────────────────────────────────────

app = FastAPI(title="Qwen3-ASR GGUF", version="2.0", lifespan=lifespan)

# Concurrency limiter for transcription requests (llama-server has limited slots)
_transcribe_semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)


class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = None
    segments: Optional[List[dict]] = None
    processing_time: float
    duration_s: Optional[float] = None
    rtf: Optional[float] = None
    request_id: Optional[str] = None


@app.get("/")
async def root():
    """Service status and configuration."""
    return {
        "status": "ok",
        "service": "Qwen3-ASR GGUF",
        "version": "2.0",
        "backend": "llama.cpp",
        "backend_type": config.LLAMA_BACKEND,
        "model_loaded": _wait_for_llama_server(timeout=2),
        "model_info": config.get_model_info(),
        "vulkan_enabled": config.ENABLE_VULKAN,
        "vulkan_detected": config.check_vulkan_available(),
        "optimizations": {
            "vulkan": config.ENABLE_VULKAN,
            "n_gpu_layers": config.N_GPU_LAYERS,
            "n_ctx": config.N_CTX,
            "chunk_duration_s": config.CHUNK_DURATION_S,
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if _wait_for_llama_server(timeout=2) else "initializing",
        "model_loaded": _wait_for_llama_server(timeout=2),
        "model_exists": config.MODEL_PATH.exists(),
        "mmproj_exists": config.MMPROJ_PATH.exists(),
    }


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
    request_id: Optional[str] = Form(None),
):
    """
    Transcribe audio file using GGUF model.

    Args:
        file: Audio file (wav, mp3, m4a, etc.)
        language: Language hint — full English name or short code. Auto-detect if omitted.
        word_timestamps: Whether to return segment timestamps
        request_id: Client-generated ID for result retrieval if the request times out
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Use client's request_id or generate one
    rid = request_id or _make_request_id(file.filename)

    async with _transcribe_semaphore:
        start_time = time.time()

        try:
            content = await file.read()
            file_size_mb = len(content) / 1024 / 1024
            logger.info("Request: file=%s size=%.1fMB backend=%s request_id=%s",
                        file.filename, file_size_mb, config.LLAMA_BACKEND, rid)

            result = _transcribe_llama_server(content, language=language, word_timestamps=word_timestamps)

            processing_time = time.time() - start_time
            logger.info("Done in %.1fs: %s", processing_time, result["text"][:80])

            segments = result.get("segments", [])
            _save_result(file.filename, result["text"], result, segments, processing_time)

            # Cache for agent retry
            cached = {
                "text": result["text"],
                "language": result.get("language"),
                "segments": result.get("segments") if word_timestamps else [],
                "processing_time": processing_time,
                "duration_s": result.get("duration_s"),
                "rtf": result.get("rtf"),
                "request_id": rid,
            }
            _cache_put(rid, cached)

            return TranscribeResponse(
                text=result["text"],
                language=result.get("language"),
                segments=result.get("segments") if word_timestamps else None,
                processing_time=processing_time,
                duration_s=result.get("duration_s"),
                rtf=result.get("rtf"),
                request_id=rid,
            )

        except HTTPException:
            raise
        except FileNotFoundError as e:
            logger.error("Model/mmproj not found: %s", e)
            raise HTTPException(500, f"Model/mmproj not found: {e}")
        except Exception as e:
            logger.exception("Transcription failed")
            raise HTTPException(500, f"Transcription failed: {e}")


@app.post("/asr")
async def transcribe_asr(file: UploadFile = File(...)):
    """Simplified endpoint (alias for /v1/transcribe)."""
    return await transcribe(file=file)


@app.get("/v1/transcribe/result/{request_id}")
async def get_transcribe_result(request_id: str):
    """
    Retrieve a previously completed transcription result by request_id.

    Used by the agent client after a timeout to fetch the result without
    re-processing the audio. Results are cached in memory for recent requests.
    """
    cached = _result_cache.get(request_id)
    if cached is None:
        raise HTTPException(404, f"No cached result found for request_id: {request_id}")
    return cached


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Qwen3-ASR GGUF server (backend=%s)", config.LLAMA_BACKEND)
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        timeout_keep_alive=30,
        log_level="info",
    )
