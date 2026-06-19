"""
Qwen3-ASR GGUF Agent 客户端
============================
兼容 GGUF 服务，也兼容 qwen3-asr-server 主服务

使用方法:
    from agent_client import ASRClient
    client = ASRClient("http://192.168.50.230:8001")
    text = client.transcribe("audio.wav")

超时策略:
    RTF ≈ 0.3 (10分钟音频约需3分钟处理)
    默认超时 = 音频时长 × 0.3 × 3 (3倍安全余量) + 30s 缓冲
    优先从音频文件读取真实时长，失败则按文件大小估算
"""
import hashlib
import logging
import requests
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("asr-client")


class ASRClient:
    """
    ASR 服务客户端 - 专为 Agent 设计

    GGUF 版本特性:
        - 支持 AMD GPU (Vulkan)
        - 模型大小 ~768MB (Q8_0 量化)
        - 基于 Qwen3-ASR 全量模型，识别准确率高
        - 兼容主服务 API

    示例:
        >>> client = ASRClient("http://192.168.50.230:8001")
        >>> text = client.transcribe("meeting.wav")
        >>> print(text)
    """

    def __init__(self, server_url: str, timeout: int = 600):
        """
        初始化客户端

        参数:
            server_url: ASR 服务地址，如 "http://192.168.50.230:8001"
            timeout: 请求超时时间（秒），长音频建议增大
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # 设置重试策略
        adapter = requests.adapters.HTTPAdapter(max_retries=2)
        self.session.mount("http://", adapter)

    def is_available(self) -> bool:
        """
        检查服务是否可用

        返回:
            True 表示服务正常
        """
        try:
            resp = self.session.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态和设备信息

        返回:
            {
                "status": "ok",
                "service": "Qwen3-ASR GGUF",
                "version": "1.0",
                "backend": "llama.cpp",
                "backend_type": "vulkan" | "cpu",
                "model_loaded": True/False,
                "vulkan_enabled": True/False,
                "model_info": {...}
            }
        """
        resp = self.session.get(f"{self.server_url}/", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def _estimate_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        估算音频时长（秒）。

        优先用 soundfile 读取真实时长，失败则按文件大小粗略估算。
        """
        path = Path(audio_path)
        if not path.exists():
            return None

        # Try soundfile first (most accurate)
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return info.duration
        except Exception:
            pass

        # Try ffprobe as fallback
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

        # Rough estimate from file size
        # Assume ~16KB/s for compressed audio (128kbps MP3 / 256kbps OPUS)
        size_bytes = path.stat().st_size
        return size_bytes / 16000  # very rough

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False
    ) -> str:
        """
        🔴 Agent 主调用方法 - 转录音频，只返回文本

        参数:
            audio_path: 音频文件路径
            language: 指定语言 — 完整英文名 (Chinese, English, Japanese...)
                     或短代码 (zh, en, ja...)，不传则自动检测
            word_timestamps: 是否需要时间戳 (默认不需要)

        返回:
            转录的文本字符串

        异常:
            requests.exceptions.RequestException: 网络错误
        """
        result = self.transcribe_full(
            audio_path,
            language=language,
            word_timestamps=word_timestamps
        )
        return result["text"]

    def _make_request_id(self, audio_path: str) -> str:
        """Generate a deterministic request_id for result retrieval on retry."""
        path = Path(audio_path)
        raw = f"{path.name}_{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _fetch_cached_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previously completed transcription result by request_id.
        Returns None if the result is not yet available or doesn't exist.
        """
        try:
            resp = self.session.get(
                f"{self.server_url}/v1/transcribe/result/{request_id}",
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def transcribe_full(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        完整转录 - 返回全部信息（文本、语言、耗时等）

        自动生成 request_id 并在请求超时/失败时尝试从服务端缓存获取结果，
        避免重复转录音频。

        返回:
            {
                "text": "转录文本",
                "language": "Chinese",
                "processing_time": 1.23,
                "segments": [...],
                "duration_s": 60.5,
                "rtf": 0.35,
                "request_id": "abc123def456"
            }
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # ── 生成 request_id，用于超时后拉取缓存 ──────────────────
        request_id = self._make_request_id(audio_path)

        # ── 根据音频时长动态计算超时 ──────────────────────────────
        # RTF ≈ 0.3，即 10分钟音频约需 3分钟处理
        # 超时 = 音频时长 × 0.3(RTF) × 3(安全余量) + 30s 缓冲
        duration_s = self._estimate_audio_duration(audio_path)
        if duration_s and duration_s > 0:
            estimated_processing_s = duration_s * 0.3
            timeout = max(self.timeout, int(estimated_processing_s * 3) + 30)
        else:
            timeout = self.timeout

        # ── 发起转录请求 ──────────────────────────────────────────
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (path.name, f)}
                data = {"request_id": request_id}
                if language:
                    data["language"] = language
                if word_timestamps:
                    data["word_timestamps"] = "true"

                resp = self.session.post(
                    f"{self.server_url}/v1/transcribe",
                    files=files,
                    data=data,
                    timeout=timeout,
                )

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            # ── 请求失败（超时/网络错误），尝试从缓存拉取结果 ──────
            logger.warning("转录请求失败: %s，尝试从缓存拉取结果 (request_id=%s)", e, request_id)

            # 轮询缓存：等待最多 2 分钟，每 5 秒检查一次
            for attempt in range(24):
                time.sleep(5)
                cached = self._fetch_cached_result(request_id)
                if cached is not None:
                    logger.info("从缓存成功获取转录结果 (request_id=%s)", request_id)
                    return cached
                logger.info("等待转录完成... (attempt %d/24)", attempt + 1)

            # 缓存中也找不到，抛出原始异常
            logger.error("无法获取转录结果 (request_id=%s)，缓存中不存在", request_id)
            raise

    def wait_for_service(self, max_wait: int = 60, check_interval: int = 2) -> bool:
        """
        等待服务启动（Agent 启动时用）

        参数:
            max_wait: 最大等待秒数
            check_interval: 检查间隔

        返回:
            True 表示服务已就绪
        """
        print(f"⏳ 等待 ASR 服务: {self.server_url}")
        start = time.time()
        while time.time() - start < max_wait:
            if self.is_available():
                print("✅ ASR 服务已就绪")
                return True
            print(f"   等待中... ({int(time.time() - start)}s)")
            time.sleep(check_interval)
        print("❌ ASR 服务超时未响应")
        return False

    def get_backend_info(self) -> Dict[str, Any]:
        """
        获取后端信息（GPU、Vulkan 等）

        返回:
            {
                "backend_type": "vulkan" | "cpu" | "cuda" | "metal",
                "vulkan_enabled": bool,
                "vulkan_detected": bool,
                "n_gpu_layers": int,
            }
        """
        status = self.get_status()
        return {
            "backend_type": status.get("backend_type", "unknown"),
            "vulkan_enabled": status.get("vulkan_enabled", False),
            "vulkan_detected": status.get("vulkan_detected", False),
            "n_gpu_layers": status.get("optimizations", {}).get("n_gpu_layers", 0),
        }


# ============================================
# 简单示例
# ============================================

if __name__ == "__main__":
    # ┌───────────────────────────────────┐
    # │  配置：改成你 Windows 主机的 IP     │
    # └───────────────────────────────────┘
    SERVER_URL = "http://192.168.50.230:8001"

    # 1. 创建客户端
    client = ASRClient(SERVER_URL)

    # 2. 查看服务状态和后端信息
    try:
        status = client.get_status()
        print(f"Service: {status['service']}")
        print(f"Backend: {status['backend']} ({status['backend_type']})")
        print(f"Model Loaded: {status['model_loaded']}")

        backend_info = client.get_backend_info()
        print(f"Vulkan: {backend_info['vulkan_enabled']}")
        print(f"Vulkan Detected: {backend_info['vulkan_detected']}")
        print(f"GPU Layers: {backend_info['n_gpu_layers']}")
    except Exception as e:
        print(f"⚠️  Cannot connect to server: {e}")

    # 3. 等待服务就绪
    if not client.wait_for_service(max_wait=30):
        print("请先启动 ASR 服务！")
        exit(1)

    # 4. 转录音频
    audio_file = "test.wav"

    if Path(audio_file).exists():
        print(f"\n🎤 正在转录: {audio_file}")
        try:
            text = client.transcribe(audio_file, language="Chinese")
            print(f"\n📝 转录结果:\n{text}")
        except Exception as e:
            print(f"❌ 转录失败: {e}")
    else:
        print(f"请先准备一个音频文件: {audio_file}")