# Qwen3-ASR GGUF Server (AMD GPU)

基于 llama-cpp-python 的 Qwen3-ASR 语音转文字服务，专为 **AMD GPU** 优化。

## 特性

| 特性 | GGUF 版本 | CPU 版本 (qwen3-asr-server) |
|------|-----------|----------------------------|
| GPU 支持 | AMD (Vulkan) ✅ | NVIDIA (CUDA) ✅ |
| CPU 兼容 | ✅ | ✅ |
| 模型大小 | ~480MB (Q4_K_M) | ~1.2GB |
| 准确率 | ~80% (简化编码器) | 100% |
| 推理速度 | Vulkan: ~0.3x 实时 | GPU: ~0.05x 实时 |
| 依赖 | llama-cpp-python | torch, transformers, qwen-asr |
| API 兼容 | ✅ 相同接口 | ✅ |

## 快速开始

### 前置条件

1. **Python 3.9+**
2. **AMD 显卡** + **Vulkan SDK**（可选，用于 GPU 加速）
3. **Visual Studio Build Tools**（用于编译 llama-cpp-python）

### 步骤 1: 安装 Vulkan SDK（可选但推荐）

下载并安装 [Vulkan SDK](https://vulkan.lunarg.com/)

验证安装：
```batch
vulkaninfo --summary
```

### 步骤 2: 启动服务

```batch
# 一键启动（自动检测环境）
start.bat
```

首次运行会自动：
1. 创建虚拟环境
2. 安装依赖
3. 检查 Vulkan 环境
4. 检查模型文件

### 步骤 3: 下载模型

如果模型未自动下载，运行：
```batch
python download_model.py
```

或手动下载：
1. 访问 https://huggingface.co/HaujetZhao/Qwen3-ASR-0.6B-GGUF
2. 下载 `qwen3-asr-0.6b-Q4_K_M.gguf`
3. 放到 `model/` 目录

### 步骤 4: 测试服务

访问 API 文档：
```
http://localhost:8000/docs
```

或使用命令行测试：
```batch
curl -X POST "http://localhost:8000/v1/transcribe" -F "file=@test.wav" -F "language=Chinese"
```

## 依赖安装详解

### 普通安装（CPU 模式）

```batch
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### AMD GPU (Vulkan) 支持

```batch
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

如果编译失败，确保已安装：
- **Vulkan SDK**
- **Visual Studio Build Tools**（包含 C++ 工具）

### 编译失败后的回退方案

如果 Vulkan 编译失败，可以回退到 CPU 模式：

```batch
# 设置环境变量禁用 Vulkan
set ASR_ENABLE_VULKAN=false

# 或在代码中修改 config.py:
# ENABLE_VULKAN = False
```

## API 接口

### POST /v1/transcribe

转录音频文件。

**参数:**
- `file`: 音频文件（支持 wav, mp3, m4a 等）
- `language` (可选): 语言提示
  - 完整英文名: `Chinese`, `English`, `Japanese`...
  - 短代码: `zh`, `en`, `ja`...
  - 不传则自动检测
- `word_timestamps` (可选): 是否返回时间戳

**返回:**
```json
{
  "text": "转录文本",
  "language": "Chinese",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "第一段文本"
    }
  ],
  "processing_time": 1.23,
  "duration_s": 60.5,
  "rtf": 0.35
}
```

### GET /

服务状态和配置信息。

**返回:**
```json
{
  "status": "ok",
  "service": "Qwen3-ASR GGUF",
  "version": "1.0",
  "backend": "llama.cpp",
  "backend_type": "vulkan",
  "model_loaded": true,
  "vulkan_enabled": true,
  "vulkan_detected": true,
  "optimizations": {
    "vulkan": true,
    "n_gpu_layers": -1,
    "n_ctx": 4096
  }
}
```

### GET /health

健康检查。

**返回:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_exists": true
}
```

## Agent 客户端

### 安装

复制 `agent_client.py` 到 Agent 项目。

### 使用示例

```python
from agent_client import ASRClient

# 创建客户端
client = ASRClient("http://192.168.50.230:8000")

# 简单用法
text = client.transcribe("meeting.wav", language="Chinese")
print(text)

# 完整用法
result = client.transcribe_full("meeting.wav", language="Chinese")
print(f"文本: {result['text']}")
print(f"耗时: {result['processing_time']}s")
print(f"RTF: {result['rtf']}")

# 查看后端信息
backend_info = client.get_backend_info()
print(f"后端: {backend_info['backend_type']}")
print(f"Vulkan: {backend_info['vulkan_enabled']}")
```

### Agent 启动时等待服务

```python
client = ASRClient("http://192.168.50.230:8000")
if not client.wait_for_service(max_wait=60):
    print("服务未就绪")
    exit(1)
```

## 环境变量

所有环境变量使用 `ASR_*` 前缀。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_ENABLE_VULKAN` | true | 是否启用 Vulkan 加速 |
| `ASR_N_GPU_LAYERS` | -1 | GPU 层数 (-1 = 全部) |
| `ASR_N_CTX` | 4096 | 上下文窗口大小 |
| `ASR_N_THREADS` | 0 | CPU 线程数 (0 = 自动) |
| `ASR_HOST` | 0.0.0.0 | 服务监听地址 |
| `ASR_PORT` | 8000 | 服务端口 |
| `ASR_MAX_FILE_SIZE_MB` | 500 | 最大文件大小 (MB) |
| `ASR_MAX_CONCURRENCY` | 2 | 最大并发数 |

**示例:**
```batch
# 禁用 Vulkan（CPU 模式）
set ASR_ENABLE_VULKAN=false

# 限制 GPU 层数（显存不足时）
set ASR_N_GPU_LAYERS=20

# 增加上下文窗口
set ASR_N_CTX=8192
```

## 与 CPU 版本的区别

### 文件结构

| qwen3-asr-server | qwen3-asr-gguf |
|------------------|----------------|
| `main.py` | `main.py` (不同实现) |
| `config.py` | `config.py` (不同配置) |
| `process_long_audio.py` | `audio_encoder.py` |
| `agent_client.py` | `agent_client.py` (兼容) |
| - | `download_model.py` |

### 依赖差异

| qwen3-asr-server | qwen3-asr-gguf |
|------------------|----------------|
| torch, torchaudio | - |
| transformers | - |
| qwen-asr | llama-cpp-python |
| bitsandbytes (GPU) | - |
| silero-vad (CPU) | - |

### 配置差异

qwen3-asr-server:
```python
DEVICE = "cuda" | "cpu"
ENABLE_4BIT = true
GPU_DTYPE = "bfloat16"
CHUNK_DURATION_S = 15
ENABLE_VAD = true
```

qwen3-asr-gguf:
```python
ENABLE_VULKAN = true
N_GPU_LAYERS = -1
LLAMA_BACKEND = "vulkan" | "cpu"
CHUNK_DURATION_S = 30
```

## 常见问题

### Q: Vulkan 编译失败

**A:** 检查以下项目：
1. 已安装 [Vulkan SDK](https://vulkan.lunarg.com/)
2. 已安装 Visual Studio Build Tools
3. 环境变量 `VULKAN_SDK` 已设置

如果仍失败，回退到 CPU 模式：
```batch
set ASR_ENABLE_VULKAN=false
start.bat
```

### Q: 模型下载太慢

**A:** 使用代理：
```batch
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
python download_model.py
```

### Q: GPU 未被使用

**A:** 检查：
1. Vulkan SDK 已安装: `vulkaninfo --summary`
2. llama-cpp-python 已用 Vulkan 编译:
   ```batch
   python -c "import llama_cpp; print(llama_cpp.__version__)"
   ```
3. 环境变量已设置: `ASR_ENABLE_VULKAN=true`

### Q: 准确率不如预期

**A:** GGUF 版本使用简化的 FBank 编码器，准确率约 80%。
如果需要 100% 准确率，使用 [qwen3-asr-server](../qwen3-asr-server/) CPU 版本。

### Q: 与 qwen3-asr-server 共存？

**A:** 可以。两个项目完全独立：
- **qwen3-asr-server**: CPU 或 NVIDIA GPU，端口 8000
- **qwen3-asr-gguf**: AMD GPU (Vulkan)，端口 8000

如果同时运行，修改端口：
```batch
set ASR_PORT=8001  # GGUF 版本使用 8001
start.bat
```

## 性能参考

| 后端 | 音频时长 | 处理时间 | RTF |
|------|----------|----------|-----|
| Vulkan (AMD) | 60s | ~18s | 0.30 |
| CPU (4 核) | 60s | ~150s | 2.50 |
| CPU (8 核) | 60s | ~90s | 1.50 |

## 许可证

本项目基于以下开源项目：
- Qwen3-ASR (Apache 2.0)
- llama.cpp (MIT)
- FastAPI (MIT)

## 相关链接

- [Qwen3-ASR 原始仓库](https://huggingface.co/Qwen/Qwen3-ASR)
- [GGUF 模型](https://huggingface.co/HaujetZhao/Qwen3-ASR-0.6B-GGUF)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Vulkan SDK](https://vulkan.lunarg.com/)