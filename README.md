# Qwen3-ASR GGUF Server (AMD GPU)

基于 llama.cpp 的 llama-server 的 Qwen3-ASR 语音转文字服务，专为 **AMD GPU (Vulkan)** 优化。

## 特性

| 特性 | GGUF 版本 | CPU 版本 (qwen3-asr-server) |
|------|-----------|----------------------------|
| GPU 支持 | AMD (Vulkan) ✅ | NVIDIA (CUDA) ✅ |
| CPU 兼容 | ✅ | ✅ |
| 模型大小 | ~767MB (Q8_0) + ~204MB (mmproj) | ~1.2GB |
| 推理速度 (20s 音频) | CPU: ~4s (RTF 0.20), Vulkan: ~2.5s (RTF 0.13) | GPU: ~0.05x 实时 |
| 依赖 | llama.cpp (已编译) | torch, transformers, qwen-asr |
| API 兼容 | ✅ 相同接口 | ✅ |

## 快速开始

### 前置条件

1. **Python 3.9+**
2. **AMD 显卡** + **Vulkan SDK**（可选，用于 GPU 加速）
3. **Visual Studio Build Tools**（用于编译 llama.cpp）
4. 已编译的 `llama.cpp/build/bin/Release/llama-server.exe`

### 步骤 1: 安装 Vulkan SDK（可选但推荐）

下载并安装 [Vulkan SDK](https://vulkan.lunarg.com/)

验证安装：
```batch
vulkaninfo --summary
```

### 步骤 2: 下载模型

```batch
python download_model.py
```

或手动下载：
1. 访问 https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF
2. 下载 `Qwen3-ASR-0.6B-Q8_0.gguf` 和 `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf`
3. 放到 `model/` 目录

### 步骤 3: 启动服务

```batch
# 一键启动（自动检测环境）
start.bat
```

首次运行会自动：
1. 创建虚拟环境
2. 安装依赖
3. 检查 Vulkan 环境
4. 检查模型文件
5. 启动内部 llama-server (端口 8080)

### 步骤 4: 测试服务

访问 API 文档：
```
http://localhost:8001/docs
```

或使用命令行测试：
```batch
curl -X POST "http://localhost:8001/v1/transcribe" -F "file=@test.wav" -F "language=Chinese"
```

## 依赖安装

```batch
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 编译 llama.cpp

如果 `llama-server.exe` 尚未编译：

```batch
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server
```

如果不需要 GPU，编译 CPU 版本：
```batch
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server
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

### GET /health

健康检查。

## Agent 客户端

复制 `agent_client.py` 到 Agent 项目。

```python
from agent_client import ASRClient

client = ASRClient("http://192.168.50.230:8001")
text = client.transcribe("meeting.wav", language="Chinese")
print(text)
```

## 环境变量

所有环境变量使用 `ASR_*` 前缀。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_ENABLE_VULKAN` | false | 是否启用 Vulkan 加速 |
| `ASR_N_GPU_LAYERS` | -1 | GPU 层数 (-1 = 全部) |
| `ASR_N_CTX` | 4096 | llama-server 上下文窗口 |
| `ASR_HOST` | 0.0.0.0 | FastAPI 监听地址 |
| `ASR_PORT` | 8001 | FastAPI 端口 |
| `ASR_LLAMA_SERVER_PORT` | 8080 | 内部 llama-server 端口 |
| `ASR_MAX_FILE_SIZE_MB` | 500 | 最大文件大小 (MB) |
| `ASR_MAX_CONCURRENCY` | 2 | 最大并发转录数 |

**示例:**
```batch
# 启用 Vulkan（AMD GPU）
set ASR_ENABLE_VULKAN=true

# 禁用 Vulkan（CPU 模式）
set ASR_ENABLE_VULKAN=false

# 限制上下文窗口（显存不足时）
set ASR_N_CTX=2048
```

## 常见问题

### Q: Vulkan 不可用

**A:** 检查：
1. 已安装 [Vulkan SDK](https://vulkan.lunarg.com/)
2. 环境变量 `VULKAN_SDK` 已设置
3. llama.cpp 已用 `-DGGML_VULKAN=ON` 编译

如果仍失败，使用 CPU 模式：
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
2. `ASR_ENABLE_VULKAN=true`
3. llama-server 日志中显示 `Vulkan0`

### Q: 与 qwen3-asr-server 共存？

**A:** 可以。两个项目完全独立。如果同时运行，修改端口。

## 性能参考

| 后端 | 音频时长 | 处理时间 | RTF |
|------|----------|----------|-----|
| Vulkan (AMD) | 20s | ~2.5s | 0.13 |
| CPU (8核) | 20s | ~4s | 0.20 |

## 相关链接

- [Qwen3-ASR 原始仓库](https://huggingface.co/Qwen/Qwen3-ASR)
- [GGUF 模型](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Vulkan SDK](https://vulkan.lunarg.com/)
