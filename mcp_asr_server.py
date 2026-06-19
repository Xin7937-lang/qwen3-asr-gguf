"""
MCP Server for Qwen3-ASR GGUF
===============================
让 MCP 兼容的 Agent（Claude Code、Cursor 等）自动发现并调用 ASR 服务。

启动:
    python mcp_asr_server.py

配置到 Agent（以 Claude Code 为例）:
    {"mcpServers": {"asr": {"command": "python", "args": ["mcp_asr_server.py"]}}}
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import requests

# ─── Configuration ────────────────────────────────────────────────────────
ASR_SERVER_URL = "http://192.168.50.230:8001"  # 改成你的 ASR 服务器地址
TIMEOUT = 600

# ─── MCP Protocol helpers ─────────────────────────────────────────────────

def send_message(message: dict):
    """Send a JSON-RPC message to the MCP client."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read_message() -> dict:
    """Read a JSON-RPC message from the MCP client."""
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


# ─── Tool: transcribe ────────────────────────────────────────────────────

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> dict:
    """
    Transcribe audio file using the ASR server.

    Args:
        audio_path: Path to the audio file (wav, mp3, m4a, etc.)
        language: Language hint (e.g., "Chinese", "English", "Japanese", or "zh", "en", "ja")
    """
    path = Path(audio_path)
    if not path.exists():
        return {"error": f"File not found: {audio_path}"}

    with open(audio_path, "rb") as f:
        files = {"file": (path.name, f)}
        data = {}
        if language:
            data["language"] = language

        resp = requests.post(
            f"{ASR_SERVER_URL}/v1/transcribe",
            files=files,
            data=data,
            timeout=TIMEOUT,
        )
    resp.raise_for_status()
    return resp.json()


# ─── Tool: check_server ───────────────────────────────────────────────────

def check_server() -> dict:
    """Check if the ASR server is running and healthy."""
    try:
        resp = requests.get(f"{ASR_SERVER_URL}/health", timeout=5)
        return {"status": "ok" if resp.status_code == 200 else "error"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


# ─── Tool definitions (OpenAI-compatible format) ──────────────────────────

TOOLS = [
    {
        "name": "transcribe",
        "description": "转录音频文件为文字。支持 wav, mp3, m4a 等格式。返回文本、语言、处理耗时等信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": "音频文件路径，如 /path/to/recording.wav",
                },
                "language": {
                    "type": "string",
                    "description": "语言提示（可选）。Chinese/zh, English/en, Japanese/ja 等。不传则自动检测。",
                },
            },
            "required": ["audio_path"],
        },
    },
    {
        "name": "check_server",
        "description": "检查 ASR 服务是否正常运行",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ─── Main loop ────────────────────────────────────────────────────────────

def main():
    # Initialize: send capabilities
    send_message({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "qwen3-asr-mcp",
                "version": "1.0.0",
            },
        },
    })

    # Wait for initialized notification
    msg = read_message()

    while True:
        msg = read_message()
        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "tools/list":
            send_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            })

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})

            try:
                if tool_name == "transcribe":
                    result = transcribe_audio(
                        audio_path=args["audio_path"],
                        language=args.get("language"),
                    )
                elif tool_name == "check_server":
                    result = check_server()
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

                send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False, indent=2),
                            }
                        ],
                    },
                })
            except Exception as e:
                send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": str(e)},
                })

        elif method == "notifications/initialized":
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
