import os
import subprocess
import sys
from pathlib import Path
from shutil import which

from dotenv import load_dotenv

load_dotenv()


def _detect_local_ip() -> str:
    """
    Resolve the Ollama IP with this priority:
      1. Auto-detect via `ipconfig getifaddr en0` on macOS
      2. Hardcoded fallback
    """
    if sys.platform == "darwin" and which("ipconfig"):
        try:
            result = subprocess.run(
                ["ipconfig", "getifaddr", "en0"],
                capture_output=True, text=True, timeout=5,
            )
            ip = result.stdout.strip()
            if ip:
                return ip
        except Exception:
            pass

    # Last resort
    return "10.207.22.21"


# Ollama Connection (always needed for embeddings)
OLLAMA_IP = os.getenv("OLLAMA_IP") or _detect_local_ip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", f"http://{OLLAMA_IP}:11434")

# LLM Provider Selection
# Supported: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")  # auto-set per provider if empty

# Model Settings (defaults vary by provider)
_DEFAULT_MODELS = {
    "ollama": "gemma3:27b",
    "afm": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "llamacpp": "gemma3",
}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "gemma3:27b"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = int(os.getenv("EMBED_DIM", 1024))
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "google/gemma-3-27b-it")

# Token Budgeting (Gemma 3 27b supports 128k context)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 128000))
SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", 64000))

# Database Settings
DB_FILE = os.getenv("DB_FILE", "logs.db")
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb")

# Telemetry Poller
# Set TELEMETRY_SOURCE_URL to enable the background poller.
# The poller will GET this URL every TELEMETRY_POLL_INTERVAL seconds
# and store the JSON response as telemetry events.
TELEMETRY_SOURCE_URL = os.getenv("TELEMETRY_SOURCE_URL", "")
TELEMETRY_POLL_INTERVAL = int(os.getenv("TELEMETRY_POLL_INTERVAL", 20))

# MCP Servers configurations
# Each entry spawns a child process the orchestrator communicates with via STDIO.
_MCP_DIR = Path(__file__).resolve().parent / "mcp_servers"

MCP_SERVERS = {
    "sqlite-query": {
        "command": sys.executable,
        "args": [str(_MCP_DIR / "sqlite_query_server.py")],
    },
    "telemetry-search": {
        "command": sys.executable,
        "args": [str(_MCP_DIR / "telemetry_search_server.py")],
    },
}
