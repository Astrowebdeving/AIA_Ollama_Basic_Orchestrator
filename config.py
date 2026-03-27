import os
import subprocess
import sys
from pathlib import Path
from shutil import which
import urllib.error
import urllib.request

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


def _normalise_host(host: str) -> str:
    value = host.strip().rstrip("/")
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    return value


def _ollama_reachable(host: str, timeout: float = 1.0) -> bool:
    if not host:
        return False

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            return getattr(response, "status", 200) == 200
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _resolve_ollama_host(primary_host: str, fallback_host: str) -> str:
    candidates: list[str] = []
    for candidate in (primary_host, fallback_host):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if _ollama_reachable(candidate):
            return candidate

    return candidates[0] if candidates else "http://127.0.0.1:11434"


# Ollama Connection (always needed for embeddings)
OLLAMA_IP = os.getenv("OLLAMA_IP") or _detect_local_ip()
_OLLAMA_PRIMARY_HOST = _normalise_host(
    os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
)
_OLLAMA_FALLBACK_HOST = _normalise_host(
    os.getenv("OLLAMA_FALLBACK_HOST", f"http://{OLLAMA_IP}:11434")
)
OLLAMA_HOST = _resolve_ollama_host(_OLLAMA_PRIMARY_HOST, _OLLAMA_FALLBACK_HOST)

# LLM Provider Selection
# Supported: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")  # auto-set per provider if empty

# Model Settings (defaults vary by provider)
_DEFAULT_MODELS = {
    "ollama": "gemma3:27b-it-qat",
    "afm": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "llamacpp": "gemma3",
}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "gemma3:27b-it-qat"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = int(os.getenv("EMBED_DIM", 1024))
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "google/gemma-3-27b-it")

# Token Budgeting (Gemma 3 27b supports 128k context)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 128000))
SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", 80000))

# LanceDB
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb")

# TSS Unity API — used by the MCP tool server for on-demand fetches
TSS_API_BASE_URL = os.getenv("TSS_API_BASE_URL", "http://127.0.0.1:8100/api/v1")
TSS_API_TIMEOUT = float(os.getenv("TSS_API_TIMEOUT", 10.0))

# MCP Servers configurations
# Each entry spawns a child process the orchestrator communicates with via STDIO.
_MCP_DIR = Path(__file__).resolve().parent / "mcp_servers"

MCP_SERVERS = {
    "tss-tools": {
        "command": sys.executable,
        "args": [str(_MCP_DIR / "tss_tools_server.py")],
    },
}
