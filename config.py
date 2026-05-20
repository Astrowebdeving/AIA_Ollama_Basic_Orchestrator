import os
import subprocess
import sys
from pathlib import Path
from shutil import which
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    # 0.0.0.0 is a server bind-all address — replace with localhost for client use
    value = value.replace("://0.0.0.0", "://localhost")
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


# Ollama Connection
# Use exactly what's configured — connectivity errors are handled per-request.
OLLAMA_HOST = _normalise_host(
    os.getenv("OLLAMA_HOST", "http://localhost:11434")
)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_PREWARM = _env_bool("OLLAMA_PREWARM", True)

# LLM Provider Selection
# Supported: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")  # auto-set per provider if empty

# Model Settings (defaults vary by provider)
_DEFAULT_MODELS = {
    "ollama": "gemma4:26b",
    "afm": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "llamacpp": "gemma4",
}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "gemma4:26b"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = int(os.getenv("EMBED_DIM", 1024))

# Tokenizer — must match the active chat model for accurate token counting.
# Auto-selected based on provider if not explicitly set.
# Tokenizer files are pre-downloaded into tokenizer_cache/ so the orchestrator
# can start without internet access (required for field / local-network use).
_DEFAULT_TOKENIZERS = {
    "ollama": "google/gemma-4-26B-A4B-it",
    "afm": "Qwen/Qwen3.5-35B-A3B",
    "llamacpp": "google/gemma-4-26B-A4B-it",
}
_TOKENIZER_CACHE_DIR = Path(__file__).resolve().parent / "tokenizer_cache"

def _resolve_tokenizer(name: str) -> str:
    """Return a local path to pre-downloaded tokenizer files if available,
    otherwise fall back to the HF model ID (requires internet)."""
    local_dir = _TOKENIZER_CACHE_DIR / name.replace("/", "--")
    if local_dir.is_dir() and (local_dir / "tokenizer.json").exists():
        return str(local_dir)
    return name

TOKENIZER_NAME = _resolve_tokenizer(
    os.getenv(
        "TOKENIZER_NAME",
        _DEFAULT_TOKENIZERS.get(LLM_PROVIDER, "google/gemma-4-26B-A4B-it"),
    )
)

# Token Budgeting (model context window — both Gemma 3 27B and Qwen 3.5 support 128k)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 128000))
SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", 80000))

# LanceDB
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb")

# TSS UDP — connects to the external NASA SUITS TSS2026 server for live telemetry.
# Set TSS_UDP_HOST to the IP of the running TSS instance (provided at test week or local dev).
TSS_UDP_HOST = os.getenv("TSS_UDP_HOST", _detect_local_ip())
TSS_UDP_PORT = int(os.getenv("TSS_UDP_PORT", 14141))
TSS_UDP_TIMEOUT = float(os.getenv("TSS_UDP_TIMEOUT", 2.0))

# Legacy HTTP config (deprecated — TSS2026 uses UDP exclusively)
# TSS_API_BASE_URL = os.getenv("TSS_API_BASE_URL", "http://127.0.0.1:8100/api/v1")
# TSS_API_TIMEOUT = float(os.getenv("TSS_API_TIMEOUT", 10.0))

# MCP Servers configurations
# Each entry spawns a child process the orchestrator communicates with via STDIO.
_MCP_DIR = Path(__file__).resolve().parent / "mcp_servers"

MCP_SERVERS = {
    "tss-tools": {
        "command": sys.executable,
        "args": [str(_MCP_DIR / "tss_tools_server.py")],
    },
}
