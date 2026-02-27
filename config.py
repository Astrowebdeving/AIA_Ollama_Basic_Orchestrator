import os
import subprocess
from dotenv import load_dotenv

load_dotenv()


def _detect_local_ip() -> str:
    """
    Resolve the Ollama IP with this priority:
      1. Auto-detect via `ipconfig getifaddr en0`
      2. OLLAMA_IP environment variable / .env
      3. Hardcoded fallback
    """
    # 1. Try live detection
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

    # 2. Fall back to env var
    env_ip = os.getenv("OLLAMA_IP")
    if env_ip:
        return env_ip

    # 3. Last resort
    return "10.207.22.21"


# Ollama Connection
OLLAMA_IP = _detect_local_ip()
OLLAMA_HOST = f"http://{OLLAMA_IP}:11434"

# Model Settings
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:27b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "google/gemma-3-27b-it")

# Token Budgeting (Gemma 3 27b supports 128k context)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 128000))

# Database Settings
DB_FILE = os.getenv("DB_FILE", "logs.db")
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb")

# MCP Servers configurations (Optional)
# e.g., {"file-system": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/expose"]}}
MCP_SERVERS = {}
