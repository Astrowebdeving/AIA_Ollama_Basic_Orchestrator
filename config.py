import os

# Ollama Connection
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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
