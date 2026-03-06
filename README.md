# Gemma 3 Orchestrator

Local LLM orchestrator built with FastAPI. Connects Ollama-hosted models (default: `gemma3:27b`) with MCP tool servers, vector-backed RAG, telemetry ingestion, and automatic context management.

## Quick Start

```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Pull required Ollama models (always needed for embeddings)
ollama pull qwen3-embedding:0.6b

# Pull a chat model (if using Ollama as the chat backend)
ollama pull gemma3:27b

# Run (defaults to Ollama backend)
uv run python main.py
```

The server starts on `http://0.0.0.0:8000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Agentic chat with MCP tool calling and RAG context |
| POST | `/telemetry` | Ingest a single telemetry event |
| POST | `/telemetry/batch` | Ingest multiple telemetry events |
| GET | `/context` | Token usage breakdown from the last `/chat` call |
| GET | `/health` | Ollama connectivity, model availability, MCP tool count |

### POST /chat

```json
{
  "messages": [
    {"role": "user", "content": "What telemetry events happened in the last hour?"}
  ],
  "stream": true
}
```

The model can autonomously call MCP tools during its response. Tool results are fed back into the conversation and the model generates a final answer.

### POST /telemetry

```json
{
  "source": "suit_sensors",
  "event_type": "pressure_reading",
  "severity": "warning",
  "payload": {"psi": 12.3, "module": "EVA-01"},
  "description": "Suit pressure dropped below nominal range"
}
```

Events are stored in SQLite (`logs.db`). The LLM can query them via MCP tools.

## Configuration

All settings are read from environment variables (with defaults). Create a `.env` file in the project root to override:

```env
# LLM provider: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:27b
LLM_API_BASE=              # auto-set per provider if empty

# Ollama (always needed for embeddings, also for chat when LLM_PROVIDER=ollama)
OLLAMA_IP=10.207.22.21
EMBED_MODEL=qwen3-embedding:0.6b
EMBED_DIM=1024

# Context management
MAX_CONTEXT_TOKENS=128000
SUMMARIZE_THRESHOLD=64000

# Telemetry poller (optional)
TELEMETRY_SOURCE_URL=http://your-telemetry-api/data
TELEMETRY_POLL_INTERVAL=20
```

### Provider Defaults

| Provider | Default API base | Default model | Notes |
|----------|-----------------|---------------|-------|
| `ollama` | `http://{auto-detected-ip}:11434` | `gemma3:27b` | Full Ollama SDK |
| `afm` | `http://localhost:9999` | `mlx-community/Qwen3.5-35B-A3B-4bit` | OpenAI-compatible (AFM/MLX) |
| `llamacpp` | `http://localhost:8080` | `gemma3` | OpenAI-compatible (llama-server) |

The Ollama host resolution order is: explicit `OLLAMA_HOST`, then `OLLAMA_IP`, then macOS auto-detection via `ipconfig getifaddr en0`, then the hardcoded default. Ollama is always required for embeddings regardless of the chat provider.

### Switching Providers

To use AFM/MLX (start the AFM server separately):
```bash
afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit -w --vlm \
  --tool-call-parser qwen3_xml --max-kv-size 32768 --kv-bits 4 \
  --enable-prefix-caching --prefill-step-size 2048 --max-tokens 16384 \
  --port 9999

LLM_PROVIDER=afm uv run python main.py
```
AFM context size is controlled by the AFM server's startup flags. The orchestrator still enforces its own local context budget, but it does not send a synthetic `max_tokens=128000` request to OpenAI-compatible backends.

To use llama.cpp (start llama-server separately with `--jinja` for tool calling):
```bash
llama-server -m model.gguf --jinja --port 8080

LLM_PROVIDER=llamacpp uv run python main.py
```
For llama.cpp as well, configure the backend context window on the server itself. The orchestrator handles local token budgeting and preserves OpenAI-style tool-call history when using compatible providers.

## Architecture

```
orchestrator/
  main.py                  FastAPI app, /chat agentic loop, /telemetry endpoints
  config.py                All configuration, env var loading, MCP server wiring
  llm_provider.py          Provider abstraction: Ollama, AFM/MLX, llama.cpp
  mcp_client.py            MCP server lifecycle, tool discovery, tool execution
  rag_service.py           Document chunking, embedding (qwen3), LanceDB storage/retrieval
  context_manager.py       Token counting, budget calculation, truncation
  context_summarizer.py    Auto-summarizes older messages when tokens exceed threshold
  db_logger.py             SQLite tables: query_logs, telemetry_events
  api_client.py            Skeleton for external JSON API access (not yet integrated)
  mcp_servers/
    sqlite_query_server.py       MCP server: read-only SQL against logs.db
    telemetry_search_server.py   MCP server: telemetry queries, on-demand fetch, RAG indexing
```

### MCP Tools Available to the LLM

| Tool | Server | Description |
|------|--------|-------------|
| `query_logs_db` | sqlite-query | Run read-only SQL against logs.db |
| `describe_logs_db` | sqlite-query | List all table schemas |
| `query_telemetry` | telemetry-search | Filter telemetry by source, type, severity, time |
| `search_telemetry` | telemetry-search | Semantic search over the RAG knowledge base |
| `telemetry_summary` | telemetry-search | Aggregate stats (counts, time range, recent events) |
| `fetch_telemetry_now` | telemetry-search | On-demand fetch from telemetry source (bypasses 20s poller) |
| `add_to_knowledge_base` | telemetry-search | LLM-driven: embed text into RAG for future retrieval |

### Data Flow

1. **Telemetry** arrives via `POST /telemetry` or the background poller (every 20s) and is stored in SQLite.
2. **Chat** requests hit `/chat`, which retrieves RAG context from LanceDB, builds the message history, optionally summarizes old messages (if > 64k tokens), and enters the agentic tool loop.
3. **Tool calls** are routed through `mcp_client.py` to the appropriate MCP server subprocess.
4. **Knowledge indexing** happens when the LLM explicitly calls `add_to_knowledge_base` -- nothing is auto-embedded.

### Embedding Model

Default: `qwen3-embedding:0.6b` (1024-dimensional vectors). This is the unquantized variant -- smaller models degrade disproportionately from quantization. The model is configurable via `EMBED_MODEL` and `EMBED_DIM` for when a finetuned version is available.

### Context Summarization

When the total token count of a conversation exceeds `SUMMARIZE_THRESHOLD` (default 64k), older messages are automatically compressed via the LLM into a single summary message. The 4 most recent messages are always kept intact.

## Security

- MCP servers can expose file system access or execute commands. Do not expose port 8000 to untrusted networks.
- The SQLite query tool restricts to `SELECT` / `PRAGMA` / `WITH` / `EXPLAIN` and opens the database in read-only mode.
- The `.env` file is in `.gitignore`.
