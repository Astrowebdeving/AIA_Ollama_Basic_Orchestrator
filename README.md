# Gemma 3 Orchestrator

Local LLM orchestrator built with FastAPI. Connects Ollama-hosted models (default: `gemma3:27b-it-qat`) with MCP tool servers and vector-backed RAG for conversation history. The LLM decides when to call tools — no auto-injection, no background polling.

## Quick Start

```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Pull required Ollama models (always needed for embeddings)
ollama pull qwen3-embedding:0.6b

# Pull a chat model (if using Ollama as the chat backend)
ollama pull gemma3:27b-it-qat

# Run (defaults to Ollama backend)
uv run python main.py
```

The server starts on `http://0.0.0.0:8000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Agentic chat with MCP tool calling and RAG context |
| GET | `/context` | Token usage breakdown from the last `/chat` call |
| GET | `/health` | LLM backend connectivity, model availability, MCP tool count |

### POST /chat

```json
{
  "messages": [
    {"role": "user", "content": "What is the EVA1 oxygen status?"}
  ],
  "stream": true
}
```

The model can autonomously call MCP tools during its response. Tool results are fed back into the conversation and the model generates a final answer. After each response, the full exchange (user message, tool calls, assistant reply) is stored into the RAG knowledge base for future retrieval.

## Configuration

All settings are read from environment variables (with defaults). Create a `.env` file in the project root to override:

```env
# LLM provider: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:27b-it-qat
LLM_API_BASE=              # auto-set per provider if empty

# Ollama (always needed for embeddings, also for chat when LLM_PROVIDER=ollama)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_IP=10.207.22.21
EMBED_MODEL=qwen3-embedding:0.6b
EMBED_DIM=1024

# Context management
MAX_CONTEXT_TOKENS=128000
SUMMARIZE_THRESHOLD=80000

# TSS Unity API (used by the get_tss_state MCP tool)
TSS_API_BASE_URL=http://127.0.0.1:8100/api/v1
TSS_API_TIMEOUT=10
```

### Provider Defaults

| Provider | Default API base | Default model | Notes |
|----------|-----------------|---------------|-------|
| `ollama` | `http://{auto-detected-ip}:11434` | `gemma3:27b-it-qat` | Full Ollama SDK |
| `afm` | `http://localhost:9999` | `mlx-community/Qwen3.5-35B-A3B-4bit` | OpenAI-compatible (AFM/MLX) |
| `llamacpp` | `http://localhost:8080` | `gemma3` | OpenAI-compatible (llama-server) |

The Ollama host resolution order is: preferred local `OLLAMA_HOST` (defaults to `http://127.0.0.1:11434`), then fallback `OLLAMA_IP` / `OLLAMA_FALLBACK_HOST`, then the first configured value if neither is reachable. Ollama is always required for embeddings regardless of the chat provider.

### Switching Providers

To use AFM/MLX (start the AFM server separately):
```bash
afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit -w --vlm \
  --tool-call-parser qwen3_xml --max-kv-size 32768 --kv-bits 4 \
  --enable-prefix-caching --prefill-step-size 2048 --max-tokens 16384 \
  --port 9999

LLM_PROVIDER=afm uv run python main.py
```

To use llama.cpp (start llama-server separately with `--jinja` for tool calling):
```bash
llama-server -m model.gguf --jinja --port 8080

LLM_PROVIDER=llamacpp uv run python main.py
```

## Architecture

```
orchestrator/
  main.py                  FastAPI app, /chat agentic loop
  config.py                All configuration, env var loading, MCP server wiring
  llm_provider.py          Provider abstraction: Ollama, AFM/MLX, llama.cpp
  mcp_client.py            MCP server lifecycle, tool discovery, tool execution
  rag_service.py           Conversation history: chunk, embed, retrieve (LanceDB)
  context_manager.py       Token counting, budget calculation, truncation
  context_summarizer.py    Auto-summarizes when conversation exceeds 80k tokens
  api_client.py            Reusable async JSON API client (httpx)
  mcp_servers/
    tss_tools_server.py    MCP server: get_tss_state, search_knowledge
  tests/
    test_api_client.py     Unit tests for ApiClient (mock transport)
    test_revamp.py         Tests for TSS tool fetch + RAG conversation storage
```

### MCP Tools Available to the LLM

| Tool | Description |
|------|-------------|
| `get_tss_state` | Fetch live TSS data on-demand. Accepts `scope` parameter: `all`, `eva`, `ltv`, `health`, `vitals`. |
| `search_knowledge` | Semantic search over past conversations and tool results stored in the RAG knowledge base. |

The LLM decides when to call these tools based on the user's prompt. There is no auto-injection of TSS state and no background polling.

### Data Flow

```
User → POST /chat
  ↓
1. Retrieve relevant past conversations from RAG (LanceDB)
2. Build message list: [system_prompt] + [RAG context] + [user messages]
3. If total tokens > 80k → summarize older conversation history
4. Send to LLM
5. LLM decides: answer directly OR call tools
   ├── get_tss_state(scope=eva) → hits TSS Unity API → returns live data
   └── search_knowledge("EVA oxygen") → queries LanceDB → returns past context
6. Tool results fed back → LLM generates final answer
7. Store full exchange (user + tools + assistant) into RAG
8. Return response
```

### Dual Storage Strategy

| What | Where | Used By |
|------|-------|---------|
| **Full exchange** (user + tool calls + results + assistant) | RAG (LanceDB) | `search_knowledge` tool — rich retrieval |
| **Compact context** (user + assistant only) | Injected into prompt | Auto-retrieved from RAG on each `/chat` call |

### TSS Unity API Integration

The `get_tss_state` MCP tool fetches live telemetry from the TSS Unity API. To use it, start the TSS server stack:

```bash
# 1. Start the core TSS server
cd ../TSS_Lunar_Lions/TSS2026-LunarLions
./server.exe

# 2. Start the Unity API wrapper
cd backend
python tss_unity_api.py --tss-host 127.0.0.1 --tss-port 14141 --api-host 0.0.0.0 --api-port 8100

# 3. Run the orchestrator
cd ../../../orchestrator
TSS_API_BASE_URL=http://127.0.0.1:8100/api/v1 uv run python main.py
```

Notes:
- The backend wrapper defaults `--tss-port` to `8080`, but the TSS server listens on `14141`. Pass `--tss-port 14141` explicitly.
- The `/vitals` endpoint is called when `scope=all` or `scope=vitals` — it gracefully handles 404 on older TSS servers that don't support it.

### Embedding Model

Default: `qwen3-embedding:0.6b` (1024-dimensional vectors). This is the unquantized variant — smaller models degrade disproportionately from quantization. Configurable via `EMBED_MODEL` and `EMBED_DIM`.

### Context Summarization

When the total token count of a conversation exceeds `SUMMARIZE_THRESHOLD` (default 80k), older messages are automatically compressed via the LLM into a single summary message. The 4 most recent messages are always kept intact.

## Testing

```bash
uv run python -m pytest tests/ -v
```

All tests use `httpx.MockTransport` and mocked RAG methods — no live server required.

## Security

- MCP servers can expose file system access or execute commands. Do not expose port 8000 to untrusted networks.
- The `.env` file is in `.gitignore`.
