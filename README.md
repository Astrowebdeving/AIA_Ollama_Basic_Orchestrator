# LLM Orchestrator

Local LLM orchestrator built with FastAPI, powered by **Gemma 4 26B** (MoE, 4B active params) via Ollama. Features native tool calling, built-in reasoning (thinking), MCP tool servers for live NASA SUITS TSS2026 telemetry over UDP, and vector-backed RAG for conversation history.

## Quick Start

```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Pull required Ollama models
ollama pull gemma4:26b           # Chat model (Q4_K_M, ~17GB)
ollama pull qwen3-embedding:0.6b # Embedding model (always needed for RAG)

# Configure TSS server connection (set to the IP/port of the running TSS instance)
# Edit .env:
#   TSS_UDP_HOST=<TSS server IP>
#   TSS_UDP_PORT=14141

# Run the orchestrator (defaults to Ollama + Gemma 4)
uv run python main.py
```

The orchestrator starts on `http://0.0.0.0:8000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Agentic chat with MCP tool calling, reasoning, and RAG context |
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

The model uses Gemma 4's native thinking to reason through the request, then autonomously calls MCP tools (e.g., fetching live TSS telemetry via UDP). The thinking trace is logged server-side but never streamed to the client. Tool results are fed back into the conversation and the model generates a final answer.

## Configuration

All settings are read from environment variables (with defaults). Create a `.env` file in the project root to override:

```env
# LLM provider: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:26b
LLM_API_BASE=              # auto-set per provider if empty

# Ollama (always needed for embeddings, also for chat when LLM_PROVIDER=ollama)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_IP=10.207.22.21
EMBED_MODEL=qwen3-embedding:0.6b
EMBED_DIM=1024

# Tokenizer — auto-selected per provider if not set.
# Must match the chat model for accurate token counting.
# TOKENIZER_NAME=google/gemma-4-26B-A4B-it    # for Gemma 4

# Context management
MAX_CONTEXT_TOKENS=128000
SUMMARIZE_THRESHOLD=80000

# TSS2026 server (UDP telemetry — used by the get_tss_state MCP tool)
# Set TSS_UDP_HOST to the IP printed by the TSS server at launch.
TSS_UDP_HOST=10.206.64.189
TSS_UDP_PORT=14141
TSS_UDP_TIMEOUT=2.0
```

### Provider Defaults

| Provider | Default API base | Default model | Default tokenizer | Notes |
|----------|-----------------|---------------|-------------------|-------|
| `ollama` | `http://{auto-detected-ip}:11434` | `gemma4:26b` | `google/gemma-4-26B-A4B-it` | Full Ollama SDK, native thinking + tools |
| `afm` | `http://localhost:9999` | `mlx-community/Qwen3.5-35B-A3B-4bit` | `Qwen/Qwen3-35B-A3B` | OpenAI-compatible (AFM/MLX) |
| `llamacpp` | `http://localhost:8080` | `gemma4` | `google/gemma-4-26B-A4B-it` | OpenAI-compatible (llama-server) |

The tokenizer is auto-selected based on `LLM_PROVIDER` when `TOKENIZER_NAME` is not set. Override it if you're running a non-default model on any provider.

The Ollama host resolution order is: preferred local `OLLAMA_HOST` (defaults to `http://127.0.0.1:11434`), then fallback `OLLAMA_IP` / `OLLAMA_FALLBACK_HOST`, then the first configured value if neither is reachable. Ollama is always required for embeddings regardless of the chat provider.

## Architecture

```
orchestrator/
  main.py                  FastAPI app, /chat agentic loop
  config.py                All configuration, env var loading, MCP server wiring
  llm_provider.py          Provider abstraction: Ollama, AFM/MLX, llama.cpp
  tss_udp_client.py        Async UDP client for TSS2026 telemetry
  mcp_client.py            MCP server lifecycle, tool discovery, tool execution
  rag_service.py           Conversation history: chunk, embed, retrieve (LanceDB)
  context_manager.py       Token counting, budget calculation, truncation
  context_summarizer.py    Auto-summarizes when conversation exceeds 80k tokens
  mcp_servers/
    tss_tools_server.py    MCP server: get_tss_state (UDP), search_knowledge
  test_tss_udp.py          Smoke test for UDP connectivity to TSS2026
```

### MCP Tools Available to the LLM

| Tool | Description |
|------|-------------|
| `get_tss_state` | Fetch live TSS2026 telemetry on-demand via UDP. Accepts `scope`: `all`, `eva`, `rover`, `ltv`, `ltv_errors`, `vitals`. |
| `search_knowledge` | Semantic search over past conversations and tool results stored in the RAG knowledge base. |

The LLM decides when to call these tools based on the user's prompt. There is no auto-injection of TSS state and no background polling — data is fetched on-demand for maximum freshness.

### TSS2026 Integration

The `get_tss_state` MCP tool communicates directly with the NASA SUITS TSS2026 server over **UDP** (port 14141). The TSS server is an external dependency managed by NASA — the orchestrator only needs its IP address and port to connect.

The TSS protocol uses big-endian binary packets: clients send an 8-byte request (`[uint32 timestamp][uint32 command]`) and receive JSON telemetry in response.

**UDP Command Map:**

| Command | Scope | Data Returned |
|---------|-------|---------------|
| 0 | `rover` | Pressurized rover telemetry (position, steering, LIDAR, cabin, battery) |
| 1 | `eva` | EVA1/EVA2 suit telemetry, DCU, UIA, IMU, errors |
| 2 | `ltv` | LTV last-known location, signal strength |
| 3 | `ltv_errors` | LTV error procedures |
| — | `vitals` | Filtered EVA data: heart rate, O₂, CO₂, temperature, battery only |
| — | `all` | Commands 0–3 combined |

**Connecting to TSS:**

Set `TSS_UDP_HOST` and `TSS_UDP_PORT` in your `.env` to match the running TSS instance. During local development you can run a local copy of TSS2026 for testing; at JSC test week, point to the official NASA-hosted instance.

```bash
# Verify connectivity to the TSS server
uv run python test_tss_udp.py
```

### Data Flow

```
User → POST /chat
  ↓
1. Retrieve relevant past conversations from RAG (LanceDB)
2. Build message list: [system_prompt] + [RAG context] + [user messages]
3. If total tokens > 80k → summarize older conversation history
4. Send to Gemma 4 (with think=True)
5. Model reasons internally (thinking trace logged, not streamed)
6. Model decides: answer directly OR call tools
   ├── get_tss_state(scope=eva) → UDP to TSS2026 → returns live JSON
   └── search_knowledge("EVA oxygen") → queries LanceDB → returns past context
7. Tool results fed back → model generates final answer
8. Store full exchange (user + tools + assistant) into RAG
9. Return response (content only, no thinking trace)
```

### Dual Storage Strategy

| What | Where | Used By |
|------|-------|---------:|
| **Full exchange** (user + tool calls + results + assistant) | RAG (LanceDB) | `search_knowledge` tool — rich retrieval |
| **Compact context** (user + assistant only) | Injected into prompt | Auto-retrieved from RAG on each `/chat` call |

### Embedding Model

Default: `qwen3-embedding:0.6b` (1024-dimensional vectors). This is the unquantized variant — smaller models degrade disproportionately from quantization. Configurable via `EMBED_MODEL` and `EMBED_DIM`.

### Context Summarization

When the total token count of a conversation exceeds `SUMMARIZE_THRESHOLD` (default 80k), older messages are automatically compressed via the LLM into a single summary message. The 4 most recent messages are always kept intact.

### Gemma 4 Reasoning

Gemma 4 supports native thinking via Ollama's `think=True` parameter. The SDK automatically separates `message.thinking` (internal reasoning) from `message.content` (final answer). The orchestrator:
- Passes `think=True` on every chat call
- Logs the thinking trace length server-side (`[THINKING] N chars`)
- Never streams or returns the thinking trace to the client
- Stores only the final answer content in RAG

## Testing

```bash
# Test TSS2026 UDP connectivity (requires running TSS server)
uv run python test_tss_udp.py

# Test Gemma 4 model + tool calling
uv run python -c "
import asyncio
from llm_provider import OllamaProvider
async def test():
    p = OllamaProvider(host='http://127.0.0.1:11434')
    r = await p.chat(model='gemma4:26b', messages=[{'role':'user','content':'Hello'}])
    print(f'Content: {r.content}')
    print(f'Thinking: {len(r.thinking)} chars' if r.thinking else 'No thinking')
asyncio.run(test())
"
```

## Security

- MCP servers can expose file system access or execute commands. Do not expose port 8000 to untrusted networks.
- The `.env` file is in `.gitignore`.
