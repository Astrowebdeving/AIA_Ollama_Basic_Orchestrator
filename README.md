# LLM Orchestrator

Local LLM orchestrator built with FastAPI, powered by Gemma 4 26B (MoE, 4B active params, Q4\_K\_M) via Ollama. Features native tool calling, built-in reasoning, and MCP tool servers for live NASA SUITS TSS2026 telemetry over UDP, reference document search (text + PDF), direct vision analysis via base64 image injection, and Magic Leap 2 headset voice/runtime context for the EVA assistant.

## Quick Start

```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Pull required Ollama models
ollama pull gemma4:26b           # Chat model (~17GB)
ollama pull qwen3-embedding:0.6b # Embedding model (needed if RAG is re-enabled)

# Configure TSS server connection (set to the IP/port of the running TSS instance)
# Edit .env:
#   TSS_UDP_HOST=<TSS server IP>
#   TSS_UDP_PORT=14141

# Run the orchestrator
uv run python main.py
```

The orchestrator starts on `http://0.0.0.0:13853`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Agentic chat with MCP tool calling and reasoning |
| GET | `/context` | Token usage breakdown from the last `/chat` call |
| GET | `/health` | LLM backend connectivity, model availability, MCP tool count |

### POST /chat

```json
// Request
{
  "messages": [
    {"role": "user", "content": "What is the EVA1 oxygen status?"}
  ],
  "stream": false
}

// Response
{
  "role": "assistant",
  "content": "EVA1: O2 94%, CO2 0.5%, HR 72 bpm, suit pressure nominal."
}
```

The model uses Gemma 4's native thinking to reason through the request, then autonomously calls MCP tools (e.g., fetching live TSS telemetry via UDP, searching reference documents). The thinking trace is logged server-side but never streamed to the client.

## Configuration

All settings are read from environment variables (with defaults). Create a `.env` file in the project root to override:

```env
# LLM provider: "ollama" (default), "afm", "llamacpp"
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:26b
LLM_API_BASE=              # auto-set per provider if empty

# Ollama host -- always localhost. If OLLAMA_HOST is set system-wide to 0.0.0.0
# (Ollama's server bind address), the orchestrator automatically remaps it to localhost.
OLLAMA_HOST=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_PREWARM=true

# Tokenizer -- auto-resolved to local tokenizer_cache/ dir if pre-downloaded.
# Startup expects local tokenizer files for offline field use.
# TOKENIZER_NAME=google/gemma-4-26B-A4B-it

# Context management
MAX_CONTEXT_TOKENS=128000
SUMMARIZE_THRESHOLD=80000

# TSS2026 server (UDP telemetry -- used by the get_tss_state MCP tool)
# Set TSS_UDP_HOST to the IP of the running TSS instance.
TSS_UDP_HOST=10.206.64.189
TSS_UDP_PORT=14141
TSS_UDP_TIMEOUT=2.0
```

`OLLAMA_PREWARM=true` sends an empty Ollama generate request at startup so the
chat model is loaded before the first Unity/headset request. `OLLAMA_KEEP_ALIVE`
is passed through on chat requests to keep the model resident between turns.

### Provider Defaults

| Provider | Default API base | Default model | Default tokenizer | Notes |
|----------|-----------------|---------------|-------------------|-------|
| `ollama` | `http://localhost:11434` | `gemma4:26b` | `google/gemma-4-26B-A4B-it` | Full Ollama SDK, native thinking + tools |
| `afm` | `http://localhost:9999` | `mlx-community/Qwen3.5-35B-A3B-4bit` | `Qwen/Qwen3.5-35B-A3B` | OpenAI-compatible (AFM/MLX) |
| `llamacpp` | `http://localhost:8080` | `gemma4` | `google/gemma-4-26B-A4B-it` | OpenAI-compatible (llama-server) |

## Offline Tokenizer Cache

The orchestrator uses a HuggingFace tokenizer for accurate token counting and context budgeting. Tokenizer files are **pre-downloaded** into `tokenizer_cache/` so the orchestrator starts without internet access — required for field deployment on a local network.

```
tokenizer_cache/
  google--gemma-4-26B-A4B-it/    # Gemma 4 tokenizer (used by ollama + llamacpp providers)
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
```

`config.py` automatically resolves the HF model ID (e.g. `google/gemma-4-26B-A4B-it`) to a local path if matching files exist in `tokenizer_cache/`. The directory name uses `--` in place of `/` (e.g. `google--gemma-4-26B-A4B-it`). `context_manager.py` loads tokenizer files with `local_files_only=True`, so field/offline startup expects the tokenizer to already be present locally.

**To pre-download a tokenizer for a new provider/model:**

```bash
uv run python -c "
from transformers import AutoTokenizer
import os
name = 'google/gemma-4-26B-A4B-it'  # change to the HF model ID
save_dir = os.path.join('tokenizer_cache', name.replace('/', '--'))
os.makedirs(save_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(name)
tok.save_pretrained(save_dir)
print(f'Saved to {save_dir}')
"
```

## Architecture

```
orchestrator/
  main.py                  FastAPI app, /chat agentic loop
  config.py                All configuration, env var loading, MCP server wiring
  llm_provider.py          Provider abstraction: Ollama, AFM/MLX, llama.cpp
  tss_udp_client.py        Async UDP client for TSS2026 telemetry
  mcp_client.py            MCP server lifecycle, tool discovery, tool execution
  context_manager.py       Token counting, budget calculation, truncation
  context_summarizer.py    Auto-summarizes when conversation exceeds 80k tokens
  tokenizer_cache/         Pre-downloaded HF tokenizer files (offline operation)
  docs/                    Mission/headset references searchable by the LLM (text, PDF, images)
  mcp_servers/
    tss_tools_server.py    MCP server: get_tss_state, search_docs, read_doc, inspect_image
  rag_service.py           Conversation history via LanceDB (disabled, preserved for later)
  test_tss_udp.py          Smoke test for UDP connectivity to TSS2026
```

### Active MCP Tools

| Tool | Description | Max Output |
|------|-------------|------------|
| `get_tss_state` | Fetch live TSS2026 telemetry via UDP. Scopes: `all`, `eva`, `rover`, `ltv`, `ltv_errors`, `vitals`. | 4K chars |
| `search_docs` | Grep docs/ for a text pattern. Searches text files and PDFs. Returns file:line:match, no surrounding context. | 50 matches |
| `read_doc` | Read a document or section. Use after search_docs to expand context around a specific line. | 2K default, 8K max |
| `inspect_image` | Load an image from docs/ directly into the LLM's visual context via base64 injection. The main LLM sees the image itself (not a second-hand description). Maps, diagrams, equipment photos. | ~3-5K vision tokens per image |

The LLM decides when to call these tools based on the user's prompt. There is no auto-injection and no background polling.

### Disabled Tools (preserved for later)

| Tool | Description | How to re-enable |
|------|-------------|------------------|
| `search_knowledge` | Semantic search over past conversations via LanceDB embeddings | Uncomment in `tss_tools_server.py` and `main.py` |

### TSS2026 Integration

The `get_tss_state` MCP tool communicates directly with the NASA SUITS TSS2026 server over UDP (port 14141). The TSS server is an external dependency managed by NASA -- the orchestrator only needs its IP address and port to connect.

The TSS protocol uses big-endian binary packets: clients send an 8-byte request (`[uint32 timestamp][uint32 command]`) and receive JSON telemetry in response.

| Command | Scope | Data Returned |
|---------|-------|---------------|
| 0 | `rover` | Pressurized rover telemetry (position, steering, LIDAR, cabin, battery) |
| 1 | `eva` | EVA1/EVA2 suit telemetry, DCU, UIA, IMU, errors |
| 2 | `ltv` | LTV last-known location, signal strength |
| 3 | `ltv_errors` | LTV error procedures |
| -- | `vitals` | Filtered EVA data: heart rate, O2, CO2, temperature, battery only |
| -- | `all` | Commands 0-3 combined |

Set `TSS_UDP_HOST` and `TSS_UDP_PORT` in `.env` to match the running TSS instance. During local development you can run a local copy of TSS2026 for testing; at JSC test week, point to the official NASA-hosted instance.

```bash
# Verify connectivity
uv run python test_tss_udp.py
```

### Document Search

Place reference documents in `docs/` (supports subdirectories). The LLM searches them via a two-step workflow:

1. `search_docs("egress procedure")` -- finds matches with file:line references
2. `read_doc("procedures/ev-team-procedure-timeline.pdf", around_line=42)` -- reads context around that match

Supported formats: Markdown, plain text, PDF (via PyMuPDF). Images can be analyzed via `inspect_image`.

Current mission/headset references include:

| Path | Purpose |
|------|---------|
| `headset_voice_commands.md` | Exact Magic Leap MLVoice/Vosk command aliases, event IDs, scene limits, and `send recording` behavior |
| `headset_unity_runtime.md` | Unity scene gating, recording-button behavior, HUD/reference-map behavior, egress/ingress automation, LTV verification, minimap caveats, and TSS runtime notes |
| `mission_description/` | NASA SUITS mission overview and acronym list |
| `procedures/` | EVA timeline, coordinates, and LTV repair procedures |
| `telemetry_ranges/` | EVA telemetry nominal/off-nominal ranges |
| `maps/` | Annotated/raw rock-yard maps, keep-out zones, and DUST map |
| `peripherals/` | UIA/DCU photos and LTV Task Board overview/close-up photos |

Note: `docs/` is in `.gitignore` — populate it locally with your mission documents. It is not committed to the repository.

### Data Flow

```
Magic Leap 2 headset / client -> POST /chat
  |
1. Build message list: [system_prompt] + [user messages]
   |-- System prompt tells Luna about Vosk/MLVoice artifacts, "send recording",
   |   Magic Leap Unity scenes, and the headset docs to consult
2. If total tokens > 80k -> summarize older conversation history
3. Send to Gemma 4 (with think=True)
4. Model reasons internally (thinking trace logged, not streamed)
5. Model decides: answer directly OR call tools
   |-- get_tss_state(scope=eva) -> UDP to TSS2026 -> live JSON
   |-- search_docs("oxygen") -> grep over docs/ -> file:line matches
   |-- search_docs("show reference map") -> exact headset voice/runtime docs
   |-- read_doc("procedures.pdf", around_line=42) -> context excerpt
   |-- inspect_image("maps/dust-map.png") -> base64 injected into LLM context
6. Tool results fed back -> model generates final answer
   |-- If images were injected, base64 is stripped after the LLM responds
   |   (images exist in context for exactly one inference call)
7. Return response (content only, no thinking trace)
```

### Context Summarization

When the total token count of a conversation exceeds `SUMMARIZE_THRESHOLD` (default 80k), older messages are automatically compressed via the LLM into a single summary message. The 4 most recent messages are always kept intact.

**Image handling during summarization:**
- Messages that carried base64 images are noted in the summary text (e.g., "attached 1 image for vision analysis") so the summary records that visual references were consulted.
- Base64 image data is **never** included in the summary input — only the text labels are summarized.
- After summarization, any images surviving in the recent (unsummarized) messages are stripped, since the LLM has already seen and responded to them.

**Immediate image stripping (pre-summarization):**
- Images are stripped from the conversation history immediately after the LLM processes them in `main.py`. This means base64 data exists in context for exactly **one** inference call, then is replaced with a lightweight text note. A single high-res map (~3,900×2,900 px) costs ~5,000 vision tokens for that one turn, not for the entire conversation.

### Gemma 4 Reasoning

Gemma 4 supports native thinking via Ollama's `think=True` parameter. The SDK automatically separates `message.thinking` (internal reasoning) from `message.content` (final answer). The orchestrator:
- Passes `think=True` on every chat call
- Logs the thinking trace length server-side
- Never streams or returns the thinking trace to the client

## Testing

```bash
# Test TSS2026 UDP connectivity (requires running TSS server)
uv run python test_tss_udp.py

# Quick model test
uv run python -c "
import asyncio
from llm_provider import OllamaProvider
async def test():
    p = OllamaProvider(host='http://localhost:11434')
    r = await p.chat(model='gemma4:26b', messages=[{'role':'user','content':'Hello'}])
    print(f'Content: {r.content}')
    print(f'Thinking: {len(r.thinking)} chars' if r.thinking else 'No thinking')
asyncio.run(test())
"
```

## Security

- MCP servers can expose file system access or execute commands. Do not expose port 13853 to untrusted networks.
- The `.env` file is in `.gitignore`.
