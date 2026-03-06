# Gemma Orchestra — Architecture & How It Works

## Overview

This is a **FastAPI Python orchestrator** that sits between clients and a local **Gemma 3 27b** instance running on Ollama. It adds three capabilities the raw Ollama API doesn't have: **deterministic token budgeting**, **RAG context injection**, and **MCP tool calling** — all managed through a single `/chat` endpoint.

## Network Topology

```
┌──────────────────────────────────────────────────────────────────┐
│  Host Machine                                                    │
│                                                                  │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐   │
│  │  Orchestrator (FastAPI) │   │  Ollama                     │   │
│  │  0.0.0.0:8000           │──▶│  0.0.0.0:11434              │   │
│  │                         │   │  gemma3:27b (Q4_K_M, 17GB)  │   │
│  └────────▲────────────────┘   └──────────▲──────────────────┘   │
│           │                               │                      │
└───────────┼───────────────────────────────┼──────────────────────┘
            │                               │
    ┌───────┴───────┐               ┌───────┴───────┐
    │ Any client on │               │ Direct Ollama  │
    │ the network   │               │ API access     │
    │ :8000         │               │ :11434         │
    └───────────────┘               └───────────────┘
```

Both services bind to `0.0.0.0`, meaning **any machine on the local network** can reach them. The orchestrator calls Ollama via the configured `OLLAMA_HOST`, which defaults from `OLLAMA_IP` or local IP auto-detection.

---

## File Structure

| File | Purpose |
|---|---|
| `config.py` | All constants: model names, 128k token limit, DB paths, Ollama host, MCP server definitions |
| `context_manager.py` | **The token budgeting engine** (see below) |
| `db_logger.py` | Async SQLite logger — records every query with its token footprint |
| `rag_service.py` | LanceDB vector store + Ollama Nomic embeddings for retrieval |
| `mcp_client.py` | MCP STDIO client: discovers tools, translates schemas, executes calls |
| `main.py` | FastAPI app: `/chat`, `/context`, `/health` endpoints |

---

## The Context Manager (Detailed)

The context manager is the **core safety mechanism** that prevents every request from exceeding Gemma's 128k context window. It uses the *exact same tokenizer* as the model (`google/gemma-3-27b-it` via HuggingFace `transformers`), so token counts are deterministic — not estimated.

### How the Budget Works

```
┌─────────────────────────────────────────────────────────────┐
│                  128,000 token window                       │
│                                                             │
│  ┌──────────┐ ┌──────┐ ┌──────────┐ ┌────────────────────┐  │
│  │  System   │ │ User │ │  Tool    │ │   DYNAMIC BUDGET   │  │
│  │  Prompt   │ │Query │ │ Schemas  │ │   (what's left)    │  │
│  │  + RAG    │ │      │ │  (JSON)  │ │                    │  │
│  └──────────┘ └──────┘ └──────────┘ └────────────────────┘  │
│  ◄── baseline_tokens ──────────────▶ ◄── dynamic_budget ──▶ │
└─────────────────────────────────────────────────────────────┘
```

1. **`count_message_tokens(messages, tool_schemas)`**
   Tokenizes the full serialized message list plus MCP tool JSON schemas. This is the non-negotiable floor for the current request.

2. **`get_dynamic_budget(baseline)`**
   Returns `128,000 - baseline`. This is how many tokens are left for RAG context, tool results, and the model's own generation.

3. **RAG fills first** — retrieved document chunks are appended *only if they fit* within the dynamic budget. Each chunk is tokenized before inclusion; once the budget is exhausted, no more chunks are added.

4. **Tool results are truncated** — when an MCP tool returns a result (e.g., a large file), `truncate_to_budget(text, remaining)` physically slices the token sequence and appends `[TRUNCATED BY ORCHESTRATOR]`. This guarantees the context window is *never* overflowed.

### Key Methods

| Method | What it does |
|---|---|
| `load_tokenizer()` | Lazy-loads the Gemma SentencePiece tokenizer (once) |
| `count_message_tokens()` | Counts tokens for the serialized message history + tool schemas |
| `get_dynamic_budget()` | `MAX_CONTEXT_TOKENS - baseline` |
| `count_tokens(text)` | Exact token count for any string |
| `truncate_to_budget(text, limit)` | Slice tokens to fit, append `[TRUNCATED BY ORCHESTRATOR]` |

---

## Request Flow

1. **Client** sends `POST /chat` with a `messages` array
2. **Token budget** is computed (baseline from system prompt + query + tool schemas)
3. **RAG** retrieves relevant document chunks from LanceDB, stopping when the budget is full
4. The **active chat backend** is called with the assembled context + any available MCP tools. On the default Ollama path, `num_ctx=128000` is passed explicitly so Ollama allocates the full window.
5. If the active chat backend returns **tool_calls**:
   - Each call is routed to the appropriate MCP server
   - The result is tokenized and truncated if over-budget
   - The result is appended to history and the chat backend is called again
   - This loops up to 10 rounds (safety valve)
6. The **final text response** is streamed back to the client

---

## API Endpoints

### `POST /chat`
Main agentic endpoint. Accepts:
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "stream": true
}
```

### `GET /context`
Returns the token breakdown from the most recent `/chat` call:
```json
{
  "max_context_tokens": 128000,
  "baseline_tokens": 342,
  "rag_tokens": 0,
  "tool_result_tokens": 1580,
  "total_message_tokens": 2100,
  "remaining_budget": 125900,
  "utilisation_pct": 1.64
}
```

### `GET /health`
Live-probes the active chat backend and reports model availability.

---

## Network Access

The orchestrator binds to **`0.0.0.0:8000`** — it is callable from any machine on the network at `http://<host-ip>:8000`.

Ollama binds to **`0.0.0.0:11434`** — it remains directly callable at `http://<host-ip>:11434` for any client that wants to bypass the orchestrator.

Both are independent. The orchestrator reaches Ollama via the configured `OLLAMA_HOST` internally.
