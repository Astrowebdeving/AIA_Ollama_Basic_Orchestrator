# Ollama Basic Orchestrator

A lightweight, local LLM orchestrator built with FastAPI, designed to integrate Ollama models (such as `gemma3:27b`) with the Model Context Protocol (MCP) and dynamic RAG (Retrieval-Augmented Generation).

## Features

- **Local Execution:** Fully local inference via Ollama, ensuring privacy and control over your data.
- **MCP Tool Integration:** Automatically discovers and connects to MCP servers via stdio, translating MCP schemas into Ollama-compatible function calling formats.
- **Dynamic Context Budgeting:** Intelligently manages context windows (e.g., 128k tokens for Gemma 3) to prevent overflow while accommodating tool results and RAG contexts.
- **Built-in RAG:** Seamless vector embedding and retrieval using LanceDB and Ollama's embedding models (e.g., `nomic-embed-text`).
- **Query Logging:** Asynchronous logging of user queries and context usage to an SQLite database (`logs.db`).

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally or accessible via network.
- Node.js (if utilizing npx for MCP servers like `@modelcontextprotocol/server-filesystem`).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Astrowebdeving/AIA_Ollama_Basic_Orchestrator.git
   cd AIA_Ollama_Basic_Orchestrator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install requirements (example, ensure you have the necessary packages):
   ```bash
   pip install fastapi uvicorn pydantic transformers aiosqlite lancedb mcp ollama
   ```

## Configuration

Configuration is managed via environment variables (or directly in `config.py`). Key settings include:

- `OLLAMA_HOST`: URL to your Ollama instance (default: `http://localhost:11434`)
- `LLM_MODEL`: The LLM to use (default: `gemma3:27b`)
- `EMBED_MODEL`: The embedding model to use (default: `nomic-embed-text`)
- `TOKENIZER_NAME`: Tokenizer for context budgeting (default: `google/gemma-3-27b-it`)
- `MAX_CONTEXT_TOKENS`: Context window limit (default: `128000`)

## Usage

Start the orchestrator:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note: Binding to `0.0.0.0` allows network access. Use `127.0.0.1` if you only want local machine access.*

### Endpoints

- **`POST /chat`**: The main interaction endpoint. Provide an array of messages and receive a streamed, tool-augmented response.
  ```json
  {
    "messages": [
      {"role": "user", "content": "What is the current status of the system?"}
    ],
    "stream": true
  }
  ```
- **`GET /context`**: Returns token breakdown and context window utilization statistics for the most recent `/chat` request.
- **`GET /health`**: System health check, reporting on Ollama connectivity, model availability, and connected MCP tools.

## Architecture

- `main.py`: FastAPI application routing and the core agentic tool loop.
- `mcp_client.py`: Manages subprocess connections to MCP servers, tool discovery, and execution.
- `rag_service.py`: Handles vectorization and document retrieval using LanceDB.
- `context_manager.py`: Manages token counting and dynamic context budget tracking to ensure the LLM does not exceed its context window.
- `db_logger.py`: Provides asynchronous logging of queries.

## Security Warning

If you configure `MCP_SERVERS` in `config.py` to expose file systems or execute commands (e.g., `@modelcontextprotocol/server-filesystem`), be extremely cautious about exposing the `0.0.0.0:8000` port to untrusted networks, as this can grant arbitrary read/write access to the host machine.
