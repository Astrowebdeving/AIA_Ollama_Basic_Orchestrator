"""
Gemma 3 27b Orchestrator — Main FastAPI Application
====================================================
Exposes endpoints:
  /chat     — Agentic chat loop with MCP tools and RAG.
  /context  — Returns current context window usage stats.
  /health   — Basic health check including Ollama connectivity.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import ollama as ollama_sdk
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import LLM_MODEL, MAX_CONTEXT_TOKENS, OLLAMA_HOST
from context_manager import context_manager
from db_logger import init_db, log_query
from mcp_client import mcp_client
from rag_service import rag_service

# ---------------------------------------------------------------
# Globals
# ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful, rigorous assistant powered by Gemma 3 27b. "
    "Respond concisely, do not be verbose. "
    "You have access to tools provided by external servers via the "
    "Model Context Protocol (MCP). Use them when needed to answer "
    "the user's request accurately. If you call a tool, wait for "
    "its result before generating your final answer."
)

MAX_TOOL_ROUNDS = 10  # Safety valve against infinite tool loops

# Explicit Ollama client pointing at the configured host
_ollama = ollama_sdk.Client(host=OLLAMA_HOST)

# Per-request context tracking (updated during /chat, queryable via /context)
_last_context_stats: dict = {}


def _coerce_tool_arguments(raw_arguments: Any) -> dict:
    """Normalise tool arguments from Ollama into a JSON object."""
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments are not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments JSON must decode to an object.")
        return parsed

    raise ValueError(
        f"Unsupported tool argument type: {type(raw_arguments).__name__}"
    )


async def _safe_log_query(query: str, baseline_tokens: int) -> None:
    """Run query logging in the background without surfacing task warnings."""
    try:
        await log_query(query, baseline_tokens)
    except Exception as exc:
        print(f"[DB] Failed to log query: {exc}")


# ---------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------

class MessagePayload(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[MessagePayload]
    stream: bool = True


# ---------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    print("[STARTUP] Initialising database …")
    await init_db()

    print("[STARTUP] Pre-loading tokenizer …")
    context_manager.load_tokenizer()

    print("[STARTUP] Verifying Ollama connectivity …")
    try:
        models = await asyncio.to_thread(_ollama.list)
        available = [m.model for m in models.models]
        if any(LLM_MODEL in m for m in available):
            print(f"[STARTUP] ✓ Model '{LLM_MODEL}' available on Ollama")
        else:
            print(
                f"[STARTUP] ⚠ Model '{LLM_MODEL}' NOT found. "
                f"Available: {available}"
            )
    except Exception as e:
        print(f"[STARTUP] ✗ Cannot reach Ollama at {OLLAMA_HOST}: {e}")

    print("[STARTUP] Connecting to MCP servers …")
    await mcp_client.connect_all()

    print("[STARTUP] Ready.")
    yield

    # ---- Shutdown ----
    print("[SHUTDOWN] Closing MCP connections …")
    await mcp_client.shutdown()


app = FastAPI(
    title="Gemma Orchestrator",
    description="Local LLM orchestrator with MCP tools & RAG",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main orchestration endpoint.
    Accepts a list of messages and returns a (streamed) response.
    """
    global _last_context_stats

    # --- Extract the latest user query ---
    user_query = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    if not user_query:
        return {"error": "No user message found in the request."}

    # --- Gather tool schemas ---
    ollama_tools = mcp_client.get_ollama_tools()

    # --- Calculate baseline token budget ---
    # We must tokenize the ENTIRE conversation history, not just the latest query,
    # otherwise long multi-turn chats will overflow the 128k context without us knowing.
    history_text = "\n".join(
        [f"{m.role}: {m.content}" for m in request.messages]
    )

    baseline = context_manager.calculate_baseline_tokens(
        system_prompt=SYSTEM_PROMPT,
        query=history_text,
        tool_schemas=ollama_tools,
    )
    dynamic_budget = context_manager.get_dynamic_budget(baseline)
    if baseline >= MAX_CONTEXT_TOKENS:
        _last_context_stats = {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "baseline_tokens": baseline,
            "rag_tokens": 0,
            "tool_result_tokens": 0,
            "total_message_tokens": baseline,
            "remaining_budget": 0,
            "utilisation_pct": 100.0,
        }
        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation already exceeds the model context limit. "
                "Please shorten message history."
            ),
        )

    # --- Log the query (fire-and-forget) ---
    asyncio.create_task(_safe_log_query(user_query, baseline))

    # --- RAG: retrieve context within budget ---
    rag_context, dynamic_budget = await rag_service.retrieve_context(
        query=user_query,
        budget_limit=dynamic_budget,
    )
    rag_tokens = 0
    if rag_context:
        rag_tokens = context_manager.count_tokens(rag_context)

    # --- Build the message history for Ollama ---
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # --- Track tokens consumed by tool results ---
    tool_result_tokens = 0

    # --- Agentic tool loop ---
    assistant_msg = None
    for _round in range(MAX_TOOL_ROUNDS):
        response = await asyncio.to_thread(
            _ollama.chat,
            model=LLM_MODEL,
            messages=messages,
            tools=ollama_tools if ollama_tools else None,
            stream=False,
            options={"num_ctx": MAX_CONTEXT_TOKENS},
        )

        assistant_msg = response.message

        # No tool calls → we have a final answer
        if not getattr(assistant_msg, "tool_calls", None):
            break

        parsed_tool_calls = []
        for tc in assistant_msg.tool_calls:
            tool_name = tc.function.name
            argument_error = None
            try:
                tool_args = _coerce_tool_arguments(tc.function.arguments)
            except ValueError as exc:
                argument_error = str(exc)
                tool_args = {}

            parsed_tool_calls.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                    "argument_error": argument_error,
                }
            )

        # Append the assistant message (with its tool_calls) to history
        messages.append(
            {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        }
                    }
                    for call in parsed_tool_calls
                ],
            }
        )

        for call in parsed_tool_calls:
            tool_name = call["name"]
            tool_args = call["arguments"]

            print(f"[TOOL CALL] {tool_name}({json.dumps(tool_args)[:120]})")

            if call["argument_error"]:
                raw_result = f"[TOOL ARGUMENT ERROR] {call['argument_error']}"
            elif dynamic_budget <= 0:
                raw_result = ""
            else:
                # Execute via MCP
                raw_result = await mcp_client.execute_tool(tool_name, tool_args)

            # Enforce token budget on the result
            result_tokens = context_manager.count_tokens(raw_result)
            if result_tokens > dynamic_budget:
                raw_result = context_manager.truncate_to_budget(
                    raw_result, dynamic_budget
                )
                result_tokens = context_manager.count_tokens(raw_result)

            dynamic_budget = max(0, dynamic_budget - result_tokens)
            tool_result_tokens += result_tokens

            messages.append({"role": "tool", "content": raw_result})

    # --- Compute total context usage for /context endpoint ---
    total_message_tokens = context_manager.count_tokens(
        json.dumps([m.get("content", "") for m in messages])
    )
    _last_context_stats = {
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "baseline_tokens": baseline,
        "rag_tokens": rag_tokens,
        "tool_result_tokens": tool_result_tokens,
        "total_message_tokens": total_message_tokens,
        "remaining_budget": dynamic_budget,
        "utilisation_pct": round(
            (1 - dynamic_budget / MAX_CONTEXT_TOKENS) * 100, 2
        ),
    }

    # --- Return the response ---
    final_content = assistant_msg.content if assistant_msg and assistant_msg.content else ""

    if request.stream:
        async def generate():
            if final_content:
                yield final_content

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        return {"role": "assistant", "content": final_content}


# ---------------------------------------------------------------
# /context — context window usage
# ---------------------------------------------------------------

@app.get("/context")
async def get_context_usage():
    """
    Returns the token breakdown from the most recent /chat call,
    showing exactly how much of the 128k context window is occupied.
    """
    if not _last_context_stats:
        return {
            "message": "No /chat call has been made yet.",
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        }
    return _last_context_stats


# ---------------------------------------------------------------
# /health — basic health check
# ---------------------------------------------------------------

@app.get("/health")
async def health():
    ollama_ok = False
    ollama_models: list[str] = []
    try:
        model_list = await asyncio.to_thread(_ollama.list)
        ollama_models = [m.model for m in model_list.models]
        ollama_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_reachable": ollama_ok,
        "ollama_host": OLLAMA_HOST,
        "model": LLM_MODEL,
        "model_available": any(LLM_MODEL in m for m in ollama_models),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "mcp_tools_count": len(mcp_client.get_ollama_tools()),
    }


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
