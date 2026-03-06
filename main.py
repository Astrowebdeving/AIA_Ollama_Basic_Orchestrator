"""
Orchestrator -- Main FastAPI Application
=========================================
Exposes endpoints:
  /chat       -- Agentic chat loop with MCP tools and RAG.
  /telemetry  -- Ingest telemetry events (stored in SQLite).
  /context    -- Returns current context window usage stats.
  /health     -- Basic health check including backend connectivity.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import (
    LLM_MODEL, LLM_PROVIDER, LLM_API_BASE, MAX_CONTEXT_TOKENS,
    OLLAMA_HOST, SUMMARIZE_THRESHOLD, TELEMETRY_POLL_INTERVAL,
    TELEMETRY_SOURCE_URL,
)
from llm_provider import get_provider
from context_manager import context_manager
from context_summarizer import context_summarizer
from db_logger import init_db, log_query, log_telemetry
from mcp_client import mcp_client
from rag_service import rag_service

# ---------------------------------------------------------------
# Globals
# ---------------------------------------------------------------

_BASE_SYSTEM_PROMPT = (
    f"You are a helpful, rigorous assistant powered by {LLM_MODEL}. "
    "Respond concisely, do not be verbose. "
    "You have access to tools provided by external servers via the "
    "Model Context Protocol (MCP). Use them when needed to answer "
    "the user's request accurately. If you call a tool, wait for "
    "its result before generating your final answer."
)


def _build_system_prompt(tools: list[dict]) -> str:
    """
    Build system prompt that includes a summary of available tools.
    This ensures the model knows what tools exist even when native
    tool calling is not supported by the backend.
    """
    if not tools:
        return _BASE_SYSTEM_PROMPT

    tool_lines = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "unknown")
        desc = fn.get("description", "No description")
        params = fn.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else "none"
        tool_lines.append(f"  - {name}({param_names}): {desc}")

    tool_block = "\n".join(tool_lines)
    return (
        f"{_BASE_SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{tool_block}\n\n"
        "Do NOT invent or hallucinate tools that are not listed above. "
        "Only reference the tools shown here."
    )

MAX_TOOL_ROUNDS = 10  # Safety valve against infinite tool loops

# LLM provider (chat). Embeddings always stay on Ollama via rag_service.
_llm = get_provider(
    LLM_PROVIDER, ollama_host=OLLAMA_HOST, api_base=LLM_API_BASE,
)

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


def _count_request_tokens(messages: list[dict], tool_schemas: list[dict]) -> int:
    """Count tokens for the message list plus tool schemas sent to Ollama."""
    return context_manager.count_message_tokens(messages, tool_schemas=tool_schemas)


def _build_assistant_history_message(
    content: str | None,
    tool_calls: list[dict] | None = None,
) -> dict:
    """Format an assistant turn in the message history for the active provider."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
    }
    if not tool_calls:
        return message

    formatted_tool_calls = []
    for index, call in enumerate(tool_calls):
        tool_call_id = call.get("id") or f"call_{index}"
        function_payload = {
            "name": call["name"],
            "arguments": call["arguments"],
        }

        if LLM_PROVIDER != "ollama":
            function_payload["arguments"] = json.dumps(
                call["arguments"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            formatted_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": call.get("type", "function"),
                    "function": function_payload,
                }
            )
        else:
            formatted_tool_calls.append({"function": function_payload})

    message["tool_calls"] = formatted_tool_calls
    return message


def _build_tool_history_message(content: str, tool_call_id: str | None) -> dict:
    """Format a tool result message for the active provider."""
    message: dict[str, Any] = {
        "role": "tool",
        "content": content,
    }
    if LLM_PROVIDER != "ollama" and tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def _tool_message_content_budget(
    messages: list[dict],
    tool_schemas: list[dict],
    tool_message_template: dict | None = None,
) -> int:
    """
    Reserve enough room for the tool message wrapper itself and return
    the remaining content budget for the tool payload.
    """
    current_tokens = _count_request_tokens(messages, tool_schemas)
    remaining_budget = context_manager.get_dynamic_budget(current_tokens)
    empty_tool_message = {"role": "tool", "content": ""}
    if tool_message_template:
        empty_tool_message.update(tool_message_template)
        empty_tool_message["content"] = ""
    empty_tool_tokens = _count_request_tokens(
        messages + [empty_tool_message],
        tool_schemas,
    ) - current_tokens

    if empty_tool_tokens > remaining_budget:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model requested a tool result, but there is no remaining "
                "context budget to fit the tool response."
            ),
        )

    return max(0, remaining_budget - empty_tool_tokens)


# ---------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------

class MessagePayload(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[MessagePayload]
    stream: bool = True


class TelemetryEvent(BaseModel):
    source: str = Field(..., description="Origin (e.g. 'suit_sensors', 'habitat_env')")
    event_type: str = Field(..., description="Category (e.g. 'temperature_reading')")
    severity: str = Field(default="info", description="debug | info | warning | critical")
    payload: dict | None = Field(default=None, description="Arbitrary JSON data")
    description: str = Field(default="", description="Human-readable summary of the event")


class TelemetryBatch(BaseModel):
    events: list[TelemetryEvent]


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

    print(f"[STARTUP] Verifying {LLM_PROVIDER} chat backend …")
    llm_health = await _llm.health_check()
    llm_host = llm_health.get("host", OLLAMA_HOST)
    if llm_health.get("reachable", False):
        available = await _llm.list_models()
        if available and any(LLM_MODEL in m for m in available):
            print(
                f"[STARTUP] ✓ Model '{LLM_MODEL}' available on "
                f"{LLM_PROVIDER} at {llm_host}"
            )
        elif available:
            print(
                f"[STARTUP] ⚠ Model '{LLM_MODEL}' NOT found on "
                f"{LLM_PROVIDER}. Available: {available}"
            )
        else:
            print(
                f"[STARTUP] ✓ {LLM_PROVIDER} reachable at {llm_host} "
                f"(model list unavailable or empty)"
            )
    else:
        error = llm_health.get("error", "health check failed")
        print(
            f"[STARTUP] ✗ Cannot reach {LLM_PROVIDER} backend "
            f"at {llm_host}: {error}"
        )

    print("[STARTUP] Connecting to MCP servers …")
    await mcp_client.connect_all()

    # ---- Background telemetry poller ----
    poller_task = None
    if TELEMETRY_SOURCE_URL:
        async def _telemetry_poller():
            """Poll telemetry source every TELEMETRY_POLL_INTERVAL seconds."""
            import httpx

            print(
                f"[POLLER] Telemetry poller started "
                f"(every {TELEMETRY_POLL_INTERVAL}s from {TELEMETRY_SOURCE_URL})"
            )
            async with httpx.AsyncClient(timeout=10) as client:
                while True:
                    try:
                        resp = await client.get(TELEMETRY_SOURCE_URL)
                        resp.raise_for_status()
                        data = resp.json()

                        # Expect either a single event dict or a list of events
                        events = data if isinstance(data, list) else [data]
                        for evt in events:
                            await log_telemetry(
                                source=evt.get("source", "telemetry_api"),
                                event_type=evt.get("event_type", "poll"),
                                severity=evt.get("severity", "info"),
                                payload=evt.get("payload"),
                                description=evt.get("description", ""),
                            )
                        if events:
                            print(f"[POLLER] Stored {len(events)} event(s)")

                    except Exception as exc:
                        print(f"[POLLER] Error: {exc}")

                    await asyncio.sleep(TELEMETRY_POLL_INTERVAL)

        poller_task = asyncio.create_task(_telemetry_poller())
    else:
        print("[STARTUP] No TELEMETRY_SOURCE_URL set — poller disabled")

    print("[STARTUP] Ready.")
    yield

    # ---- Shutdown ----
    if poller_task:
        poller_task.cancel()
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
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    # --- Gather tool schemas ---
    ollama_tools = mcp_client.get_ollama_tools()

    # --- Calculate baseline token budget ---
    system_prompt = _build_system_prompt(ollama_tools)
    base_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        base_messages.append({"role": msg.role, "content": msg.content})

    baseline = _count_request_tokens(base_messages, ollama_tools)
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

    # --- Build the message history ---
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    messages.extend(base_messages[1:])

    # --- Context summarization when above threshold ---
    pre_summary_tokens = _count_request_tokens(messages, ollama_tools)
    if context_summarizer.should_summarize(pre_summary_tokens):
        messages = await context_summarizer.summarize_history(
            _llm, messages
        )

    pre_tool_tokens = _count_request_tokens(messages, ollama_tools)
    dynamic_budget = context_manager.get_dynamic_budget(pre_tool_tokens)
    if pre_tool_tokens >= MAX_CONTEXT_TOKENS:
        _last_context_stats = {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "baseline_tokens": max(0, pre_tool_tokens - rag_tokens),
            "rag_tokens": rag_tokens,
            "tool_result_tokens": 0,
            "total_message_tokens": pre_tool_tokens,
            "remaining_budget": 0,
            "utilisation_pct": 100.0,
        }
        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation and retrieved context exceed the model context "
                "limit. Please shorten message history or reduce retrieved context."
            ),
        )

    # --- Track tokens consumed by tool results ---
    tool_result_tokens = 0

    # --- Agentic tool loop ---
    assistant_msg = None
    completed_response = False
    for _round in range(MAX_TOOL_ROUNDS):
        current_tokens = _count_request_tokens(messages, ollama_tools)
        if current_tokens >= MAX_CONTEXT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail="Conversation exceeded the model context limit during tool execution.",
            )

        response = await _llm.chat(
            model=LLM_MODEL,
            messages=messages,
            tools=ollama_tools if ollama_tools else None,
            max_context=MAX_CONTEXT_TOKENS,
        )

        assistant_msg = response

        # No tool calls -> we have a final answer
        if not assistant_msg.tool_calls:
            completed_response = True
            break

        parsed_tool_calls = []
        for call_index, tc in enumerate(assistant_msg.tool_calls):
            tool_name = tc.function.name
            argument_error = None
            tool_call_id = tc.id or f"call_{_round}_{call_index}"
            try:
                tool_args = _coerce_tool_arguments(tc.function.arguments)
            except ValueError as exc:
                argument_error = str(exc)
                tool_args = {}

            parsed_tool_calls.append(
                {
                    "name": tool_name,
                    "id": tool_call_id,
                    "type": tc.type,
                    "arguments": tool_args,
                    "argument_error": argument_error,
                }
            )

        # Append the assistant message (with its tool_calls) to history
        messages.append(
            _build_assistant_history_message(
                assistant_msg.content,
                parsed_tool_calls,
            )
        )

        for call in parsed_tool_calls:
            tool_name = call["name"]
            tool_args = call["arguments"]

            print(f"[TOOL CALL] {tool_name}({json.dumps(tool_args)[:120]})")

            if call["argument_error"]:
                raw_result = f"[TOOL ARGUMENT ERROR] {call['argument_error']}"
            else:
                # Execute via MCP
                raw_result = await mcp_client.execute_tool(tool_name, tool_args)

            tool_message_template = {}
            if LLM_PROVIDER != "ollama":
                tool_message_template["tool_call_id"] = call["id"]
            tool_content_budget = _tool_message_content_budget(
                messages,
                ollama_tools,
                tool_message_template=tool_message_template,
            )
            raw_result = context_manager.truncate_to_budget(
                raw_result, tool_content_budget
            )

            result_tokens = context_manager.count_tokens(raw_result)
            tool_result_tokens += result_tokens

            messages.append(
                _build_tool_history_message(raw_result, call["id"])
            )
            dynamic_budget = context_manager.get_dynamic_budget(
                _count_request_tokens(messages, ollama_tools)
            )

    # --- Compute total context usage for /context endpoint ---
    total_message_tokens = _count_request_tokens(messages, ollama_tools)
    _last_context_stats = {
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "baseline_tokens": max(0, pre_tool_tokens - rag_tokens),
        "rag_tokens": rag_tokens,
        "tool_result_tokens": tool_result_tokens,
        "total_message_tokens": total_message_tokens,
        "remaining_budget": dynamic_budget,
        "utilisation_pct": round(
            (total_message_tokens / MAX_CONTEXT_TOKENS) * 100, 2
        ),
    }

    # --- Return the response ---
    if completed_response and assistant_msg and assistant_msg.content:
        final_content = assistant_msg.content
    elif completed_response:
        final_content = ""
    else:
        final_content = "Unable to complete the request within the tool-call limit."

    if request.stream:
        async def generate():
            if final_content:
                yield final_content

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        return {"role": "assistant", "content": final_content}


# ---------------------------------------------------------------
# /telemetry — ingest telemetry events
# ---------------------------------------------------------------

@app.post("/telemetry")
async def ingest_telemetry(event: TelemetryEvent):
    """Ingest a single telemetry event. Stored in SQLite only."""
    row_id = await log_telemetry(
        source=event.source,
        event_type=event.event_type,
        severity=event.severity,
        payload=event.payload,
        description=event.description,
    )
    return {"status": "ok", "id": row_id}


@app.post("/telemetry/batch")
async def ingest_telemetry_batch(batch: TelemetryBatch):
    """Ingest multiple telemetry events at once. Stored in SQLite only."""
    ids = []
    for event in batch.events:
        row_id = await log_telemetry(
            source=event.source,
            event_type=event.event_type,
            severity=event.severity,
            payload=event.payload,
            description=event.description,
        )
        ids.append(row_id)
    return {"status": "ok", "ids": ids, "count": len(ids)}


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
    llm_health = await _llm.health_check()
    llm_reachable = llm_health.get("reachable", False)

    # Check model availability
    model_available = False
    if llm_reachable:
        try:
            models = await _llm.list_models()
            model_available = any(LLM_MODEL in m for m in models)
        except Exception:
            pass

    return {
        "status": "ok" if llm_reachable else "degraded",
        "llm_provider": LLM_PROVIDER,
        "llm_backend": llm_health,
        "model": LLM_MODEL,
        "model_available": model_available,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "mcp_tools_count": len(mcp_client.get_ollama_tools()),
    }


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
