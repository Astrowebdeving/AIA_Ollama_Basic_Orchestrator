#!/usr/bin/env python3
"""
MCP Server — Telemetry Search & Knowledge Tools
================================================
Provides the LLM with tools for:
  - Structured telemetry queries (SQLite)
  - Semantic telemetry search (LanceDB)
  - On-demand telemetry fetch from source
  - Adding items to the RAG knowledge base

The orchestrator connects to this server via STDIO transport.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_DEFAULT_DB = os.getenv(
    "SQLITE_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "logs.db"),
)
_MAX_ROWS = 200

# Telemetry source URL (same as main poller uses)
_TELEMETRY_SOURCE_URL = os.getenv("TELEMETRY_SOURCE_URL", "")

# ---------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------

server = Server("telemetry-search")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{_DEFAULT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _get_write_connection() -> sqlite3.Connection:
    """Writable connection for inserting fetched telemetry."""
    conn = sqlite3.connect(_DEFAULT_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _coerce_bounded_int(
    value,
    *,
    default: int,
    minimum: int = 1,
    maximum: int = _MAX_ROWS,
) -> int:
    """Best-effort integer coercion for LLM-supplied tool arguments."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(number, maximum))


# ---------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_telemetry",
            description=(
                "Query telemetry events from the database using "
                "structured filters. Filter by time range, source, "
                "event type, and severity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Filter by source "
                            "(e.g. 'suit_sensors', 'habitat_env')"
                        ),
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["debug", "info", "warning", "critical"],
                        "description": "Minimum severity level.",
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO 8601 timestamp lower bound.",
                    },
                    "until": {
                        "type": "string",
                        "description": "ISO 8601 timestamp upper bound.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50).",
                    },
                },
            },
        ),
        Tool(
            name="search_telemetry",
            description=(
                "Semantic search over telemetry descriptions using "
                "natural language via vector embeddings. "
                "Example: 'crew suit pressure anomalies'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default 10).",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="telemetry_summary",
            description=(
                "Aggregate telemetry statistics: counts by source, "
                "severity, event type, and time range covered."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="fetch_telemetry_now",
            description=(
                "Immediately fetch fresh telemetry data from the "
                "configured source, outside the regular 20-second "
                "polling cycle. Use this when you need the very "
                "latest readings right now."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="add_to_knowledge_base",
            description=(
                "Add a piece of text to the RAG semantic knowledge "
                "base so it can be retrieved in future conversations. "
                "Use this when you encounter important information "
                "that should be remembered — telemetry insights, "
                "crew observations, mission decisions, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The text to embed and store. Should be a "
                            "meaningful, self-contained passage."
                        ),
                    },
                    "metadata": {
                        "type": "string",
                        "description": (
                            "Optional JSON metadata (source, category, etc.)."
                        ),
                    },
                },
                "required": ["text"],
            },
        ),
    ]


# ---------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    arguments = arguments or {}
    handlers = {
        "query_telemetry": _handle_query,
        "search_telemetry": _handle_search,
        "telemetry_summary": _handle_summary,
        "fetch_telemetry_now": _handle_fetch_now,
        "add_to_knowledge_base": _handle_add_knowledge,
    }
    handler = handlers.get(name)
    if not handler:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    # Some handlers are async, some are sync
    import asyncio
    if asyncio.iscoroutinefunction(handler):
        return await handler(arguments)
    return handler(arguments)


# ---------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------


def _handle_query(arguments: dict) -> list[TextContent]:
    """Structured telemetry query with filters."""
    try:
        conn = _get_connection()
        conditions: list[str] = []
        params: list = []

        if source := arguments.get("source"):
            conditions.append("source = ?")
            params.append(source)
        if event_type := arguments.get("event_type"):
            conditions.append("event_type = ?")
            params.append(event_type)
        if severity := arguments.get("severity"):
            severity_order = {"debug": 0, "info": 1, "warning": 2, "critical": 3}
            min_level = severity_order.get(severity, 0)
            matching = [s for s, v in severity_order.items() if v >= min_level]
            placeholders = ", ".join(["?"] * len(matching))
            conditions.append(f"severity IN ({placeholders})")
            params.extend(matching)
        if since := arguments.get("since"):
            conditions.append("timestamp >= ?")
            params.append(since)
        if until := arguments.get("until"):
            conditions.append("timestamp <= ?")
            params.append(until)

        limit = _coerce_bounded_int(arguments.get("limit"), default=50)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT id, timestamp, source, event_type, severity, "
            f"payload_json, description FROM telemetry_events"
            f"{where} ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        result = {
            "events": [dict(zip(columns, row)) for row in rows],
            "count": len(rows),
            "truncated": len(rows) == limit,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


async def _handle_search(arguments: dict) -> list[TextContent]:
    """Semantic search over the knowledge base (includes any telemetry the LLM has indexed)."""
    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Error: empty search query.")]

    top_k = _coerce_bounded_int(arguments.get("top_k"), default=10)
    try:
        from rag_service import rag_service

        # Use the general knowledge base retrieval with a generous budget
        context, _ = await rag_service.retrieve_context(
            query=query, budget_limit=50000, top_k=top_k
        )
        if not context:
            return [TextContent(type="text", text="No matching entries in knowledge base.")]
        return [TextContent(type="text", text=context)]
    except Exception as exc:
        return [TextContent(type="text", text=f"Search error: {exc}")]


def _handle_summary(arguments: dict) -> list[TextContent]:
    """Aggregate telemetry statistics."""
    try:
        conn = _get_connection()
        total = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]

        if total == 0:
            conn.close()
            return [TextContent(type="text", text=json.dumps({
                "total_events": 0,
                "message": "No telemetry events stored yet.",
            }))]

        by_source = dict(conn.execute(
            "SELECT source, COUNT(*) FROM telemetry_events GROUP BY source"
        ).fetchall())
        by_severity = dict(conn.execute(
            "SELECT severity, COUNT(*) FROM telemetry_events GROUP BY severity"
        ).fetchall())
        by_type = dict(conn.execute(
            "SELECT event_type, COUNT(*) FROM telemetry_events "
            "GROUP BY event_type ORDER BY COUNT(*) DESC LIMIT 20"
        ).fetchall())
        time_range = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM telemetry_events"
        ).fetchone()
        recent = conn.execute(
            "SELECT timestamp, source, event_type, severity, description "
            "FROM telemetry_events ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        recent_cols = ["timestamp", "source", "event_type", "severity", "description"]
        conn.close()

        result = {
            "total_events": total,
            "by_source": by_source,
            "by_severity": by_severity,
            "top_event_types": by_type,
            "time_range": {"earliest": time_range[0], "latest": time_range[1]},
            "most_recent": [dict(zip(recent_cols, row)) for row in recent],
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


async def _handle_fetch_now(arguments: dict) -> list[TextContent]:
    """
    Immediately fetch fresh telemetry from the configured source.
    Bypasses the regular 20-second polling cycle.
    """
    if not _TELEMETRY_SOURCE_URL:
        return [TextContent(
            type="text",
            text=(
                "Error: TELEMETRY_SOURCE_URL is not configured. "
                "Set it in .env to enable on-demand fetching."
            ),
        )]

    try:
        import httpx
        from datetime import datetime, timezone

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_TELEMETRY_SOURCE_URL)
            resp.raise_for_status()
            data = resp.json()

        events = data if isinstance(data, list) else [data]

        # Write to SQLite
        conn = _get_write_connection()
        ids = []
        for evt in events:
            cursor = conn.execute(
                "INSERT INTO telemetry_events "
                "(timestamp, source, event_type, severity, payload_json, description) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    evt.get("source", "telemetry_api"),
                    evt.get("event_type", "on_demand"),
                    evt.get("severity", "info"),
                    json.dumps(evt.get("payload")) if evt.get("payload") else None,
                    evt.get("description", ""),
                ),
            )
            ids.append(cursor.lastrowid)
        conn.commit()
        conn.close()

        return [TextContent(type="text", text=json.dumps({
            "status": "ok",
            "fetched": len(events),
            "ids": ids,
            "source": _TELEMETRY_SOURCE_URL,
        }, indent=2))]

    except Exception as exc:
        return [TextContent(type="text", text=f"Fetch error: {exc}")]


async def _handle_add_knowledge(arguments: dict) -> list[TextContent]:
    """
    Add text to the RAG semantic knowledge base.
    The LLM calls this when it determines information is worth
    indexing for future retrieval.
    """
    text = arguments.get("text", "").strip()
    if not text:
        return [TextContent(type="text", text="Error: empty text.")]

    metadata = arguments.get("metadata", "{}")

    try:
        from rag_service import rag_service
        await rag_service.add_document(text=text, metadata=metadata)
        return [TextContent(type="text", text=json.dumps({
            "status": "ok",
            "message": "Text added to knowledge base.",
            "text_preview": text[:100],
        }))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(main())
