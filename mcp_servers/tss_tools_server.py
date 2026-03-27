#!/usr/bin/env python3
"""
MCP Server — TSS Tools & Knowledge Search
==========================================
Provides the LLM with two tools:
  - get_tss_state: on-demand fetch of live TSS telemetry data
  - search_knowledge: semantic search over conversation history (LanceDB)

The orchestrator connects to this server via STDIO transport.
"""

import sys
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TSS_API_BASE_URL, TSS_API_TIMEOUT
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from api_client import ApiClient, ApiClientError

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_TSS_BASE_URL = TSS_API_BASE_URL.rstrip("/")
_TSS_TIMEOUT = TSS_API_TIMEOUT

# Scopes and their endpoint mappings
_SCOPE_ENDPOINTS: dict[str, list[str]] = {
    "all": ["/health", "/eva", "/ltv", "/vitals"],
    "eva": ["/eva"],
    "ltv": ["/ltv"],
    "health": ["/health"],
    "vitals": ["/health", "/vitals"],
}

_VALID_SCOPES = set(_SCOPE_ENDPOINTS.keys())

# ---------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------

server = Server("tss-tools")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _flatten(prefix: str, value, lines: list[str]) -> None:
    """Recursively flatten a dict/list into 'key: value' lines."""
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, lines)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _flatten(f"{prefix}[{index}]", child, lines)
    else:
        lines.append(f"{prefix}: {_format_scalar(value)}")


def _format_tss_response(results: dict[str, dict | None]) -> str:
    """Format fetched TSS data into a human-readable text block."""
    sections: list[str] = []

    for endpoint, data in results.items():
        if data is None:
            sections.append(f"## {endpoint}\nNot available (endpoint returned error or 404)")
            continue

        lines: list[str] = []
        _flatten("", data, lines)
        sections.append(f"## {endpoint}\n" + "\n".join(lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_tss_state",
            description=(
                "Fetch live telemetry data from the TSS Unity API. "
                "Use this when the user asks about EVA status, LTV location, "
                "crew vitals, suit telemetry, or any real-time mission data. "
                "The 'scope' parameter controls which endpoints are queried."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "eva", "ltv", "health", "vitals"],
                        "description": (
                            "What data to fetch. "
                            "'all' returns health + EVA + LTV + vitals. "
                            "'eva' returns EVA telemetry only. "
                            "'ltv' returns LTV location/signal only. "
                            "'health' returns server health only. "
                            "'vitals' returns crew vitals + health."
                        ),
                    },
                },
            },
        ),
        Tool(
            name="search_knowledge",
            description=(
                "Search the knowledge base for information from past "
                "conversations, including previous tool call results and "
                "assistant responses. Use natural language queries. "
                "Example: 'EVA1 oxygen levels from earlier' or "
                "'what did we discuss about the LTV errors'."
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
                        "description": "Max results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


# ---------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    arguments = arguments or {}

    if name == "get_tss_state":
        return await _handle_get_tss_state(arguments)
    if name == "search_knowledge":
        return await _handle_search_knowledge(arguments)

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------

async def _handle_get_tss_state(arguments: dict) -> list[TextContent]:
    """Fetch live TSS data with scoped endpoint selection."""
    scope = (arguments.get("scope") or "all").strip().lower()
    if scope not in _VALID_SCOPES:
        return [TextContent(
            type="text",
            text=f"Invalid scope '{scope}'. Valid: {', '.join(sorted(_VALID_SCOPES))}",
        )]

    if not _TSS_BASE_URL:
        return [TextContent(
            type="text",
            text=(
                "Error: TSS_API_BASE_URL is not configured. "
                "Set it in .env or environment variables."
            ),
        )]

    endpoints = _SCOPE_ENDPOINTS[scope]
    results: dict[str, dict | None] = {}

    try:
        async with ApiClient(
            base_url=_TSS_BASE_URL,
            timeout=_TSS_TIMEOUT,
        ) as client:
            for endpoint in endpoints:
                try:
                    data = await client.fetch_json(endpoint)
                    results[endpoint] = data
                except ApiClientError as exc:
                    # Graceful handling — endpoint may not exist (e.g. /vitals on older servers)
                    error_msg = str(exc)
                    if "404" in error_msg:
                        results[endpoint] = None
                    else:
                        results[endpoint] = None
                        print(f"[TSS] Error fetching {endpoint}: {exc}")
                except Exception as exc:
                    results[endpoint] = None
                    print(f"[TSS] Unexpected error fetching {endpoint}: {exc}")

        formatted = _format_tss_response(results)
        header = f"TSS State (scope={scope}, source={_TSS_BASE_URL})"
        return [TextContent(type="text", text=f"{header}\n\n{formatted}")]

    except Exception as exc:
        return [TextContent(
            type="text",
            text=f"Failed to connect to TSS server at {_TSS_BASE_URL}: {exc}",
        )]


async def _handle_search_knowledge(arguments: dict) -> list[TextContent]:
    """Semantic search over the conversation history knowledge base."""
    query = (arguments.get("query") or "").strip()
    if not query:
        return [TextContent(type="text", text="Error: empty search query.")]

    raw_top_k = arguments.get("top_k", 10)
    try:
        top_k = max(1, min(int(raw_top_k), 50))
    except (TypeError, ValueError):
        top_k = 10

    try:
        from rag_service import rag_service

        context, _ = await rag_service.retrieve_context(
            query=query, budget_limit=50000, top_k=top_k,
        )
        if not context:
            return [TextContent(
                type="text",
                text="No matching entries found in the knowledge base.",
            )]
        return [TextContent(type="text", text=context)]
    except Exception as exc:
        return [TextContent(type="text", text=f"Search error: {exc}")]


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
