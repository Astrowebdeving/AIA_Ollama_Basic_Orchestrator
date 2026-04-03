#!/usr/bin/env python3
"""
MCP Server — TSS Tools & Knowledge Search
==========================================
Provides the LLM with two tools:
  - get_tss_state: on-demand fetch of live TSS telemetry via UDP
  - search_knowledge: semantic search over conversation history (LanceDB)

The orchestrator connects to this server via STDIO transport.
"""

import sys
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TSS_UDP_HOST, TSS_UDP_PORT, TSS_UDP_TIMEOUT
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from tss_udp_client import TssUdpClient, TssUdpError

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_udp_client = TssUdpClient(
    host=TSS_UDP_HOST,
    port=TSS_UDP_PORT,
    timeout=TSS_UDP_TIMEOUT,
)

# Scopes map to one or more UDP command numbers.
# Command 0=ROVER, 1=EVA, 2=LTV, 3=LTV_ERRORS
_SCOPE_COMMANDS: dict[str, list[tuple[str, int]]] = {
    "all":    [("rover", 0), ("eva", 1), ("ltv", 2), ("ltv_errors", 3)],
    "rover":  [("rover", 0)],
    "eva":    [("eva", 1)],
    "ltv":    [("ltv", 2)],
    "ltv_errors": [("ltv_errors", 3)],
    "vitals": [("eva", 1)],  # fetches EVA, post-filtered to vitals only
}

_VALID_SCOPES = set(_SCOPE_COMMANDS.keys())

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

    for label, data in results.items():
        if data is None:
            sections.append(f"## {label}\nNot available (request failed or timed out)")
            continue

        lines: list[str] = []
        _flatten("", data, lines)
        sections.append(f"## {label}\n" + "\n".join(lines))

    return "\n\n".join(sections)


_VITALS_KEYS = {
    "heart_rate", "oxy_consumption", "co2_production", "temperature",
    "suit_pressure_oxy", "suit_pressure_co2", "suit_pressure_other",
    "suit_pressure_total", "helmet_pressure_co2",
    "primary_battery_level", "secondary_battery_level", "battery_level",
    "oxy_pri_storage", "oxy_sec_storage",
    "coolant_storage", "coolant_liquid_pressure",
    "fan_pri_rpm", "fan_sec_rpm",
    "eva_elapsed_time",
}


def _extract_vitals(eva_data: dict) -> dict:
    """Filter full EVA JSON down to vitals-only fields per crew member."""
    telemetry = eva_data.get("telemetry", {})
    result = {}
    for eva_id, readings in telemetry.items():
        if isinstance(readings, dict):
            result[eva_id] = {
                k: v for k, v in readings.items() if k in _VITALS_KEYS
            }
    return {"vitals": result}


# ---------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_tss_state",
            description=(
                "Fetch live telemetry data from the TSS2026 server via UDP. "
                "Use this when the user asks about EVA status, LTV location, "
                "rover telemetry, crew vitals, suit telemetry, or any real-time "
                "mission data. The 'scope' parameter controls which data sets "
                "are queried."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(_VALID_SCOPES),
                        "description": (
                            "What data to fetch. "
                            "'all' returns rover + EVA + LTV + LTV errors. "
                            "'rover' returns pressurized rover telemetry only. "
                            "'eva' returns EVA suit telemetry, DCU, UIA, IMU. "
                            "'vitals' returns crew vitals only (heart rate, O2, CO2, temp, battery). "
                            "'ltv' returns LTV location/signal only. "
                            "'ltv_errors' returns LTV error states only."
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
    """Fetch live TSS data on-demand via UDP."""
    scope = (arguments.get("scope") or "all").strip().lower()
    if scope not in _VALID_SCOPES:
        return [TextContent(
            type="text",
            text=f"Invalid scope '{scope}'. Valid: {', '.join(sorted(_VALID_SCOPES))}",
        )]

    commands = _SCOPE_COMMANDS[scope]
    results: dict[str, dict | None] = {}

    for label, cmd_number in commands:
        try:
            data = await _udp_client.request_json(cmd_number)
            if scope == "vitals":
                data = _extract_vitals(data)
            results[label] = data
        except TssUdpError as exc:
            results[label] = None
            print(f"[TSS] UDP error fetching {label} (cmd={cmd_number}): {exc}")
        except Exception as exc:
            results[label] = None
            print(f"[TSS] Unexpected error fetching {label}: {exc}")

    formatted = _format_tss_response(results)
    header = f"TSS State (scope={scope}, source={_udp_client.host}:{_udp_client.port})"
    return [TextContent(type="text", text=f"{header}\n\n{formatted}")]


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
