#!/usr/bin/env python3
"""
MCP Server — TSS Tools, Document Search & Knowledge
=====================================================
Provides the LLM with tools for:
  - get_tss_state: on-demand fetch of live TSS telemetry via UDP
  - search_docs / read_doc: grep + read reference documents (text + PDF)
  - inspect_image: base64-encode an image and return it as ImageContent
    so the orchestrator injects it directly into the main LLM context
    (no separate vision call — the main model sees the actual image)
  - search_knowledge: semantic search over conversation history (LanceDB)

The orchestrator connects to this server via STDIO transport.
"""

import base64
import mimetypes
import sys
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TSS_UDP_HOST, TSS_UDP_PORT, TSS_UDP_TIMEOUT, OLLAMA_HOST, LLM_MODEL
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool

import ollama as ollama_sdk
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

_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

_PDF_EXTENSIONS = {".pdf"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif", ".bmp"}

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


def _list_docs() -> list[str]:
    """List available doc filenames relative to docs/."""
    if not _DOCS_DIR.is_dir():
        return []
    return [
        str(f.relative_to(_DOCS_DIR))
        for f in sorted(_DOCS_DIR.rglob("*"))
        if f.is_file() and not f.name.startswith(".")
    ]


def _read_file_text(filepath: Path) -> str | None:
    """
    Read text content from a file. Supports plain text and PDFs.
    Returns None if the file can't be read as text.
    """
    if filepath.suffix.lower() in _PDF_EXTENSIONS:
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(filepath))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except Exception as exc:
            print(f"[DOCS] PDF read error for {filepath.name}: {exc}")
            return None
    elif filepath.suffix.lower() in _IMAGE_EXTENSIONS:
        return None  # images aren't text-searchable
    else:
        try:
            return filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


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
            name="search_docs",
            description=(
                "Grep reference documents in docs/ for a text pattern. "
                "Searches text files and PDFs. "
                "Returns compact file:line:match results with NO surrounding context. "
                "To expand context around a specific result, call read_doc with around_line."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text pattern to search for (case-insensitive).",
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "Optional glob to narrow search (default: all files). Example: '*.md'",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_doc",
            description=(
                "Read a document or section from docs/. Supports text and PDF files. "
                "Use after search_docs to expand context around a specific line, "
                "or to read a short document in full. "
                "Example: read_doc('procedures/ev-team-procedure-timeline.pdf', around_line=42, max_chars=2000)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path relative to docs/ (e.g. 'procedures.md').",
                    },
                    "around_line": {
                        "type": "integer",
                        "description": "Line number to center the read around (from search_docs results).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default: 2000, max: 8000).",
                    },
                },
                "required": ["filename"],
            },
        ),
        Tool(
            name="inspect_image",
            description=(
                "Load an image from docs/ directly into your visual context. "
                "Use for maps, diagrams, equipment photos, or any visual reference. "
                "The image will be injected into your context so you can see it yourself. "
                "Ask your question about the image in your NEXT message after receiving the result. "
                "Example: inspect_image('maps/annotated/dust-map.png')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Image path relative to docs/ (e.g. 'maps/annotated/dust-map.png').",
                    },
                },
                "required": ["filename"],
            },
        ),
        # --- search_knowledge disabled for MVP — re-enable for RAG ---
        # Tool(
        #     name="search_knowledge",
        #     description=(
        #         "Search the knowledge base for information from past "
        #         "conversations, including previous tool call results and "
        #         "assistant responses. Use natural language queries. "
        #         "Example: 'EVA1 oxygen levels from earlier' or "
        #         "'what did we discuss about the LTV errors'."
        #     ),
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "query": {
        #                 "type": "string",
        #                 "description": "Natural language search query.",
        #             },
        #             "top_k": {
        #                 "type": "integer",
        #                 "description": "Max results to return (default 10).",
        #             },
        #         },
        #         "required": ["query"],
        #     },
        # ),
    ]


# ---------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    arguments = arguments or {}

    if name == "get_tss_state":
        return await _handle_get_tss_state(arguments)
    if name == "search_docs":
        return await _handle_search_docs(arguments)
    if name == "read_doc":
        return await _handle_read_doc(arguments)
    if name == "inspect_image":
        return await _handle_inspect_image(arguments)
    # if name == "search_knowledge":  # disabled for MVP
    #     return await _handle_search_knowledge(arguments)

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
    output = f"{header}\n\n{formatted}"

    # Cap total output to prevent context bloat on scope=all
    max_output = 4000
    if len(output) > max_output:
        output = output[:max_output] + f"\n\n... [truncated at {max_output} chars. Use a narrower scope like 'eva' or 'vitals']"

    return [TextContent(type="text", text=output)]


async def _handle_search_docs(arguments: dict) -> list[TextContent]:
    """Lean grep: file:line:matched_line — no surrounding context."""
    query = (arguments.get("query") or "").strip()
    if not query:
        return [TextContent(type="text", text="Error: empty search query.")]

    file_filter = (arguments.get("file_filter") or "*").strip()

    if not _DOCS_DIR.is_dir():
        return [TextContent(type="text", text="No docs/ directory found.")]

    query_lower = query.lower()
    matches: list[str] = []
    max_matches = 50

    for filepath in sorted(_DOCS_DIR.rglob(file_filter)):
        if not filepath.is_file() or filepath.name.startswith("."):
            continue

        text = _read_file_text(filepath)
        if text is None:
            continue

        lines = text.splitlines()

        rel_path = filepath.relative_to(_DOCS_DIR)
        for line_num, line in enumerate(lines, 1):
            if query_lower in line.lower():
                matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                if len(matches) >= max_matches:
                    break
        if len(matches) >= max_matches:
            break

    if not matches:
        available = _list_docs()
        if available:
            file_list = "\n".join(f"  - {f}" for f in available[:30])
            return [TextContent(type="text", text=f"No matches for '{query}'. Available documents:\n{file_list}")]
        return [TextContent(type="text", text=f"No matches for '{query}' and no documents found in docs/.")]

    return [TextContent(
        type="text",
        text=f"Found {len(matches)} match(es) for '{query}':\n" + "\n".join(matches),
    )]


async def _handle_read_doc(arguments: dict) -> list[TextContent]:
    """Read a document or section from docs/."""
    filename = (arguments.get("filename") or "").strip()
    if not filename:
        return [TextContent(type="text", text="Error: filename is required.")]

    if ".." in filename:
        return [TextContent(type="text", text="Error: '..' not allowed in filename.")]

    filepath = _DOCS_DIR / filename
    if not filepath.is_file():
        available = _list_docs()
        if available:
            file_list = "\n".join(f"  - {f}" for f in available[:30])
            return [TextContent(type="text", text=f"File '{filename}' not found. Available:\n{file_list}")]
        return [TextContent(type="text", text=f"File '{filename}' not found and docs/ is empty.")]

    try:
        text = _read_file_text(filepath)
        if text is None:
            return [TextContent(type="text", text=f"Cannot read '{filename}' as text. Use inspect_image for image files.")]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error reading '{filename}': {exc}")]

    max_chars = min(max(int(arguments.get("max_chars") or 2000), 100), 8000)
    around_line = arguments.get("around_line")

    if around_line is not None:
        lines = text.splitlines(keepends=True)
        target = max(1, int(around_line)) - 1  # 0-indexed

        # Expand outward from target line until we hit max_chars
        start = target
        end = min(target + 1, len(lines))
        char_count = len(lines[target]) if target < len(lines) else 0

        while char_count < max_chars:
            expanded = False
            if start > 0:
                start -= 1
                char_count += len(lines[start])
                expanded = True
            if end < len(lines):
                char_count += len(lines[end])
                end += 1
                expanded = True
            if not expanded:
                break

        excerpt = "".join(lines[start:end])
        header = f"--- {filename} (lines {start + 1}-{end}, around line {around_line}) ---"
        return [TextContent(type="text", text=f"{header}\n{excerpt}")]
    else:
        excerpt = text[:max_chars]
        truncated = len(text) > max_chars
        header = f"--- {filename} ({len(text)} chars total) ---"
        suffix = ""
        if truncated:
            suffix = f"\n\n... [truncated, {len(text) - max_chars} more chars]"
        return [TextContent(type="text", text=f"{header}\n{excerpt}{suffix}")]


# --- search_knowledge disabled for MVP — re-enable for RAG ---
# async def _handle_search_knowledge(arguments: dict) -> list[TextContent]:
#     """Semantic search over the conversation history knowledge base."""
#     query = (arguments.get("query") or "").strip()
#     if not query:
#         return [TextContent(type="text", text="Error: empty search query.")]
#
#     raw_top_k = arguments.get("top_k", 10)
#     try:
#         top_k = max(1, min(int(raw_top_k), 50))
#     except (TypeError, ValueError):
#         top_k = 10
#
#     try:
#         from rag_service import rag_service
#
#         context, _ = await rag_service.retrieve_context(
#             query=query, budget_limit=8000, top_k=top_k,
#         )
#         if not context:
#             return [TextContent(
#                 type="text",
#                 text="No matching entries found in the knowledge base.",
#             )]
#         return [TextContent(type="text", text=context)]
#     except Exception as exc:
#         return [TextContent(type="text", text=f"Search error: {exc}")]


async def _handle_inspect_image(arguments: dict) -> list[TextContent | ImageContent]:
    """Return base64-encoded image for direct injection into the main LLM context."""
    filename = (arguments.get("filename") or "").strip()
    if not filename:
        return [TextContent(type="text", text="Error: filename is required.")]

    if ".." in filename:
        return [TextContent(type="text", text="Error: '..' not allowed in filename.")]

    filepath = _DOCS_DIR / filename
    if not filepath.is_file():
        images = [
            str(f.relative_to(_DOCS_DIR))
            for f in sorted(_DOCS_DIR.rglob("*"))
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        if images:
            file_list = "\n".join(f"  - {f}" for f in images[:20])
            return [TextContent(type="text", text=f"Image '{filename}' not found. Available images:\n{file_list}")]
        return [TextContent(type="text", text=f"Image '{filename}' not found.")]

    if filepath.suffix.lower() not in _IMAGE_EXTENSIONS:
        return [TextContent(type="text", text=f"'{filename}' is not an image file. Use read_doc for text/PDF files.")]

    try:
        image_bytes = filepath.read_bytes()
        b64_data = base64.b64encode(image_bytes).decode("ascii")

        mime_type = mimetypes.guess_type(str(filepath))[0] or "image/png"

        return [
            TextContent(type="text", text=f"[Image loaded: {filename}]"),
            ImageContent(type="image", data=b64_data, mimeType=mime_type),
        ]
    except Exception as exc:
        return [TextContent(type="text", text=f"Image read error: {exc}")]


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
