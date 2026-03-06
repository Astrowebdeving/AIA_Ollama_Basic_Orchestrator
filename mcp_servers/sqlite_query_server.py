#!/usr/bin/env python3
"""
MCP Server — SQLite Query Tool
===============================
Exposes the orchestrator's logs.db to the LLM via MCP,
allowing it to run read-only SQL queries and inspect schema.

Run:
    python mcp_servers/sqlite_query_server.py

The orchestrator connects to this server via STDIO transport.
"""

import json
import os
import sqlite3
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Default to the logs.db in the parent directory (orchestrator root)
_DEFAULT_DB = os.getenv(
    "SQLITE_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "logs.db"),
)

# Safety: maximum rows returned per query
_MAX_ROWS = 200

# ---------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------

server = Server("sqlite-query")


def _get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a read-only connection to the SQLite database."""
    path = db_path or _DEFAULT_DB
    # "file:" URI with "?mode=ro" ensures read-only access
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_logs_db",
            description=(
                "Execute a read-only SQL query against the orchestrator's "
                "logs.db database. Returns results as JSON. "
                "Use this to inspect past queries, token usage, timestamps, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A read-only SQL query (SELECT only). "
                            "Example: SELECT * FROM query_logs ORDER BY id DESC LIMIT 10"
                        ),
                    },
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="describe_logs_db",
            description=(
                "List all tables and their schemas in logs.db. "
                "Call this first to understand the database structure."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_logs_db":
        return _handle_query(arguments)
    elif name == "describe_logs_db":
        return _handle_describe()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------


def _handle_query(arguments: dict) -> list[TextContent]:
    """Execute a read-only SQL query and return results as JSON."""
    sql = (arguments.get("sql") or "").strip()
    if not sql:
        return [TextContent(type="text", text="Error: empty SQL query.")]

    # Safety: reject anything that isn't a SELECT / WITH / EXPLAIN
    first_word = sql.split()[0].upper()
    if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA"):
        return [
            TextContent(
                type="text",
                text=(
                    f"Error: only SELECT queries are allowed "
                    f"(got '{first_word}…'). This is a read-only tool."
                ),
            )
        ]

    try:
        conn = _get_connection()
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(_MAX_ROWS)
        conn.close()

        result = {
            "columns": columns,
            "rows": [dict(zip(columns, row)) for row in rows],
            "row_count": len(rows),
            "truncated": len(rows) == _MAX_ROWS,
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except sqlite3.Error as exc:
        return [TextContent(type="text", text=f"SQL Error: {exc}")]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


def _handle_describe() -> list[TextContent]:
    """Return the full schema of the database."""
    try:
        conn = _get_connection()
        cursor = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        )
        tables = cursor.fetchall()
        conn.close()

        if not tables:
            return [TextContent(type="text", text="Database has no tables.")]

        lines = ["# Database Schema\n"]
        for table in tables:
            lines.append(f"## {table['name']}\n```sql\n{table['sql']}\n```\n")

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())
