"""
MCP Client Module
=================
Manages connections to MCP servers, discovers available tools,
translates MCP tool schemas into Ollama-compatible format,
and executes tool calls on behalf of the orchestrator.
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_SERVERS


class McpClient:
    """
    Manages one or more MCP server connections via STDIO transport.
    Provides tool discovery, schema translation, and tool execution.
    """

    def __init__(self):
        # server_name -> ClientSession
        self._sessions: dict[str, ClientSession] = {}
        # Ollama-formatted tool list (combined from all servers)
        self._ollama_tools: list[dict] = []
        # Mapping: tool_name -> server_name (so we know where to route calls)
        self._tool_to_server: dict[str, str] = {}
        # Keep the exit stack alive for the lifetime of the client
        self._exit_stack = AsyncExitStack()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect_all(self):
        """
        Connect to every MCP server declared in config.MCP_SERVERS.
        Each entry should be:
            {
                "server-name": {
                    "command": "npx",        # or "python", "node", etc.
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {}                # optional extra env vars
                }
            }
        """
        for server_name, server_cfg in MCP_SERVERS.items():
            try:
                await self._connect_one(server_name, server_cfg)
                print(f"[MCP] Connected to server: {server_name}")
            except Exception as e:
                print(f"[MCP] Failed to connect to {server_name}: {e}")

        # After all connections, discover tools
        await self._discover_all_tools()

    async def _connect_one(self, name: str, cfg: dict):
        """Spawn the server process via STDIO and open a ClientSession."""
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env"),
        )

        # stdio_client is an async context manager that yields (read, write)
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport

        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._sessions[name] = session

    async def shutdown(self):
        """Gracefully close all MCP server connections."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._ollama_tools.clear()
        self._tool_to_server.clear()

    # ------------------------------------------------------------------
    # Tool discovery & schema translation
    # ------------------------------------------------------------------

    async def _discover_all_tools(self):
        """Fetch tool lists from every connected server and translate them."""
        self._ollama_tools = []
        self._tool_to_server = {}

        for server_name, session in self._sessions.items():
            result = await session.list_tools()
            for mcp_tool in result.tools:
                ollama_tool = self._mcp_to_ollama_tool(mcp_tool)
                self._ollama_tools.append(ollama_tool)
                self._tool_to_server[mcp_tool.name] = server_name

        print(f"[MCP] Discovered {len(self._ollama_tools)} tool(s) across {len(self._sessions)} server(s)")

    @staticmethod
    def _mcp_to_ollama_tool(mcp_tool) -> dict:
        """
        Translate an MCP Tool object into the dict format expected by
        the Ollama Python SDK:

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            }

        MCP Tool fields:
            - name: str
            - description: str | None
            - inputSchema: dict   (JSON Schema)
        """
        input_schema = mcp_tool.inputSchema or {}

        # Ensure the schema has the right top-level type
        parameters = {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
        }
        if "required" in input_schema:
            parameters["required"] = input_schema["required"]

        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "parameters": parameters,
            },
        }

    def get_ollama_tools(self) -> list[dict]:
        """Return all discovered tools in Ollama-compatible format."""
        return self._ollama_tools

    def get_tool_schemas_json(self) -> str:
        """Serialised form for token counting."""
        return json.dumps(self._ollama_tools)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Route a tool call to the correct MCP server and return the
        result as a plain string (ready for the LLM context).
        """
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"[ERROR] Unknown tool: {tool_name}"

        session = self._sessions.get(server_name)
        if not session:
            return f"[ERROR] No active session for server: {server_name}"

        try:
            result = await session.call_tool(tool_name, arguments)

            # CallToolResult.content is a list of content blocks
            # Each block has a .type and .text (for text blocks)
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    # Fallback: serialise the block
                    parts.append(str(block))

            output = "\n".join(parts)

            if result.isError:
                return f"[TOOL ERROR] {output}"
            return output

        except Exception as e:
            return f"[TOOL EXECUTION ERROR] {e}"


# Global instance
mcp_client = McpClient()
