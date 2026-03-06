"""
LLM Provider Abstraction
=========================
Unified interface for chat backends so the orchestrator can switch
between Ollama, AFM/MLX, and llama.cpp without changing core logic.

Embeddings always stay on Ollama regardless of the chat provider.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import ollama as ollama_sdk


# ---------------------------------------------------------------
# Normalized response types
# ---------------------------------------------------------------

@dataclass
class ToolCallFunction:
    name: str
    arguments: Any  # dict or str (callers must coerce)


@dataclass
class ToolCall:
    function: ToolCallFunction
    id: str | None = None
    type: str = "function"


@dataclass
class ChatResponse:
    """Provider-agnostic chat completion response."""
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


# ---------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------

class LLMProvider(ABC):
    """Interface that every chat backend must implement."""

    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_context: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return a normalized response."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return a list of available model identifiers."""

    @abstractmethod
    async def health_check(self) -> dict:
        """Return provider health status."""


# ---------------------------------------------------------------
# Ollama provider (default)
# ---------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Chat via the Ollama Python SDK. Current default."""

    def __init__(self, host: str):
        self._client = ollama_sdk.Client(host=host)
        self.host = host

    async def chat(
        self, *, model, messages, tools=None, max_context=None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
        if max_context:
            kwargs["options"] = {"num_ctx": max_context}

        try:
            response = await asyncio.to_thread(self._client.chat, **kwargs)
        except ollama_sdk.ResponseError as exc:
            if "does not support tools" in str(exc) and tools:
                print(
                    f"[PROVIDER] {model} does not support tools via Ollama. "
                    f"Retrying without tools."
                )
                kwargs.pop("tools", None)
                response = await asyncio.to_thread(self._client.chat, **kwargs)
            else:
                raise

        msg = response.message

        # Normalize tool calls
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                ToolCall(
                    id=getattr(tc, "id", None),
                    type=getattr(tc, "type", "function"),
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
                for tc in msg.tool_calls
            ]

        return ChatResponse(content=msg.content, tool_calls=tool_calls)

    async def list_models(self) -> list[str]:
        result = await asyncio.to_thread(self._client.list)
        return [m.model for m in result.models]

    async def health_check(self) -> dict:
        try:
            models = await self.list_models()
            return {"reachable": True, "host": self.host, "models": models}
        except Exception as exc:
            return {"reachable": False, "host": self.host, "error": str(exc)}


# ---------------------------------------------------------------
# OpenAI-compatible provider (AFM / llama.cpp / llamaswap / etc.)
# ---------------------------------------------------------------

class OpenAICompatProvider(LLMProvider):
    """
    Chat via any server that exposes OpenAI-compatible
    /v1/chat/completions (AFM, llama.cpp, llamaswap, vLLM, etc.).
    """

    def __init__(self, base_url: str, provider_label: str = "openai-compat"):
        # Ensure base_url ends without trailing slash
        self.base_url = base_url.rstrip("/")
        self.label = provider_label

    @staticmethod
    def _normalise_messages(messages: list[dict]) -> list[dict]:
        """
        Convert orchestrator message history into the shape expected by
        OpenAI-compatible chat backends.
        """
        normalised_messages: list[dict] = []

        for msg_index, message in enumerate(messages):
            normalised = dict(message)

            if normalised.get("role") == "assistant" and normalised.get("tool_calls"):
                tool_calls = []
                for call_index, tool_call in enumerate(normalised["tool_calls"]):
                    function = dict(tool_call.get("function") or {})
                    arguments = function.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(
                            arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )

                    tool_calls.append(
                        {
                            "id": tool_call.get("id")
                            or f"call_{msg_index}_{call_index}",
                            "type": tool_call.get("type", "function"),
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    )

                normalised["tool_calls"] = tool_calls

            normalised_messages.append(normalised)

        return normalised_messages

    async def chat(
        self, *, model, messages, tools=None, max_context=None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._normalise_messages(messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        # OpenAI-compatible backends manage context size at server startup.
        # The orchestrator already enforces its own context budget client-side.
        _ = max_context

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg = choice.get("message", {})

        # Normalize tool calls from OpenAI format
        tool_calls = None
        raw_tool_calls = msg.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                # Arguments come as a JSON string in OpenAI format
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass  # Leave as string, callers will coerce
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id"),
                        type=tc.get("type", "function"),
                        function=ToolCallFunction(
                            name=fn.get("name", ""),
                            arguments=args,
                        )
                    )
                )

        return ChatResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
        )

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
            return [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            return []

    async def health_check(self) -> dict:
        error = ""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return {
                            "reachable": True,
                            "host": self.base_url,
                            "provider": self.label,
                        }
                    error = f"/health returned status {resp.status_code}"
                except Exception as exc:
                    error = str(exc)

                models_resp = await client.get(f"{self.base_url}/v1/models")
                models_resp.raise_for_status()
                return {
                    "reachable": True,
                    "host": self.base_url,
                    "provider": self.label,
                }
        except Exception as exc:
            if not error:
                error = str(exc)

        return {
            "reachable": False,
            "host": self.base_url,
            "provider": self.label,
            "error": error,
        }


# ---------------------------------------------------------------
# Factory
# ---------------------------------------------------------------

def get_provider(
    provider_name: str,
    *,
    ollama_host: str = "",
    api_base: str = "",
) -> LLMProvider:
    """
    Instantiate the appropriate provider.

    Parameters
    ----------
    provider_name : str
        One of "ollama", "afm", "llamacpp".
    ollama_host : str
        Ollama API base URL (only used when provider_name == "ollama").
    api_base : str
        Override the API base URL for OpenAI-compatible providers.
    """
    name = provider_name.lower().strip()

    if name == "ollama":
        return OllamaProvider(host=ollama_host)

    if name == "afm":
        base = api_base or "http://localhost:9999"
        return OpenAICompatProvider(base_url=base, provider_label="afm")

    if name == "llamacpp":
        base = api_base or "http://localhost:8080"
        return OpenAICompatProvider(base_url=base, provider_label="llamacpp")

    raise ValueError(
        f"Unknown LLM provider: {name!r}. "
        f"Supported: ollama, afm, llamacpp"
    )
