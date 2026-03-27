"""
Async JSON API client helpers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx


class ApiClientError(RuntimeError):
    """Raised when an API request fails or returns an unexpected payload."""


class ApiClient:
    """
    Async HTTP client for object-shaped JSON APIs.

    Usage:
        async with ApiClient(base_url="https://api.example.com/v1") as client:
            data = await client.fetch_json("/users", params={"page": 1})
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ApiClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def connect(self) -> "ApiClient":
        """Initialise the underlying HTTP client if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                transport=self.transport,
            )
        return self

    async def close(self) -> None:
        """Gracefully close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.connect()
        assert self._client is not None
        return self._client

    # ----------------------------------------------------------------
    # Requests
    # ----------------------------------------------------------------

    async def fetch_json(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make a GET request and return a parsed JSON object."""
        return await self._request_json("GET", endpoint, params=params)

    async def post_json(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make a POST request with a JSON body and return a JSON object."""
        return await self._request_json("POST", endpoint, json=payload)

    async def _request_json(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        client = await self._ensure_client()

        last_response: httpx.Response | None = None
        for attempt in range(2):
            response = await client.request(method, endpoint, **kwargs)
            last_response = response
            if response.status_code != 429 or attempt == 1:
                return self._handle_response(response)
            await asyncio.sleep(self._retry_after_seconds(response))

        assert last_response is not None
        return self._handle_response(last_response)

    # ----------------------------------------------------------------
    # Response handling
    # ----------------------------------------------------------------

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Validate an HTTP response and extract the JSON body."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text.strip()
            if len(body) > 200:
                body = f"{body[:200]}..."
            detail = body or str(exc)
            raise ApiClientError(
                f"{response.request.method} {response.request.url} "
                f"failed with status {response.status_code}: {detail}"
            ) from exc

        return self._parse_json(response.text)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float:
        raw_retry_after = response.headers.get("Retry-After", "1")
        try:
            return max(0.0, float(raw_retry_after))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse a raw JSON string into a JSON object with error handling."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected a JSON object response, received {type(parsed).__name__}."
            )
        return parsed
