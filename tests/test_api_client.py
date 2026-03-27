from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api_client import ApiClient, ApiClientError


class ApiClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_json_returns_parsed_object(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.path, "/api/v1/health")
            return httpx.Response(200, json={"ok": True}, request=request)

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=httpx.MockTransport(handler),
        ) as client:
            data = await client.fetch_json("/health")

        self.assertEqual(data, {"ok": True})

    async def test_post_json_sends_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            body = json.loads(request.content.decode("utf-8"))
            self.assertEqual(body, {"hello": "world"})
            return httpx.Response(200, json={"status": "ok"}, request=request)

        async with ApiClient(
            base_url="http://example.local/api/v1",
            transport=httpx.MockTransport(handler),
        ) as client:
            data = await client.post_json("/telemetry", {"hello": "world"})

        self.assertEqual(data, {"status": "ok"})

    async def test_fetch_json_retries_once_on_rate_limit(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "0"},
                    json={"detail": "slow down"},
                    request=request,
                )
            return httpx.Response(200, json={"ok": True}, request=request)

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=httpx.MockTransport(handler),
        ) as client:
            data = await client.fetch_json("/health")

        self.assertEqual(attempts, 2)
        self.assertEqual(data, {"ok": True})

    async def test_fetch_json_rejects_non_object_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[1, 2, 3], request=request)

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=httpx.MockTransport(handler),
        ) as client:
            with self.assertRaises(ValueError) as ctx:
                await client.fetch_json("/health")

        self.assertIn("Expected a JSON object response", str(ctx.exception))

    async def test_fetch_json_raises_descriptive_error_on_http_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                503,
                json={"detail": "backend unavailable"},
                request=request,
            )

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=httpx.MockTransport(handler),
        ) as client:
            with self.assertRaises(ApiClientError) as ctx:
                await client.fetch_json("/health")

        message = str(ctx.exception)
        self.assertIn("503", message)
        self.assertIn("/api/v1/health", message)


if __name__ == "__main__":
    unittest.main()
