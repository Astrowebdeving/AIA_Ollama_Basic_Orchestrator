from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api_client import ApiClient, ApiClientError


def _make_tss_transport() -> httpx.MockTransport:
    """Mock transport that simulates the TSS Unity API endpoints."""
    payloads = {
        "/api/v1/health": {"ok": True, "source_online": True},
        "/api/v1/eva": {
            "status": {"started": True},
            "telemetry": {"eva1": {"heart_rate": 90, "oxy_pri_storage": 95.0}},
        },
        "/api/v1/ltv": {
            "location": {"last_known_x": 1.0, "last_known_y": 2.0},
            "signal": {"strength": -42},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in payloads:
            return httpx.Response(200, json=payloads[path], request=request)
        # /vitals not available on this mock server
        return httpx.Response(404, json={"detail": "Not found"}, request=request)

    return httpx.MockTransport(handler)


class TssToolFetchTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the TSS tool server's fetch logic using mocked HTTP."""

    async def test_fetch_all_scopes_returns_health_eva_ltv(self) -> None:
        """Simulate what get_tss_state(scope=all) does internally."""
        transport = _make_tss_transport()

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=transport,
        ) as client:
            health = await client.fetch_json("/health")
            eva = await client.fetch_json("/eva")
            ltv = await client.fetch_json("/ltv")

        self.assertTrue(health["ok"])
        self.assertEqual(eva["telemetry"]["eva1"]["heart_rate"], 90)
        self.assertEqual(ltv["location"]["last_known_x"], 1.0)

    async def test_vitals_endpoint_graceful_404(self) -> None:
        """The /vitals endpoint may not exist — should handle 404 gracefully."""
        transport = _make_tss_transport()

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=transport,
        ) as client:
            with self.assertRaises(ApiClientError) as ctx:
                await client.fetch_json("/vitals")

            self.assertIn("404", str(ctx.exception))

    async def test_scoped_eva_only_fetch(self) -> None:
        """Scope=eva should only hit the /eva endpoint."""
        calls = []

        def tracking_handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(
                200,
                json={"telemetry": {"eva1": {"heart_rate": 90}}},
                request=request,
            )

        async with ApiClient(
            base_url="http://tss.local/api/v1",
            transport=httpx.MockTransport(tracking_handler),
        ) as client:
            data = await client.fetch_json("/eva")

        self.assertEqual(calls, ["/api/v1/eva"])
        self.assertEqual(data["telemetry"]["eva1"]["heart_rate"], 90)


class RagConversationStorageTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the RAG conversation turn storage."""

    async def test_store_conversation_turn_formats_correctly(self) -> None:
        """Verify the text document structure for a turn with tool calls."""
        from rag_service import RagService

        service = RagService()

        # Capture what gets passed to add_document
        captured_text = None
        captured_metadata = None

        async def mock_add_document(text, metadata="{}", max_chunk_tokens=512):
            nonlocal captured_text, captured_metadata
            captured_text = text
            captured_metadata = metadata

        service.add_document = mock_add_document

        await service.store_conversation_turn(
            user_message="What is EVA1 status?",
            assistant_reply="EVA1 heart rate is 90 bpm, oxygen at 95%.",
            tool_calls_log=[
                {
                    "name": "get_tss_state",
                    "arguments": {"scope": "eva"},
                    "result": "heart_rate: 90, oxy_pri: 95.0",
                }
            ],
        )

        self.assertIsNotNone(captured_text)
        self.assertIn("User: What is EVA1 status?", captured_text)
        self.assertIn("Tool: get_tss_state(scope=eva)", captured_text)
        self.assertIn("heart_rate: 90", captured_text)
        self.assertIn("Assistant: EVA1 heart rate is 90 bpm", captured_text)

        metadata = json.loads(captured_metadata)
        self.assertEqual(metadata["kind"], "conversation_turn")
        self.assertTrue(metadata["has_tool_calls"])

    async def test_store_conversation_turn_without_tools(self) -> None:
        """A simple user+assistant exchange with no tool calls."""
        from rag_service import RagService

        service = RagService()

        captured_text = None

        async def mock_add_document(text, metadata="{}", max_chunk_tokens=512):
            nonlocal captured_text
            captured_text = text

        service.add_document = mock_add_document

        await service.store_conversation_turn(
            user_message="Hello",
            assistant_reply="Hi there!",
        )

        self.assertIsNotNone(captured_text)
        self.assertIn("User: Hello", captured_text)
        self.assertIn("Assistant: Hi there!", captured_text)
        self.assertNotIn("Tool:", captured_text)


if __name__ == "__main__":
    unittest.main()
