"""
TSS UDP Client — On-demand telemetry fetcher for TSS2026.
==========================================================
Sends big-endian binary packets over UDP and decodes the JSON response.

Protocol (from TSS2026 README):
  Request:  [uint32 timestamp][uint32 command]           (8 bytes)
  Request:  [uint32 timestamp][uint32 command][float32]   (12 bytes, for POST)
  Response: raw UTF-8 JSON bytes (variable length)

Command map (GET):
  0 -> ROVER.json    1 -> EVA.json    2 -> LTV.json    3 -> LTV_ERRORS.json

Command map (POST):
  1107 -> Brakes      1109 -> Throttle    1110 -> Steering
  1103 -> Cabin Heat   1104 -> Cabin Cool   1106 -> Headlights
  2050 -> LTV Ping     2051 -> LTV Ping (unlimited)
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any


class TssUdpError(RuntimeError):
    """Raised when a TSS UDP request fails."""


class TssUdpClient:
    """
    Async, on-demand UDP client for TSS2026.

    Usage:
        client = TssUdpClient("10.206.64.189", 14141)
        data = await client.request_json(1)   # EVA telemetry
        await client.close()
    """

    # GET commands that return JSON
    CMD_ROVER = 0
    CMD_EVA = 1
    CMD_LTV = 2
    CMD_LTV_ERRORS = 3

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 14141,
        timeout: float = 2.0,
        retries: int = 2,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = max(1, retries)

    async def request_json(self, command: int) -> dict[str, Any]:
        """
        Send a GET command and return the parsed JSON response.

        Args:
            command: UDP command number (0=ROVER, 1=EVA, 2=LTV, 3=LTV_ERRORS).

        Returns:
            Parsed JSON dict from the server.

        Raises:
            TssUdpError: On timeout, connection refusal, or invalid JSON.
        """
        packet = self._pack_get(command)
        raw = await self._send_receive(packet)
        return self._decode_json(raw)

    async def send_command(
        self, command: int, value: float
    ) -> bool:
        """
        Send a POST command with a float value and return success status.

        Args:
            command: UDP command number (e.g. 1109 for Throttle).
            value:   Float value to send.

        Returns:
            True if the server acknowledged success, False otherwise.

        Raises:
            TssUdpError: On timeout or connection issues.
        """
        packet = self._pack_post(command, value)
        raw = await self._send_receive(packet)
        if len(raw) < 4:
            return False
        status = struct.unpack("<I", raw[:4])[0]
        return status == 1

    async def close(self) -> None:
        """No persistent resources to clean up for on-demand UDP."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_get(command: int) -> bytes:
        """Pack an 8-byte GET request: [timestamp:u32][command:u32] big-endian."""
        ts = int(time.time()) & 0xFFFFFFFF
        return struct.pack(">II", ts, command)

    @staticmethod
    def _pack_post(command: int, value: float) -> bytes:
        """Pack a 12-byte POST request: [timestamp:u32][command:u32][value:f32] big-endian."""
        ts = int(time.time()) & 0xFFFFFFFF
        return struct.pack(">IIf", ts, command, value)

    async def _send_receive(self, packet: bytes) -> bytes:
        """
        Send a UDP packet and wait for the response with retries.
        Uses asyncio's low-level transport for true async I/O.
        """
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                return await asyncio.wait_for(
                    self._udp_roundtrip(packet),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                last_error = TssUdpError(
                    f"TSS UDP request timed out after {self.timeout}s "
                    f"(attempt {attempt + 1}/{self.retries}, "
                    f"target={self.host}:{self.port})"
                )
            except OSError as exc:
                last_error = TssUdpError(
                    f"TSS UDP network error: {exc} "
                    f"(target={self.host}:{self.port})"
                )

        raise last_error  # type: ignore[misc]

    async def _udp_roundtrip(self, packet: bytes) -> bytes:
        """Single send-receive cycle using asyncio datagram transport."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        class _Protocol(asyncio.DatagramProtocol):
            def __init__(self):
                self.transport: asyncio.DatagramTransport | None = None

            def connection_made(self, transport):
                self.transport = transport
                transport.sendto(packet)

            def datagram_received(self, data, addr):
                if not future.done():
                    future.set_result(data)

            def error_received(self, exc):
                if not future.done():
                    future.set_exception(
                        TssUdpError(f"Datagram error: {exc}")
                    )

            def connection_lost(self, exc):
                if not future.done() and exc:
                    future.set_exception(
                        TssUdpError(f"Connection lost: {exc}")
                    )

        transport, _ = await loop.create_datagram_endpoint(
            _Protocol,
            remote_addr=(self.host, self.port),
        )

        try:
            data = await future
            return data
        finally:
            transport.close()

    @staticmethod
    def _decode_json(raw: bytes) -> dict[str, Any]:
        """
        Decode response bytes into a JSON dict.
        Scans forward for the opening '{' to skip any binary prefix.
        """
        # Find the start of the JSON object
        start = -1
        for i, b in enumerate(raw):
            if b == ord("{"):
                start = i
                break

        if start < 0:
            # Fallback: TSS2026 responses should always start with '{',
            # but some older stacks may prepend an 8-byte binary header.
            start = 8 if len(raw) > 8 else 0

        text = raw[start:].decode("utf-8", errors="replace").strip().rstrip("\x00")

        if not text:
            raise TssUdpError("Empty response from TSS server")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TssUdpError(f"Invalid JSON from TSS: {exc}") from exc

        if not isinstance(parsed, dict):
            raise TssUdpError(
                f"Expected JSON object, got {type(parsed).__name__}"
            )
        return parsed
