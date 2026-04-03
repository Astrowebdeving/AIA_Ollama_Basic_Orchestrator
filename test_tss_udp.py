#!/usr/bin/env python3
"""Quick smoke-test: hit the running TSS2026 server via our new UDP client."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tss_udp_client import TssUdpClient, TssUdpError


async def main():
    host = os.getenv("TSS_UDP_HOST", "10.206.64.189")
    port = int(os.getenv("TSS_UDP_PORT", 14141))

    print(f"Testing TSS UDP client against {host}:{port}\n")
    client = TssUdpClient(host=host, port=port, timeout=2.0)

    commands = [
        (0, "ROVER"),
        (1, "EVA"),
        (2, "LTV"),
        (3, "LTV_ERRORS"),
    ]

    all_passed = True
    for cmd, label in commands:
        try:
            data = await client.request_json(cmd)
            top_keys = list(data.keys())
            print(f"  ✓ CMD {cmd} ({label}): OK — top-level keys: {top_keys}")
        except TssUdpError as exc:
            print(f"  ✗ CMD {cmd} ({label}): FAILED — {exc}")
            all_passed = False

    print()
    if all_passed:
        print("All commands succeeded! UDP client is working correctly.")
    else:
        print("Some commands failed. Check TSS server status and IP/port.")

    await client.close()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
