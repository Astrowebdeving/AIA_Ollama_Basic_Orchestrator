"""
Database Logger Module
======================
Handles SQLite storage for:
  - Query logs (orchestrator /chat usage)
  - Telemetry events (streamed from external sources)
"""

import json
import aiosqlite
from datetime import datetime, timezone
from config import DB_FILE


async def init_db():
    """Create all tables if they don't exist."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                baseline_tokens INTEGER
            )
        ''')

        await db.execute('''
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT DEFAULT 'info',
                payload_json TEXT,
                description TEXT
            )
        ''')

        # Indexes for common telemetry queries
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
            ON telemetry_events (timestamp)
        ''')
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_telemetry_source
            ON telemetry_events (source)
        ''')
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_telemetry_severity
            ON telemetry_events (severity)
        ''')
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_telemetry_event_type
            ON telemetry_events (event_type)
        ''')

        await db.commit()


async def log_query(query: str, baseline_tokens: int):
    """Log an orchestrator /chat query."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO query_logs (timestamp, query, baseline_tokens) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), query, baseline_tokens),
        )
        await db.commit()


# TODO: Update this to actual Telemetry APIs
async def log_telemetry(
    source: str,
    event_type: str,
    severity: str = "info",
    payload: dict | None = None,
    description: str = "",
) -> int:
    """
    Log a telemetry event and return its row id.

    Parameters
    ----------
    source : str
        Origin of the event (e.g. "suit_sensors", "habitat_env", "crew_vitals").
    event_type : str
        Category (e.g. "temperature_reading", "pressure_alert", "heartrate").
    severity : str
        One of: "debug", "info", "warning", "critical".
    payload : dict, optional
        Arbitrary JSON data (sensor values, coordinates, etc.).
    description : str
        Human-readable summary of the event.
    """
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "INSERT INTO telemetry_events "
            "(timestamp, source, event_type, severity, payload_json, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                event_type,
                severity,
                json.dumps(payload) if payload else None,
                description,
            ),
        )
        await db.commit()
        return cursor.lastrowid
