import aiosqlite
from datetime import datetime
from config import DB_FILE

async def init_db():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                baseline_tokens INTEGER
            )
        ''')
        await db.commit()

async def log_query(query: str, baseline_tokens: int):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO query_logs (timestamp, query, baseline_tokens) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), query, baseline_tokens)
        )
        await db.commit()
