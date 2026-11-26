"""Where accounts live: one SQLite file with a users table and a small key-value table.

The key-value table holds everything short-lived (login sessions, e-mail codes, reset
tokens, rate-limit counters). Each row has an expiry time, so a restart never logs
anyone out and old rows are ignored once they expire.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    verified_at   TEXT,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS kv (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    expires REAL NOT NULL
);
"""


class KV(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...
    def delete(self, key: str) -> None: ...
    def incr(self, key: str, ttl_seconds: int) -> int: ...


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


class SqliteUsers:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.db = connection
        self.lock = threading.Lock()

    def _one(self, sql: str, value: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute(sql, (value,)).fetchone()
        return dict(row) if row else None

    def _write(self, sql: str, values: tuple[Any, ...]) -> None:
        with self.lock:
            self.db.execute(sql, values)
            self.db.commit()

    def by_email(self, email: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM users WHERE email = ?", email)

    def by_id(self, user_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM users WHERE id = ?", user_id)

    def create(self, user_id: str, email: str, name: str, password_hash: str) -> None:
        sql = "INSERT INTO users (id, email, name, password_hash) VALUES (?, ?, ?, ?)"
        self._write(sql, (user_id, email, name, password_hash))

    def set_password(self, user_id: str, password_hash: str) -> None:
        self._write("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))

    def set_verified(self, user_id: str) -> None:
        sql = "UPDATE users SET verified_at = CURRENT_TIMESTAMP WHERE id = ? AND verified_at IS NULL"
        self._write(sql, (user_id,))


class SqliteKV:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.db = connection
        self.lock = threading.Lock()

    def _alive(self, key: str) -> tuple[str, float] | None:
        row = self.db.execute("SELECT value, expires FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None or row["expires"] < time.time():
            return None
        return row["value"], row["expires"]

    def get(self, key: str) -> str | None:
        with self.lock:
            found = self._alive(key)
        return found[0] if found else None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        with self.lock:
            row = (key, value, time.time() + ttl_seconds)
            self.db.execute("REPLACE INTO kv (key, value, expires) VALUES (?, ?, ?)", row)
            self.db.execute("DELETE FROM kv WHERE expires < ?", (time.time(),))
            self.db.commit()

    def delete(self, key: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM kv WHERE key = ?", (key,))
            self.db.commit()

    def incr(self, key: str, ttl_seconds: int) -> int:
        """Add one to a counter. The window starts at the first hit and is not extended by later ones."""
        with self.lock:
            found = self._alive(key)
            count = int(found[0]) + 1 if found else 1
            expires = found[1] if found else time.time() + ttl_seconds
            self.db.execute("REPLACE INTO kv (key, value, expires) VALUES (?, ?, ?)", (key, str(count), expires))
            self.db.commit()
        return count


class MemoryKV:
    """Same four methods, kept in a dict. Used by the tests."""

    def __init__(self) -> None:
        self.data: dict[str, tuple[str, float]] = {}

    def _alive(self, key: str) -> tuple[str, float] | None:
        found = self.data.get(key)
        return found if found and found[1] >= time.time() else None

    def get(self, key: str) -> str | None:
        found = self._alive(key)
        return found[0] if found else None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.data[key] = (value, time.time() + ttl_seconds)

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def incr(self, key: str, ttl_seconds: int) -> int:
        found = self._alive(key)
        count = int(found[0]) + 1 if found else 1
        self.data[key] = (str(count), found[1] if found else time.time() + ttl_seconds)
        return count
