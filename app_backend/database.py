"""
database.py

Very small SQLite layer for the two things the app actually needs
persisted server-side: user accounts (for cross-device sign in) and
health profiles (for cross-device sync of the health diary).

SQLite (not Postgres/MySQL/Firebase) is a deliberate choice here, not
a shortcut:
  - PythonAnywhere's free tier does not allow outbound connections to
    external managed databases, so a hosted Postgres/MySQL instance
    would not work on the free plan anyway.
  - The whole dataset for an FYP demo (a handful of test users) is
    tiny -- SQLite handles this without any extra moving parts.
  - It's a single file, so it's trivial to explain, inspect, and back
    up during a viva ("here is wellness_saheli.db, here's the schema").

If you outgrow this later (e.g. real multi-user production load),
swap DB_PATH's sqlite3 connection for a Postgres connection behind the
same functions -- nothing above this file needs to change.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "wellness_saheli.db"


def init_db() -> None:
    """Creates the users and profiles tables if they don't exist yet.
    Safe to call every time the app starts (CREATE TABLE IF NOT EXISTS)."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_connection():
    """Context-managed SQLite connection. Using a fresh connection per
    request (rather than one global connection) avoids the classic
    'SQLite object created in one thread used in another' error that
    FastAPI's threaded request handling would otherwise trigger."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------
# Users
# ---------------------------------------------------------------

def get_user_by_email(email: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()


def create_user(name: str, email: str, password_hash: str) -> sqlite3.Row:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash),
        )
        conn.commit()
        return conn.execute(
            "SELECT * FROM users WHERE id = ?", (cur.lastrowid,)
        ).fetchone()


def update_password(email: str, new_password_hash: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (new_password_hash, email),
        )
        conn.commit()


# ---------------------------------------------------------------
# Health profiles
# ---------------------------------------------------------------
# Stored as a single JSON blob per user_id rather than a normalized
# relational schema. This matches the Flutter side deliberately --
# lib/models/health_profile.dart's own docstring says the model is
# "intentionally a loose, nested JSON-friendly structure ... because
# the fields here will keep growing" and that adding a field "just
# means adding it to the map, no migration needed on the backend."
# A normalized schema (separate tables per nested object) would fight
# that design instead of matching it, and would need a migration every
# time a field is added to the Dart model -- not worth it for an FYP.

def get_profile(user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT data FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None


def upsert_profile(user_id: str, profile_json: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO profiles (user_id, data, last_updated)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                data = excluded.data,
                last_updated = excluded.last_updated
            """,
            (user_id, json.dumps(profile_json), profile_json.get("last_updated", "")),
        )
        conn.commit()