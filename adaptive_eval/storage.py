"""Persistence: response cache + append-only session event log (SQLite).

Design A uses one SQLite file accessed from a single asyncio thread. Each write is
milliseconds, so blocking the loop briefly is acceptable here. Design B replaces
this with Postgres + Redis behind the same methods.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, run_name TEXT, model TEXT, provider TEXT,
    config TEXT, status TEXT, theta REAL, se REAL, n_items INTEGER,
    started_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, step INTEGER NOT NULL, type TEXT NOT NULL,
    item_id TEXT NOT NULL, correct INTEGER, cached INTEGER, cost_usd REAL, ts REAL,
    UNIQUE(session_id, step, type));          -- idempotency key
CREATE TABLE IF NOT EXISTS call_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, step INTEGER, attempt INTEGER, provider TEXT, model TEXT,
    item_id TEXT, outcome TEXT NOT NULL, started_at REAL NOT NULL, latency_s REAL);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)   # autocommit: each write is durable
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


class ResponseCache:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.hits = self.misses = 0

    @staticmethod
    def make_key(**fields) -> str:
        """Everything that could change the answer must be in the key: model (+version),
        item content hash, prompt template version, decoding params, sample index."""
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: dict) -> None:
        self.conn.execute("INSERT OR IGNORE INTO cache VALUES (?,?,?)",
                          (key, json.dumps(value), time.time()))


@dataclass
class SessionState:
    """Rebuilt purely from the event log. Ability is derived, never stored as truth."""
    answered: list[tuple[str, int]] = field(default_factory=list)   # (item_id, correct)
    pending: tuple[int, str] | None = None                          # (step, item_id)
    done: bool = False


class EventStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def ensure_session(self, session_id, run_name, model, provider, config: dict) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, run_name, model, provider, config,"
            " status, started_at) VALUES (?,?,?,?,?, 'running', ?)",
            (session_id, run_name, model, provider, json.dumps(config), time.time()))

    def append(self, session_id, step, type_, item_id, correct=None, cached=None, cost=None) -> bool:
        """Returns False if this (session, step, type) already exists -- safe to retry."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO events (session_id, step, type, item_id, correct, cached,"
            " cost_usd, ts) VALUES (?,?,?,?,?,?,?,?)",
            (session_id, step, type_, item_id, correct, cached, cost, time.time()))
        return cur.rowcount == 1

    def load_state(self, session_id) -> SessionState:
        st = SessionState()
        row = self.conn.execute("SELECT status FROM sessions WHERE session_id=?",
                                (session_id,)).fetchone()
        st.done = bool(row and row[0] == "done")
        selected, answers = {}, {}
        for step, type_, item_id, correct in self.conn.execute(
                "SELECT step, type, item_id, correct FROM events WHERE session_id=? ORDER BY step",
                (session_id,)):
            if type_ == "item_selected":
                selected[step] = item_id
            else:
                answers[step] = correct
        for step in sorted(selected):
            if step in answers:
                st.answered.append((selected[step], answers[step]))
            else:
                st.pending = (step, selected[step])
        return st

    def log_attempt(self, session_id, step, attempt, provider, model, item_id,
                    outcome, started_at, latency_s) -> None:
        """One row per provider call, failures included. Deliberately not idempotent:
        every attempt consumed rate limit (and maybe money), even if it gets repeated."""
        self.conn.execute(
            "INSERT INTO call_attempts (session_id, step, attempt, provider, model, item_id,"
            " outcome, started_at, latency_s) VALUES (?,?,?,?,?,?,?,?,?)",
            (session_id, step, attempt, provider, model, item_id, outcome, started_at, latency_s))

    def finish(self, session_id, theta, se, n_items) -> None:
        self.conn.execute(
            "UPDATE sessions SET status='done', theta=?, se=?, n_items=?, finished_at=?"
            " WHERE session_id=?", (theta, se, n_items, time.time(), session_id))
