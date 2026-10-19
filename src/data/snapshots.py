"""
Point-in-time snapshot store for Screener fundamentals.

Why this exists: the Screener cache is a single JSON file per company, and every
refresh overwrites it. That throws away exactly what a quant model needs —
what the data looked like on a given date. Without history there is no
backtest, no "what changed since last quarter", and no record of how a verdict
moved.

Design
------
* One SQLite file, one table. SQLite ships with Python, needs no server, and a
  few hundred snapshots of ~50 KB each is trivially small.
* A snapshot is written only when the *financial content* changes, identified
  by a hash of the four statement sections. Refreshing unchanged data is a no-op,
  and `current_price` (which moves daily) is deliberately excluded from the hash.
* `fetched_at` is when we observed the data, not when the company filed it.
  Backtests must still apply a reporting lag on top (see src/data/company.py).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB = Path("data/snapshots/screener.db")

# Sections whose change constitutes a new snapshot. Price and peers are noise.
_HASHED_SECTIONS = ("profit_loss", "balance_sheet", "cash_flow", "ratios")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    symbol      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,          -- ISO-8601 UTC
    data_hash   TEXT NOT NULL,
    payload     TEXT NOT NULL,          -- full Screener JSON
    PRIMARY KEY (symbol, data_hash)
);
CREATE INDEX IF NOT EXISTS idx_snap_symbol_time ON snapshots (symbol, fetched_at);
"""


def content_hash(data: dict[str, Any]) -> str:
    """Stable hash of the financial statements only."""
    core = {k: data.get(k) for k in _HASHED_SECTIONS}
    blob = json.dumps(core, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class SnapshotStore:
    """Append-only history of Screener payloads, de-duplicated by content."""

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, symbol: str, data: dict[str, Any],
             fetched_at: Optional[datetime] = None) -> bool:
        """
        Record a snapshot. Returns True if it was new content, False if identical
        content was already stored (in which case nothing is written).
        """
        when = (fetched_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        h    = content_hash(data)
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO snapshots (symbol, fetched_at, data_hash, payload) "
                "VALUES (?, ?, ?, ?)",
                (symbol, when, h, json.dumps(data, default=str)),
            )
            return cur.rowcount == 1

    # ── Read ──────────────────────────────────────────────────────────────────

    def history(self, symbol: str) -> list[dict[str, str]]:
        """All snapshots for a symbol, oldest first (metadata only)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT fetched_at, data_hash FROM snapshots "
                "WHERE symbol = ? ORDER BY fetched_at",
                (symbol,),
            ).fetchall()
        return [{"fetched_at": t, "data_hash": h} for t, h in rows]

    def as_of(self, symbol: str, when: datetime | str) -> Optional[dict[str, Any]]:
        """The newest snapshot observed on or before `when`, or None."""
        cutoff = when.isoformat(timespec="seconds") if isinstance(when, datetime) else when
        with self._connect() as con:
            row = con.execute(
                "SELECT payload FROM snapshots WHERE symbol = ? AND fetched_at <= ? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (symbol, cutoff),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def latest(self, symbol: str) -> Optional[dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                "SELECT payload FROM snapshots WHERE symbol = ? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def symbols(self) -> list[str]:
        with self._connect() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM snapshots")]

    def count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]


def backfill_from_cache(cache_dir: str | Path = "data/raw/screener",
                        store: Optional[SnapshotStore] = None) -> int:
    """
    Seed the store from the current Screener JSON files, dated by file mtime.

    Idempotent: content already stored is skipped. Returns snapshots added.
    """
    store = store or get_snapshot_store()
    added = 0
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not any((data.get(s) or {}).get("rows") for s in _HASHED_SECTIONS):
            continue
        when = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        added += store.save(path.stem, data, fetched_at=when)
    return added


_store: Optional[SnapshotStore] = None
_store_lock = threading.Lock()


def get_snapshot_store() -> SnapshotStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SnapshotStore()
    return _store
