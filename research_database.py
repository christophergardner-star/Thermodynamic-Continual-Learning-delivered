"""
SQLite-backed research trace store for the Cruxy loop.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class ResearchEntry:
    session: str
    task_type: str
    prompt: str
    response: str
    score: float = 0.0
    success: bool = False
    metadata: dict = field(default_factory=dict)
    created_at: Optional[str] = None
    id: Optional[int] = None


class ResearchDatabase:
    def __init__(self, path: str = "research_db.sqlite"):
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                session TEXT NOT NULL,
                task_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                score REAL NOT NULL,
                success INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_entry(self, entry: ResearchEntry) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO research_entries (
                session, task_type, prompt, response, score, success, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.session,
                entry.task_type,
                entry.prompt,
                entry.response,
                float(entry.score),
                int(entry.success),
                json.dumps(entry.metadata),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def iter_entries(self, session: Optional[str] = None) -> Iterable[ResearchEntry]:
        sql = "SELECT * FROM research_entries"
        params = ()
        if session:
            sql += " WHERE session = ?"
            params = (session,)
        sql += " ORDER BY id ASC"
        for row in self.conn.execute(sql, params):
            yield ResearchEntry(
                id=row["id"],
                created_at=row["created_at"],
                session=row["session"],
                task_type=row["task_type"],
                prompt=row["prompt"],
                response=row["response"],
                score=row["score"],
                success=bool(row["success"]),
                metadata=json.loads(row["metadata_json"]),
            )

    def high_quality_entries(self, min_score: float = 0.5) -> list[ResearchEntry]:
        return [entry for entry in self.iter_entries() if entry.score >= min_score]

    def export_jsonl(self, output_path: str, min_score: float = 0.0) -> int:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("w", encoding="utf-8") as handle:
            for entry in self.iter_entries():
                if entry.score < min_score:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "session": entry.session,
                            "task_type": entry.task_type,
                            "prompt": entry.prompt,
                            "response": entry.response,
                            "score": entry.score,
                            "success": entry.success,
                            "metadata": entry.metadata,
                        }
                    )
                    + "\n"
                )
                count += 1
        return count

    def close(self) -> None:
        self.conn.close()
