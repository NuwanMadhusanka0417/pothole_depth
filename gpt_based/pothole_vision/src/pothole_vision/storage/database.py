"""SQLite results database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ResultsDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                width INTEGER,
                height INTEGER,
                fps REAL,
                frame_count INTEGER,
                duration_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                track_id TEXT,
                start_frame INTEGER,
                end_frame INTEGER,
                status TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pothole_id INTEGER,
                frame_index INTEGER,
                confidence REAL,
                bbox TEXT,
                area_pixels INTEGER,
                FOREIGN KEY (pothole_id) REFERENCES potholes(id)
            );
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pothole_id INTEGER UNIQUE,
                metric_depth_available INTEGER,
                maximum_depth_cm REAL,
                width_cm REAL,
                length_cm REAL,
                overall_confidence REAL,
                measurement_status TEXT,
                json_blob TEXT,
                FOREIGN KEY (pothole_id) REFERENCES potholes(id)
            );
        """)
        self._conn.commit()

    def insert_video(self, path: str, meta: dict[str, Any]) -> int:
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO videos (path, width, height, fps, frame_count, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (path, meta["width"], meta["height"], meta["fps"], meta["frame_count"], meta["duration_seconds"]),
        )
        self._conn.commit()
        cur.execute("SELECT id FROM videos WHERE path = ?", (path,))
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
