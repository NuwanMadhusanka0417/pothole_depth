"""SQLite results database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ResultsDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                width INTEGER,
                height INTEGER,
                fps REAL,
                frame_count INTEGER,
                duration_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY,
                pothole_id TEXT UNIQUE,
                video_id INTEGER,
                start_frame INTEGER,
                end_frame INTEGER,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                pothole_id TEXT,
                frame_index INTEGER,
                confidence REAL,
                bbox TEXT,
                area_pixels INTEGER
            );
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY,
                pothole_id TEXT UNIQUE,
                metric_depth_available INTEGER,
                maximum_depth_cm REAL,
                width_cm REAL,
                length_cm REAL,
                overall_confidence REAL,
                measurement_status TEXT,
                json_blob TEXT
            );
        """)
        self._conn.commit()

    def insert_video(self, path: str, meta: dict) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO videos (path, width, height, fps, frame_count, duration_seconds) VALUES (?,?,?,?,?,?)",
            (path, meta["width"], meta["height"], meta["fps"], meta["frame_count"], meta["duration_seconds"]),
        )
        self._conn.commit()
        return cur.lastrowid

    def insert_measurement(self, pothole_id: str, data: dict[str, Any]) -> None:
        import json
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO measurements
            (pothole_id, metric_depth_available, maximum_depth_cm, width_cm, length_cm,
             overall_confidence, measurement_status, json_blob)
            VALUES (?,?,?,?,?,?,?,?)""",
            (
                pothole_id,
                int(data.get("metric_depth_available", False)),
                data.get("maximum_depth_cm"),
                data.get("width_cm"),
                data.get("length_cm"),
                data.get("overall_confidence"),
                data.get("measurement_status"),
                json.dumps(data),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
