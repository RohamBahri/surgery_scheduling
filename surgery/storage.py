"""Atomic checkpoints and a persistent solve ledger."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serial,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + '\n')
    temporary.replace(path)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'checkpoints.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS solves (id INTEGER PRIMARY KEY, time REAL, value TEXT)')
        self.db.commit()

    def get(self, key):
        row = self.db.execute('SELECT value FROM tasks WHERE key=?', (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO tasks VALUES (?,?)',
                            (key, json.dumps(value, default=serial, allow_nan=False)))

    def log(self, value):
        with self.db:
            self.db.execute('INSERT INTO solves(time,value) VALUES (?,?)',
                            (time.time(), json.dumps(value, default=serial, allow_nan=False)))

    def solve_rows(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT value FROM solves ORDER BY id')]

    def export_log(self, path):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            for row in self.db.execute('SELECT value FROM solves ORDER BY id'):
                stream.write(row[0] + '\n')
        temporary.replace(path)

    def close(self):
        self.db.close()
