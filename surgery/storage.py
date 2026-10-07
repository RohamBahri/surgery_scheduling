"""Atomic checkpoints and a persistent solve ledger."""
import hashlib
import multiprocessing
from contextlib import contextmanager, closing
import logging
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import ProcessPoolExecutor
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
        self.db = sqlite3.connect(self.root / 'checkpoints.sqlite', timeout=60)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS solves (id INTEGER PRIMARY KEY, time REAL, value TEXT)')
        self.db.commit()

    @contextmanager
    def day(self, key):
        child = Store(self.root / 'days' / digest(key)[:20])
        try:
            yield child
        finally:
            child.close()


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

    def iter_solves(self):
        for row in self.db.execute('SELECT time,value FROM solves ORDER BY id'):
            yield dict(json.loads(row[1]), logged_at=row[0])
        for path in sorted((self.root / 'days').glob('*/checkpoints.sqlite')):
            with closing(sqlite3.connect(path)) as db:
                for row in db.execute('SELECT time,value FROM solves ORDER BY id'):
                    yield dict(json.loads(row[1]), logged_at=row[0])

    def solve_rows(self):
        return list(self.iter_solves())

    def export_log(self, path):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            for row in self.iter_solves():
                stream.write(json.dumps(row, default=serial, allow_nan=False) + '\n')
        temporary.replace(path)

    def close(self):
        self.db.close()


def _day_group(job):
    function, items, root = job
    store = Store(Path(root) / 'days' / digest(items[0][1][0].key)[:20]) if root is not None else None
    try:
        return [(index, function(*args, store=store)) for index, args in items]
    finally:
        if store:
            store.close()


def map_days(function, jobs, store, workers=1):
    """One writer per day checkpoint; duplicate day jobs run in the same process."""
    groups = {}
    for index, job in enumerate(jobs):
        groups.setdefault(job[0].key, []).append((index, job))
    inputs = [(function, items, store.root if store else None) for items in groups.values()]
    completed, next_index, ready = 0, 0, {}
    context = multiprocessing.get_context('spawn')
    parallel = workers > 1
    while completed < len(inputs):
        try:
            if not parallel:
                results = map(_day_group, inputs[completed:])
                pool = None
            else:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
                results = pool.map(_day_group, inputs[completed:])
            try:
                for batch in results:
                    completed += 1
                    ready.update(batch)
                    while next_index in ready:
                        yield ready.pop(next_index)
                        next_index += 1
            finally:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
        except (BrokenProcessPool, sqlite3.OperationalError) as exc:
            if workers == 1:
                raise
            workers = max(1, workers // 2)
            logging.getLogger('surgery').warning('Worker failed (%s); resuming saved day tasks with %d worker(s)', exc, workers)
