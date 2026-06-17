"""
Async Bulk Processing for Resume Parser.

Provides SQLite-backed job queue with background processing.
Zero external dependencies — uses built-in sqlite3 + threading.
Process-safe across Gunicorn workers via SQLite WAL mode.
"""

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid

from groq_parser import parse_resume, extract_text_from_file

# --- Configuration ---
DATA_DIR = os.environ.get("BULK_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DB_PATH = os.path.join(DATA_DIR, "jobs.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
RATE_LIMIT_INTERVAL = float(os.environ.get("BULK_RATE_INTERVAL", "2.0"))
JOB_TTL_HOURS = float(os.environ.get("BULK_JOB_TTL_HOURS", "24"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    total_files INTEGER NOT NULL,
    completed_files INTEGER NOT NULL DEFAULT 0,
    failed_files INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS job_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result_json TEXT,
    error TEXT,
    processing_time_ms INTEGER,
    created_at REAL NOT NULL,
    completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_job_files_job_id ON job_files(job_id);
CREATE INDEX IF NOT EXISTS idx_job_files_status ON job_files(status);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


class JobStore:
    """Process-safe SQLite storage for bulk parsing jobs."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self):
        """One connection per thread (SQLite requirement)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def create_job(self, files_info):
        """Create a new job with file records.

        Args:
            files_info: list of (original_filename, stored_path) tuples

        Returns:
            job_id string
        """
        job_id = uuid.uuid4().hex
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO jobs (id, status, total_files, created_at, updated_at) "
            "VALUES (?, 'processing', ?, ?, ?)",
            (job_id, len(files_info), now, now),
        )
        for filename, stored_path in files_info:
            conn.execute(
                "INSERT INTO job_files (job_id, filename, stored_path, status, created_at) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (job_id, filename, stored_path, now),
            )
        conn.commit()
        return job_id

    def get_job_status(self, job_id):
        """Return job status dict for polling endpoint."""
        conn = self._get_conn()
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            return None
        total = max(job["total_files"], 1)
        done = job["completed_files"] + job["failed_files"]
        return {
            "job_id": job["id"],
            "status": job["status"],
            "total_files": job["total_files"],
            "completed_files": job["completed_files"],
            "failed_files": job["failed_files"],
            "progress_pct": round(done / total * 100, 1),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "completed_at": job["completed_at"],
        }

    def claim_next_file(self):
        """Atomically claim one pending file for processing.

        Returns sqlite3.Row or None.
        """
        conn = self._get_conn()
        # Find a pending file from any processing job
        row = conn.execute(
            "SELECT jf.id, jf.job_id, jf.filename, jf.stored_path "
            "FROM job_files jf "
            "JOIN jobs j ON jf.job_id = j.id "
            "WHERE jf.status = 'pending' AND j.status = 'processing' "
            "ORDER BY jf.created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        # Attempt atomic claim
        cursor = conn.execute(
            "UPDATE job_files SET status = 'processing' "
            "WHERE id = ? AND status = 'pending'",
            (row["id"],),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None  # another worker claimed it
        return row

    def complete_file(self, file_id, job_id, result_json=None, error=None, processing_time_ms=0):
        """Mark a file as completed/failed and update job counters."""
        now = time.time()
        status = "failed" if error else "completed"
        conn = self._get_conn()
        conn.execute(
            "UPDATE job_files SET status=?, result_json=?, error=?, "
            "processing_time_ms=?, completed_at=? WHERE id=?",
            (status, result_json, error, processing_time_ms, now, file_id),
        )
        if error:
            conn.execute(
                "UPDATE jobs SET failed_files = failed_files + 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET completed_files = completed_files + 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        conn.commit()
        self._maybe_complete_job(job_id)

    def _maybe_complete_job(self, job_id):
        """If all files are done, mark job completed and write results file."""
        conn = self._get_conn()
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job or job["status"] != "processing":
            return
        done = job["completed_files"] + job["failed_files"]
        if done >= job["total_files"]:
            now = time.time()
            conn.execute(
                "UPDATE jobs SET status='completed', completed_at=?, updated_at=? WHERE id=?",
                (now, now, job_id),
            )
            conn.commit()
            self._write_results_file(job_id)

    def _write_results_file(self, job_id):
        """Assemble all file results into a single JSON file."""
        conn = self._get_conn()
        files = conn.execute(
            "SELECT * FROM job_files WHERE job_id = ? ORDER BY created_at",
            (job_id,),
        ).fetchall()
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

        results = []
        for f in files:
            entry = {
                "filename": f["filename"],
                "status": f["status"],
                "processing_time_ms": f["processing_time_ms"],
            }
            if f["result_json"]:
                try:
                    entry["result"] = json.loads(f["result_json"])
                except json.JSONDecodeError:
                    entry["result"] = f["result_json"]
            if f["error"]:
                entry["error"] = f["error"]
            results.append(entry)

        output = {
            "job_id": job_id,
            "total_files": job["total_files"],
            "successful": job["completed_files"],
            "failed": job["failed_files"],
            "results": results,
        }

        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"{job_id}.json")
        with open(path, "w") as fp:
            json.dump(output, fp, indent=2, ensure_ascii=False)

    def get_results(self, job_id):
        """Load completed results from file."""
        path = os.path.join(RESULTS_DIR, f"{job_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as fp:
            return json.load(fp)

    def cleanup_old_jobs(self, ttl_hours=None):
        """Delete jobs older than ttl_hours and their files."""
        ttl = ttl_hours if ttl_hours is not None else JOB_TTL_HOURS
        cutoff = time.time() - (ttl * 3600)
        conn = self._get_conn()
        old_jobs = conn.execute(
            "SELECT id FROM jobs WHERE created_at < ?", (cutoff,)
        ).fetchall()
        for job in old_jobs:
            jid = job["id"]
            upload_dir = os.path.join(UPLOAD_DIR, jid)
            if os.path.isdir(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
            results_file = os.path.join(RESULTS_DIR, f"{jid}.json")
            if os.path.exists(results_file):
                try:
                    os.remove(results_file)
                except OSError:
                    pass
            conn.execute("DELETE FROM job_files WHERE job_id = ?", (jid,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (jid,))
        conn.commit()

    def recover_stuck_files(self, stuck_minutes=10):
        """Reset files stuck in 'processing' back to 'pending'."""
        cutoff = time.time() - (stuck_minutes * 60)
        conn = self._get_conn()
        conn.execute(
            "UPDATE job_files SET status = 'pending' "
            "WHERE status = 'processing' AND created_at < ?",
            (cutoff,),
        )
        conn.commit()


class BulkProcessor:
    """Background daemon thread that processes bulk parse jobs."""

    def __init__(self, store, rate_interval=None):
        self.store = store
        self.rate_interval = rate_interval if rate_interval is not None else RATE_LIMIT_INTERVAL
        self._poll_interval = 5.0
        self._cleanup_interval = 3600
        self._last_cleanup = time.time()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True, name="BulkProcessor")
        t.start()

    def _run(self):
        while True:
            try:
                # Periodic cleanup
                if time.time() - self._last_cleanup > self._cleanup_interval:
                    self.store.cleanup_old_jobs()
                    self.store.recover_stuck_files()
                    self._last_cleanup = time.time()

                file_row = self.store.claim_next_file()
                if file_row:
                    self._process_one(file_row)
                    time.sleep(self.rate_interval)
                else:
                    time.sleep(self._poll_interval)
            except Exception as e:
                print(f"[BulkProcessor] error: {e}")
                time.sleep(self._poll_interval)

    def _process_one(self, file_row):
        """Process a single file: extract text, call Groq, store result."""
        start = time.time()
        file_id = file_row["id"]
        job_id = file_row["job_id"]
        filepath = file_row["stored_path"]

        try:
            resume_text = extract_text_from_file(filepath)
            if not resume_text or len(resume_text.strip()) < 50:
                self.store.complete_file(
                    file_id, job_id, error="Could not extract text from file",
                    processing_time_ms=int((time.time() - start) * 1000),
                )
                return

            result = parse_resume(resume_text)
            elapsed = int((time.time() - start) * 1000)

            if "error" in result and "_metadata" not in result:
                self.store.complete_file(
                    file_id, job_id,
                    result_json=json.dumps(result),
                    error=result["error"],
                    processing_time_ms=elapsed,
                )
            else:
                self.store.complete_file(
                    file_id, job_id,
                    result_json=json.dumps(result, ensure_ascii=False),
                    processing_time_ms=elapsed,
                )
        except Exception as e:
            self.store.complete_file(
                file_id, job_id, error=str(e),
                processing_time_ms=int((time.time() - start) * 1000),
            )


def init_bulk_processing(app):
    """Initialize directories, database, and background processor.

    Call from app.py after creating the Flask app.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    store = JobStore()
    store.recover_stuck_files()

    processor = BulkProcessor(store)
    processor.start()

    app.bulk_store = store
    return store
