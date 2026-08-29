#!/usr/bin/env python3
"""
SESSION STORE — persists past scoring sessions to disk.
=========================================================
Each time app.py finishes scoring a figure, it calls save_session() with
the swimmer ID, the scorer's result dict, and the paths to the video/CSV
files that were just produced (still inside the temp dir at that point).
This copies those files into a permanent `sessions/` folder and appends a
summary record to `sessions/index.json`, so a "Previous Sessions" list can
survive after the temp dir is cleaned up.

NOTE on hosting: if this app is deployed to a free-tier Hugging Face
Space, the Space's disk is NOT persistent across restarts/sleep — sessions
saved here will be lost when the Space goes to sleep and wakes back up.
For long-term storage, back this with a real database or object storage
instead. For local `streamlit run app.py` use, this just works as a
normal folder on disk.
"""

import json
import os
import shutil
import time
from pathlib import Path

SESSIONS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "sessions"
INDEX_FILE = SESSIONS_DIR / "index.json"

# Deduction dict keys that are metadata, not a deduction amount itself —
# skipped when building the "top issues" summary string.
_NON_DEDUCTION_SUFFIXES = ("_abs", "_rel", "_degrees", "_value")


def _ensure_dirs():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("[]")


def load_sessions():
    """Newest-first list of session summary dicts."""
    _ensure_dirs()
    try:
        return json.loads(INDEX_FILE.read_text())
    except Exception:
        return []


def _save_index(sessions):
    INDEX_FILE.write_text(json.dumps(sessions, indent=2))


def _top_issues_summary(deductions, keys, n=3):
    scored = [
        (k, v) for k, v in deductions.items()
        if k in keys and isinstance(v, (int, float)) and v > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return "No major deductions"
    return ", ".join(f"{k.replace('_', ' ')} -{v:.2f}" for k, v in scored[:n])


def save_session(swimmer_id, mode, score_result, file_paths, official_keys):
    """
    swimmer_id: str
    mode: 'Walticam' / 'Above-Water' / 'Underwater' / 'Above+Below'
    score_result: the dict returned by BarracudaScorer.score_figure() /
        score_single_pair() (has 'score', 'base_score', 'total_deduction',
        'deductions')
    file_paths: dict, any of {"video", "above_video", "below_video",
        "above_csv", "below_csv"} -> path (existing paths only get copied)
    official_keys: list of deduction keys that count toward the score
        (pass BarracudaScorer()._deduction_keys() from the caller) — used
        just to build the "top issues" summary line.
    """
    _ensure_dirs()

    session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{(swimmer_id or 'figure').replace(' ', '_')}"
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}
    for key, path in (file_paths or {}).items():
        if path and Path(path).exists():
            dest = session_dir / Path(path).name
            shutil.copy(path, dest)
            saved_files[key] = f"{session_id}/{dest.name}"

    deductions = score_result.get("deductions", {})
    summary = _top_issues_summary(deductions, official_keys)

    record = {
        "id": session_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "swimmer_id": swimmer_id or "Unknown",
        "mode": mode,
        "score": score_result.get("score"),
        "base_score": score_result.get("base_score"),
        "total_deduction": score_result.get("total_deduction"),
        "summary": summary,
        "files": saved_files,
    }

    sessions = load_sessions()
    sessions.insert(0, record)  # newest first
    _save_index(sessions)
    return record


def delete_session(session_id):
    sessions = [s for s in load_sessions() if s["id"] != session_id]
    _save_index(sessions)
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)


def clear_all_sessions():
    _save_index([])
    if SESSIONS_DIR.exists():
        for entry in SESSIONS_DIR.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)


def session_file_path(relative_path):
    """relative_path is the value stored in a record's files dict, e.g.
    '20260818_1530_Swimmer12/foo_above_tracking.mp4'."""
    return SESSIONS_DIR / relative_path
