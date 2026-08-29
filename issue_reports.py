#!/usr/bin/env python3
"""
ISSUE REPORTS — stores bug reports submitted from the Help page.
==================================================================
Since this app doesn't have SMTP/email credentials configured, a report
is:
  1. Saved locally to reports/index.json (survives local `streamlit run`
     use; NOT persistent on free-tier hosting, same caveat as
     session_store.py's sessions/).
  2. Also offered as a pre-filled mailto: link, so the person can send it
     directly from their own email client to DEVELOPER_EMAIL below — this
     works with zero server-side email setup.

SET THIS before deploying:
"""
DEVELOPER_EMAIL = "YOUR_EMAIL_HERE@example.com"  # <-- Mona: put your real email here

import json
import os
import time
import urllib.parse
from pathlib import Path

REPORTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "reports"
INDEX_FILE = REPORTS_DIR / "index.json"


def _ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("[]")


def load_reports():
    _ensure_dirs()
    try:
        return json.loads(INDEX_FILE.read_text())
    except Exception:
        return []


def _save_index(reports):
    INDEX_FILE.write_text(json.dumps(reports, indent=2))


def save_report(name, contact_email, description, page_context=""):
    """Saves a report locally and returns (record, mailto_url)."""
    _ensure_dirs()
    record = {
        "id": time.strftime("%Y%m%d_%H%M%S"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name or "Anonymous",
        "contact_email": contact_email or "",
        "description": description,
        "page_context": page_context,
    }
    reports = load_reports()
    reports.insert(0, record)
    _save_index(reports)

    subject = f"Sync.AI issue report from {record['name']}"
    body_lines = [
        f"Name: {record['name']}",
        f"Contact email: {record['contact_email'] or '(not provided)'}",
        f"Page: {page_context or '(not specified)'}",
        "",
        "Description:",
        description,
    ]
    body = "\n".join(body_lines)
    mailto_url = (
        f"mailto:{DEVELOPER_EMAIL}"
        f"?subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(body)}"
    )
    return record, mailto_url
