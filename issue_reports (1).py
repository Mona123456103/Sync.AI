#!/usr/bin/env python3
"""
ISSUE REPORTS — persists Help-page bug reports to disk, and builds a
mailto: link so a report can also reach the developer by email even on
hosting where the disk doesn't persist (see session_store.py's note on
free-tier Streamlit Community Cloud / Hugging Face Spaces).

app.py calls:
    save_report(name, contact_email, description, page_context)
        -> (record, mailto_url)
    load_reports()
        -> newest-first list of report dicts

Same on-disk pattern as session_store.py: a JSON index file in a folder
next to app.py.
"""

import json
import os
import time
import urllib.parse
from pathlib import Path

REPORTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "reports"
INDEX_FILE = REPORTS_DIR / "index.json"

# Where the "email this report" link is addressed to. Change this to the
# address that should actually receive bug reports.
DEVELOPER_EMAIL = "your-email@example.com"


def _ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("[]")


def load_reports():
    """Newest-first list of report dicts."""
    _ensure_dirs()
    try:
        return json.loads(INDEX_FILE.read_text())
    except Exception:
        return []


def _save_index(reports):
    INDEX_FILE.write_text(json.dumps(reports, indent=2))


def _build_mailto(name, contact_email, description, page_context):
    subject = f"Sync.AI issue report — {page_context}"
    body_lines = [
        f"Page: {page_context}",
        f"Name: {name or '(not given)'}",
        f"Reply-to: {contact_email or '(not given)'}",
        "",
        "Description:",
        description,
    ]
    body = "\n".join(body_lines)
    query = urllib.parse.urlencode({"subject": subject, "body": body})
    return f"mailto:{DEVELOPER_EMAIL}?{query}"


def save_report(name, contact_email, description, page_context):
    """
    name: str, optional
    contact_email: str, optional
    description: str, required (caller already validates non-empty)
    page_context: str, one of the Help-page selectbox options

    Returns (record, mailto_url).
    """
    _ensure_dirs()

    record = {
        "id": time.strftime("%Y%m%d_%H%M%S") + f"_{len(load_reports())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name or "Anonymous",
        "contact_email": contact_email or "",
        "page_context": page_context,
        "description": description,
    }

    reports = load_reports()
    reports.insert(0, record)  # newest first
    _save_index(reports)

    mailto_url = _build_mailto(name, contact_email, description, page_context)
    return record, mailto_url
