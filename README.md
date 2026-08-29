---
title: Barracuda Above-Water Tracker
emoji: 🏊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Barracuda Tracker + Scorer (Web App)

Upload synchronized swimming footage (WaltiCam split-screen, or separate
above/below videos) and get back:
- An annotated tracking video
- Downloadable CSV tracking data
- A live FINA-style barracuda score with a deduction breakdown

## Files

- `app.py` — Streamlit UI (upload, settings, progress, download, scoring)
- `tracker_core.py` — the tracking engine (tent masking, waterline detection,
  swimmer locking, smoothing, Kalman filtering)
- `scorer.py` — the FINA-aligned barracuda scorer (height-based base score +
  vertical alignment / backpike / leg extension deductions)
- `requirements.txt` / `packages.txt` — dependencies

## What's new: automatic scoring

After a video (or video pair) finishes processing, the app now automatically
scores the figure and displays:
- Final score out of 10
- Base score (from jump height / foot clearance)
- Total deduction
- A breakdown table: ascent alignment, descent alignment, backpike, and leg
  extension (measured but not counted, matching your original scorer)

This uses `BarracudaScorer.score_single_pair(...)`, a small addition to your
existing `scorer.py` that scores one figure directly from the tracking CSVs
the app just produced — no folder of files needed, and no change to your
original `score_all()` / `print_summary_table()` / `save_html_report()`
batch workflow, which still works exactly as before if you run
`scorer.py` directly against a folder of CSVs.

## Running locally first (recommended before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Deploy to Hugging Face Spaces (free)

1. Go to https://huggingface.co/new-space
2. Name your Space (e.g. `barracuda-tracker`)
3. **SDK**: Streamlit, **Hardware**: free "CPU basic" tier, **Visibility**: Public
4. Click "Create Space"
5. Upload all files in this folder (`app.py`, `tracker_core.py`, `scorer.py`,
   `requirements.txt`, `packages.txt`) via the Files tab, or push with git:

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/barracuda-tracker
   cp app.py tracker_core.py scorer.py requirements.txt packages.txt barracuda-tracker/
   cd barracuda-tracker
   git add .
   git commit -m "Add scoring step"
   git push
   ```

6. The Space builds automatically (a few minutes; RTMPose model files also
   download on first use).
7. Your app is live at:
   `https://huggingface.co/spaces/YOUR_USERNAME/barracuda-tracker`

## Session history (new)

Every scored figure now gets saved automatically to a `sessions/` folder
(created next to `app.py`), with:
- The video(s) and CSV(s) that were produced
- A summary: swimmer ID, timestamp, score, base score, total deduction,
  and the top deduction categories

This shows up as a **"Previous Sessions" panel in the sidebar** — expand
any entry to see the summary and re-download its files, or delete it.

**Important limitation:** on a free-tier Hugging Face Space, the disk is
**not persistent** — sessions are lost whenever the Space restarts or goes
to sleep (after ~48h of inactivity). This works reliably for local
`streamlit run app.py` use. For durable history on a hosted deployment,
swap `session_store.py`'s file-based storage for a real database or
object storage (S3, etc.) — the function signatures (`save_session`,
`load_sessions`, `delete_session`) are written so that swap only touches
one file.

## Known issues being worked on

- **Base (height) score can be off.** It's computed from *foot clearance*
  — how far the ankles rise above the waterline, as a fraction of frame
  height — which is camera-distance/zoom dependent, not a real-world
  measurement. If your recordings vary in camera distance/zoom, the same
  physical jump height will produce different foot-clearance numbers and
  therefore different base scores. Each score's "🔍 Debug info" expander
  now shows the raw `foot_clearance` value feeding into the base score,
  to help pin down whether a specific video's setup is the cause.
- **Tracker could lock onto the wrong person** (e.g. a reflection in a
  window) for part of a clip. `tracker_core.py`'s swimmer-matching logic
  was tightened to re-validate (position, anatomy, in-water check) instead
  of trusting proximity-to-last-frame alone, plus a periodic re-check
  every ~1.5s to catch slow drift. If misidentification still happens on
  a particular video, that's useful signal for further tuning
  `ignore_top_percent`, the position bounds, or the water-color check.


- **Hardware**: 2 vCPU, 16GB RAM, 50GB (non-persistent) disk — no cost.
- **Sleep**: free Spaces sleep after 48 hours of no traffic, ~30s cold start
  on next visit.
- **Speed**: CPU-only inference. "Fast" mode is default; "Most Accurate" is
  available in the sidebar but slower per video.

## .gitignore suggestion

If you push this to a git repo/Space, add:
```
sessions/
__pycache__/
*.pyc
```
so scored videos/CSVs and their session metadata don't get committed.
