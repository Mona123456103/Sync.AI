#!/usr/bin/env python3
"""
Sync.AI — Web App
===========================================================================
Streamlit front-end for tracker_core.py + scorer.py.

Pages (session_state-based routing, no reliance on newer st.navigation
APIs, so this works across Streamlit versions):
  - Home     : landing page, explains the product, links to Analyze
  - Analyze  : the actual tracking + scoring workflow (Walticam /
               Just Above / Above + Below)
  - History  : previously scored sessions
  - Help     : report an issue

Run locally:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import time
import pandas as pd
import io
import zipfile
from pathlib import Path

import tracker_core as tc
from scorer import BarracudaScorer
import session_store
import issue_reports

APP_NAME = "Sync.AI"
APP_DIR = Path(__file__).resolve().parent

DEDUCTION_LABELS = {
    "ascent_alignment": "Ascent alignment",
    "descent_alignment": "Descent alignment",
    "backpike": "Backpike",
    "leg_extension": "Leg extension",
    "ankle_extension": "Ankle extension",
    "back_roundness": "Back roundness",
    "travel": "Travel",
    "unroll_speed": "Unroll speed",
    "head_tuck": "Head tuck",
}

# Speed fix — loading the pose model is the single biggest cost in this
# app, and it was happening from scratch on EVERY "Process video" click,
# even for the second, third, tenth video of the same session, and TWICE
# per click for Above+Below mode (once for the above tracker, once for
# underwater — same model, same settings, loaded independently). Cached
# with st.cache_resource, which is Streamlit's standard mechanism for
# exactly this — expensive objects (ML models, DB connections) that
# should be built once and reused across reruns/videos within the same
# running app, not once per interaction.
#
# above_below_role is only used to give Walticam's above/below trackers
# SEPARATE cached instances rather than one shared one — they're called
# interleaved (alternating every single frame in the tracking loop), and
# reusing one live model object across rapidly alternating calls on
# unrelated image regions isn't something verifiable without the actual
# rtmlib library, so this stays cautious there specifically. The
# non-Walticam Above/Below path calls its two trackers sequentially (one
# fully finishes before the other starts), which is a much lower-risk
# reuse pattern, so those two safely share one cached instance.
@st.cache_resource(show_spinner=False)
def _cached_pose_tracker(mode, det_frequency):
    return tc.make_pose_tracker(mode, det_frequency)


@st.cache_resource(show_spinner=False)
def _cached_pose_tracker_halpe(mode, det_frequency, above_below_role):
    return tc.make_pose_tracker_halpe(mode, det_frequency)


# Files bundled into the "download this app" zip on the Home page, so
# someone can run their own copy with none of the shared hosted app's
# resource limits (see the concurrent-users discussion this came out of).
# packages.txt is deliberately NOT included — that file is only for the
# hosted Streamlit Community Cloud environment's Linux system packages,
# not needed for a local Windows/Mac/Linux setup.
SOURCE_FILES_FOR_DOWNLOAD = [
    "app.py",
    "tracker_core.py",
    "scorer.py",
    "session_store.py",
    "issue_reports.py",
    "requirements.txt",
    "camera_angle_guide.png",
    "framing_example.png",
    "LOCAL_SETUP.md",
]


@st.cache_data(show_spinner=False)
def _build_source_zip():
    """Zips up everything needed to run this app on someone else's own
    computer, read directly from the files sitting next to this script —
    so the download always matches whatever's actually deployed, not a
    stale hand-copied version. Any listed file that doesn't exist yet
    (e.g. before LOCAL_SETUP.md has been added to the repo) is skipped
    rather than failing the whole zip."""
    buffer = io.BytesIO()
    included = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in SOURCE_FILES_FOR_DOWNLOAD:
            path = APP_DIR / filename
            if path.exists():
                zf.write(path, arcname=filename)
                included.append(filename)
    buffer.seek(0)
    return buffer.getvalue(), included


def render_download_source_control():
    zip_bytes, included_files = _build_source_zip()
    st.download_button(
        "Download this app to run on your own computer",
        data=zip_bytes,
        file_name="sync-ai-source.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
    if "LOCAL_SETUP.md" in included_files:
        st.caption(
            "Includes a step-by-step setup guide (LOCAL_SETUP.md inside "
            "the zip) — no coding experience needed. Running your own "
            "copy means no waiting on anyone else and no shared resource "
            "limits."
        )
    else:
        st.caption(
            "Running your own copy means no waiting on anyone else and no "
            "shared resource limits. You'll need Python installed — "
            "`pip install -r requirements.txt` then `streamlit run app.py`."
        )


st.set_page_config(
    page_title=APP_NAME,
    layout="centered",
)

# ── Global visual styling (no emojis anywhere in this app) ────────────────
st.markdown(
    """
    <style>
    /* Fill the sides instead of leaving them stark white on wide screens.
       color is set here too, at the true root of everything Streamlit
       renders — a lot of native components (progress bar text, spinner
       text, etc.) use color:inherit rather than setting their own
       value, and chasing each one's specific data-testid has been
       unreliable (Streamlit's internal DOM structure differs enough
       between versions that a selector verified against one version
       doesn't always match what's actually deployed). Setting the
       color at the root instead relies only on ordinary CSS
       inheritance, which works regardless of exactly how many wrapper
       elements sit in between. Elements that set their OWN explicit
       color (tables, radio option text, widget labels) still need
       their own direct override below — this only fixes the ones that
       were never overriding it in the first place. */
    .stApp {
        background: radial-gradient(circle at 12% 8%, rgba(8,145,178,0.05), transparent 45%),
                    radial-gradient(circle at 88% 92%, rgba(14,116,144,0.05), transparent 45%),
                    #fbfdfe;
        color: #0f172a !important;
    }
    .block-container { padding-top: 4rem; padding-bottom: 3rem; max-width: 1040px; }
    h1, h2, h3 { letter-spacing: -0.01em; }

    /* Top bar */
    .sa-topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 10px; margin-bottom: 8px; border-bottom: 1px solid rgba(120,120,120,0.18);
    }
    .sa-logo {
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: #0e7490;
        line-height: 1.3; padding: 6px 0 10px 0; margin: 0;
        display: inline-block; text-decoration: none; cursor: pointer;
    }
    .sa-logo:hover { color: #0891b2; text-decoration: none; }

    /* Nudge the nav buttons down so they line up with the bigger logo
       instead of sitting above its vertical center */
    div[data-testid="column"]:has(.sa-logo) { padding-top: 4px; }

    /* Hero */
    .sa-hero {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 45%, #155e75 100%);
        border-radius: 18px;
        padding: 34px 32px;
        margin: 10px 0 22px 0;
        color: white;
        box-shadow: 0 8px 24px rgba(8, 145, 178, 0.25);
    }
    .sa-hero h1 { color: white; margin: 0 0 8px 0; font-size: 2.1rem; }
    .sa-hero p { color: rgba(255,255,255,0.92) !important; margin: 0; font-size: 1.05rem; line-height: 1.5; }
    .sa-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
    .sa-badge {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: white !important;
        display: inline-block;
    }

    /* Feature cards */
    .sa-card {
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }
    .sa-card h4 { margin: 0 0 6px 0; font-size: 1.02rem; }
    .sa-card p { margin: 0; font-size: 0.92rem; color: #475569 !important; line-height: 1.5; }

    /* Filming-tip images */
    .sa-fig-caption { font-size: 0.85rem; color: #64748b; text-align: center; margin-top: 6px; }
    div[data-testid="stImage"] img { border-radius: 12px; border: 1px solid rgba(120,120,120,0.15); }

    /* Segmented control */
    div[role="radiogroup"] {
        display: flex; gap: 4px; background: rgba(120,120,120,0.12);
        padding: 4px; border-radius: 999px; width: fit-content;
    }
    div[role="radiogroup"] label { border-radius: 999px !important; padding: 6px 20px !important; margin: 0 !important; }

    /* Score card */
    .sa-score-card {
        background: linear-gradient(180deg, rgba(8,145,178,0.06) 0%, rgba(8,145,178,0.01) 100%);
        border: 1px solid rgba(8,145,178,0.18);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 8px 0 18px 0;
        text-align: center;
    }
    .sa-score-number { font-size: 3.2rem; font-weight: 700; color: #0e7490 !important; line-height: 1; }
    .sa-score-max { font-size: 1.1rem; color: #64748b !important; font-weight: 500; }
    .sa-score-assessment {
        display: inline-block; margin-top: 10px; padding: 4px 14px;
        border-radius: 999px; font-size: 0.85rem; font-weight: 600;
    }

    .sa-section-label {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: #64748b; margin: 18px 0 6px 0;
    }

    .sa-footer { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 40px; }

    /* ---------------------------------------------------------------
       AESTHETIC FIXES ONLY — none of this touches tracking, scoring,
       or speed. The platform's current default styling for plain text,
       widget labels, buttons, and the file uploader was rendering dark
       (or invisible) regardless of page background, so these are
       pinned explicitly using the *real* Streamlit component
       selectors (verified against Streamlit's own bundled source, not
       guessed) instead of relying on defaults.

       :where() keeps the broad text-color reset at zero specificity so
       the sa-* rules above (which set explicit colors for text that's
       meant to stay white, like the hero and badges) always win
       without depending on source order.
       --------------------------------------------------------------- */
    :where(
        .stMarkdown,
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stProgress"],
        [data-testid="stSpinner"]
    ) { color: #0f172a !important; }

    /* st.table() renders an actual <table> that sets its own explicit
       text color (not inherited), so the broad reset above can't reach
       it — this was the "Official deductions" / "Coaching feedback"
       tables rendering nearly invisible. */
    table[data-testid="stTableStyledTable"] { color: #0f172a !important; }

    /* Radio option text ("Walticam" / "Above / Below" etc.) sits in an
       inner <div> that sets its own color directly from the same theme
       token that was broken elsewhere — it doesn't inherit from the
       label, so it has to be targeted directly rather than via the
       ancestor. */
    div[role="radiogroup"] label * { color: #0f172a !important; }

    button[data-testid^="stBaseButton-secondary"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid rgba(8,145,178,0.35) !important;
    }
    button[data-testid^="stBaseButton-secondary"]:hover {
        background-color: #ecfeff !important;
        border-color: #0e7490 !important;
        color: #0e7490 !important;
    }
    button[data-testid^="stBaseButton-secondary"] * { color: inherit !important; }

    button[data-testid^="stBaseButton-primary"] {
        background-color: #0e7490 !important;
        border-color: #0e7490 !important;
        color: #ffffff !important;
    }
    button[data-testid^="stBaseButton-primary"]:hover {
        background-color: #0c6280 !important;
        border-color: #0c6280 !important;
    }
    button[data-testid^="stBaseButton-primary"] * { color: #ffffff !important; }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: rgba(8,145,178,0.25) !important;
    }

    /* File uploader — target the stable stFileUploader wrapper and its
       native <section> dropzone (the inner stFileUploaderDropzone
       testid lives on that <section>, not a <div>, which is why a
       div-only selector never matched it). Every descendant is set
       dark first, then the browse button's text is re-lightened —
       source order makes the later, more specific rule win for
       elements matched by both. */
    div[data-testid="stFileUploader"] * { color: #0f172a !important; }
    div[data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px dashed rgba(8,145,178,0.3) !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #0f172a !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] button * { color: #ffffff !important; }
    div[data-testid="stFileChip"] {
        background-color: #f8fafc !important;
        border: 1px solid rgba(8,145,178,0.15) !important;
    }
    div[data-testid="stFileChip"] * { color: #0f172a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page routing ────────────────────────────────────────────────────────
# Uses the URL's ?page= query param as the source of truth, so a plain
# HTML link (like the logo below) can navigate with a normal click/reload
# instead of needing a Python callback wired to it.
_valid_pages = {"home", "analyze", "history", "help"}
_qp_page = st.query_params.get("page")

if "page" not in st.session_state:
    st.session_state.page = _qp_page if _qp_page in _valid_pages else "home"
elif _qp_page in _valid_pages and _qp_page != st.session_state.page:
    st.session_state.page = _qp_page


def go_to(page_name):
    st.session_state.page = page_name
    st.query_params["page"] = page_name


# ── Top bar (present on every page) ────────────────────────────────────
top_left, top_a, top_b, top_c = st.columns([5, 1.4, 1.4, 1.2])
with top_left:
    st.markdown(
        f'<a href="?page=home" target="_self" class="sa-logo">{APP_NAME}</a>',
        unsafe_allow_html=True,
    )
with top_a:
    if st.button("Analyze", use_container_width=True, type="primary" if st.session_state.page == "analyze" else "secondary"):
        go_to("analyze")
with top_b:
    if st.button("History", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
        go_to("history")
with top_c:
    if st.button("Help", use_container_width=True, type="primary" if st.session_state.page == "help" else "secondary"):
        go_to("help")
st.markdown('<div class="sa-topbar"></div>', unsafe_allow_html=True)


# ── Shared: filming-guide images + popup, used from both Home and Analyze ──
def _show_figure(col, filename, caption):
    path = APP_DIR / filename
    with col:
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.markdown(f'<div class="sa-fig-caption">{caption}</div>', unsafe_allow_html=True)
        else:
            st.warning(
                f"Image not found: {filename}. It needs to sit in the same "
                f"folder as app.py in the repo (looked in: {APP_DIR})."
            )


@st.dialog("How should I film this?")
def show_filming_guide_dialog():
    st.write(
        "Camera angle is the single biggest factor in tracking accuracy. "
        "Shoot from eye level with the water, not from above looking "
        "down — a downward angle foreshortens the swimmer's rise out of "
        "the water, which throws off the height measurement the base "
        "score depends on."
    )
    dcol1, dcol2 = st.columns(2)
    _show_figure(dcol1, "camera_angle_guide.png", "Shoot level with the water, not down at it.")
    _show_figure(
        dcol2, "framing_example.png",
        "Keep the swimmer centered with clear space above and below for the full rise and entry.",
    )
    if st.button("Got it", type="primary"):
        st.rerun()


# ============================================================================
# HOME PAGE
# ============================================================================
def render_home():
    st.markdown(
        f"""
        <div class="sa-hero">
            <h1>{APP_NAME}</h1>
            <p>
                Judging a barracuda figure by eye is fast, but it is also
                inconsistent — the same swimmer can score differently
                depending on the angle, the light, and which judge happens
                to be watching that second. {APP_NAME} takes the exact same
                footage a judge already sees and turns it into a
                repeatable, explainable measurement: how high the swimmer
                rose, how straight their line was on the way up and down,
                how much their legs and ankles bent, and how their back
                held shape — every one of those numbers, every time, from
                the same video.
            </p>
            <div class="sa-badges">
                <span class="sa-badge">AI pose tracking</span>
                <span class="sa-badge">Above and underwater</span>
                <span class="sa-badge">FINA-aligned scoring</span>
                <span class="sa-badge">Kalman-smoothed data</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Analyze a video", type="primary", use_container_width=True):
            go_to("analyze")
    with col2:
        if st.button("See past results", use_container_width=True):
            go_to("history")

    st.markdown('<div class="sa-section-label">Why this matters</div>', unsafe_allow_html=True)
    st.write(
        "A coach reviewing footage after a meet is doing the same thing "
        "over and over by eye: rewinding, guessing degrees, comparing "
        "memory of one swimmer against another. That does not scale past "
        "a handful of swimmers, and it is not something a swimmer can use "
        "on their own between practices. A tool that reads the same "
        "footage and reports the same measurements every time gives "
        "coaches a second opinion they can trust, and gives swimmers "
        "something concrete to work on — not \"looked a little off\" but "
        "\"your ascent was tilted eight degrees, here is the frame where "
        "it happened.\""
    )

    st.markdown('<div class="sa-section-label">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("Upload footage", "A single WaltiCam split-screen video, or separate above-water and underwater clips — whatever you have."),
        ("AI tracks the swimmer", "A pose-detection model finds and locks onto the swimmer frame by frame, rejecting background people, reflections, and pool-deck noise along the way."),
        ("The figure gets measured", "Waterline, jump height, body alignment, backpike, leg and ankle extension, back roundness, and more — all extracted directly from the tracked joints."),
        ("A score comes back", "A FINA-aligned base score from height, minus deductions for each technical category, plus a clear breakdown of exactly where points were lost."),
    ]
    for i, (title, body) in enumerate(steps, start=1):
        c1, c2 = st.columns([0.6, 6])
        with c1:
            st.markdown(f"**{i}**")
        with c2:
            st.markdown(f"**{title}**  \n{body}")

    st.markdown('<div class="sa-section-label">Camera setup that actually works</div>', unsafe_allow_html=True)
    if st.button("How should I film this?", type="primary"):
        show_filming_guide_dialog()
    st.write(
        "Tracking accuracy starts before the video is even uploaded. The "
        "single biggest factor is camera angle: shoot from eye level, not "
        "from above looking down. A downward angle foreshortens the "
        "swimmer's rise out of the water, which throws off exactly the "
        "height measurement the base score depends on — a horizontal, "
        "eye-level camera keeps that rise measurable and consistent."
    )
    img_col1, img_col2 = st.columns(2)
    _show_figure(img_col1, "camera_angle_guide.png", "Shoot level with the water, not down at it.")
    _show_figure(
        img_col2, "framing_example.png",
        "Keep the swimmer centered with clear space above and below for the full rise and entry.",
    )

    st.markdown('<div class="sa-section-label">Built for how you actually film</div>', unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown(
            """<div class="sa-card"><h4>WaltiCam</h4>
            <p>One split-screen video, above and below in the same frame.
            Tracked with a dedicated model that reads real heel, toe, and
            neck positions, and scored with a height calibration built
            specifically for a split-frame camera's narrower field of
            view.</p></div>""",
            unsafe_allow_html=True,
        )
    with fcol2:
        st.markdown(
            """<div class="sa-card"><h4>Above water only</h4>
            <p>Just a single above-water camera. Tracked with a model
            tuned for a full, dedicated view of the swimmer, with tent and
            pool-deck masking built in so background people cannot get
            mistaken for the swimmer.</p></div>""",
            unsafe_allow_html=True,
        )
    with fcol3:
        st.markdown(
            """<div class="sa-card"><h4>Above and below</h4>
            <p>Both cameras, tracked separately. The official score comes
            from the above-water footage, and the underwater footage adds
            coaching-only feedback — details like a bent knee mid-figure
            that a judge on deck would never get to see.</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sa-section-label">What gets measured</div>', unsafe_allow_html=True)
    st.write(
        "Ascent and descent alignment, backpike, leg extension, ankle "
        "extension, back roundness, travel, unroll speed, underwater bent "
        "knee, back layout depth, and head tuck all count toward the "
        "score, blended between a fixed technical standard and how the "
        "figure compares to others scored in the same batch. Nothing is "
        "hidden — every measured number is shown, and any figure worth 0.5 "
        "or more on a category gets a plain-language note on what to work "
        "on."
    )

    st.markdown('<div class="sa-section-label">Run it without the wait</div>', unsafe_allow_html=True)
    st.write(
        "This hosted version runs on shared, resource-limited servers — "
        "usually fine, but only one person can really process a video at "
        "a time, and heavy use can slow it down for everyone. Running "
        "your own copy on your own computer sidesteps that completely: "
        "no sharing, no limits, and your session history actually sticks "
        "around instead of disappearing if the hosted app goes to sleep."
    )
    render_download_source_control()

    if st.button("Start analyzing", type="primary"):
        go_to("analyze")


# ============================================================================
# ANALYZE PAGE
# ============================================================================
def render_model_preload_control():
    """Lets the user trigger the slow one-time model load ahead of time
    (e.g. while they're still getting their video ready), instead of it
    always landing on whichever video they process first. Calls the same
    cached factory functions the actual processing calls use, so once
    warmed here, processing a video afterward reuses the exact same
    loaded model with zero extra load time."""
    if st.session_state.get("models_preloaded"):
        st.success("Pose model loaded and ready.")
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        st.caption(
            "The pose model takes about a minute to load the first time "
            "it's used in a session. Load it now if you'd rather not wait "
            "once you're ready to process a video — if it's already warm "
            "from a recent video on this app, this will be quick instead."
        )
    with col2:
        if st.button("Load pose model now", use_container_width=True):
            with st.spinner("Loading pose model..."):
                _cached_pose_tracker("lightweight", 4)
                _cached_pose_tracker_halpe("lightweight", 4, "above")
                _cached_pose_tracker_halpe("lightweight", 4, "below")
            st.session_state.models_preloaded = True
            st.rerun()


def render_analyze():
    st.markdown('<div class="sa-section-label">Analyze</div>', unsafe_allow_html=True)
    top_row1, top_row2 = st.columns([4, 2])
    with top_row1:
        st.write(
            "Upload footage below. Each of the three modes uses its own "
            "tracking model and its own height calibration — see the Home "
            "page for why."
        )
    with top_row2:
        if st.button("How should I film this?", use_container_width=True):
            show_filming_guide_dialog()

    render_model_preload_control()

    # det_frequency controls how often the underlying model re-runs full
    # person detection vs. reusing/tracking a previous frame's result.
    # Was 4 — nudged down slightly (a little slower, per direct request
    # for a bit more accuracy) specifically because a too-infrequent
    # detector was letting the tracker drift onto — and then stay
    # locked onto — background clutter for several frames at a time
    # instead of getting a fresh chance to re-identify the real
    # swimmer. mode stays "lightweight" (unchanged).
    chosen = dict(mode="lightweight", det_frequency=3)
    waterline_value = None
    max_duration = 60

    source = st.radio(
        "Camera source", options=["Walticam", "Above / Below"],
        horizontal=True, label_visibility="collapsed",
    )

    swimmer_id = st.text_input("Swimmer ID (optional, shown on the score)", value="")

    above_water = True
    below_water = False
    if source == "Above / Below":
        sub_mode = st.radio(
            "What footage do you have?",
            options=["Just Above", "Above + Below"],
            horizontal=True,
        )
        below_water = (sub_mode == "Above + Below")
        if below_water:
            st.caption(
                "The underwater video adds coaching-only feedback (bent "
                "knee, back layout depth). The official score always "
                "comes from the above-water footage."
            )

    def run_with_progress(label):
        progress_bar = st.progress(0.0, text=f"Starting {label}...")
        start_time = time.time()

        def update_progress(frame_count, total_frames):
            if total_frames > 0:
                pct = min(frame_count / total_frames, 1.0)
                elapsed = time.time() - start_time
                fps_proc = frame_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
                progress_bar.progress(
                    pct, text=f"{label}: frame {frame_count}/{total_frames} (about {eta:.0f}s remaining)"
                )
        return progress_bar, update_progress

    def show_results(label, video_path, csv_path, landmarks):
        kalman_csv = tc.apply_kalman_filter_to_csv(csv_path, landmarks)
        st.success(f"{label} processing complete.")
        vcol, _ = st.columns([1, 1])
        with vcol:
            st.video(str(video_path))
        col1, col2 = st.columns(2)
        with col1:
            with open(video_path, "rb") as f:
                st.download_button(
                    f"Download {label} video", data=f.read(),
                    file_name=Path(video_path).name, mime="video/mp4",
                    use_container_width=True, key=f"video_{label}",
                )
        with col2:
            with open(kalman_csv, "rb") as f:
                st.download_button(
                    f"Download {label} data (CSV)", data=f.read(),
                    file_name=Path(kalman_csv).name, mime="text/csv",
                    use_container_width=True, key=f"csv_{label}",
                )
        return kalman_csv

    def assessment_style(score):
        if score >= 9.5: return "Excellent / Near Perfect", "#059669", "#ecfdf5"
        if score >= 8.5: return "Very Good", "#0891b2", "#ecfeff"
        if score >= 7.5: return "Good", "#2563eb", "#eff6ff"
        if score >= 6.5: return "Competent", "#7c3aed", "#f5f3ff"
        if score >= 5.5: return "Satisfactory", "#d97706", "#fffbeb"
        if score >= 4.5: return "Deficient", "#dc2626", "#fef2f2"
        return "Weak", "#991b1b", "#fef2f2"

    def show_score(above_kalman_csv, below_kalman_csv=None, name="figure", source_mode="above"):
        try:
            result = BarracudaScorer.score_single_pair(
                above_kalman_csv, below_kalman_csv, name=name, source_mode=source_mode
            )
        except Exception as e:
            st.warning(f"Could not compute a score for this figure: {e}")
            return None

        st.markdown('<div class="sa-section-label">Result</div>', unsafe_allow_html=True)

        score = result["score"]
        assess, color, bg = assessment_style(score)

        st.markdown(
            f"""
            <div class="sa-score-card">
                <div style="color:#64748b; font-size:0.85rem; font-weight:600; letter-spacing:0.04em;">
                    {(name or "FIGURE").upper()}
                </div>
                <div><span class="sa-score-number">{score:.2f}</span><span class="sa-score-max"> / 10.0</span></div>
                <div class="sa-score-assessment" style="color:{color}; background:{bg};">{assess}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("Base score (height)", f"{result['base_score']:.2f}")
        col2.metric("Total deduction", f"-{result['total_deduction']:.2f}")
        st.caption(f"Height metric used: {result.get('base_score_metric_used', 'n/a')}")

        with st.expander("Debug info (height calculation inputs)"):
            st.write(
                "The base score comes from foot clearance — how far the "
                "swimmer's ankles rise above the detected waterline. "
                "Above / Above+Below modes use the raw frame-fraction "
                "value; Walticam mode uses a body-length-normalized "
                "version instead, since a split-frame camera has a "
                "different field of view than a dedicated above-water "
                "camera (see the Home page for why)."
            )
            st.json({
                "source_mode": result.get("source_mode"),
                "base_score_metric_used": result.get("base_score_metric_used"),
                "foot_clearance (frame-fraction)": result.get("foot_clearance"),
                "foot_clearance_normalized (body-lengths)": result.get("foot_clearance_normalized"),
                "body_scale (shoulder-to-ankle, px-frame-units)": result.get("body_scale"),
                "base_score": result.get("base_score"),
                "ascent_tilt_median (deg)": result.get("ascent_tilt_median"),
                "descent_tilt_median (deg)": result.get("descent_tilt_median"),
                "knee_angle_median (deg)": result.get("knee_angle_median"),
                "frames": result.get("frames"),
            })

        d = result["deductions"]
        # All categories now count toward the score (INCLUDE_COACHING_IN_SCORE
        # is True in scorer.py), so they're shown as one combined breakdown
        # rather than split into "official" vs "coaching-only" tables.
        all_rows = [
            ("Ascent alignment", "ascent_alignment", "ascent_alignment_degrees"),
            ("Descent alignment", "descent_alignment", "descent_alignment_degrees"),
            ("Backpike", "backpike", "backpike_degrees"),
            ("Leg extension", "leg_extension", "leg_extension_degrees"),
            ("Ankle extension", "ankle_extension", "ankle_extension_degrees"),
            ("Back roundness", "back_roundness", "back_roundness_degrees"),
            ("Travel", "travel", "travel_degrees"),
            ("Unroll speed", "unroll_speed", "unroll_speed_degrees"),
            ("Underwater bent knee", "underwater_bent_knee", "underwater_bent_knee_degrees"),
            ("Back layout depth (estimated — not judge-confirmed)", "back_layout_depth", "back_layout_depth_value"),
            ("Head tuck (estimated — not judge-confirmed)", "head_tuck", "head_tuck_degrees"),
        ]
        st.markdown("**Deductions**")
        st.table({
            "Category": [label for label, _, _ in all_rows],
            "Deduction": [f"-{d.get(key, 0):.2f}" if d.get(key, 0) else "None" for _, key, _ in all_rows],
            "Measured": [
                f"{d[deg_key]:.2f}" if d.get(deg_key) is not None else "None"
                for _, _, deg_key in all_rows
            ],
        })

        # Coaching feedback: whichever categories are actually costing
        # real points (>=0.5) get a plain-language "what to work on" tip
        # instead of just a number, so this reads as something to act on
        # rather than another row of the same table.
        FIX_TIPS = {
            "ascent_alignment": "Keep your body vertical on the way up — avoid leaning to either side during the rise.",
            "descent_alignment": "Stay vertical on the way down too — a tilt during descent is graded the same as one during ascent.",
            "backpike": "Keep hips and legs in one straight line after the peak — avoid piking or bending at the hips on the way down.",
            "leg_extension": "Work on fully straightening your knees — eliminate any bend during the figure.",
            "ankle_extension": "Point your toes and fully extend your ankles.",
            "back_roundness": "Keep your back flat in the layout position — avoid arching or rounding at the stomach.",
            "travel": "Work on staying in one spot — minimize lateral drift during the figure.",
            "unroll_speed": "Control your descent — come down slower than you went up, don't rush the unroll.",
            "underwater_bent_knee": "Keep your knees straight underwater too — it affects body line even where it's harder for a judge to see.",
            "back_layout_depth": "Try not to sink too deep in the layout — stay closer to the surface.",
            "head_tuck": "Keep your head in a neutral position — avoid tucking your chin toward your chest.",
        }
        significant = [
            (label, key, d.get(key, 0)) for label, key, _ in all_rows
            if d.get(key, 0) >= 0.5
        ]
        st.markdown("**Worth working on** (deductions of 0.5 or more)")
        if significant:
            for label, key, value in sorted(significant, key=lambda r: r[2], reverse=True):
                tip = FIX_TIPS.get(key, "")
                st.markdown(f"- **{label}** (-{value:.2f}): {tip}")
        else:
            st.caption("No deductions of 0.5 or more on this figure.")

        if below_kalman_csv is None:
            st.caption(
                "No underwater video was processed, so underwater-only "
                "coaching feedback is not available."
            )

        return result

    MAX_SIZE_MB = 300

    if source == "Walticam":
        uploaded_file = st.file_uploader(
            "Upload your WaltiCam video (split-screen: top=above, bottom=below)",
            type=["mp4", "mov", "m4v", "avi"],
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"{uploaded_file.name} ({file_size_mb:.1f} MB)")

            if file_size_mb > MAX_SIZE_MB:
                st.error(f"File is too large ({file_size_mb:.0f} MB). Please upload a video under {MAX_SIZE_MB} MB.")
            elif st.button("Process video", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    input_path = Path(tmp_dir) / uploaded_file.name
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    output_path = Path(tmp_dir) / f"{input_path.stem}_walticam_tracking.mp4"

                    progress_bar, update_progress = run_with_progress("Walticam")
                    try:
                        with st.spinner("Loading pose model (first video in a session takes about a minute; reused after that)..."):
                            above_pt = _cached_pose_tracker_halpe(chosen["mode"], chosen["det_frequency"], "above")
                            below_pt = _cached_pose_tracker_halpe(chosen["mode"], chosen["det_frequency"], "below")
                            st.session_state.models_preloaded = True
                            video_file, above_csv, below_csv = tc.process_video_walticam(
                                str(input_path), output_path,
                                mode=chosen["mode"], det_frequency=chosen["det_frequency"],
                                max_duration=max_duration, progress_callback=update_progress,
                                above_pose_tracker=above_pt, below_pose_tracker=below_pt,
                            )
                        progress_bar.progress(1.0, text="Done.")

                        above_kalman = tc.apply_kalman_filter_to_csv(above_csv, tc.ALL_LANDMARKS_ABOVE)
                        below_kalman = tc.apply_kalman_filter_to_csv(below_csv, tc.ALL_LANDMARKS_ABOVE)

                        st.success("Walticam processing complete.")
                        vcol, _ = st.columns([1, 1])
                        with vcol:
                            st.video(str(video_file))

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            with open(video_file, "rb") as f:
                                st.download_button("Combined video", data=f.read(),
                                                    file_name=Path(video_file).name, mime="video/mp4",
                                                    use_container_width=True)
                        with col2:
                            with open(above_kalman, "rb") as f:
                                st.download_button("Above data (CSV)", data=f.read(),
                                                    file_name=Path(above_kalman).name, mime="text/csv",
                                                    use_container_width=True)
                        with col3:
                            with open(below_kalman, "rb") as f:
                                st.download_button("Below data (CSV)", data=f.read(),
                                                    file_name=Path(below_kalman).name, mime="text/csv",
                                                    use_container_width=True)

                        score_result = show_score(above_kalman, below_kalman, name=swimmer_id or "figure", source_mode="walticam")

                        if score_result is not None:
                            session_store.save_session(
                                swimmer_id=swimmer_id or "figure",
                                mode="Walticam",
                                score_result=score_result,
                                file_paths={
                                    "video": video_file,
                                    "above_csv": above_kalman,
                                    "below_csv": below_kalman,
                                },
                                official_keys=BarracudaScorer.__new__(BarracudaScorer)._deduction_keys(),
                            )
                            st.caption("Saved to History.")

                    except Exception as e:
                        st.error(f"Something went wrong during processing: {e}")
                        st.exception(e)
        else:
            st.info("Upload a WaltiCam split-screen video to get started.")

    else:
        above_file, below_file = None, None

        above_file = st.file_uploader(
            "Upload above-water video", type=["mp4", "mov", "m4v", "avi"], key="above_upload"
        )
        if below_water:
            below_file = st.file_uploader(
                "Upload underwater video", type=["mp4", "mov", "m4v", "avi"], key="below_upload"
            )

        ready = (above_file is not None) and (not below_water or below_file is not None)

        if ready and above_file is not None:
            oversized = []
            if above_file.size / (1024 * 1024) > MAX_SIZE_MB:
                oversized.append(above_file.name)
            if below_file is not None and below_file.size / (1024 * 1024) > MAX_SIZE_MB:
                oversized.append(below_file.name)

            if oversized:
                st.error(f"These files are too large (over {MAX_SIZE_MB} MB): {', '.join(oversized)}")
            elif st.button("Process video(s)", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    above_kalman_result = None
                    below_kalman_result = None
                    above_video_file = None
                    below_video_file = None
                    try:
                        with st.spinner("Loading pose model (first video in a session takes about a minute; reused after that)..."):
                            shared_pt = _cached_pose_tracker(chosen["mode"], chosen["det_frequency"])
                            st.session_state.models_preloaded = True

                            above_input = Path(tmp_dir) / above_file.name
                            with open(above_input, "wb") as f:
                                f.write(above_file.getbuffer())
                            above_output = Path(tmp_dir) / f"{above_input.stem}_above_tracking.mp4"

                            _, update_above = run_with_progress("Above-water")
                            video_file, csv_file = tc.process_video_above_water(
                                str(above_input), above_output, waterline_value,
                                mode=chosen["mode"], det_frequency=chosen["det_frequency"],
                                max_duration=max_duration, progress_callback=update_above,
                                pose_tracker=shared_pt,
                            )
                            above_video_file = video_file
                            above_kalman_result = show_results(
                                "Above-water", video_file, csv_file, tc.ALL_LANDMARKS_ABOVE
                            )

                            if below_water and below_file is not None:
                                below_input = Path(tmp_dir) / below_file.name
                                with open(below_input, "wb") as f:
                                    f.write(below_file.getbuffer())
                                below_output = Path(tmp_dir) / f"{below_input.stem}_below_tracking.mp4"

                                _, update_below = run_with_progress("Underwater")
                                video_file, csv_file = tc.process_video_underwater(
                                    str(below_input), below_output, waterline_value,
                                    mode=chosen["mode"], det_frequency=chosen["det_frequency"],
                                    max_duration=max_duration, progress_callback=update_below,
                                    pose_tracker=shared_pt,
                                )
                                below_video_file = video_file
                                below_kalman_result = show_results(
                                    "Underwater", video_file, csv_file, tc.ALL_LANDMARKS_UNDERWATER
                                )

                        if above_kalman_result is not None:
                            source_mode = "above_below" if below_kalman_result is not None else "above"
                            score_result = show_score(
                                above_kalman_result, below_kalman_result,
                                name=swimmer_id or "figure", source_mode=source_mode,
                            )

                            if score_result is not None:
                                mode_label = "Above+Below" if below_kalman_result is not None else "Above-Water"
                                session_store.save_session(
                                    swimmer_id=swimmer_id or "figure",
                                    mode=mode_label,
                                    score_result=score_result,
                                    file_paths={
                                        "above_video": above_video_file,
                                        "below_video": below_video_file,
                                        "above_csv": above_kalman_result,
                                        "below_csv": below_kalman_result,
                                    },
                                    official_keys=BarracudaScorer.__new__(BarracudaScorer)._deduction_keys(),
                                )
                                st.caption("Saved to History.")

                    except Exception as e:
                        st.error(f"Something went wrong during processing: {e}")
                        st.exception(e)
        else:
            st.info("Upload your above-water video (and underwater too, if you have it) to get started.")


# ============================================================================
# HISTORY PAGE
# ============================================================================
def render_athlete_progress(sessions):
    """Score-over-time chart plus a notepad of still-outstanding deductions
    for a chosen athlete, built from stored session history."""
    swimmer_ids = sorted({s.get("swimmer_id", "Unknown") for s in sessions})
    if not swimmer_ids:
        return

    st.markdown('<div class="sa-section-label">Athlete progress</div>', unsafe_allow_html=True)
    chosen_swimmer = st.selectbox("Athlete", options=swimmer_ids, key="athlete_progress_select")

    athlete_sessions = [s for s in sessions if s.get("swimmer_id", "Unknown") == chosen_swimmer]
    # session_store returns newest-first; sort ascending (oldest to newest)
    # for a chronological trend line.
    athlete_sessions = sorted(athlete_sessions, key=lambda s: s.get("timestamp", ""))

    scored = [s for s in athlete_sessions if s.get("score") is not None]
    if len(scored) >= 2:
        chart_df = pd.DataFrame({
            "Score": [s["score"] for s in scored],
            "Base score": [s.get("base_score") for s in scored],
        }, index=pd.to_datetime([s["timestamp"] for s in scored]))
        st.line_chart(chart_df)
        improvement = scored[-1]["score"] - scored[0]["score"]
        direction = "up" if improvement > 0 else ("down" if improvement < 0 else "unchanged")
        st.caption(
            f"{len(scored)} scored sessions for {chosen_swimmer}. "
            f"Score is {direction} {abs(improvement):.2f} from the first to the most recent session."
        )
    elif len(scored) == 1:
        st.info(f"Only one scored session for {chosen_swimmer} so far — a trend line needs at least two.")
    else:
        st.info(f"No scored sessions for {chosen_swimmer} yet.")

    st.markdown("**Still needs work**")
    if not athlete_sessions:
        return

    latest = athlete_sessions[-1]
    deductions = latest.get("deductions")
    official_keys = latest.get("official_keys")

    if deductions is None or official_keys is None:
        st.caption(
            "This athlete's most recent session was saved before deduction "
            "details started being tracked — analyze a new video for them "
            "to enable this list."
        )
        return

    outstanding = [
        (key, deductions.get(key, 0), deductions.get(f"{key}_degrees"))
        for key in official_keys
        if isinstance(deductions.get(key), (int, float)) and deductions.get(key, 0) > 0
    ]
    outstanding.sort(key=lambda x: x[1], reverse=True)

    if not outstanding:
        st.success(f"No outstanding deductions in {chosen_swimmer}'s most recent session ({latest['timestamp']}).")
        return

    st.caption(f"From the most recent session ({latest['timestamp']}):")
    for key, value, degrees in outstanding:
        label = DEDUCTION_LABELS.get(key, key.replace("_", " ").title())
        measured = f" — measured {degrees:.2f}" if degrees is not None else ""
        st.markdown(f"- **{label}**: -{value:.2f}{measured}")


def render_history():
    st.markdown('<div class="sa-section-label">History</div>', unsafe_allow_html=True)
    st.caption(
        "Not persistent on free-tier hosting — sessions are lost if the "
        "app restarts or sleeps. Local `streamlit run` keeps them on disk."
    )

    sessions = session_store.load_sessions()
    if not sessions:
        st.info("No sessions scored yet. Analyze a video to see it here.")
        if st.button("Go to Analyze", type="primary"):
            go_to("analyze")
        return

    st.write(f"{len(sessions)} figure{'s' if len(sessions) != 1 else ''} scored")
    if st.button("Clear all sessions"):
        session_store.clear_all_sessions()
        st.rerun()

    render_athlete_progress(sessions)

    st.markdown('<div class="sa-section-label">All sessions</div>', unsafe_allow_html=True)
    for s in sessions:
        score_str = f"{s['score']:.2f}/10" if s.get("score") is not None else "None"
        with st.expander(f"{s['swimmer_id']} - {score_str} ({s['timestamp']})"):
            st.write(f"Mode: {s['mode']}")
            base = s.get("base_score")
            ded = s.get("total_deduction")
            if base is not None and ded is not None:
                st.write(f"Base: {base:.2f}  |  Deduction: -{ded:.2f}")
            st.write(f"Top issues: {s['summary']}")
            for key, rel_path in s.get("files", {}).items():
                fpath = session_store.session_file_path(rel_path)
                if fpath.exists():
                    with open(fpath, "rb") as f:
                        mime = "video/mp4" if "video" in key else "text/csv"
                        st.download_button(
                            key, data=f.read(), file_name=fpath.name,
                            mime=mime, key=f"hist_{s['id']}_{key}",
                            use_container_width=True,
                        )
            if st.button("Delete", key=f"del_{s['id']}"):
                session_store.delete_session(s["id"])
                st.rerun()


# ============================================================================
# HELP PAGE
# ============================================================================
def render_help():
    st.markdown('<div class="sa-section-label">Help</div>', unsafe_allow_html=True)
    st.write(
        "Something wrong with a score, the waterline, or the tracked "
        "skeleton? Describe it below — this is sent directly for review "
        "and fixing."
    )

    with st.form("issue_report_form"):
        name = st.text_input("Your name (optional)")
        contact_email = st.text_input("Your email, if you want a reply (optional)")
        page_context = st.selectbox(
            "Which part of the app is this about?",
            options=["Analyze - Walticam", "Analyze - Just Above", "Analyze - Above + Below",
                     "History", "Scoring / results", "Something else"],
        )
        description = st.text_area(
            "What happened?",
            placeholder="Example: the waterline was drawn well above the swimmer for this video, "
                        "or the score seemed too high given the deductions shown.",
            height=140,
        )
        submitted = st.form_submit_button("Send report", type="primary")

    if submitted:
        if not description.strip():
            st.warning("Please describe the issue before sending.")
        else:
            record, mailto_url = issue_reports.save_report(name, contact_email, description, page_context)
            st.success("Report saved.")
            st.markdown(
                f'<a href="{mailto_url}" target="_blank">Click here to also send this by email</a>',
                unsafe_allow_html=True,
            )
            st.caption(
                "The report is saved locally on this app's server. Clicking the "
                "link above additionally opens your email client with the "
                "report pre-filled, so it reaches the developer even if this "
                "app's storage does not persist."
            )

    with st.expander("Previously submitted reports (this session's storage)"):
        reports = issue_reports.load_reports()
        if not reports:
            st.caption("No reports submitted yet.")
        else:
            for r in reports:
                st.write(f"{r['timestamp']} - {r['name']} - {r['page_context']}")
                st.caption(r["description"])


# ============================================================================
# ROUTING
# ============================================================================
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "analyze":
    render_analyze()
elif st.session_state.page == "history":
    render_history()
elif st.session_state.page == "help":
    render_help()
else:
    render_home()

if st.session_state.page != "analyze":
    st.divider()
    with st.expander("About " + APP_NAME):
        st.markdown(
            """
            This tool uses RTMPose (via `rtmlib`) for pose detection, with
            three modes:

            - Walticam: a single split-screen video (top half = above-water,
              bottom half = underwater), tracked with two RTMPose instances
              running in tandem, using real heel/toe/neck keypoints (Halpe26).
            - Above / Below: separate above-water and/or underwater videos,
              each run through a dedicated tracker with:
                - Physical tent/background masking (above-water only)
                - Waterline detected from shoulder and hip position
                - Validated swimmer locking, with a periodic re-check
                  requiring two consecutive disagreements before switching
                  who is tracked
                - Multi-level smoothing and Kalman filtering
                - A synthesized mid_spine point (underwater only) for
                  back-curvature measurements

            Scoring is FINA-aligned: a base score from jump height, minus
            deductions for alignment, backpike, leg/ankle extension, back
            roundness, travel, unroll speed, underwater bent knee, back
            layout depth, and head tuck — blended 70 percent absolute
            (fixed thresholds) and 30 percent relative (how this figure
            compares to others scored in the same batch). Walticam mode
            uses a body-length-normalized height metric instead of the raw
            frame-fraction one, since a split-frame camera has a different
            field of view than a dedicated above-water camera.

            Every scored figure is saved to History for later reference.
            """
        )

    st.markdown(
        f'<div class="sa-footer">{APP_NAME} — RTMPose tracking, Kalman filtering, FINA-aligned scoring</div>',
        unsafe_allow_html=True,
    )
