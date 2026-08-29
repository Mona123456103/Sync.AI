#!/usr/bin/env python3
"""
Barracuda Lens — web front-end for tracker_core.py + scorer.py
===========================================================================
Four views, switched via a `?page=` query param (kept simple and
version-portable rather than depending on newer st.navigation APIs):

    home     landing page — what this does and why
    analyze  upload + track + score a figure
    history  past sessions, saved by session_store.py
    help     report a problem

Tracking speed is intentionally NOT user-configurable here — every run
uses FAST_SETTINGS below. See the comment on that constant for why.

Run locally:
    streamlit run app.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

import issue_reports
import session_store
import tracker_core as tc
from scorer import BarracudaScorer

APP_NAME = "Barracuda Lens"
APP_DIR = Path(__file__).resolve().parent

# Tracking always runs at this setting — it's not exposed as a control.
# "lightweight" mode + det_frequency=4 is the fast end of what rtmlib
# supports; this keeps a figure processing in well under a minute on
# shared CPU hardware instead of several minutes. Nothing downstream
# reads a different value, so changing this one constant is enough if
# that trade-off ever needs revisiting.
FAST_SETTINGS = {"mode": "lightweight", "det_frequency": 4}
MAX_CLIP_SECONDS = 60

# Labels for the deduction categories, used in the History page's
# progress panel.
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

# Everything a from-scratch local install needs, bundled by the
# "download this app" button on the Home page. packages.txt is left out
# on purpose — it's Linux system packages for the hosted deployment
# only, not something a local Windows/Mac/Linux setup needs.
LOCAL_COPY_FILES = [
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _local_copy_zip():
    """Builds the downloadable zip straight from whatever's sitting next
    to this script, so it can never drift out of sync with what's
    actually deployed. Missing optional files (e.g. no LOCAL_SETUP.md
    yet) are skipped rather than failing the zip."""
    buf = io.BytesIO()
    included = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in LOCAL_COPY_FILES:
            path = APP_DIR / filename
            if path.exists():
                zf.write(path, arcname=filename)
                included.append(filename)
    buf.seek(0)
    return buf.getvalue(), included


def local_copy_button():
    zip_bytes, included = _local_copy_zip()
    st.download_button(
        "Download a copy to run on your own machine",
        data=zip_bytes,
        file_name="barracuda-lens.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
    if "LOCAL_SETUP.md" in included:
        st.caption(
            "Includes a setup guide (LOCAL_SETUP.md in the zip). Your own "
            "copy isn't sharing CPU with anyone else's uploads."
        )
    else:
        st.caption(
            "Your own copy isn't sharing CPU with anyone else's uploads. "
            "You'll need Python: `pip install -r requirements.txt`, then "
            "`streamlit run app.py`."
        )


def figure_image(col, filename, caption):
    path = APP_DIR / filename
    with col:
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.markdown(f'<p class="bl-caption">{caption}</p>', unsafe_allow_html=True)
        else:
            st.warning(f"{filename} isn't in the app folder yet ({APP_DIR}).")


@st.dialog("Filming tips")
def filming_tips_dialog():
    st.write(
        "Camera angle matters more than almost anything else here. Shoot "
        "level with the water rather than down at it — a downward angle "
        "compresses how far the swimmer actually rises, and that rise is "
        "exactly what the base score is built from."
    )
    c1, c2 = st.columns(2)
    figure_image(c1, "camera_angle_guide.png", "Level with the waterline, not angled down.")
    figure_image(c2, "framing_example.png", "Leave headroom above and below for the full rise and entry.")
    if st.button("Close", type="primary"):
        st.rerun()


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_NAME, layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(30,58,95,0.05), transparent 45%),
            radial-gradient(circle at 90% 90%, rgba(217,119,6,0.05), transparent 45%),
            #fbfaf8;
    }
    .block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1020px; }

    .bl-nav {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 10px; margin-bottom: 10px;
        border-bottom: 1px solid rgba(30,58,95,0.15);
    }
    .bl-wordmark {
        font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
        color: #1e3a5f; text-decoration: none; cursor: pointer;
    }
    .bl-wordmark:hover { color: #2b5480; text-decoration: none; }
    div[data-testid="column"]:has(.bl-wordmark) { padding-top: 6px; }

    .bl-hero {
        background: linear-gradient(120deg, #1e3a5f 0%, #24476f 55%, #163150 100%);
        border-radius: 16px; padding: 32px 30px; margin: 8px 0 20px 0;
        color: white; box-shadow: 0 8px 22px rgba(30,58,95,0.28);
    }
    .bl-hero h1 { color: white; margin: 0 0 8px 0; font-size: 2rem; }
    .bl-hero p { color: rgba(255,255,255,0.9); margin: 0; font-size: 1.03rem; line-height: 1.55; }
    .bl-chips { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
    .bl-chip {
        background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.26);
        border-radius: 999px; padding: 4px 12px; font-size: 0.76rem; color: white;
    }

    .bl-tile {
        border: 1px solid rgba(30,58,95,0.15); border-radius: 12px;
        padding: 16px 18px; height: 100%;
    }
    .bl-tile h4 { margin: 0 0 6px 0; font-size: 1rem; color: #1e3a5f; }
    .bl-tile p { margin: 0; font-size: 0.9rem; color: #475569; line-height: 1.5; }

    .bl-caption { font-size: 0.83rem; color: #64748b; text-align: center; margin-top: 6px; }
    div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid rgba(30,58,95,0.12); }

    div[role="radiogroup"] {
        display: flex; gap: 4px; background: rgba(30,58,95,0.08);
        padding: 4px; border-radius: 999px; width: fit-content;
    }
    div[role="radiogroup"] label { border-radius: 999px !important; padding: 6px 18px !important; margin: 0 !important; }

    .bl-scorebox {
        background: linear-gradient(180deg, rgba(30,58,95,0.06) 0%, rgba(30,58,95,0.01) 100%);
        border: 1px solid rgba(30,58,95,0.16); border-radius: 14px;
        padding: 22px 26px; margin: 8px 0 16px 0; text-align: center;
    }
    .bl-scorebox .who { color: #64748b; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.05em; }
    .bl-scorebox .num { font-size: 3.1rem; font-weight: 700; color: #1e3a5f; line-height: 1; }
    .bl-scorebox .max { font-size: 1.05rem; color: #64748b; font-weight: 500; }
    .bl-scorebox .tag { display: inline-block; margin-top: 8px; padding: 4px 14px; border-radius: 999px; font-size: 0.83rem; font-weight: 600; }

    .bl-label {
        font-size: 0.76rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        color: #64748b; margin: 16px 0 6px 0;
    }
    .bl-foot { text-align: center; color: #94a3b8; font-size: 0.78rem; margin-top: 36px; }
    </style>
    """,
    unsafe_allow_html=True,
)

_pages = {"home", "analyze", "history", "help"}
_from_url = st.query_params.get("page")
if "page" not in st.session_state:
    st.session_state.page = _from_url if _from_url in _pages else "home"
elif _from_url in _pages and _from_url != st.session_state.page:
    st.session_state.page = _from_url


def go(page):
    st.session_state.page = page
    st.query_params["page"] = page


nav_logo, nav_a, nav_b, nav_c = st.columns([5, 1.4, 1.4, 1.2])
with nav_logo:
    st.markdown(f'<a href="?page=home" target="_self" class="bl-wordmark">{APP_NAME}</a>', unsafe_allow_html=True)
with nav_a:
    if st.button("Analyze", use_container_width=True, type="primary" if st.session_state.page == "analyze" else "secondary"):
        go("analyze")
with nav_b:
    if st.button("History", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
        go("history")
with nav_c:
    if st.button("Help", use_container_width=True, type="primary" if st.session_state.page == "help" else "secondary"):
        go("help")
st.markdown('<div class="bl-nav"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def page_home():
    st.markdown(
        f"""
        <div class="bl-hero">
            <h1>{APP_NAME}</h1>
            <p>
                A judge scoring a barracuda figure live has one look, from
                one angle, under whatever light the deck happens to have —
                and two judges watching the same swimmer can still land on
                different numbers. {APP_NAME} runs the same footage through
                pose tracking instead: how high the swimmer rose, how
                straight the line was going up and coming down, how bent
                the legs and ankles were, how the back held its shape —
                measured the same way, from the same video, every time.
            </p>
            <div class="bl-chips">
                <span class="bl-chip">Pose tracking</span>
                <span class="bl-chip">Above &amp; underwater</span>
                <span class="bl-chip">FINA-aligned scoring</span>
                <span class="bl-chip">Kalman-smoothed data</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Analyze a video", type="primary", use_container_width=True):
            go("analyze")
    with c2:
        if st.button("Browse past sessions", use_container_width=True):
            go("history")

    st.markdown('<div class="bl-label">Why bother automating this</div>', unsafe_allow_html=True)
    st.write(
        "Reviewing footage by eye means rewinding the same clip, guessing "
        "at angles, and comparing swimmers from memory — it doesn't scale "
        "past a handful of athletes, and a swimmer can't really do it on "
        "their own between practices. Reading the same numbers off the "
        "same footage every time gives a coach a second opinion worth "
        "trusting, and gives a swimmer something concrete to fix instead "
        "of \"that looked a bit off\" — an actual tilt in degrees, and the "
        "frame it happened on."
    )

    st.markdown('<div class="bl-label">How a video becomes a score</div>', unsafe_allow_html=True)
    steps = [
        ("Upload footage", "One WaltiCam split-screen clip, or separate above-water / underwater clips — whatever you filmed with."),
        ("The swimmer gets tracked", "A pose model follows the swimmer frame by frame, filtering out background people, reflections, and pool-deck clutter."),
        ("The figure gets measured", "Waterline, jump height, alignment on the way up and down, backpike, leg and ankle extension, back shape — pulled straight from the tracked joints."),
        ("A score comes back", "A height-based base score, minus deductions per category, with the full breakdown shown — not just the final number."),
    ]
    for i, (title, body) in enumerate(steps, start=1):
        sc1, sc2 = st.columns([0.5, 6])
        with sc1:
            st.markdown(f"**{i}**")
        with sc2:
            st.markdown(f"**{title}**  \n{body}")

    st.markdown('<div class="bl-label">Getting the camera right</div>', unsafe_allow_html=True)
    if st.button("Filming tips", type="primary"):
        filming_tips_dialog()
    st.write(
        "Camera angle is the single biggest factor in how accurate the "
        "tracking ends up. Shoot level with the water, not down at it — a "
        "downward angle foreshortens the rise out of the water, which is "
        "exactly the measurement the base score depends on."
    )
    ic1, ic2 = st.columns(2)
    figure_image(ic1, "camera_angle_guide.png", "Level with the waterline, not angled down.")
    figure_image(ic2, "framing_example.png", "Leave headroom above and below for the full rise and entry.")

    st.markdown('<div class="bl-label">Whatever footage you have</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            """<div class="bl-tile"><h4>WaltiCam</h4>
            <p>One split-screen clip, above and below at once — tracked
            by two trackers running side by side on the same video.</p></div>""",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """<div class="bl-tile"><h4>Above water only</h4>
            <p>A single above-water camera, with tent and pool-deck
            masking so background people don't get mistaken for the
            swimmer.</p></div>""",
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """<div class="bl-tile"><h4>Above and below</h4>
            <p>Both cameras, tracked separately. The score always comes
            from the above-water footage; the underwater clip adds
            coaching notes a judge on deck would never see.</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="bl-label">What actually gets measured</div>', unsafe_allow_html=True)
    st.write(
        "Ascent and descent alignment, backpike, leg extension, ankle "
        "extension, back roundness, travel, and unroll speed all count "
        "toward the score. Head tuck is measured but not yet counted — "
        "there's no calibrated threshold for it yet — and it's still "
        "shown rather than hidden. The base height score currently comes "
        "from how far the ankles clear the waterline as a fraction of the "
        "frame, which is sensitive to how far back the camera was set up; "
        "a body-length-normalized version of that same measurement is "
        "shown in the debug panel for comparison, but isn't wired into "
        "the score yet."
    )

    st.markdown('<div class="bl-label">Skip the shared server</div>', unsafe_allow_html=True)
    st.write(
        "This hosted version runs on shared, limited hardware — fine for "
        "one video at a time, slower if several people are using it at "
        "once. Running your own copy sidesteps that: no queue, no shared "
        "limits, and session history that doesn't vanish when a free "
        "hosted instance goes to sleep."
    )
    local_copy_button()

    if st.button("Start analyzing", type="primary"):
        go("analyze")


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def warm_up_control():
    """The pose model's weights take about a minute to download the
    first time rtmlib needs them in a session. This just triggers that
    download ahead of time so it doesn't land on whichever video gets
    processed first — it does NOT keep a tracker object alive for reuse,
    since tracker_core builds its own tracker per video; it only makes
    sure the weight files are already on disk by the time that happens."""
    if st.session_state.get("model_warmed"):
        st.success("Pose model weights are already downloaded for this session.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption(
            "First use of the pose model in a session downloads its "
            "weights (about a minute). Warm it up now if you'd rather not "
            "wait once your video is ready."
        )
    with c2:
        if st.button("Warm up model", use_container_width=True):
            with st.spinner("Downloading pose model weights..."):
                tc.make_pose_tracker(FAST_SETTINGS["mode"], FAST_SETTINGS["det_frequency"])
            st.session_state.model_warmed = True
            st.rerun()


def progress_tracker(label):
    bar = st.progress(0.0, text=f"Starting {label}...")

    def update(frame_count, total_frames):
        if total_frames > 0:
            pct = min(frame_count / total_frames, 1.0)
            frames_left = max(total_frames - frame_count, 0)
            bar.progress(pct, text=f"{label}: {frames_left} frame(s) left")

    return bar, update


def deliver_result(label, video_path, csv_path, landmarks):
    kalman_csv = tc.apply_kalman_filter_to_csv(csv_path, landmarks)
    st.success(f"{label} processing complete.")
    st.video(str(video_path))
    c1, c2 = st.columns(2)
    with c1:
        with open(video_path, "rb") as f:
            st.download_button(
                f"Download {label} video", data=f.read(),
                file_name=Path(video_path).name, mime="video/mp4",
                use_container_width=True, key=f"video_{label}",
            )
    with c2:
        with open(kalman_csv, "rb") as f:
            st.download_button(
                f"Download {label} data (CSV)", data=f.read(),
                file_name=Path(kalman_csv).name, mime="text/csv",
                use_container_width=True, key=f"csv_{label}",
            )
    return kalman_csv


def score_band(score):
    if score >= 9.5: return "Excellent", "#0f766e", "#ecfdf5"
    if score >= 8.5: return "Very good", "#1e3a5f", "#eef4fb"
    if score >= 7.5: return "Good", "#2563eb", "#eff6ff"
    if score >= 6.5: return "Competent", "#7c3aed", "#f5f3ff"
    if score >= 5.5: return "Satisfactory", "#d97706", "#fffbeb"
    if score >= 4.5: return "Deficient", "#dc2626", "#fef2f2"
    return "Weak", "#991b1b", "#fef2f2"


def render_score(above_csv, below_csv=None, swimmer_name="figure"):
    try:
        result = BarracudaScorer.score_single_pair(above_csv, below_csv, name=swimmer_name)
    except Exception as e:
        st.warning(f"Couldn't compute a score for this figure: {e}")
        return None

    st.markdown('<div class="bl-label">Result</div>', unsafe_allow_html=True)

    score = result["score"]
    tag, color, bg = score_band(score)
    st.markdown(
        f"""
        <div class="bl-scorebox">
            <div class="who">{(swimmer_name or "FIGURE").upper()}</div>
            <div><span class="num">{score:.2f}</span><span class="max"> / 10.0</span></div>
            <div class="tag" style="color:{color}; background:{bg};">{tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Base score (height)", f"{result['base_score']:.2f}")
    c2.metric("Total deduction", f"-{result['total_deduction']:.2f}")

    with st.expander("Debug info (what the height score is built from)"):
        st.write(
            "The base score comes from foot clearance — how far the "
            "ankles rise above the detected waterline, as a fraction of "
            "the frame. A body-length-normalized version is shown here "
            "too for reference; it isn't used for scoring yet."
        )
        st.json({
            "frames": result.get("frames"),
            "foot_clearance (frame-fraction)": result.get("foot_clearance"),
            "foot_clearance_normalized (body-lengths)": result.get("foot_clearance_normalized"),
            "body_scale (shoulder-to-ankle, px-frame-units)": result.get("body_scale"),
            "base_score": result.get("base_score"),
            "ascent_tilt_median (deg)": result.get("ascent_tilt_median"),
            "descent_tilt_median (deg)": result.get("descent_tilt_median"),
            "knee_angle_median (deg)": result.get("knee_angle_median"),
        })

    d = result["deductions"]
    official_rows = [
        ("Ascent alignment", "ascent_alignment"),
        ("Descent alignment", "descent_alignment"),
        ("Backpike", "backpike"),
        ("Leg extension", "leg_extension"),
        ("Ankle extension", "ankle_extension"),
        ("Back roundness", "back_roundness"),
        ("Travel", "travel"),
        ("Unroll speed", "unroll_speed"),
        ("Head tuck (not yet calibrated)", "head_tuck"),
    ]
    st.markdown("**Deductions counted toward the score**")
    st.table({
        "Category": [label for label, _ in official_rows],
        "Deduction": [f"-{d.get(key, 0):.2f}" if d.get(key, 0) else "None" for _, key in official_rows],
        "Measured": [
            f"{d[f'{key}_degrees']:.2f}" if d.get(f"{key}_degrees") is not None else "None"
            for _, key in official_rows
        ],
    })

    coaching_rows = [
        ("Underwater bent knee", "underwater_bent_knee", "underwater_bent_knee_degrees"),
        ("Back layout depth (not yet calibrated)", "back_layout_depth", "back_layout_depth_value"),
    ]
    st.markdown("**Coaching notes** (measured, not counted toward the score)")
    st.table({
        "Category": [label for label, _, _ in coaching_rows],
        "Deduction": [f"-{d.get(key, 0):.2f}" if d.get(key, 0) else "None" for _, key, _ in coaching_rows],
        "Measured": [
            f"{d[deg_key]:.2f}" if d.get(deg_key) is not None else "None"
            for _, _, deg_key in coaching_rows
        ],
    })

    if below_csv is None:
        st.caption("No underwater video this time, so there are no underwater-only coaching notes.")

    return result


def page_analyze():
    st.markdown('<div class="bl-label">Analyze</div>', unsafe_allow_html=True)
    top1, top2 = st.columns([4, 2])
    with top1:
        st.write("Upload footage below. Every mode uses the same pose-tracking speed.")
    with top2:
        if st.button("Filming tips", use_container_width=True):
            filming_tips_dialog()

    warm_up_control()

    source = st.radio(
        "Camera source", options=["Walticam", "Above / Below"],
        horizontal=True, label_visibility="collapsed",
    )
    swimmer_name = st.text_input("Swimmer ID (optional, shown on the score)", value="")

    below_water = False
    if source == "Above / Below":
        footage = st.radio(
            "What footage do you have?",
            options=["Just above-water", "Above + underwater"],
            horizontal=True,
        )
        below_water = (footage == "Above + underwater")
        if below_water:
            st.caption(
                "The underwater clip adds coaching-only notes (bent knee, "
                "back layout depth). The score itself always comes from "
                "the above-water footage."
            )

    MAX_SIZE_MB = 300

    if source == "Walticam":
        upload = st.file_uploader(
            "Upload a WaltiCam video (split-screen: top = above, bottom = below)",
            type=["mp4", "mov", "m4v", "avi"],
        )
        if upload is None:
            st.info("Upload a WaltiCam split-screen video to get started.")
            return

        size_mb = upload.size / (1024 * 1024)
        st.info(f"{upload.name} ({size_mb:.1f} MB)")
        if size_mb > MAX_SIZE_MB:
            st.error(f"That file is too large ({size_mb:.0f} MB) — please stay under {MAX_SIZE_MB} MB.")
            return
        if not st.button("Process video", type="primary", use_container_width=True):
            return

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path = Path(tmp_dir) / upload.name
            with open(in_path, "wb") as f:
                f.write(upload.getbuffer())
            out_path = Path(tmp_dir) / f"{in_path.stem}_walticam_tracking.mp4"

            bar, update = progress_tracker("Walticam")
            try:
                video_file, above_csv, below_csv = tc.process_video_walticam(
                    str(in_path), out_path,
                    mode=FAST_SETTINGS["mode"], det_frequency=FAST_SETTINGS["det_frequency"],
                    max_duration=MAX_CLIP_SECONDS, progress_callback=update,
                )
                bar.progress(1.0, text="Done.")
                st.session_state.model_warmed = True

                above_kalman = tc.apply_kalman_filter_to_csv(above_csv, tc.ALL_LANDMARKS_ABOVE)
                below_kalman = tc.apply_kalman_filter_to_csv(below_csv, tc.ALL_LANDMARKS_ABOVE)

                st.success("Walticam processing complete.")
                st.video(str(video_file))

                v1, v2, v3 = st.columns(3)
                with v1:
                    with open(video_file, "rb") as f:
                        st.download_button("Combined video", data=f.read(),
                                            file_name=Path(video_file).name, mime="video/mp4",
                                            use_container_width=True)
                with v2:
                    with open(above_kalman, "rb") as f:
                        st.download_button("Above data (CSV)", data=f.read(),
                                            file_name=Path(above_kalman).name, mime="text/csv",
                                            use_container_width=True)
                with v3:
                    with open(below_kalman, "rb") as f:
                        st.download_button("Below data (CSV)", data=f.read(),
                                            file_name=Path(below_kalman).name, mime="text/csv",
                                            use_container_width=True)

                result = render_score(above_kalman, below_kalman, swimmer_name=swimmer_name or "figure")
                if result is not None:
                    session_store.save_session(
                        swimmer_id=swimmer_name or "figure",
                        mode="Walticam",
                        score_result=result,
                        file_paths={"video": video_file, "above_csv": above_kalman, "below_csv": below_kalman},
                        official_keys=BarracudaScorer.__new__(BarracudaScorer)._deduction_keys(),
                    )
                    st.caption("Saved to History.")

            except Exception as e:
                st.error(f"Something went wrong during processing: {e}")
                st.exception(e)

    else:
        above_file = st.file_uploader(
            "Upload above-water video", type=["mp4", "mov", "m4v", "avi"], key="above_upload"
        )
        below_file = None
        if below_water:
            below_file = st.file_uploader(
                "Upload underwater video", type=["mp4", "mov", "m4v", "avi"], key="below_upload"
            )

        ready = above_file is not None and (not below_water or below_file is not None)
        if not ready:
            st.info("Upload your above-water video (and underwater too, if you have it) to get started.")
            return

        oversized = []
        if above_file.size / (1024 * 1024) > MAX_SIZE_MB:
            oversized.append(above_file.name)
        if below_file is not None and below_file.size / (1024 * 1024) > MAX_SIZE_MB:
            oversized.append(below_file.name)
        if oversized:
            st.error(f"These files are over {MAX_SIZE_MB} MB: {', '.join(oversized)}")
            return
        if not st.button("Process video(s)", type="primary", use_container_width=True):
            return

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            above_kalman = None
            below_kalman = None
            above_video = None
            below_video = None
            try:
                above_in = Path(tmp_dir) / above_file.name
                with open(above_in, "wb") as f:
                    f.write(above_file.getbuffer())
                above_out = Path(tmp_dir) / f"{above_in.stem}_above_tracking.mp4"

                _, update_above = progress_tracker("Above-water")
                video_file, csv_file = tc.process_video_above_water(
                    str(above_in), above_out,
                    mode=FAST_SETTINGS["mode"], det_frequency=FAST_SETTINGS["det_frequency"],
                    max_duration=MAX_CLIP_SECONDS, progress_callback=update_above,
                )
                st.session_state.model_warmed = True
                above_video = video_file
                above_kalman = deliver_result("Above-water", video_file, csv_file, tc.ALL_LANDMARKS_ABOVE)

                if below_water and below_file is not None:
                    below_in = Path(tmp_dir) / below_file.name
                    with open(below_in, "wb") as f:
                        f.write(below_file.getbuffer())
                    below_out = Path(tmp_dir) / f"{below_in.stem}_below_tracking.mp4"

                    _, update_below = progress_tracker("Underwater")
                    video_file, csv_file = tc.process_video_underwater(
                        str(below_in), below_out,
                        mode=FAST_SETTINGS["mode"], det_frequency=FAST_SETTINGS["det_frequency"],
                        max_duration=MAX_CLIP_SECONDS, progress_callback=update_below,
                    )
                    below_video = video_file
                    below_kalman = deliver_result("Underwater", video_file, csv_file, tc.ALL_LANDMARKS_UNDERWATER)

                if above_kalman is not None:
                    result = render_score(above_kalman, below_kalman, swimmer_name=swimmer_name or "figure")
                    if result is not None:
                        mode_label = "Above+Below" if below_kalman is not None else "Above-Water"
                        session_store.save_session(
                            swimmer_id=swimmer_name or "figure",
                            mode=mode_label,
                            score_result=result,
                            file_paths={
                                "above_video": above_video, "below_video": below_video,
                                "above_csv": above_kalman, "below_csv": below_kalman,
                            },
                            official_keys=BarracudaScorer.__new__(BarracudaScorer)._deduction_keys(),
                        )
                        st.caption("Saved to History.")

            except Exception as e:
                st.error(f"Something went wrong during processing: {e}")
                st.exception(e)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def athlete_progress_panel(sessions):
    swimmer_ids = sorted({s.get("swimmer_id", "Unknown") for s in sessions})
    if not swimmer_ids:
        return

    st.markdown('<div class="bl-label">Progress by athlete</div>', unsafe_allow_html=True)
    swimmer = st.selectbox("Athlete", options=swimmer_ids, key="progress_swimmer")

    their_sessions = sorted(
        (s for s in sessions if s.get("swimmer_id", "Unknown") == swimmer),
        key=lambda s: s.get("timestamp", ""),
    )
    scored = [s for s in their_sessions if s.get("score") is not None]

    if len(scored) >= 2:
        chart = pd.DataFrame(
            {"Score": [s["score"] for s in scored], "Base score": [s.get("base_score") for s in scored]},
            index=pd.to_datetime([s["timestamp"] for s in scored]),
        )
        st.line_chart(chart)
        delta = scored[-1]["score"] - scored[0]["score"]
        direction = "up" if delta > 0 else ("down" if delta < 0 else "unchanged")
        st.caption(
            f"{len(scored)} scored sessions for {swimmer}. Score is "
            f"{direction} {abs(delta):.2f} from first to most recent."
        )
    elif len(scored) == 1:
        st.info(f"Only one scored session for {swimmer} so far — a trend needs at least two.")
    else:
        st.info(f"No scored sessions for {swimmer} yet.")

    if their_sessions:
        latest = their_sessions[-1]
        st.markdown("**Top issues in the most recent session**")
        st.caption(f"{latest['timestamp']}: {latest.get('summary', 'No summary available.')}")


def page_history():
    st.markdown('<div class="bl-label">History</div>', unsafe_allow_html=True)
    st.caption(
        "Not persistent on free-tier hosting — sessions disappear if the "
        "app restarts or goes to sleep. A local `streamlit run` keeps "
        "them on disk."
    )

    sessions = session_store.load_sessions()
    if not sessions:
        st.info("Nothing scored yet — analyze a video and it'll show up here.")
        if st.button("Go to Analyze", type="primary"):
            go("analyze")
        return

    st.write(f"{len(sessions)} figure{'s' if len(sessions) != 1 else ''} scored")
    if st.button("Clear all sessions"):
        session_store.clear_all_sessions()
        st.rerun()

    athlete_progress_panel(sessions)

    st.markdown('<div class="bl-label">All sessions</div>', unsafe_allow_html=True)
    for s in sessions:
        score_str = f"{s['score']:.2f}/10" if s.get("score") is not None else "None"
        with st.expander(f"{s['swimmer_id']} - {score_str} ({s['timestamp']})"):
            st.write(f"Mode: {s['mode']}")
            if s.get("base_score") is not None and s.get("total_deduction") is not None:
                st.write(f"Base: {s['base_score']:.2f}  |  Deduction: -{s['total_deduction']:.2f}")
            st.write(f"Top issues: {s['summary']}")
            for key, rel_path in s.get("files", {}).items():
                fpath = session_store.session_file_path(rel_path)
                if fpath.exists():
                    with open(fpath, "rb") as f:
                        mime = "video/mp4" if "video" in key else "text/csv"
                        st.download_button(
                            key, data=f.read(), file_name=fpath.name, mime=mime,
                            key=f"hist_{s['id']}_{key}", use_container_width=True,
                        )
            if st.button("Delete", key=f"del_{s['id']}"):
                session_store.delete_session(s["id"])
                st.rerun()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def page_help():
    st.markdown('<div class="bl-label">Help</div>', unsafe_allow_html=True)
    st.write(
        "Something off with a score, the waterline, or the tracked "
        "skeleton? Describe it below and it's logged for review."
    )

    with st.form("issue_report_form"):
        name = st.text_input("Your name (optional)")
        contact_email = st.text_input("Your email, if you'd like a reply (optional)")
        page_context = st.selectbox(
            "Which part of the app is this about?",
            options=["Analyze - Walticam", "Analyze - Just above-water", "Analyze - Above + underwater",
                     "History", "Scoring / results", "Something else"],
        )
        description = st.text_area(
            "What happened?",
            placeholder="Example: the waterline was drawn well above the swimmer in this video, "
                        "or the score seemed off given the deductions shown.",
            height=140,
        )
        submitted = st.form_submit_button("Send report", type="primary")

    if submitted:
        if not description.strip():
            st.warning("Add a description before sending.")
        else:
            record, mailto_url = issue_reports.save_report(name, contact_email, description, page_context)
            st.success("Report saved.")
            st.markdown(f'<a href="{mailto_url}" target="_blank">Also send this by email</a>', unsafe_allow_html=True)
            st.caption(
                "Saved on this app's server. The email link opens your "
                "mail client with the report pre-filled, so it still "
                "reaches the developer even if this app's storage doesn't "
                "persist."
            )

    with st.expander("Previously submitted reports (this server's storage)"):
        reports = issue_reports.load_reports()
        if not reports:
            st.caption("No reports yet.")
        else:
            for r in reports:
                st.write(f"{r['timestamp']} - {r['name']} - {r['page_context']}")
                st.caption(r["description"])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_dispatch = {"home": page_home, "analyze": page_analyze, "history": page_history, "help": page_help}
_dispatch.get(st.session_state.page, page_home)()

if st.session_state.page != "analyze":
    st.divider()
    with st.expander("About " + APP_NAME):
        st.markdown(
            """
            Pose detection runs on RTMPose (via `rtmlib`), always at the
            same fast setting, in one of three modes:

            - **Walticam** — a single split-screen video (top = above,
              bottom = below), tracked with two trackers running side by
              side on the same frame.
            - **Above / Below** — separate above-water and/or underwater
              videos, each through its own tracker: tent/pool-deck
              masking and waterline detection above water, a synthesized
              mid-spine point underwater for back-curvature measurements,
              with Kalman filtering applied to both.

            Scoring is FINA-aligned: a base score from jump height, minus
            deductions for alignment, backpike, leg/ankle extension, back
            roundness, travel, and unroll speed — blended 70% against a
            fixed technical standard and 30% relative to other figures
            scored in the same batch (a single-figure run in this app
            uses the fixed standard only). Head tuck and back-layout
            depth are measured but not yet counted toward the score.

            Every scored figure is saved to History automatically.
            """
        )
    st.markdown(
        f'<div class="bl-foot">{APP_NAME} — RTMPose tracking, Kalman filtering, FINA-aligned scoring</div>',
        unsafe_allow_html=True,
    )
