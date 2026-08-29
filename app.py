#!/usr/bin/env python3
"""
Barracuda Tracker — Web App
===========================================================================
Streamlit front-end for tracker_core.py.

Three modes:
  - Walticam: single split-screen video (top=above, bottom=below)
  - Above / Below: separate above-water and/or underwater videos

Run locally:
    streamlit run app.py

Deploy: push this folder to a Hugging Face Space or Streamlit Cloud.
"""

import streamlit as st
import tempfile
import time
from pathlib import Path

import tracker_core as tc
from scorer import BarracudaScorer
import session_store

st.set_page_config(
    page_title="Barracuda Tracker",
    page_icon="🏊",
    layout="centered",
)

st.title("🏊 Barracuda Tracker")
st.caption(
    "Upload synchronized swimming footage to get pose tracking, waterline "
    "detection, and downloadable data."
)

# ── Sidebar: previous sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Previous Sessions")
    st.caption(
        "⚠️ Not persistent on free-tier hosting — sessions are lost if the "
        "app restarts or sleeps. Local `streamlit run` keeps them on disk."
    )
    _sessions = session_store.load_sessions()
    if not _sessions:
        st.caption("No sessions scored yet.")
    else:
        if st.button("🗑️ Clear all sessions"):
            session_store.clear_all_sessions()
            st.rerun()
        for _s in _sessions[:25]:
            _score_str = f"{_s['score']:.2f}/10" if _s.get("score") is not None else "—"
            with st.expander(f"{_s['swimmer_id']} — {_score_str} ({_s['timestamp']})"):
                st.write(f"**Mode:** {_s['mode']}")
                base = _s.get("base_score")
                ded = _s.get("total_deduction")
                st.write(
                    f"**Base:** {base:.2f}  |  **Deduction:** -{ded:.2f}"
                    if base is not None and ded is not None else ""
                )
                st.write(f"**Top issues:** {_s['summary']}")
                for _key, _rel_path in _s.get("files", {}).items():
                    _fpath = session_store.session_file_path(_rel_path)
                    if _fpath.exists():
                        with open(_fpath, "rb") as _f:
                            _mime = "video/mp4" if "video" in _key else "text/csv"
                            st.download_button(
                                f"⬇️ {_key}", data=_f.read(), file_name=_fpath.name,
                                mime=_mime, key=f"hist_{_s['id']}_{_key}",
                                use_container_width=True,
                            )
                if st.button("🗑️ Delete", key=f"del_{_s['id']}"):
                    session_store.delete_session(_s['id'])
                    st.rerun()

# ── Sidebar settings ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    speed_choice = st.select_slider(
        "Speed vs. accuracy",
        options=["Fast", "Balanced", "Most Accurate"],
        value="Fast",
        help=(
            "Fast: quickest results, good for previewing.\n"
            "Balanced: recommended default.\n"
            "Most Accurate: best joint precision, slowest (can take several "
            "minutes per video on this free server)."
        ),
    )
    speed_map = {
        "Fast": dict(mode="lightweight", det_frequency=4),
        "Balanced": dict(mode="balanced", det_frequency=2),
        "Most Accurate": dict(mode="performance", det_frequency=1),
    }
    chosen = speed_map[speed_choice]

    st.divider()
    manual_waterline = st.checkbox("Set waterline manually (Above/Below only)", value=False)
    waterline_value = None
    if manual_waterline:
        waterline_value = st.slider(
            "Waterline position (fraction from top of frame)",
            min_value=0.30, max_value=0.95, value=0.70, step=0.01,
        )

    st.divider()
    max_duration = st.slider(
        "Max figure duration to track (seconds)",
        min_value=10, max_value=90, value=60, step=5,
        help="Processing stops after this many seconds to keep runtimes reasonable.",
    )

    st.divider()
    st.caption(
        "⏱️ This app runs on shared CPU hardware. A 30–60 second clip can "
        "take a few minutes to process, especially on 'Most Accurate'."
    )

# ── Pill-button styling for the segmented controls below ──────────────────
# NOTE: previously this hid `div:first-child` inside each radio label to
# remove the circle indicator for a cleaner "pill" look. That targets
# Streamlit's internal (non-public, version-dependent) DOM structure —
# on some Streamlit versions that first child is the OPTION TEXT, not the
# circle, which makes the whole button render blank/invisible. Removed
# that rule; kept only the safe container/spacing styling.
st.markdown(
    """
    <style>
    div[role="radiogroup"] {
        display: flex; gap: 4px; background: rgba(120,120,120,0.12);
        padding: 4px; border-radius: 999px; width: fit-content;
    }
    div[role="radiogroup"] label { border-radius: 999px !important; padding: 6px 20px !important; margin: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Source toggle: Walticam vs Above/Below ────────────────────────────────
source = st.radio(
    "Camera source", options=["Walticam", "Above / Below"],
    horizontal=True, label_visibility="collapsed",
)

swimmer_id = st.text_input("Swimmer ID (optional, shown on the score)", value="")

above_water = True
below_water = True
if source == "Above / Below":
    col1, col2 = st.columns(2)
    with col1:
        above_water = st.checkbox("Above Water", value=True)
    with col2:
        below_water = st.checkbox("Below Water", value=True)
    if not above_water and not below_water:
        st.warning("Select at least one of Above Water / Below Water.")


def run_with_progress(label):
    """Returns (progress_bar, update_progress_fn) pair for a processing step."""
    progress_bar = st.progress(0.0, text=f"Starting {label}...")
    start_time = time.time()

    def update_progress(frame_count, total_frames):
        if total_frames > 0:
            pct = min(frame_count / total_frames, 1.0)
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
            progress_bar.progress(
                pct, text=f"{label}: frame {frame_count}/{total_frames} (~{eta:.0f}s remaining)"
            )
    return progress_bar, update_progress


def show_results(label, video_path, csv_path, landmarks):
    kalman_csv = tc.apply_kalman_filter_to_csv(csv_path, landmarks)
    st.success(f"✅ {label} processing complete!")
    st.video(str(video_path))
    col1, col2 = st.columns(2)
    with col1:
        with open(video_path, "rb") as f:
            st.download_button(
                f"⬇️ Download {label} Video", data=f.read(),
                file_name=Path(video_path).name, mime="video/mp4",
                use_container_width=True, key=f"video_{label}",
            )
    with col2:
        with open(kalman_csv, "rb") as f:
            st.download_button(
                f"⬇️ Download {label} Data (CSV)", data=f.read(),
                file_name=Path(kalman_csv).name, mime="text/csv",
                use_container_width=True, key=f"csv_{label}",
            )
    return kalman_csv


def show_score(above_kalman_csv, below_kalman_csv=None, name="figure"):
    """Run the barracuda scorer on the processed CSV(s) and display the
    score, base, and deduction breakdown. Requires an above-water CSV;
    the below-water CSV is optional and only adds informational metrics."""
    try:
        result = BarracudaScorer.score_single_pair(
            above_kalman_csv, below_kalman_csv, name=name
        )
    except Exception as e:
        st.warning(f"Could not compute a score for this figure: {e}")
        return

    st.divider()
    st.subheader("🏆 Barracuda Score")

    score = result["score"]
    if score >= 9.5:   assess = "Excellent / Near Perfect"
    elif score >= 8.5: assess = "Very Good"
    elif score >= 7.5: assess = "Good"
    elif score >= 6.5: assess = "Competent"
    elif score >= 5.5: assess = "Satisfactory"
    elif score >= 4.5: assess = "Deficient"
    else:              assess = "Weak"

    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{score:.2f} / 10")
    col2.metric("Base (height)", f"{result['base_score']:.2f}")
    col3.metric("Total Deduction", f"-{result['total_deduction']:.2f}")
    st.caption(f"Assessment: **{assess}**")

    with st.expander("🔍 Debug info (height calculation inputs)"):
        st.write(
            "The base score comes from **foot clearance** — how far the "
            "swimmer's ankles rise above the detected waterline, as a "
            "fraction of the frame height. If this looks wrong, it's "
            "usually one of: the waterline was detected in the wrong "
            "place, the tracker locked onto the wrong person for part of "
            "the clip, or the camera's distance/zoom differs from the "
            "footage the height chart was originally calibrated against "
            "(this measurement is in *frame-relative* units, not real "
            "world cm — a closer/farther camera changes the number for "
            "the same real jump height)."
        )
        st.json({
            "foot_clearance (frame-fraction, camera-dependent)": result.get("foot_clearance"),
            "foot_clearance_normalized (body-lengths, camera-independent)": result.get("foot_clearance_normalized"),
            "body_scale (shoulder-to-ankle, px-frame-units)": result.get("body_scale"),
            "base_score (currently uses foot_clearance, NOT the normalized version)": result.get("base_score"),
            "ascent_tilt_median (deg)": result.get("ascent_tilt_median"),
            "descent_tilt_median (deg)": result.get("descent_tilt_median"),
            "knee_angle_median (deg)": result.get("knee_angle_median"),
            "frames": result.get("frames"),
        })
        st.caption(
            "Since your camera distance/zoom varies between videos, compare "
            "`foot_clearance` (frame-fraction) against "
            "`foot_clearance_normalized` (body-lengths) across a few of "
            "your videos with known judge scores. If the normalized value "
            "tracks the real scores more consistently, that confirms "
            "camera distance is the issue — the next step is recalibrating "
            "the base-score chart around the normalized metric."
        )

    d = result["deductions"]
    scored_rows = [
        ("Ascent alignment", d.get("ascent_alignment", 0), d.get("ascent_alignment_degrees")),
        ("Descent alignment", d.get("descent_alignment", 0), d.get("descent_alignment_degrees")),
        ("Backpike", d.get("backpike", 0), d.get("backpike_degrees")),
        ("Leg extension", d.get("leg_extension", 0), d.get("leg_extension_degrees")),
        ("Ankle extension", d.get("ankle_extension", 0), d.get("ankle_extension_degrees")),
        ("Back roundness", d.get("back_roundness", 0), d.get("back_roundness_degrees")),
        ("Travel", d.get("travel", 0), d.get("travel_degrees")),
        ("Unroll speed", d.get("unroll_speed", 0), d.get("unroll_speed_degrees")),
        ("Head tuck (not yet calibrated)", d.get("head_tuck", 0), d.get("head_tuck_degrees")),
    ]
    st.markdown("**Official deductions**")
    st.table({
        "Category": [r[0] for r in scored_rows],
        "Deduction": [f"-{r[1]:.2f}" if r[1] else "—" for r in scored_rows],
        "Measured": [f"{r[2]:.2f}" if r[2] is not None else "—" for r in scored_rows],
    })

    coaching_rows = [
        ("Underwater bent knee", d.get("underwater_bent_knee", 0), d.get("underwater_bent_knee_degrees")),
        ("Back layout depth (not yet calibrated)", d.get("back_layout_depth", 0), d.get("back_layout_depth_value")),
    ]
    st.markdown("**Coaching feedback** _(measured, not counted toward the score)_")
    st.table({
        "Category": [r[0] for r in coaching_rows],
        "Deduction": [f"-{r[1]:.2f}" if r[1] else "—" for r in coaching_rows],
        "Measured": [f"{r[2]:.2f}" if r[2] is not None else "—" for r in coaching_rows],
    })

    st.caption(
        "ℹ️ Leg/ankle/underwater-knee tiers, back roundness cutoff, and the "
        "coaching-vs-official split come from the judges' meeting notes. "
        "Head tuck and back layout depth are measured placeholders (always "
        "0) pending exact point values from the judges."
    )

    if below_kalman_csv is None:
        st.caption(
            "ℹ️ No underwater video was processed, so underwater-only "
            "coaching feedback (bent knee, back layout depth) isn't available."
        )

    return result


MAX_SIZE_MB = 300

# ============================================================================
# WALTICAM MODE — single split-screen video
# ============================================================================
if source == "Walticam":
    uploaded_file = st.file_uploader(
        "Upload your WaltiCam video (split-screen: top=above, bottom=below)",
        type=["mp4", "mov", "m4v", "avi"],
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 {uploaded_file.name} ({file_size_mb:.1f} MB)")

        if file_size_mb > MAX_SIZE_MB:
            st.error(f"File is too large ({file_size_mb:.0f} MB). Please upload a video under {MAX_SIZE_MB} MB.")
        elif st.button("🚀 Process Video", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_path = Path(tmp_dir) / uploaded_file.name
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                output_path = Path(tmp_dir) / f"{input_path.stem}_walticam_tracking.mp4"

                progress_bar, update_progress = run_with_progress("Walticam")
                try:
                    with st.spinner("Loading pose model (first run downloads it, ~1 min)..."):
                        video_file, above_csv, below_csv = tc.process_video_walticam(
                            str(input_path), output_path,
                            mode=chosen["mode"], det_frequency=chosen["det_frequency"],
                            max_duration=max_duration, progress_callback=update_progress,
                        )
                    progress_bar.progress(1.0, text="Done!")

                    above_kalman = tc.apply_kalman_filter_to_csv(above_csv, tc.ALL_LANDMARKS_ABOVE)
                    below_kalman = tc.apply_kalman_filter_to_csv(below_csv, tc.ALL_LANDMARKS_ABOVE)

                    st.success("✅ Walticam processing complete!")
                    st.video(str(video_file))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with open(video_file, "rb") as f:
                            st.download_button("⬇️ Combined Video", data=f.read(),
                                                file_name=Path(video_file).name, mime="video/mp4",
                                                use_container_width=True)
                    with col2:
                        with open(above_kalman, "rb") as f:
                            st.download_button("⬇️ Above Data (CSV)", data=f.read(),
                                                file_name=Path(above_kalman).name, mime="text/csv",
                                                use_container_width=True)
                    with col3:
                        with open(below_kalman, "rb") as f:
                            st.download_button("⬇️ Below Data (CSV)", data=f.read(),
                                                file_name=Path(below_kalman).name, mime="text/csv",
                                                use_container_width=True)

                    score_result = show_score(above_kalman, below_kalman, name=swimmer_id or "figure")

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
                        st.caption("💾 Saved to session history (see sidebar).")

                except Exception as e:
                    st.error(f"Something went wrong during processing: {e}")
                    st.exception(e)

# ============================================================================
# ABOVE / BELOW MODE — separate videos, per checkbox
# ============================================================================
else:
    above_file, below_file = None, None

    if above_water:
        above_file = st.file_uploader(
            "Upload above-water video", type=["mp4", "mov", "m4v", "avi"], key="above_upload"
        )
    if below_water:
        below_file = st.file_uploader(
            "Upload underwater video", type=["mp4", "mov", "m4v", "avi"], key="below_upload"
        )

    ready = (not above_water or above_file is not None) and (not below_water or below_file is not None)
    any_selected = above_water or below_water

    if any_selected and ready and (above_file is not None or below_file is not None):
        oversized = []
        if above_file is not None and above_file.size / (1024 * 1024) > MAX_SIZE_MB:
            oversized.append(above_file.name)
        if below_file is not None and below_file.size / (1024 * 1024) > MAX_SIZE_MB:
            oversized.append(below_file.name)

        if oversized:
            st.error(f"These files are too large (over {MAX_SIZE_MB} MB): {', '.join(oversized)}")
        elif st.button("🚀 Process Video(s)", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                above_kalman_result = None
                below_kalman_result = None
                above_video_file = None
                below_video_file = None
                try:
                    with st.spinner("Loading pose model (first run downloads it, ~1 min)..."):
                        if above_water and above_file is not None:
                            above_input = Path(tmp_dir) / above_file.name
                            with open(above_input, "wb") as f:
                                f.write(above_file.getbuffer())
                            above_output = Path(tmp_dir) / f"{above_input.stem}_above_tracking.mp4"

                            _, update_above = run_with_progress("Above-water")
                            video_file, csv_file = tc.process_video_above_water(
                                str(above_input), above_output, waterline_value,
                                mode=chosen["mode"], det_frequency=chosen["det_frequency"],
                                max_duration=max_duration, progress_callback=update_above,
                            )
                            above_video_file = video_file
                            above_kalman_result = show_results(
                                "Above-Water", video_file, csv_file, tc.ALL_LANDMARKS_ABOVE
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
                            )
                            below_video_file = video_file
                            below_kalman_result = show_results(
                                "Underwater", video_file, csv_file, tc.ALL_LANDMARKS_UNDERWATER
                            )

                    # Score the figure once tracking is done. Scoring needs
                    # the above-water CSV at minimum; the underwater CSV
                    # (if present) adds informational hold/descent metrics.
                    if above_kalman_result is not None:
                        score_result = show_score(above_kalman_result, below_kalman_result,
                                    name=swimmer_id or "figure")

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
                            st.caption("💾 Saved to session history (see sidebar).")

                except Exception as e:
                    st.error(f"Something went wrong during processing: {e}")
                    st.exception(e)

st.divider()
with st.expander("ℹ️ About this tracker"):
    st.markdown(
        """
        This tool uses **RTMPose-x** (via `rtmlib`) for pose detection, with three modes:

        - **Walticam**: a single split-screen video (top half = above-water,
          bottom half = underwater), tracked with two lightweight RTMPose
          instances running in tandem.
        - **Above / Below**: separate above-water and/or underwater videos,
          each run through a dedicated tracker with:
            - Physical tent/background masking (above-water only)
            - Blue water color validation to reject tents and pool deck (above-water only)
            - Automatic waterline detection
            - Swimmer locking across frames
            - Multi-level smoothing and Kalman filtering
            - A synthesized `mid_spine` point (underwater only) for back-curvature measurements

        Output includes an annotated video with the detected skeleton and
        waterline, plus a CSV of per-frame joint positions for further analysis.
        """
    )
