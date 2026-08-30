#!/usr/bin/env python3
"""
BARRACUDA FIGURE SCORER — FINA Height-Based + Blended Deductions
=================================================================
Scoring approach:
  1. HEIGHT establishes a BASE SCORE from the FINA height chart
  2. DEDUCTIONS are 70% absolute (FINA standard) + 30% relative (group rank)
  3. Deductions are on a smooth 0.1-increment scale (interpolated between
     calibration anchor points) instead of big 0.2/0.5/1.0 jumps.

OFFICIAL deduction categories (count toward the score):
  1. Vertical alignment  — body tilt during ascent and descent (above water)
  2. Backpike            — body line post-peak
  3. Leg extension        — knee bend
  4. Ankle extension      — ankle/foot flex (mirrors leg extension tiers)
  5. Back roundness       — back layout curvature
  6. Travel               — lateral drift during the figure
  7. Unroll speed         — descent should be slower than the initial rise
  8. Head tuck            — STUB, no measurable criteria yet (see TODOs)

COACHING-ONLY categories (measured, shown as feedback, NOT counted toward
the official score by default — see INCLUDE_COACHING_IN_SCORE):
  9.  Underwater bent knee — same knee-bend tiers, from the underwater feed
  10. Back layout depth    — STUB, no measurable criteria yet (see TODOs)

============================================================================
CALIBRATION LOG — from the 2026 judges' meeting (Lara + 1 other judge)
============================================================================
DECIDED (implemented below with real numbers):
  - Leg extension tiers: 1-5° = small, 5-14° = medium, 15°+ = large
    deviation. Judges named the tiers as "small/medium/large" but did NOT
    give exact point values — this file assumes small=-0.1, medium=-0.3,
    large=-0.5 as a first pass. CONFIRM these three numbers with the judges.
  - Ankle extension uses the *same* tiers as leg extension (judges said
    "same for ankle extension").
  - Underwater bent knee uses the *same* tiers again, applied to the
    underwater camera's knee angle.
  - All deductions should be smooth/graduated in 0.1 steps, not jump by
    0.2/0.5/1.0 — implemented via _graduated_deduction() below.
  - Back roundness: "less than 30 degrees back will be rounded — look at
    the stomach." Only the 30° cutoff was given; the deduction MAGNITUDE
    below that cutoff is this file's placeholder guess. CONFIRM tier sizes.

STILL OPEN (measured where possible, deduction values are TODO stubs — do
NOT treat these numbers as judge-approved until confirmed):
  - Head tuck: no measurement definition or point values given yet.
    UPDATE: now measured (shoulder-vertex angle between hip and nose,
    deviation from 180°) and scored using the same small/medium/large
    tiers judges gave for leg/ankle extension, as a placeholder
    convention — NOT a judge-confirmed number. See _abs_head_tuck.
  - Back layout depth: "how far under should the swimmer be to receive a
    deduction, and how much deduction" — both blank in the notes.
    UPDATE: now scored against a guessed depth breakpoint chart — NOT
    judge-confirmed. See _abs_back_layout_depth /
    _BACK_LAYOUT_DEPTH_BREAKPOINTS.
  - Unrolling motion quality (hips unroll smoothly, head follows, descent
    slower than the initial rise): implemented as a rough speed-ratio
    check, but no judge-approved thresholds/point values yet.
  - Travel: no distance thresholds or point values given yet.
  - Whether underwater-only mistakes (judges can't normally see them)
    should count toward the OFFICIAL score or only show as COACHING
    feedback: still an open question from the meeting. This file defaults
    to "coaching feedback only" (INCLUDE_COACHING_IN_SCORE = False) for
    underwater_bent_knee and back_layout_depth — flip that flag once the
    meeting settles the question.
  - Base (height) score is still computed from `foot_clearance`, a
    fraction of FRAME height — which varies with camera distance/zoom.
    A camera-independent alternative, `foot_clearance_normalized` (foot
    clearance measured in units of the swimmer's own shoulder-to-ankle
    body length instead of frame height), is now measured and exposed
    per-figure, but NOT yet used for base_score — there's no judge-
    compared data yet to calibrate a new breakpoint chart around it.
    Compare `foot_clearance` vs `foot_clearance_normalized` across your
    varied-camera-distance videos; once `foot_clearance_normalized` looks
    like the more consistent predictor, we can build a new
    `_height_base_score` chart calibrated around it.

============================================================================
PEAK-FRAME AVERAGING (added after the above calibration log)
============================================================================
`foot_clearance` (and therefore `base_score`) used to come from a single
frame — whichever frame had the single highest ankle point in the whole
clip. A tracking glitch on exactly that one frame could throw the whole
base score off with nothing to catch it. `foot_clearance` is now the
MEDIAN of the top PEAK_AVERAGE_WINDOW (default 4) highest-ankle frames in
a small window around the detected peak, instead of that one frame's raw
value — median specifically because it's robust to one remaining outlier
even within the small averaged window. `peak_frame` itself (used to
window the ascent/descent/backpike measurements) is unchanged, so this
only smooths the height number, not everything else. `body_scale`
already worked this way (median over a window around peak_frame); this
just brings `foot_clearance` in line with that same approach.

USAGE:
    scorer = BarracudaScorer('/Users/mona/.../WaltiCam')
    scorer.score_all()
    scorer.print_summary_table()
    scorer.save_html_report()
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


class BarracudaScorer:

    ABSOLUTE_WEIGHT = 0.70
    RELATIVE_WEIGHT = 0.30

    # Whether coaching-only (underwater-invisible-to-judges) deductions get
    # subtracted from the official score, or only shown as feedback.
    # DEFAULT: feedback only. Flip to True once the judges decide.
    INCLUDE_COACHING_IN_SCORE = False

    # How many of the highest-ankle frames near the peak get averaged
    # (via median) into foot_clearance, instead of trusting one frame.
    PEAK_AVERAGE_WINDOW = 4

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.figures = {}
        self.results = {}
        if self.data_dir is not None:
            self._find_figures()

    # ── Figure discovery ──

    _LEGACY_ABOVE_RE = re.compile(
        r'^(?P<name>.+)_above_(?P<variant>POSE|FAST)_tracking(?:_data)?(?P<kalman>_KALMAN)?\.csv$')
    _LEGACY_BELOW_RE = re.compile(
        r'^(?P<name>.+)_below_(?P<variant>UNDERWATER)_tracking(?:_data)?(?P<kalman>_KALMAN)?\.csv$')

    _ABOVE_VARIANT_RANK = {('POSE', True): 0, ('POSE', False): 1,
                            ('FAST', True): 2, ('FAST', False): 3}
    _BELOW_VARIANT_RANK = {('UNDERWATER', True): 0, ('UNDERWATER', False): 1}

    def _find_figures_legacy_naming(self):
        above_candidates = {}
        below_candidates = {}

        for path in self.data_dir.rglob('*.csv'):
            m = self._LEGACY_ABOVE_RE.match(path.name)
            if m:
                name = m.group('name')
                variant = m.group('variant')
                kalman = bool(m.group('kalman'))
                rank = self._ABOVE_VARIANT_RANK[(variant, kalman)]
                above_candidates.setdefault(name, []).append((rank, path))
                continue
            m = self._LEGACY_BELOW_RE.match(path.name)
            if m:
                name = m.group('name')
                variant = m.group('variant')
                kalman = bool(m.group('kalman'))
                rank = self._BELOW_VARIANT_RANK[(variant, kalman)]
                below_candidates.setdefault(name, []).append((rank, path))

        if not above_candidates:
            return False

        print(f"  Using legacy naming convention (POSE/FAST/UNDERWATER "
              f"+ optional _KALMAN).")
        print(f"  Variant preference: POSE+KALMAN > POSE > FAST+KALMAN > FAST "
              f"(above), UNDERWATER+KALMAN > UNDERWATER (below).\n")

        for name in sorted(above_candidates.keys()):
            candidates = sorted(above_candidates[name], key=lambda c: c[0])
            best_rank, best_path = candidates[0]
            below_path = None
            if name in below_candidates:
                below_candidates[name].sort(key=lambda c: c[0])
                below_path = below_candidates[name][0][1]

            self.figures[name] = {'above': best_path, 'below': below_path}

            if len(candidates) > 1:
                skipped = ', '.join(p.name for _, p in candidates[1:])
                print(f"  Found: {name}")
                print(f"    above -> {best_path.name}  (also available, not used: {skipped})")
            else:
                print(f"  Found: {name}")
                print(f"    above -> {best_path.name}")
            print(f"    below -> {below_path.name if below_path else '(none found)'}")

        print(f"\n  Total: {len(self.figures)} figures (legacy naming)\n")
        return True

    def _find_figures(self):
        if not self.data_dir.exists():
            print(f"  ⚠ Directory does not exist: {self.data_dir}")
            print(f"    Double-check the path — it must point to the folder")
            print(f"    containing your *_above_tracking_data.csv files.\n")
            return

        above_files = sorted(self.data_dir.glob('*_above_tracking_data.csv'))
        searched_recursively = False
        if not above_files:
            above_files = sorted(self.data_dir.rglob('*_above_tracking_data.csv'))
            searched_recursively = True

        for ab_path in above_files:
            name = ab_path.name.replace('_above_tracking_data.csv', '')
            uw_path = ab_path.parent / f'{name}_below_tracking_data.csv'
            if name in self.figures:
                print(f"  ⚠ Duplicate figure name '{name}' found at {ab_path} "
                      f"— keeping the first one found, skipping this one.")
                continue
            self.figures[name] = {
                'above': ab_path,
                'below': uw_path if uw_path.exists() else None
            }
            print(f"  Found: {name}" + (f"  ({ab_path.parent})" if searched_recursively else ""))
        print(f"  Total: {len(self.figures)} figures"
              + (" (found via recursive subfolder search)\n" if searched_recursively and self.figures else "\n"))

        if not self.figures:
            if self._find_figures_legacy_naming():
                return

            print(f"  ⚠ No '*_above_tracking_data.csv' files found in:")
            print(f"    {self.data_dir}")
            print(f"    (searched this folder and all subfolders)")
            print(f"    Check that this is the same folder your tracker script")
            print(f"    writes its output CSVs to, and that filenames end in")
            print(f"    '_above_tracking_data.csv' / '_below_tracking_data.csv'.\n")

            try:
                entries = sorted(self.data_dir.iterdir())
            except Exception as e:
                entries = []
                print(f"    (couldn't list directory contents: {e})\n")

            if entries:
                print(f"    Contents of {self.data_dir}:")
                for e in entries[:25]:
                    tag = '/' if e.is_dir() else ''
                    print(f"      {e.name}{tag}")
                if len(entries) > 25:
                    print(f"      ... and {len(entries) - 25} more")
                print()

                csv_like = [e for e in entries if e.suffix.lower() == '.csv']
                if csv_like:
                    print(f"    Note: this folder DOES contain {len(csv_like)} CSV file(s),")
                    print(f"    but none end in '_above_tracking_data.csv'. If your tracker")
                    print(f"    uses different naming, either rename the files or update")
                    print(f"    the pattern in _find_figures().\n")
            else:
                print(f"    (folder exists but is empty)\n")

    # ── Helpers ──

    def _avg_lr(self, row, name):
        ly, ry = row.get(f'left_{name}_y'), row.get(f'right_{name}_y')
        lv = row.get(f'left_{name}_visibility')
        rv = row.get(f'right_{name}_visibility')
        l_ok = pd.notna(ly) and (pd.isna(lv) or lv > 0.1)
        r_ok = pd.notna(ry) and (pd.isna(rv) or rv > 0.1)
        if l_ok and r_ok: return (ly + ry) / 2
        elif l_ok: return ly
        elif r_ok: return ry
        return None

    def _avg_lr_x(self, row, name):
        lx, rx = row.get(f'left_{name}_x'), row.get(f'right_{name}_x')
        if pd.notna(lx) and pd.notna(rx): return (lx + rx) / 2
        elif pd.notna(lx): return lx
        elif pd.notna(rx): return rx
        return None

    def _single(self, row, name):
        """For unpaired landmarks like 'nose' (stored as plain nose_x/
        nose_y, not left_/right_-prefixed) — _avg_lr/_avg_lr_x don't
        apply since there's nothing to average."""
        v = row.get(f'{name}_y')
        vis = row.get(f'{name}_visibility')
        if pd.isna(v):
            return None
        if pd.notna(vis) and vis <= 0.1:
            return None
        return v

    def _single_x(self, row, name):
        v = row.get(f'{name}_x')
        return v if pd.notna(v) else None

    def _collect(self, df, joint):
        return [v for v in (self._avg_lr(df.iloc[i], joint)
                for i in range(len(df))) if v is not None]

    def _joint_angle(self, p1, p2, p3):
        v1 = (p1[0]-p2[0], p1[1]-p2[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        m1 = np.sqrt(v1[0]**2 + v1[1]**2)
        m2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if m1 < 0.001 or m2 < 0.001: return 180.0
        return np.degrees(np.arccos(np.clip(dot/(m1*m2), -1, 1)))

    # ── Height chart → base score ──

    # Height chart, in frame-fraction units — used for Above and
    # Above+Below modes, where the above-water camera has a full,
    # dedicated view of the swimmer.
    _HEIGHT_BASE_BREAKPOINTS_FRAME_FRACTION = [
        (0.33, 10.0), (0.30, 9.5), (0.27, 9.0), (0.24, 8.5),
        (0.21, 8.0), (0.18, 7.5), (0.15, 7.0), (0.12, 6.5),
        (0.09, 6.0), (0.06, 5.0), (0.03, 4.0), (0.00, 3.0),
    ]

    # WALTICAM-SPECIFIC height chart, in body-length units.
    #
    # A WaltiCam splits ONE camera's frame into top (above-water) and
    # bottom (underwater) halves — so the above-water half has roughly
    # HALF the vertical field of view of a dedicated above-water camera
    # pointed at the same swimmer. The same real jump therefore occupies
    # a LARGER fraction of a WaltiCam half-frame than of a full dedicated
    # above-water frame. Using the frame-fraction chart directly on
    # WaltiCam data would systematically overestimate the base score.
    #
    # foot_clearance_normalized (body-lengths, not frame-fraction)
    # sidesteps this: both the swimmer's body and their clearance are
    # measured in the SAME frame, so field-of-view differences cancel
    # out. This is why Walticam mode uses a different metric AND a
    # different chart from Above / Above+Below.
    #
    # PLACEHOLDER CALIBRATION: this chart is derived by converting the
    # frame-fraction chart through an ASSUMED typical body_scale of 0.35
    # (shoulder-to-ankle length as a fraction of frame height, from the
    # original above-water calibration videos) — NOT real Walticam-
    # specific calibration data. Confirm/recalibrate once you have
    # Walticam videos with known judge scores to check against.
    _ASSUMED_TYPICAL_BODY_SCALE_FOR_CONVERSION = 0.35
    _HEIGHT_BASE_BREAKPOINTS_BODY_LENGTHS = [
        (round(x / 0.35, 4), s)
        for x, s in _HEIGHT_BASE_BREAKPOINTS_FRAME_FRACTION
    ]

    def _height_base_score(self, clearance_value, breakpoints):
        if clearance_value >= breakpoints[0][0]:
            return breakpoints[0][1]
        if clearance_value <= breakpoints[-1][0]:
            return breakpoints[-1][1]
        for i in range(len(breakpoints) - 1):
            cl_hi, sc_hi = breakpoints[i]
            cl_lo, sc_lo = breakpoints[i + 1]
            if clearance_value >= cl_lo:
                t = (clearance_value - cl_lo) / (cl_hi - cl_lo)
                return sc_lo + t * (sc_hi - sc_lo)
        return 3.0

    def _compute_base_score(self, m, source_mode):
        """Picks the right height metric + chart for the given source
        mode. Returns (base_score, metric_name_used)."""
        if source_mode == 'walticam' and m.get('foot_clearance_normalized') is not None:
            base = self._height_base_score(
                m['foot_clearance_normalized'], self._HEIGHT_BASE_BREAKPOINTS_BODY_LENGTHS
            )
            return round(base, 2), 'foot_clearance_normalized (body-lengths, walticam mode)'
        base = self._height_base_score(
            m.get('foot_clearance', 0), self._HEIGHT_BASE_BREAKPOINTS_FRAME_FRACTION
        )
        return round(base, 2), 'foot_clearance (frame-fraction)'

    # ── Graduated (0.1-increment) deduction scale ──

    def _graduated_deduction(self, value, breakpoints, cap=1.0, unknown_default=0.5):
        """breakpoints: ascending list of (value, deduction) anchor pairs."""
        if value is None:
            return round(min(cap, unknown_default), 2)
        if value <= breakpoints[0][0]:
            d = breakpoints[0][1]
        elif value >= breakpoints[-1][0]:
            d = breakpoints[-1][1]
        else:
            d = breakpoints[-1][1]
            for i in range(len(breakpoints) - 1):
                v0, d0 = breakpoints[i]
                v1, d1 = breakpoints[i + 1]
                if v0 <= value <= v1:
                    t = (value - v0) / (v1 - v0) if v1 > v0 else 0.0
                    d = d0 + t * (d1 - d0)
                    break
        d = round(min(cap, max(0.0, d)) / 0.1) * 0.1
        return round(d, 2)

    _VERTICAL_ALIGNMENT_BREAKPOINTS = [
        (0, 0.0), (3, 0.0), (5, 0.2), (7, 0.4), (9, 0.6), (12, 0.8), (20, 1.0)
    ]
    _BACKPIKE_BREAKPOINTS = [
        (0, 0.0), (5, 0.2), (10, 0.3), (20, 0.5), (30, 0.8), (45, 1.0)
    ]
    _BEND_DEVIATION_BREAKPOINTS = [
        (0, 0.0), (1, 0.1), (5, 0.3), (15, 0.5),
    ]

    def _abs_vertical_alignment(self, tilt):
        return self._graduated_deduction(tilt, self._VERTICAL_ALIGNMENT_BREAKPOINTS)

    def _abs_backpike(self, bp):
        return self._graduated_deduction(bp, self._BACKPIKE_BREAKPOINTS, unknown_default=0.0)

    def _abs_bend_deviation(self, deviation_degrees):
        """Shared by leg extension, ankle extension, and underwater bent
        knee — all use the same judge-given tiers."""
        return self._graduated_deduction(
            deviation_degrees, self._BEND_DEVIATION_BREAKPOINTS, unknown_default=0.3
        )

    # FIX: previously took the raw joint angle (180° = straight) directly,
    # with breakpoints covering only 0-30°. A real swimmer's back angle is
    # almost always 100-180° in a back layout, which is entirely outside
    # that 0-30° domain — so every real figure silently clamped to 0
    # deduction regardless of actual roundness. Now uses DEVIATION from
    # straight (180 - angle) instead, matching how leg/ankle extension
    # already work, so the breakpoint domain overlaps with realistic
    # values. The judges' one confirmed number — "less than 30 degrees
    # back will be rounded" — is read here as "30°+ of deviation from
    # straight counts as rounded." Point values beyond that are still this
    # file's placeholder guess (see calibration log at the top).
    _BACK_ROUNDNESS_BREAKPOINTS = [
        (0, 0.0), (15, 0.2), (30, 0.4), (50, 0.6),
    ]

    def _abs_back_roundness(self, deviation_degrees):
        return self._graduated_deduction(deviation_degrees, self._BACK_ROUNDNESS_BREAKPOINTS, unknown_default=0.3)

    _TRAVEL_BREAKPOINTS = [
        (0.00, 0.0), (0.05, 0.2), (0.10, 0.5), (0.15, 1.0),
    ]
    _UNROLL_SPEED_RATIO_BREAKPOINTS = [
        (0.5, 0.0), (0.8, 0.1), (1.0, 0.3), (1.3, 0.5),
    ]

    def _abs_travel(self, hip_x_range):
        return self._graduated_deduction(hip_x_range, self._TRAVEL_BREAKPOINTS, unknown_default=0.0)

    def _abs_unroll_speed(self, descent_to_ascent_speed_ratio):
        return self._graduated_deduction(
            descent_to_ascent_speed_ratio, self._UNROLL_SPEED_RATIO_BREAKPOINTS, unknown_default=0.0
        )

    # ESTIMATED — no judge-given point values exist for head tuck (see
    # calibration log at the top: "no measurement definition or point
    # values given yet"). This reuses the SAME small/medium/large tiers
    # judges gave for leg/ankle extension (_BEND_DEVIATION_BREAKPOINTS),
    # since that's the only real judge-approved deduction scale in this
    # file and head tuck is the same kind of "how far off straight"
    # measurement. This is a placeholder convention, NOT a judge-
    # confirmed number — replace with real thresholds once the judges
    # give them, the same way leg/ankle extension were confirmed.
    def _abs_head_tuck(self, deviation_degrees):
        return self._graduated_deduction(
            deviation_degrees, self._BEND_DEVIATION_BREAKPOINTS, unknown_default=0.0
        )

    # ESTIMATED — the meeting notes left both "how far under" and "how
    # much deduction" blank. These breakpoints are a first-pass guess:
    # back_layout_depth_start is the swimmer's median hip depth (as a
    # frame-fraction below the water_level=0.05 convention) during the
    # first ~10 underwater frames, so 0.15-0.20 assumes a fairly shallow
    # layout is normal and depths beyond that start costing points.
    # NOT judge-confirmed — recalibrate once real numbers exist.
    _BACK_LAYOUT_DEPTH_BREAKPOINTS = [
        (0.15, 0.0), (0.25, 0.2), (0.35, 0.4), (0.50, 0.6),
    ]

    def _abs_back_layout_depth(self, depth_value):
        return self._graduated_deduction(
            depth_value, self._BACK_LAYOUT_DEPTH_BREAKPOINTS, unknown_default=0.0
        )

    # ── Relative deduction ──

    def _relative_deduction(self, value, all_values, max_deduction, higher_is_worse=True):
        valid = [v for v in all_values if v is not None]
        if not valid or value is None:
            return 0.0
        if len(valid) == 1:
            return 0.0
        best = min(valid) if higher_is_worse else max(valid)
        worst = max(valid) if higher_is_worse else min(valid)
        if best == worst:
            return 0.0
        if higher_is_worse:
            t = (value - best) / (worst - best)
        else:
            t = (best - value) / (best - worst)
        t = np.clip(t, 0, 1)
        return round(t * max_deduction, 2)

    # ── Measurements ──

    def _angle_series(self, df, p1_name, p2_name, p3_name, frame_range):
        angles = []
        for fn in frame_range:
            if fn < 0 or fn >= len(df):
                continue
            row = df.iloc[fn]
            p1 = (self._avg_lr_x(row, p1_name), self._avg_lr(row, p1_name))
            p2 = (self._avg_lr_x(row, p2_name), self._avg_lr(row, p2_name))
            p3 = (self._avg_lr_x(row, p3_name), self._avg_lr(row, p3_name))
            if all(v is not None for v in p1 + p2 + p3):
                angles.append(self._joint_angle(p1, p2, p3))
        return np.median(angles) if angles else None

    def _extract_measurements(self, name):
        paths = self.figures[name]
        ab = pd.read_csv(paths['above'])
        uw = pd.read_csv(paths['below']) if paths['below'] else None
        wl_ab = ab['water_level'].median()

        m = {'name': name, 'frames': len(ab)}

        if len(ab) > 1:
            dt = ab.iloc[1]['time_seconds'] - ab.iloc[0]['time_seconds']
            m['fps'] = 1.0 / dt if dt > 0 else 30.0
        else:
            m['fps'] = 30.0

        ab_ankles = self._collect(ab, 'ankle')

        # peak_frame: found via a lightly SMOOTHED ankle series (rolling
        # median, window=5) instead of the raw single-frame minimum. The
        # raw version was susceptible to one glitchy frame becoming the
        # single most extreme value in the whole clip — and since EVERY
        # other above-water measurement (ascent tilt, descent tilt,
        # backpike, ankle extension, body_scale) windows around
        # peak_frame, a misplaced peak_frame doesn't just skew one number,
        # it can misplace all of them at once. foot_clearance already got
        # its own glitch-resistance (median of the top N near-peak
        # frames); this fixes the shared root cause upstream of that.
        peak_frame = None
        min_ankle = min(ab_ankles) if ab_ankles else None
        if ab_ankles:
            ankle_series = pd.Series(
                [self._avg_lr(ab.iloc[i], 'ankle') for i in range(len(ab))]
            )
            smoothed = ankle_series.rolling(window=5, center=True, min_periods=1).median()
            if smoothed.notna().any():
                peak_frame = int(smoothed.idxmin())

        # foot_clearance: median of the PEAK_AVERAGE_WINDOW highest-ankle
        # frames in a small window around peak_frame, instead of trusting
        # peak_frame's single value alone. Guards against a tracking
        # glitch on exactly the peak frame skewing the base score. Median
        # (not mean) so one remaining outlier even within this small
        # window still can't dominate. peak_frame ITSELF is unchanged —
        # ascent/descent/backpike windows below still center on it.
        if ab_ankles:
            if peak_frame is not None:
                half = self.PEAK_AVERAGE_WINDOW // 2
                window_vals = []
                for fn in range(max(0, peak_frame - half - 1), min(len(ab), peak_frame + half + 2)):
                    a = self._avg_lr(ab.iloc[fn], 'ankle')
                    if a is not None:
                        window_vals.append(a)
                if window_vals:
                    window_vals.sort()  # ascending y = highest point first
                    top_n = window_vals[:self.PEAK_AVERAGE_WINDOW]
                    peak_ankle_y = float(np.median(top_n))
                else:
                    peak_ankle_y = min_ankle
            else:
                peak_ankle_y = min_ankle
            m['foot_clearance'] = wl_ab - peak_ankle_y
        else:
            m['foot_clearance'] = 0

        # Ascent tilt + knee angle
        ascent_tilts = []
        knee_angles = []
        ascent_window = range(max(0, (peak_frame or 0) - 7), min(len(ab), (peak_frame or 0) + 8)) \
            if peak_frame is not None else range(0)
        if peak_frame is not None:
            for fn in ascent_window:
                hx = self._avg_lr_x(ab.iloc[fn], 'hip')
                hy = self._avg_lr(ab.iloc[fn], 'hip')
                ax = self._avg_lr_x(ab.iloc[fn], 'ankle')
                ay = self._avg_lr(ab.iloc[fn], 'ankle')
                kx = self._avg_lr_x(ab.iloc[fn], 'knee')
                ky = self._avg_lr(ab.iloc[fn], 'knee')
                if all(v is not None for v in [hx, hy, ax, ay]):
                    dx = ax - hx; dy = ay - hy
                    ascent_tilts.append(abs(np.degrees(np.arctan2(dx, abs(dy)))))
                if all(v is not None for v in [hx, hy, kx, ky, ax, ay]):
                    knee_angles.append(self._joint_angle((hx, hy), (kx, ky), (ax, ay)))

        m['ascent_tilt_median'] = np.median(ascent_tilts) if ascent_tilts else None

        # Descent tilt
        descent_tilts = []
        descent_window = range(peak_frame, min(len(ab), peak_frame + 40)) if peak_frame is not None else range(0)
        if peak_frame is not None:
            for fn in descent_window:
                hx = self._avg_lr_x(ab.iloc[fn], 'hip')
                hy = self._avg_lr(ab.iloc[fn], 'hip')
                ax = self._avg_lr_x(ab.iloc[fn], 'ankle')
                ay = self._avg_lr(ab.iloc[fn], 'ankle')
                if all(v is not None for v in [hx, hy, ax, ay]):
                    if ay < wl_ab:
                        dx = ax - hx; dy = ay - hy
                        descent_tilts.append(abs(np.degrees(np.arctan2(dx, abs(dy)))))

        m['descent_tilt_median'] = np.median(descent_tilts) if descent_tilts else None

        ascent = m['ascent_tilt_median']
        descent = m['descent_tilt_median']
        if ascent is not None and descent is not None:
            m['worst_tilt'] = max(ascent, descent)
        elif ascent is not None:
            m['worst_tilt'] = ascent
        elif descent is not None:
            m['worst_tilt'] = descent
        else:
            m['worst_tilt'] = None

        m['knee_angle_median'] = np.median(knee_angles) if knee_angles else None
        m['leg_extension_deviation'] = (
            abs(180.0 - m['knee_angle_median']) if m['knee_angle_median'] is not None else None
        )

        # Ankle extension (above water), same window as knee
        ankle_angle = None
        for foot_pt in ('heel', 'foot_index', 'foot_best'):
            col_check = f'left_{foot_pt}_y'
            if col_check in ab.columns:
                ankle_angle = self._angle_series(ab, 'knee', 'ankle', foot_pt, ascent_window)
                if ankle_angle is not None:
                    break
        m['ankle_angle_median'] = ankle_angle
        m['ankle_extension_deviation'] = (
            abs(180.0 - ankle_angle) if ankle_angle is not None else None
        )

        # Back roundness — first ~10 frames as a stand-in for "back layout"
        layout_window = range(0, min(10, len(ab)))
        m['back_angle_median'] = self._angle_series(ab, 'shoulder', 'hip', 'knee', layout_window)
        m['back_roundness_deviation'] = (
            abs(180.0 - m['back_angle_median']) if m['back_angle_median'] is not None else None
        )

        # Head tuck (ESTIMATED — see _abs_head_tuck for why). Angle at
        # the shoulder between the spine line (down to the hip) and the
        # neck/head line (up to the nose), same layout_window as back
        # roundness. In a neutral streamlined position the nose roughly
        # continues the spine's line (~180°); a tucked chin bends this
        # angle away from that, same "deviation from straight" pattern
        # used for leg/ankle extension and back roundness.
        head_tuck_angles = []
        for fn in layout_window:
            row = ab.iloc[fn]
            hx, hy = self._avg_lr_x(row, 'hip'), self._avg_lr(row, 'hip')
            sx, sy = self._avg_lr_x(row, 'shoulder'), self._avg_lr(row, 'shoulder')
            nx, ny = self._single_x(row, 'nose'), self._single(row, 'nose')
            if all(v is not None for v in [hx, hy, sx, sy, nx, ny]):
                head_tuck_angles.append(self._joint_angle((hx, hy), (sx, sy), (nx, ny)))
        m['head_tuck_angle_median'] = np.median(head_tuck_angles) if head_tuck_angles else None
        m['head_tuck_deviation'] = (
            abs(180.0 - m['head_tuck_angle_median']) if m['head_tuck_angle_median'] is not None else None
        )

        # Travel: hip x-range over the whole above-water clip
        hip_xs = [v for v in (self._avg_lr_x(ab.iloc[i], 'hip') for i in range(len(ab))) if v is not None]
        m['hip_travel_range'] = (max(hip_xs) - min(hip_xs)) if hip_xs else None

        # Unroll speed ratio: descent vertical speed vs ascent vertical speed
        def _avg_vertical_speed(frame_range):
            ys = []
            for fn in frame_range:
                if 0 <= fn < len(ab):
                    hy = self._avg_lr(ab.iloc[fn], 'hip')
                    if hy is not None:
                        ys.append(hy)
            if len(ys) < 2:
                return None
            diffs = [abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
            return np.mean(diffs) if diffs else None

        ascent_speed = _avg_vertical_speed(ascent_window)
        descent_speed = _avg_vertical_speed(descent_window)
        if ascent_speed and ascent_speed > 1e-6 and descent_speed is not None:
            m['unroll_speed_ratio'] = descent_speed / ascent_speed
        else:
            m['unroll_speed_ratio'] = None

        # NEW (from last round) — camera-distance-independent version of
        # foot clearance: swimmer's own shoulder-to-ankle length near the
        # peak, used to express clearance in body-lengths instead of
        # frame-height fraction. NOT used for base_score yet.
        body_scale = None
        if peak_frame is not None:
            scale_window = range(max(0, peak_frame - 5), min(len(ab), peak_frame + 6))
            lengths = []
            for fn in scale_window:
                row = ab.iloc[fn]
                sx, sy = self._avg_lr_x(row, 'shoulder'), self._avg_lr(row, 'shoulder')
                ax, ay = self._avg_lr_x(row, 'ankle'), self._avg_lr(row, 'ankle')
                if all(v is not None for v in [sx, sy, ax, ay]):
                    lengths.append(np.sqrt((ax - sx) ** 2 + (ay - sy) ** 2))
            body_scale = np.median(lengths) if lengths else None
        m['body_scale'] = body_scale
        m['foot_clearance_normalized'] = (
            m['foot_clearance'] / body_scale if body_scale and body_scale > 1e-6 else None
        )

        # Backpike
        if peak_frame is not None:
            bp_angles = []
            for fn in descent_window:
                hx = self._avg_lr_x(ab.iloc[fn], 'hip')
                hy = self._avg_lr(ab.iloc[fn], 'hip')
                kx = self._avg_lr_x(ab.iloc[fn], 'knee')
                ky = self._avg_lr(ab.iloc[fn], 'knee')
                if all(v is not None for v in [hx, hy, kx, ky]):
                    if ky < wl_ab - 0.05:
                        dx = kx - hx
                        dy = hy - ky
                        if dy > 0.01:
                            bp_angles.append(abs(np.degrees(np.arctan2(dx, dy))))
            if bp_angles:
                m['backpike_worst'] = max(bp_angles)
                m['backpike_sustained'] = np.median(sorted(bp_angles, reverse=True)[:5])
                m['backpike_score'] = m['backpike_worst'] * 0.6 + m['backpike_sustained'] * 0.4
            else:
                m['backpike_worst'] = 0
                m['backpike_sustained'] = 0
                m['backpike_score'] = 0
        else:
            m['backpike_worst'] = 0
            m['backpike_sustained'] = 0
            m['backpike_score'] = 0

        # Underwater metrics
        if uw is not None:
            uw_hips = [(i, self._avg_lr(uw.iloc[i], 'hip')) for i in range(len(uw))]
            uw_hips = [(i, h) for i, h in uw_hips if h is not None]

            if uw_hips:
                peak_i, peak_h = min(uw_hips, key=lambda x: x[1])

                threshold = peak_h + 0.10
                hold_frames = sum(1 for i, h in uw_hips if h <= threshold and i >= peak_i)
                m['hold_duration_sec'] = round(hold_frames / m['fps'], 2)

                post_peak = [(i, h) for i, h in uw_hips if i > peak_i + 3 and i < peak_i + 25]
                if len(post_peak) >= 3:
                    m['descent_rate'] = (post_peak[-1][1] - post_peak[0][1]) / \
                                        (post_peak[-1][0] - post_peak[0][0])
                else:
                    m['descent_rate'] = 0.01

                hold_start = peak_i + 3
                hold_end = min(peak_i + 25, len(uw))
                hold_hips = [h for i, h in uw_hips if hold_start <= i < hold_end]
                m['hold_stability_std'] = np.std(hold_hips) if hold_hips else 0.15

                uw_window = range(max(0, peak_i - 7), min(len(uw), peak_i + 8))
                uw_knee_angle = self._angle_series(uw, 'hip', 'knee', 'ankle', uw_window)
                m['underwater_knee_angle_median'] = uw_knee_angle
                m['underwater_knee_deviation'] = (
                    abs(180.0 - uw_knee_angle) if uw_knee_angle is not None else None
                )

                start_window = range(0, min(10, len(uw)))
                start_hip_depths = [
                    self._avg_lr(uw.iloc[i], 'hip') for i in start_window
                    if self._avg_lr(uw.iloc[i], 'hip') is not None
                ]
                m['back_layout_depth_start'] = (
                    np.median(start_hip_depths) if start_hip_depths else None
                )
            else:
                m['underwater_knee_deviation'] = None
                m['back_layout_depth_start'] = None
        else:
            m['underwater_knee_deviation'] = None
            m['back_layout_depth_start'] = None

        return m

    # ── Deduction key groups ──

    def _deduction_keys(self):
        """OFFICIAL categories — subtracted from the score."""
        keys = ['ascent_alignment', 'descent_alignment', 'backpike',
                'leg_extension', 'ankle_extension', 'back_roundness',
                'travel', 'unroll_speed', 'head_tuck']
        if self.INCLUDE_COACHING_IN_SCORE:
            keys = keys + self._coaching_deduction_keys()
        return keys

    def _coaching_deduction_keys(self):
        """Underwater-only categories judges normally can't see."""
        return ['underwater_bent_knee', 'back_layout_depth']

    def _compute_all_deductions(self, m, group_values=None):
        d = {}
        gv = group_values or {}

        def blended(key, value, all_key, abs_fn, higher_is_worse=True):
            abs_ded = abs_fn(value)
            if group_values is not None:
                rel_ded = self._relative_deduction(
                    value, gv.get(all_key, []), 1.0, higher_is_worse=higher_is_worse)
                total = min(1.0, round(self.ABSOLUTE_WEIGHT * abs_ded + self.RELATIVE_WEIGHT * rel_ded, 2))
            else:
                rel_ded = 0.0
                total = round(abs_ded, 2)
            d[key] = total
            d[f'{key}_abs'] = abs_ded
            d[f'{key}_rel'] = round(rel_ded, 2)
            d[f'{key}_degrees'] = round(value, 1) if value is not None else None

        blended('ascent_alignment', m.get('ascent_tilt_median'), 'ascent',
                 self._abs_vertical_alignment, higher_is_worse=True)
        blended('descent_alignment', m.get('descent_tilt_median'), 'descent',
                 self._abs_vertical_alignment, higher_is_worse=True)
        blended('backpike', m.get('backpike_score', 0), 'backpike',
                 self._abs_backpike, higher_is_worse=True)
        blended('leg_extension', m.get('leg_extension_deviation'), 'leg_ext',
                 self._abs_bend_deviation, higher_is_worse=True)
        blended('ankle_extension', m.get('ankle_extension_deviation'), 'ankle_ext',
                 self._abs_bend_deviation, higher_is_worse=True)
        blended('back_roundness', m.get('back_roundness_deviation'), 'back_roundness',
                 self._abs_back_roundness, higher_is_worse=True)
        blended('travel', m.get('hip_travel_range'), 'travel',
                 self._abs_travel, higher_is_worse=True)
        blended('unroll_speed', m.get('unroll_speed_ratio'), 'unroll',
                 self._abs_unroll_speed, higher_is_worse=True)
        blended('head_tuck', m.get('head_tuck_deviation'), 'head_tuck',
                 self._abs_head_tuck, higher_is_worse=True)

        hd = self._abs_bend_deviation(m.get('underwater_knee_deviation'))
        d['underwater_bent_knee'] = round(hd, 2)
        d['underwater_bent_knee_degrees'] = (
            round(m['underwater_knee_deviation'], 1) if m.get('underwater_knee_deviation') is not None else None
        )
        d['back_layout_depth'] = round(self._abs_back_layout_depth(m.get('back_layout_depth_start')), 2)
        d['back_layout_depth_value'] = m.get('back_layout_depth_start')

        return d

    # ── Two-pass scoring: extract all, then compute blended deductions ──

    def score_all(self, source_mode='above'):
        if not self.figures:
            print("  ⚠ No figures loaded — nothing to score.")
            print("    Check the data_dir path passed to BarracudaScorer() and")
            print("    that it contains '*_above_tracking_data.csv' files.\n")
            return self.results

        measurements = {name: self._extract_measurements(name) for name in sorted(self.figures.keys())}

        group_values = {
            'ascent': [measurements[n].get('ascent_tilt_median') for n in measurements],
            'descent': [measurements[n].get('descent_tilt_median') for n in measurements],
            'backpike': [measurements[n].get('backpike_score', 0) for n in measurements],
            'leg_ext': [measurements[n].get('leg_extension_deviation') for n in measurements],
            'ankle_ext': [measurements[n].get('ankle_extension_deviation') for n in measurements],
            'back_roundness': [measurements[n].get('back_roundness_deviation') for n in measurements],
            'travel': [measurements[n].get('hip_travel_range') for n in measurements],
            'unroll': [measurements[n].get('unroll_speed_ratio') for n in measurements],
            'head_tuck': [measurements[n].get('head_tuck_deviation') for n in measurements],
        }

        for name in sorted(self.figures.keys()):
            m = measurements[name]
            d = self._compute_all_deductions(m, group_values=group_values)

            base, metric_used = self._compute_base_score(m, source_mode)
            m['base_score'] = base
            m['base_score_metric_used'] = metric_used
            m['source_mode'] = source_mode

            total_ded = sum(d.get(k, 0) for k in self._deduction_keys())
            m['deductions'] = d
            m['total_deduction'] = round(total_ded, 2)
            m['score'] = round(max(0.0, base - total_ded) * 20) / 20

            self.results[name] = m

        return self.results

    def score_figure(self, name, source_mode='above'):
        """Score a single figure (no relative/group component).

        source_mode: 'walticam', 'above', or 'above_below' — determines
        which height metric/chart is used for the base score (see
        _compute_base_score) and is recorded on the result for display.
        """
        m = self._extract_measurements(name)
        d = self._compute_all_deductions(m, group_values=None)

        base, metric_used = self._compute_base_score(m, source_mode)
        m['base_score'] = base
        m['base_score_metric_used'] = metric_used
        m['source_mode'] = source_mode

        total_ded = sum(d.get(k, 0) for k in self._deduction_keys())
        m['deductions'] = d
        m['total_deduction'] = round(total_ded, 2)
        m['score'] = round(max(0.0, base - total_ded) * 20) / 20

        self.results[name] = m
        return m

    @classmethod
    def score_single_pair(cls, above_csv_path, below_csv_path=None, name="figure", source_mode='above'):
        """Score one figure directly from its above/below CSV paths, with
        no folder scanning — what the web app calls right after tracking
        finishes.

        source_mode: 'walticam', 'above', or 'above_below'. Walticam uses
        a body-length-normalized height metric instead of the raw frame-
        fraction one, since a WaltiCam half-frame has a different field
        of view than a dedicated above-water camera (see
        _HEIGHT_BASE_BREAKPOINTS_BODY_LENGTHS above)."""
        scorer = cls.__new__(cls)
        scorer.data_dir = None
        scorer.figures = {
            name: {
                'above': Path(above_csv_path),
                'below': Path(below_csv_path) if below_csv_path else None,
            }
        }
        scorer.results = {}
        return scorer.score_figure(name, source_mode=source_mode)

    # ── Output ──

    def summary_dataframe(self):
        if not self.results:
            self.score_all()
        if not self.results:
            return pd.DataFrame()

        rows = []
        for name, r in self.results.items():
            d = r['deductions']

            labels = {
                'ascent_alignment': 'ascent align.', 'descent_alignment': 'descent align.',
                'backpike': 'backpike', 'leg_extension': 'leg ext.',
                'ankle_extension': 'ankle ext.', 'back_roundness': 'back round.',
                'travel': 'travel', 'unroll_speed': 'unroll speed', 'head_tuck': 'head tuck',
            }
            top = sorted(
                [(k, d[k]) for k in self._deduction_keys() if d.get(k, 0) > 0],
                key=lambda x: x[1], reverse=True)
            top_str = ', '.join(f"{labels.get(k, k)} -{v:.2f}" for k, v in top) or '—'

            coaching = sorted(
                [(k, d[k]) for k in self._coaching_deduction_keys() if d.get(k, 0) > 0],
                key=lambda x: x[1], reverse=True)
            coaching_labels = {'underwater_bent_knee': 'uw bent knee', 'back_layout_depth': 'back layout depth'}
            coaching_str = ', '.join(f"{coaching_labels.get(k, k)} -{v:.2f}" for k, v in coaching) or '—'

            s = r['score']
            if s >= 9.5:   assess = 'Excellent'
            elif s >= 8.5: assess = 'Very Good'
            elif s >= 7.5: assess = 'Good'
            elif s >= 6.5: assess = 'Competent'
            elif s >= 5.5: assess = 'Satisfactory'
            elif s >= 4.5: assess = 'Deficient'
            else:          assess = 'Weak'

            rows.append({
                'Figure': name, 'Score': s, 'Assessment': assess,
                'Base': r['base_score'], 'Deduction': r['total_deduction'],
                'Top Deductions': top_str,
                'Coaching Feedback (not scored)': coaching_str,
            })

        df = pd.DataFrame(rows).sort_values('Score', ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Rank'
        for c in ['Score', 'Base', 'Deduction']:
            df[c] = pd.to_numeric(df[c], errors='coerce').round(2)
        return df

    def print_summary_table(self):
        df = self.summary_dataframe()
        if df.empty:
            print("  ⚠ Nothing to display — no figures were found or scored.")
            print(f"    data_dir: {self.data_dir}")
            return

        print(f"\n{'='*100}")
        print(f"  BARRACUDA SCORER — Summary ({len(df)} Figures)")
        print(f"  Coaching-only deductions included in score: {self.INCLUDE_COACHING_IN_SCORE}")
        print(f"{'='*100}\n")

        with pd.option_context('display.max_colwidth', None, 'display.width', None):
            print(df[['Figure', 'Score', 'Assessment', 'Base', 'Deduction']].to_string(na_rep='—'))

        print(f"\n{'-'*100}")
        print(f"  Top scored deductions / Coaching-only feedback per figure:")
        print(f"{'-'*100}")
        name_w = max(len(n) for n in df['Figure']) + 2
        for rank, row in df.iterrows():
            print(f"  {rank}. {row['Figure']:<{name_w}} Scored: {row['Top Deductions']}")
            print(f"     {' ' * name_w} Coaching: {row['Coaching Feedback (not scored)']}")

        print(f"\n{'='*100}")
        print(f"  Note: head_tuck and back_layout_depth are measured-but-not-yet-")
        print(f"  calibrated STUBS (always contribute 0) pending judge input.")
        print(f"{'='*100}\n")

    def save_html_report(self, output_path=None):
        df = self.summary_dataframe()
        if df.empty:
            print("  ⚠ Nothing to export — no figures were found or scored.")
            return None

        if output_path is None:
            out_dir = self.data_dir / 'scoring_results_with_html'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / 'barracuda_summary.html'
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        styled = (
            df.style
            .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=10)
            .format({'Score': '{:.2f}', 'Base': '{:.2f}', 'Deduction': '{:.2f}'}, na_rep='—')
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#2b2b2b'), ('color', 'white'),
                    ('padding', '8px 12px'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [
                    ('padding', '6px 12px'), ('text-align', 'center'),
                    ('font-family', 'Helvetica, Arial, sans-serif')]},
                {'selector': 'table', 'props': [
                    ('border-collapse', 'collapse'), ('margin', '20px auto')]},
            ])
            .set_properties(subset=['Figure', 'Top Deductions', 'Coaching Feedback (not scored)'],
                             **{'text-align': 'left'})
        )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Barracuda Scoring Summary</title></head>
<body style="font-family: Helvetica, Arial, sans-serif; background: #f5f5f5;">
<h2 style="text-align:center;">Barracuda Figure Scoring — {len(df)} Figures</h2>
{styled.to_html()}
<p style="text-align:center; color:#666; font-size:0.9em;">
Coaching-feedback columns (underwater bent knee, back layout depth) are
measured but NOT counted toward the score. head_tuck is a measured-but-
uncalibrated stub (always 0) pending judge input on point values.
</p>
</body></html>"""

        output_path.write_text(html)
        print(f"  ✓ HTML report saved: {output_path}")
        return output_path


if __name__ == "__main__":
    data_dir = '/Users/mona/Desktop/Science fairs/Science fair 2026/Barracuda folders/Jmeet figures.nosync'
    print(f"\n{'='*80}")
    print(f"  BARRACUDA FIGURE SCORER — FINA-Aligned Deductions")
    print(f"  Directory: {data_dir}")
    print(f"{'='*80}\n")
    scorer = BarracudaScorer(data_dir)
    scorer.score_all()
    scorer.print_summary_table()
    scorer.save_html_report()
