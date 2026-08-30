#!/usr/bin/env python3
"""
TRACKER CORE — RTMPose engine for the Barracuda web app
===========================================================================
Three processing modes, all producing CSVs the scorer already expects
(`{name}_above_tracking_data.csv`, `{name}_below_tracking_data.csv`):

  - process_video_above_water()  — single above-water video (COCO-17 Body)
  - process_video_underwater()   — single underwater video (COCO-17 Body)
  - process_video_walticam()     — one split-screen video, top=above/
                                    bottom=below (Halpe26 BodyWithFeet)

===========================================================================
REBUILD NOTE
===========================================================================
This file was reconstructed from conversation history after a sandbox
reset wiped the working copy — it is NOT a direct edit of your live repo
file. Two different confidence levels apply:

  - process_video_above_water() / process_video_underwater() and their
    tracker classes: reconstructed from the fragments viewed earlier in
    this conversation plus the standalone above_water_rtmpose_tracker.py
    script you shared (clearly the same lineage — matching docstrings,
    same tent-masking/locking/smoothing design). The locking fix from
    earlier in this conversation (anatomy+size validated matching, 2-in-a
    -row relock safety) is included. TEST THIS against your existing
    single-video workflows before trusting it in place of your last known
    -good deployed version.

  - process_video_walticam() and its trackers: this is the v2 WaltiCam
    logic from earlier in this conversation, which WAS unit-tested
    (locking, anatomy validation, size-based foreground filtering,
    2-in-a-row relock safety, Kalman smoothing all verified in isolation)
    before being wired in here. Higher confidence.

If anything behaves differently than your last deployed version, the
fastest fix is diffing this against your live repo's tracker_core.py
directly, if you still have it, rather than guessing from output alone.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import time

# ============================================================================
# SHARED CONFIG
# ============================================================================

IGNORE_TOP_PERCENT = 0.35   # above-water single-video tent zone mask

# Speed optimization — caps the SAVED/downloadable annotated video's width.
# Detection always runs on the full-resolution source frame first (this
# has zero effect on tracking or scoring, which reads the CSV, never the
# video pixels); only the final rendered frame gets scaled down before
# being handed to the video encoder. Encoding cost scales with pixel
# count, so for a 4K or 1080p phone video this cuts encoding time
# substantially. Videos already at or under this width are written
# unchanged.
MAX_OUTPUT_VIDEO_WIDTH = 1280


def _output_video_size(w, h, max_width=MAX_OUTPUT_VIDEO_WIDTH):
    if w <= max_width:
        return w, h
    scale = max_width / w
    return max_width, max(2, int(round(h * scale)) // 2 * 2)  # keep height even (codec-friendly)

MAX_FRAMES_LOST = 30
SHORT_GAP_HOLD_FRAMES = 5   # hold last known position through brief gaps only
RELOCK_CHECK_INTERVAL = 45
RELOCK_DRIFT_THRESHOLD = 0.15
LOCK_MATCH_DISTANCE = 0.15
MIN_FOREGROUND_SIZE_RATIO = 0.60


def make_pose_tracker(mode='performance', det_frequency=1):
    """COCO-17 (Body) tracker — used by the single above/underwater
    (non-WaltiCam) video paths."""
    from rtmlib import PoseTracker, Body
    return PoseTracker(
        Body, mode=mode, det_frequency=det_frequency,
        backend='onnxruntime', device='cpu', to_openpose=False,
    )


def make_pose_tracker_halpe(mode='performance', det_frequency=1):
    """Halpe26 (BodyWithFeet) tracker — used by the WaltiCam split-screen
    path, since it gives real heel/toe/neck keypoints instead of
    synthesized offsets."""
    from rtmlib import PoseTracker, BodyWithFeet
    return PoseTracker(
        BodyWithFeet, mode=mode, det_frequency=det_frequency,
        backend='onnxruntime', device='cpu', to_openpose=False,
    )


# ============================================================================
# LANDMARK SCHEMES
# ============================================================================

# COCO-17 (single above/underwater videos) — no native feet, so heel/
# foot_index/foot_best are synthesized from the ankle.
COCO_TO_LANDMARKS = {
    0: 'nose',
    5: 'left_shoulder', 6: 'right_shoulder',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle',
}

ALL_LANDMARKS_ABOVE = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
    'left_foot_best', 'right_foot_best',
]

ALL_LANDMARKS_UNDERWATER = ALL_LANDMARKS_ABOVE + ['mid_spine']

COCO_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Halpe26 (WaltiCam) — real heel/toe/neck keypoints. First 17 indices
# match COCO-17, so validate_pose_anatomy_coco / calculate_pose_size_coco
# work unmodified on Halpe26 arrays too.
HALPE26_TO_LANDMARKS = {
    0: 'nose',
    5: 'left_shoulder', 6: 'right_shoulder',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle',
    18: 'neck',
    20: 'left_foot_index', 21: 'right_foot_index',
    22: 'left_small_toe', 23: 'right_small_toe',
    24: 'left_heel', 25: 'right_heel',
}

ALL_LANDMARKS_WALTICAM_ABOVE = [
    'nose', 'neck', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

ALL_LANDMARKS_WALTICAM_BELOW = ALL_LANDMARKS_WALTICAM_ABOVE + [
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
    'left_small_toe', 'right_small_toe',
]

HALPE_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 20), (15, 24), (16, 21), (16, 25),
    (18, 5), (18, 6),
]

MIN_KEYPOINT_CONF = 0.15
MIN_KEYPOINT_CONF_LOWER_BODY = 0.08
LOWER_BODY_LANDMARK_NAMES = {
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
    'left_small_toe', 'right_small_toe',
}


def resolve_landmark(name, x_norm, y_norm, conf, water_level):
    min_conf = MIN_KEYPOINT_CONF_LOWER_BODY if name in LOWER_BODY_LANDMARK_NAMES else MIN_KEYPOINT_CONF
    if conf <= min_conf:
        return None
    return {
        'x': x_norm, 'y': y_norm, 'z': 0.0, 'visibility': conf,
        'above_water': y_norm < water_level,
    }


# ============================================================================
# IMAGE ENHANCEMENT
# ============================================================================

def quick_enhance(frame):
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=15)


def enhance_underwater(frame):
    b, g, r = cv2.split(frame)
    r_boosted = cv2.convertScaleAbs(r, alpha=1.4, beta=10)
    g_adjusted = cv2.convertScaleAbs(g, alpha=1.1, beta=0)
    b_reduced = cv2.convertScaleAbs(b, alpha=0.85, beta=0)
    corrected = cv2.merge([b_reduced, g_adjusted, r_boosted])

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b_ch]), cv2.COLOR_LAB2BGR)

    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.addWeighted(sharpened, 0.6, enhanced, 0.4, 0)


# ============================================================================
# VALIDATION / SELECTION HELPERS (shared across all trackers)
# ============================================================================

def validate_pose_anatomy_coco(person_kps, person_scores, h, w):
    """Pure-geometry sanity check. Works on COCO-17 or Halpe26 arrays —
    only indices 0-16 are used, which match between the two schemes."""
    if person_scores[5] > 0.10 and person_scores[11] > 0.10:
        ls_y = person_kps[5][1] / h
        lh_y = person_kps[11][1] / h
        if ls_y > lh_y + 0.35:
            return False
    if person_scores[6] > 0.10 and person_scores[12] > 0.10:
        rs_y = person_kps[6][1] / h
        rh_y = person_kps[12][1] / h
        if rs_y > rh_y + 0.35:
            return False
    if person_scores[11] > 0.10 and person_scores[12] > 0.10:
        lh_x = person_kps[11][0] / w
        rh_x = person_kps[12][0] / w
        hip_width = abs(lh_x - rh_x)
        if hip_width < 0.02 or hip_width > 0.60:
            return False
    return True


def calculate_pose_size_coco(person_kps, person_scores, h, w):
    """Rough foreground/background size estimate. Returns None if hips
    aren't detected at all."""
    if person_scores[11] < 0.10 or person_scores[12] < 0.10:
        return None

    lh_x, lh_y = person_kps[11][0] / w, person_kps[11][1] / h
    rh_x, rh_y = person_kps[12][0] / w, person_kps[12][1] / h

    torso_height = torso_width = 0
    has_shoulders = person_scores[5] > 0.10 and person_scores[6] > 0.10
    if has_shoulders:
        ls_y, rs_y = person_kps[5][1] / h, person_kps[6][1] / h
        ls_x = person_kps[5][0] / w
        hip_y = (lh_y + rh_y) / 2
        shoulder_y = (ls_y + rs_y) / 2
        torso_height = abs(hip_y - shoulder_y)
        torso_width = abs(((lh_x + rh_x) / 2) - ls_x)

    hip_width = abs(lh_x - rh_x)

    full_height, has_feet = 0, False
    for ankle_idx in [15, 16]:
        if person_scores[ankle_idx] > 0.10:
            has_feet = True
            hip_y = (lh_y + rh_y) / 2
            ankle_y = person_kps[ankle_idx][1] / h
            full_height = max(full_height, abs(hip_y - ankle_y))
            break

    size = 0
    if has_shoulders:
        size += (torso_height + torso_width) * 3.0
    size += hip_width * 2.0
    if has_feet:
        size += full_height * 1.5
    return {
        'size': size / 6.5, 'hip_width': hip_width,
        'torso_height': torso_height, 'torso_width': torso_width,
        'full_height': full_height, 'has_shoulders': has_shoulders,
        'has_feet': has_feet,
    }


def get_hip_position(person_kps, person_scores, h, w):
    if person_scores[11] < 0.05 or person_scores[12] < 0.05:
        return None
    return {
        'x': (person_kps[11][0] + person_kps[12][0]) / 2 / w,
        'y': (person_kps[11][1] + person_kps[12][1]) / 2 / h,
    }


def is_in_water_lenient(hip_x_norm, hip_y_norm, water_level, frame):
    """Absolute HSV water check — used by the single-video (non-WaltiCam)
    trackers, where a fixed calibrated color range is a reasonable
    tradeoff since it's one camera/venue per video."""
    h, w = frame.shape[:2]
    hip_x = int(hip_x_norm * w)
    hip_y = int(hip_y_norm * h)

    if hip_y_norm < 0.35:
        return False
    if hip_y_norm < water_level - 0.15:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    water_lower1 = np.array([85, 30, 30]); water_upper1 = np.array([135, 255, 255])
    water_lower2 = np.array([70, 30, 30]); water_upper2 = np.array([105, 255, 255])

    sample_radius = 60
    below_region = hsv[min(h, hip_y + 20):min(h, hip_y + sample_radius + 40),
                       max(0, hip_x - sample_radius):min(w, hip_x + sample_radius)]
    if below_region.size > 0:
        mask1 = cv2.inRange(below_region, water_lower1, water_upper1)
        mask2 = cv2.inRange(below_region, water_lower2, water_upper2)
        below_water_pct = (np.sum(mask1 > 0) + np.sum(mask2 > 0)) / below_region.shape[0] / below_region.shape[1]
        if below_water_pct < 0.08:
            return False

    sample_at_hip = hsv[max(0, hip_y - 20):min(h, hip_y + 20),
                        max(0, hip_x - 40):min(w, hip_x + 40)]
    if sample_at_hip.size > 0:
        mask1_hip = cv2.inRange(sample_at_hip, water_lower1, water_upper1)
        mask2_hip = cv2.inRange(sample_at_hip, water_lower2, water_upper2)
        at_hip_water_pct = (np.sum(mask1_hip > 0) + np.sum(mask2_hip > 0)) / sample_at_hip.shape[0] / sample_at_hip.shape[1]
        if at_hip_water_pct < 0.05:
            return False

    return True


def is_water_relative(hip_x_norm, hip_y_norm, sub_frame, margin=25):
    """Relative (not calibrated-absolute) water check — used by the
    WaltiCam path, deliberately, since calibrated absolute color ranges
    are the category of fix that caused cross-venue regressions before."""
    h, w = sub_frame.shape[:2]
    x = int(np.clip(hip_x_norm, 0, 1) * w)
    y = int(np.clip(hip_y_norm, 0, 1) * h)
    y0, y1 = min(h, y + 5), min(h, y + 5 + margin)
    x0, x1 = max(0, x - margin), min(w, x + margin)
    region = sub_frame[y0:y1, x0:x1]
    if region.size == 0:
        return True
    region_i16 = region.astype(np.int16)
    b, r = region_i16[:, :, 0], region_i16[:, :, 2]
    blue_frac = ((b - r) > 10).mean()
    return blue_frac > 0.35


def select_best_swimmer_coco(all_keypoints, all_scores, water_level, frame, ignore_top_percent=IGNORE_TOP_PERCENT):
    """Fresh (unlocked) selection for the single-video trackers: size-
    based foreground filter + full validation (anatomy, position bounds,
    water check)."""
    if all_keypoints is None or len(all_keypoints) == 0:
        return None
    h, w = frame.shape[:2]

    candidates = []
    for i, (kps, sc) in enumerate(zip(all_keypoints, all_scores)):
        size_info = calculate_pose_size_coco(kps, sc, h, w)
        if size_info:
            candidates.append((i, kps, sc, size_info))
    if not candidates:
        return None

    candidates.sort(key=lambda c: c[3]['size'], reverse=True)
    largest_size = candidates[0][3]['size']
    min_foreground_size = largest_size * MIN_FOREGROUND_SIZE_RATIO

    for idx, kps, sc, size_info in candidates:
        if size_info['size'] < min_foreground_size or size_info['size'] < 0.15:
            continue
        if sc[11] < 0.15 or sc[12] < 0.15:
            continue
        hip_y = (kps[11][1] + kps[12][1]) / 2 / h
        hip_x = (kps[11][0] + kps[12][0]) / 2 / w
        if hip_y < ignore_top_percent or hip_y < 0.35:
            continue
        if hip_x < 0.15 or hip_x > 0.85:
            continue
        if hip_y < water_level - 0.10:
            continue
        if not is_in_water_lenient(hip_x, hip_y, water_level, frame):
            continue
        if not validate_pose_anatomy_coco(kps, sc, h, w):
            continue
        visible_count = sum(1 for j in [0, 5, 6, 11, 12, 13, 14, 15, 16] if sc[j] > 0.15)
        if visible_count < 3:
            continue
        return idx
    return None


# ============================================================================
# SHARED LOCKING (used by the WaltiCam trackers; unit-tested earlier)
# ============================================================================

class SwimmerLock:
    def __init__(self):
        self.locked = None
        self.frames_since_detection = 0
        self._frames_since_relock_check = 0
        self._pending_relock_idx = None
        self._pending_relock_streak = 0

    def select(self, keypoints, scores, h, w, is_candidate_valid_fn, sub_frame=None):
        if keypoints is None or len(keypoints) == 0:
            self.frames_since_detection += 1
            if self.frames_since_detection > MAX_FRAMES_LOST:
                self.locked = None
            return None

        if self.locked is not None:
            idx = self._find_matching(keypoints, scores, h, w, is_candidate_valid_fn, sub_frame)
            if idx is not None:
                self.frames_since_detection = 0
                self.locked = get_hip_position(keypoints[idx], scores[idx], h, w)
                idx = self._periodic_relock_check(keypoints, scores, h, w, is_candidate_valid_fn, sub_frame, idx)
                return idx
            self.frames_since_detection += 1
            if self.frames_since_detection > MAX_FRAMES_LOST:
                idx = self._fresh_select(keypoints, scores, h, w, is_candidate_valid_fn, sub_frame)
                if idx is not None:
                    self.locked = get_hip_position(keypoints[idx], scores[idx], h, w)
                    self.frames_since_detection = 0
                return idx
            return None
        else:
            idx = self._fresh_select(keypoints, scores, h, w, is_candidate_valid_fn, sub_frame)
            if idx is not None:
                self.locked = get_hip_position(keypoints[idx], scores[idx], h, w)
                self.frames_since_detection = 0
            return idx

    def _find_matching(self, keypoints, scores, h, w, is_valid_fn, sub_frame):
        best_idx, best_dist = None, LOCK_MATCH_DISTANCE
        for i, (kps, sc) in enumerate(zip(keypoints, scores)):
            hip = get_hip_position(kps, sc, h, w)
            if hip is None or not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            if not is_valid_fn(kps, sc, hip, sub_frame):
                continue
            dist = np.hypot(hip['x'] - self.locked['x'], hip['y'] - self.locked['y'])
            if dist < best_dist:
                best_dist, best_idx = dist, i
        return best_idx

    def _fresh_select(self, keypoints, scores, h, w, is_valid_fn, sub_frame):
        candidates = []
        for i, (kps, sc) in enumerate(zip(keypoints, scores)):
            size = calculate_pose_size_coco(kps, sc, h, w)
            if size is not None:
                candidates.append((i, kps, sc, size['size']))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[3], reverse=True)
        min_size = candidates[0][3] * MIN_FOREGROUND_SIZE_RATIO
        for i, kps, sc, size in candidates:
            if size < min_size or not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            hip = get_hip_position(kps, sc, h, w)
            if hip is None or not is_valid_fn(kps, sc, hip, sub_frame):
                continue
            return i
        return None

    def _periodic_relock_check(self, keypoints, scores, h, w, is_valid_fn, sub_frame, current_idx):
        self._frames_since_relock_check += 1
        if self._frames_since_relock_check < RELOCK_CHECK_INTERVAL:
            return current_idx
        self._frames_since_relock_check = 0

        fresh_idx = self._fresh_select(keypoints, scores, h, w, is_valid_fn, sub_frame)
        if fresh_idx is None or fresh_idx == current_idx:
            self._pending_relock_idx, self._pending_relock_streak = None, 0
            return current_idx

        fresh_hip = get_hip_position(keypoints[fresh_idx], scores[fresh_idx], h, w)
        drift = np.hypot(fresh_hip['x'] - self.locked['x'], fresh_hip['y'] - self.locked['y'])
        if drift <= RELOCK_DRIFT_THRESHOLD:
            self._pending_relock_idx, self._pending_relock_streak = None, 0
            return current_idx

        if self._pending_relock_idx == fresh_idx:
            self._pending_relock_streak += 1
        else:
            self._pending_relock_idx, self._pending_relock_streak = fresh_idx, 1

        if self._pending_relock_streak >= 2:
            self.locked = fresh_hip
            self._pending_relock_idx, self._pending_relock_streak = None, 0
            return fresh_idx
        return current_idx


# ============================================================================
# KALMAN FILTERING (shared by all three paths)
# ============================================================================

class ImprovedKalmanFilter1D:
    # ACCURACY TUNING (no effect on inference speed — this is pure
    # post-processing on keypoints the model already produced, not an
    # extra model call, so it doesn't touch mode/det_frequency at all).
    #
    # outlier_threshold was 0.15 (15% of frame height/width). A
    # barracuda figure's whole point is an explosive, fast vertical
    # jump — exactly the kind of large frame-to-frame position change
    # this threshold exists to reject as tracking noise. At 0.15, a
    # genuinely fast rise could get misclassified as an outlier and
    # replaced with the filter's more conservative predicted value
    # instead of the real measurement, clipping the true peak — which
    # would show up as an UNDERESTIMATED foot_clearance, and therefore
    # a lower base_score, for exactly the swimmers with the most
    # explosive jumps. Raised to 0.22 so real jump speed passes through
    # while still catching genuine tracking glitches (a mis-lock onto a
    # different point is typically a much larger, near-instantaneous
    # jump than this).
    def __init__(self, process_var=0.001, measurement_var=0.05, outlier_threshold=0.22):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.Q = np.array([[process_var, 0], [0, process_var * 0.1]])
        self.R = np.array([[measurement_var]])
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.initialized = False
        self.outlier_threshold = outlier_threshold
        self.last_measurement = None
        self.consecutive_predictions = 0
        self.max_predictions = 20

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]

    def is_outlier(self, measurement):
        if self.last_measurement is None:
            return False
        return abs(measurement - self.last_measurement) > self.outlier_threshold

    def update(self, measurement, confidence=1.0, force=False):
        if not self.initialized:
            self.x[0] = measurement
            self.last_measurement = measurement
            self.initialized = True
            self.consecutive_predictions = 0
            return measurement
        if not force and self.is_outlier(measurement):
            return self.predict()
        R_adjusted = self.R * (2.0 - confidence)
        y = measurement - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + R_adjusted
        K = self.P @ self.H.T / S
        self.x = self.x + (K * y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.last_measurement = measurement
        self.consecutive_predictions = 0
        return self.x[0]

    def filter(self, measurement, confidence=1.0):
        predicted = self.predict()
        if measurement is not None and not np.isnan(measurement):
            return self.update(measurement, confidence)
        self.consecutive_predictions += 1
        if self.consecutive_predictions > self.max_predictions:
            return np.nan
        return predicted


def _kalman_filter_dataframe(df, landmarks):
    """Same Kalman math and same sequential order as before — only the
    per-row access pattern changed. The old version called df.iterrows()
    once per joint-axis pass (~30 passes for a typical joint set), and
    iterrows() reconstructs the ENTIRE row (every column, not just the 2
    actually used) into a pandas Series on every single row, every pass.
    For a ~600-frame video that's roughly 18,000 full-row reconstructions
    of a 70-100 column dataframe just to read 2 values each time. Pulling
    the needed columns out as plain numpy arrays once per pass removes
    that overhead entirely — the numeric output is identical, only the
    access pattern changed (verified: see the numeric-equivalence test
    this was checked against before deploying)."""
    n = len(df)
    for joint in landmarks:
        for axis in ['y', 'x']:
            col = f'{joint}_{axis}'
            vis_col = f'{joint}_visibility'
            if col not in df.columns:
                continue
            first_idx = df[col].first_valid_index()
            if first_idx is None:
                continue
            kf = ImprovedKalmanFilter1D()
            kf.x[0] = df.loc[first_idx, col]
            kf.initialized = True

            col_values = df[col].to_numpy()
            vis_values = df[vis_col].to_numpy() if vis_col in df.columns else None

            filtered = [None] * n
            for i in range(n):
                raw_m = col_values[i]
                m = None if pd.isna(raw_m) else raw_m
                if vis_values is not None:
                    raw_c = vis_values[i]
                    c = 1.0 if pd.isna(raw_c) else raw_c
                else:
                    c = 1.0
                if m is not None and kf.is_outlier(m):
                    m = None
                filtered[i] = kf.filter(m, c)

            df[f'{joint}_{axis}_raw'] = df[col]
            df[col] = filtered
    return df


def apply_kalman_filter_to_csv(csv_path, landmarks):
    """Called by app.py after single above/underwater video tracking.
    Reads the raw CSV, Kalman-filters it, writes a `_KALMAN.csv` sibling,
    returns its path."""
    df = pd.read_csv(csv_path)
    df = _kalman_filter_dataframe(df, landmarks)
    output_path = Path(csv_path).parent / (Path(csv_path).stem + "_KALMAN.csv")
    df.to_csv(output_path, index=False, float_format='%.4f')
    return output_path


# ============================================================================
# ABOVE-WATER TRACKER — single video, COCO-17, full frame
# ============================================================================

class AboveWaterRTMPoseTracker:
    def __init__(self, mode='performance', det_frequency=1, ignore_top_percent=IGNORE_TOP_PERCENT, pose_tracker=None):
        # pose_tracker: pass a pre-built one (e.g. cached by the caller
        # across videos/trackers) to skip re-loading the model. Defaults
        # to building its own, so this class still works standalone.
        self.pose_tracker = pose_tracker if pose_tracker is not None else make_pose_tracker(mode, det_frequency)
        self.ignore_top_percent = ignore_top_percent
        self.tracking_data = []

        self.locked_swimmer = None
        self.frames_since_detection = 0
        self.max_frames_lost = MAX_FRAMES_LOST
        self.relock_check_interval = RELOCK_CHECK_INTERVAL
        self.relock_drift_threshold = RELOCK_DRIFT_THRESHOLD
        self._frames_since_relock_check = 0
        self._pending_relock_idx = None
        self._pending_relock_streak = 0

        self.position_history = []
        self.history_size = 5
        self.foot_history = {}
        self.foot_history_size = 7
        self.hip_history = {}
        self.hip_history_size = 12
        self.toe_history = {}
        self.toe_history_size = 9

        self.last_known = {}
        self.max_jump = 0.25
        self.hip_correction_ratio = 0.20

    def _get_swimmer_position(self, person_kps, person_scores, h, w):
        return get_hip_position(person_kps, person_scores, h, w)

    def _find_matching_swimmer(self, all_keypoints, all_scores, locked_position, water_level, frame, h, w):
        """Dropped the per-frame water-color check here (kept for
        fresh/initial selection via select_best_swimmer_coco only). Real
        tracking output showed the swimmer getting lost for 335 of 422
        frames because that check has to pass on EVERY frame to count as
        a match, and it's fragile against splash/foam/lighting. Position
        bounds + anatomy stay — pure geometry, still block a background
        person from stealing the lock."""
        best_match, best_distance = None, LOCK_MATCH_DISTANCE
        for i, (kps, sc) in enumerate(zip(all_keypoints, all_scores)):
            if sc[11] < 0.10 or sc[12] < 0.10:
                continue
            hip_y = (kps[11][1] + kps[12][1]) / 2 / h
            hip_x = (kps[11][0] + kps[12][0]) / 2 / w
            if hip_y < self.ignore_top_percent or hip_y < 0.35:
                continue
            if hip_x < 0.15 or hip_x > 0.85:
                continue
            if not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            distance = np.sqrt((hip_x - locked_position['x'])**2 + (hip_y - locked_position['y'])**2)
            if distance < best_distance:
                best_distance, best_match = distance, i
        return best_match

    def process_frame(self, frame, frame_num, fps, water_level):
        h, w = frame.shape[:2]
        frame_masked = frame.copy()
        mask_height = int(h * self.ignore_top_percent)
        frame_masked[0:mask_height, :] = 0
        enhanced = quick_enhance(frame_masked)
        keypoints, scores = self.pose_tracker(enhanced)

        best_person_idx = None
        if keypoints is not None and len(keypoints) > 0:
            if self.locked_swimmer is not None:
                best_person_idx = self._find_matching_swimmer(
                    keypoints, scores, self.locked_swimmer, water_level, frame_masked, h, w
                )
                if best_person_idx is not None:
                    self.frames_since_detection = 0
                    self.locked_swimmer = self._get_swimmer_position(keypoints[best_person_idx], scores[best_person_idx], h, w)

                    self._frames_since_relock_check += 1
                    if self._frames_since_relock_check >= self.relock_check_interval:
                        self._frames_since_relock_check = 0
                        fresh_idx = select_best_swimmer_coco(keypoints, scores, water_level, frame_masked, self.ignore_top_percent)
                        if fresh_idx is not None and fresh_idx != best_person_idx:
                            fresh_pos = self._get_swimmer_position(keypoints[fresh_idx], scores[fresh_idx], h, w)
                            if fresh_pos is not None:
                                drift = np.sqrt((fresh_pos['x'] - self.locked_swimmer['x'])**2 + (fresh_pos['y'] - self.locked_swimmer['y'])**2)
                                if drift > self.relock_drift_threshold:
                                    if self._pending_relock_idx == fresh_idx:
                                        self._pending_relock_streak += 1
                                    else:
                                        self._pending_relock_idx, self._pending_relock_streak = fresh_idx, 1
                                    if self._pending_relock_streak >= 2:
                                        best_person_idx = fresh_idx
                                        self.locked_swimmer = fresh_pos
                                        self._pending_relock_idx, self._pending_relock_streak = None, 0
                                else:
                                    self._pending_relock_idx, self._pending_relock_streak = None, 0
                else:
                    self.frames_since_detection += 1
                    if self.frames_since_detection > self.max_frames_lost:
                        best_person_idx = select_best_swimmer_coco(keypoints, scores, water_level, frame_masked, self.ignore_top_percent)
                        if best_person_idx is not None:
                            self.locked_swimmer = self._get_swimmer_position(keypoints[best_person_idx], scores[best_person_idx], h, w)
                            self.frames_since_detection = 0
            else:
                best_person_idx = select_best_swimmer_coco(keypoints, scores, water_level, frame_masked, self.ignore_top_percent)
                if best_person_idx is not None:
                    self.locked_swimmer = self._get_swimmer_position(keypoints[best_person_idx], scores[best_person_idx], h, w)
                    self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_frames_lost:
                self.locked_swimmer = None

        best_person = (keypoints[best_person_idx], scores[best_person_idx]) if best_person_idx is not None else None

        frame_data = {
            'frame': frame_num,
            'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4),
            'tracking_locked': self.locked_swimmer is not None,
            'frames_since_detection': self.frames_since_detection,
        }

        if best_person is not None:
            person_kps, person_scores = best_person
            current_positions = {}
            for coco_idx, landmark_name in COCO_TO_LANDMARKS.items():
                conf = float(person_scores[coco_idx])
                if conf > 0.05:
                    x_norm = person_kps[coco_idx][0] / w
                    y_norm = person_kps[coco_idx][1] / h
                    current_positions[landmark_name] = {
                        'x': x_norm, 'y': y_norm, 'z': 0.0, 'visibility': conf,
                        'above_water': y_norm < water_level,
                    }

            for side in ['left', 'right']:
                hip_name, knee_name = f'{side}_hip', f'{side}_knee'
                if hip_name in current_positions and knee_name in current_positions:
                    hip, knee = current_positions[hip_name], current_positions[knee_name]
                    r = self.hip_correction_ratio
                    corrected_y = hip['y'] + r * (knee['y'] - hip['y'])
                    current_positions[hip_name] = {
                        'x': hip['x'] + r * (knee['x'] - hip['x']), 'y': corrected_y, 'z': hip['z'],
                        'visibility': min(hip['visibility'], knee['visibility']),
                        'above_water': corrected_y < water_level,
                    }

            for side in ['left', 'right']:
                ankle_name = f'{side}_ankle'
                if ankle_name in current_positions:
                    ankle = current_positions[ankle_name]
                    current_positions[f'{side}_heel'] = {
                        'x': ankle['x'], 'y': ankle['y'] - 0.005, 'z': 0.0,
                        'visibility': ankle['visibility'] * 0.8, 'above_water': (ankle['y'] - 0.005) < water_level,
                    }
                    current_positions[f'{side}_foot_index'] = {
                        'x': ankle['x'], 'y': ankle['y'] + 0.005, 'z': 0.0,
                        'visibility': ankle['visibility'] * 0.8, 'above_water': (ankle['y'] + 0.005) < water_level,
                    }
                    current_positions[f'{side}_foot_best'] = {
                        'x': ankle['x'], 'y': ankle['y'], 'z': 0.0,
                        'visibility': ankle['visibility'], 'above_water': ankle['y'] < water_level,
                        'source': ankle_name,
                    }

            current_positions = self._filter_jumps(current_positions)

            for side in ['left', 'right']:
                foot_landmarks = [(f'{side}_foot_index', 10.0), (f'{side}_ankle', 7.0), (f'{side}_heel', 5.0)]
                best_foot, best_score = None, 0
                for lm_name, weight in foot_landmarks:
                    if lm_name in current_positions:
                        score = current_positions[lm_name]['visibility'] * weight
                        if score > best_score:
                            best_score, best_foot = score, lm_name
                if best_foot:
                    key = f'{side}_foot_best'
                    self.foot_history.setdefault(key, []).append({
                        'x': current_positions[best_foot]['x'], 'y': current_positions[best_foot]['y'],
                        'z': current_positions[best_foot]['z'], 'visibility': current_positions[best_foot]['visibility'],
                        'above_water': current_positions[best_foot]['above_water'], 'source': best_foot,
                    })
                    if len(self.foot_history[key]) > self.foot_history_size:
                        self.foot_history[key].pop(0)

            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                if hip_name in current_positions:
                    key = f'{side}_hip_ultra'
                    self.hip_history.setdefault(key, []).append(dict(current_positions[hip_name]))
                    if len(self.hip_history[key]) > self.hip_history_size:
                        self.hip_history[key].pop(0)

            for side in ['left', 'right']:
                toe_name = f'{side}_foot_index'
                if toe_name in current_positions:
                    key = f'{side}_toe_ultra'
                    self.toe_history.setdefault(key, []).append(dict(current_positions[toe_name]))
                    if len(self.toe_history[key]) > self.toe_history_size:
                        self.toe_history[key].pop(0)

            self.position_history.append(current_positions)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            smoothed_positions = self._smooth_all(water_level)
            for joint_name, pos in smoothed_positions.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos['z'], 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']

        elif self.position_history and self.frames_since_detection <= SHORT_GAP_HOLD_FRAMES:
            # Brief detection gaps (a couple frames, not a real loss)
            # previously left this row's joint columns entirely empty,
            # which is what produced the "there, gone, back again"
            # flicker. Hold the last known position with decaying
            # confidence for SHORT gaps only — a sustained loss still
            # reads as low-confidence/lost, not disguised as real data.
            last = self.position_history[-1]
            decay = 0.6 ** self.frames_since_detection
            for joint_name, pos in last.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos.get('z', 0.0), 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'] * decay, 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current_positions):
        filtered = {}
        for joint_name, pos in current_positions.items():
            if joint_name in self.last_known:
                last = self.last_known[joint_name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    held = {'x': last['x'], 'y': last['y'], 'z': pos['z'],
                            'visibility': pos['visibility'] * 0.3, 'above_water': pos['above_water']}
                    if 'source' in pos:
                        held['source'] = pos['source']
                    filtered[joint_name] = held
                    continue
            filtered[joint_name] = pos
            self.last_known[joint_name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth_all(self, water_level):
        smoothed = {}

        def _weighted_avg(history, power):
            weights = [(i + 1) ** power for i in range(len(history))]
            tw = sum(weights)
            result = {
                'x': sum(p['x'] * w for p, w in zip(history, weights)) / tw,
                'y': sum(p['y'] * w for p, w in zip(history, weights)) / tw,
                'z': sum(p.get('z', 0) * w for p, w in zip(history, weights)) / tw,
                'visibility': sum(p['visibility'] * w for p, w in zip(history, weights)) / tw,
            }
            result['above_water'] = result['y'] < water_level
            if 'source' in history[-1]:
                result['source'] = history[-1]['source']
            return result

        for joint_name in ALL_LANDMARKS_ABOVE:
            is_foot = 'foot' in joint_name or 'ankle' in joint_name or 'heel' in joint_name
            hist = [f[joint_name] for f in self.position_history if joint_name in f]
            if hist:
                smoothed[joint_name] = _weighted_avg(hist, 2.0 if is_foot else 1.0)

        for side in ['left', 'right']:
            key = f'{side}_foot_best'
            if key in self.foot_history and self.foot_history[key]:
                smoothed[key] = _weighted_avg(self.foot_history[key], 2.0)
            hip_key = f'{side}_hip_ultra'
            if hip_key in self.hip_history and self.hip_history[hip_key]:
                smoothed[f'{side}_hip'] = _weighted_avg(self.hip_history[hip_key], 3.0)
            toe_key = f'{side}_toe_ultra'
            if toe_key in self.toe_history and self.toe_history[toe_key]:
                smoothed[f'{side}_foot_index'] = _weighted_avg(self.toe_history[toe_key], 2.8)

        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


# ============================================================================
# UNDERWATER TRACKER — single video, COCO-17, full frame
# ============================================================================

class RTMPoseUnderwaterTracker:
    def __init__(self, mode='performance', det_frequency=1, pose_tracker=None):
        self.pose_tracker = pose_tracker if pose_tracker is not None else make_pose_tracker(mode, det_frequency)
        self.tracking_data = []

        self.locked_swimmer = None
        self.frames_since_detection = 0
        self.max_frames_lost = MAX_FRAMES_LOST
        self.relock_check_interval = RELOCK_CHECK_INTERVAL
        self.relock_drift_threshold = RELOCK_DRIFT_THRESHOLD
        self._frames_since_relock_check = 0
        self._pending_relock_idx = None
        self._pending_relock_streak = 0

        self.position_history = []
        self.history_size = 4
        self.hip_history = {'left': [], 'right': []}
        self.hip_history_size = 6
        self.last_known = {}
        self.max_jump = 0.25
        self.hip_correction_ratio = 0.20

    def _select_best_underwater(self, keypoints, scores, h, w):
        best_idx, best_score = None, -1
        for i, (kps, sc) in enumerate(zip(keypoints, scores)):
            if not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            major = [5, 6, 11, 12, 13, 14, 15, 16]
            avg_conf = np.mean([sc[j] for j in major])
            hip_width = abs(kps[11][0] - kps[12][0]) / w
            total_score = avg_conf + hip_width * 2.0
            if total_score > best_score:
                best_score, best_idx = total_score, i
        return best_idx

    def _find_matching_underwater(self, keypoints, scores, locked_position, h, w):
        best_match, best_distance = None, LOCK_MATCH_DISTANCE
        for i, (kps, sc) in enumerate(zip(keypoints, scores)):
            if sc[11] < 0.10 or sc[12] < 0.10:
                continue
            if not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            hip_y = (kps[11][1] + kps[12][1]) / 2 / h
            hip_x = (kps[11][0] + kps[12][0]) / 2 / w
            distance = np.sqrt((hip_x - locked_position['x'])**2 + (hip_y - locked_position['y'])**2)
            if distance < best_distance:
                best_distance, best_match = distance, i
        return best_match

    def process_frame(self, frame, frame_num, fps, water_level):
        h, w = frame.shape[:2]
        enhanced = enhance_underwater(frame)
        keypoints, scores = self.pose_tracker(enhanced)

        best_idx = None
        if keypoints is not None and len(keypoints) > 0:
            if self.locked_swimmer is not None:
                best_idx = self._find_matching_underwater(keypoints, scores, self.locked_swimmer, h, w)
                if best_idx is not None:
                    self.frames_since_detection = 0
                    self._frames_since_relock_check += 1
                    if self._frames_since_relock_check >= self.relock_check_interval:
                        self._frames_since_relock_check = 0
                        fresh_idx = self._select_best_underwater(keypoints, scores, h, w)
                        if fresh_idx is not None and fresh_idx != best_idx:
                            fresh_pos = get_hip_position(keypoints[fresh_idx], scores[fresh_idx], h, w)
                            cur_pos = get_hip_position(keypoints[best_idx], scores[best_idx], h, w)
                            if fresh_pos is not None and cur_pos is not None:
                                drift = np.sqrt((fresh_pos['x'] - cur_pos['x'])**2 + (fresh_pos['y'] - cur_pos['y'])**2)
                                if drift > self.relock_drift_threshold:
                                    if self._pending_relock_idx == fresh_idx:
                                        self._pending_relock_streak += 1
                                    else:
                                        self._pending_relock_idx, self._pending_relock_streak = fresh_idx, 1
                                    if self._pending_relock_streak >= 2:
                                        best_idx = fresh_idx
                                        self._pending_relock_idx, self._pending_relock_streak = None, 0
                                else:
                                    self._pending_relock_idx, self._pending_relock_streak = None, 0
                else:
                    self.frames_since_detection += 1
                    if self.frames_since_detection > self.max_frames_lost:
                        best_idx = self._select_best_underwater(keypoints, scores, h, w)
                        if best_idx is not None:
                            self.frames_since_detection = 0
            else:
                best_idx = self._select_best_underwater(keypoints, scores, h, w)
                if best_idx is not None:
                    self.frames_since_detection = 0

            if best_idx is not None:
                self.locked_swimmer = get_hip_position(keypoints[best_idx], scores[best_idx], h, w)
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_frames_lost:
                self.locked_swimmer = None

        best_person = (keypoints[best_idx], scores[best_idx]) if best_idx is not None else None

        frame_data = {
            'frame': frame_num,
            'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4),
            'tracking_locked': best_person is not None,
            'frames_since_detection': self.frames_since_detection,
        }

        if best_person is not None:
            person_kps, person_scores = best_person
            current = {}
            for coco_idx, name in COCO_TO_LANDMARKS.items():
                conf = float(person_scores[coco_idx])
                if conf > 0.05:
                    x_norm, y_norm = person_kps[coco_idx][0] / w, person_kps[coco_idx][1] / h
                    current[name] = {'x': x_norm, 'y': y_norm, 'z': 0.0, 'visibility': conf, 'above_water': y_norm < water_level}

            for side in ['left', 'right']:
                hip, knee = f'{side}_hip', f'{side}_knee'
                if hip in current and knee in current:
                    r = self.hip_correction_ratio
                    hp, kn = current[hip], current[knee]
                    corrected_y = hp['y'] + r * (kn['y'] - hp['y'])
                    current[hip] = {
                        'x': hp['x'] + r * (kn['x'] - hp['x']), 'y': corrected_y, 'z': 0.0,
                        'visibility': min(hp['visibility'], kn['visibility']), 'above_water': corrected_y < water_level,
                    }

            # Synthesize a mid_spine point (underwater-only) for back
            # curvature measurements — midpoint of hip and shoulder,
            # nudged toward the knee side slightly for a rough spine curve.
            if 'left_hip' in current and 'right_hip' in current and 'left_shoulder' in current and 'right_shoulder' in current:
                hip_mid = {
                    'x': (current['left_hip']['x'] + current['right_hip']['x']) / 2,
                    'y': (current['left_hip']['y'] + current['right_hip']['y']) / 2,
                }
                sh_mid = {
                    'x': (current['left_shoulder']['x'] + current['right_shoulder']['x']) / 2,
                    'y': (current['left_shoulder']['y'] + current['right_shoulder']['y']) / 2,
                }
                mid_x = (hip_mid['x'] + sh_mid['x']) / 2
                mid_y = (hip_mid['y'] + sh_mid['y']) / 2
                current['mid_spine'] = {
                    'x': mid_x, 'y': mid_y, 'z': 0.0,
                    'visibility': min(current['left_hip']['visibility'], current['left_shoulder']['visibility']),
                    'above_water': mid_y < water_level,
                }

            current = self._filter_jumps(current)

            for side in ['left', 'right']:
                hip = f'{side}_hip'
                if hip in current:
                    self.hip_history[side].append(dict(current[hip]))
                    if len(self.hip_history[side]) > self.hip_history_size:
                        self.hip_history[side].pop(0)

            self.position_history.append(current)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            smoothed = self._smooth(water_level)
            for name, pos in smoothed.items():
                frame_data[f'{name}_x'] = round(pos['x'], 4)
                frame_data[f'{name}_y'] = round(pos['y'], 4)
                frame_data[f'{name}_z'] = round(pos['z'], 4)
                frame_data[f'{name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{name}_above_water'] = pos['above_water']

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current):
        filtered = {}
        for name, pos in current.items():
            if name in self.last_known:
                last = self.last_known[name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    filtered[name] = {**pos, 'x': last['x'], 'y': last['y'], 'visibility': pos['visibility'] * 0.3}
                    continue
            filtered[name] = pos
            self.last_known[name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth(self, water_level):
        smoothed = {}

        def _wavg(hist, power=1.0):
            weights = [(i + 1) ** power for i in range(len(hist))]
            tw = sum(weights)
            r = {
                'x': sum(p['x'] * w for p, w in zip(hist, weights)) / tw,
                'y': sum(p['y'] * w for p, w in zip(hist, weights)) / tw,
                'z': 0.0,
                'visibility': sum(p['visibility'] * w for p, w in zip(hist, weights)) / tw,
            }
            r['above_water'] = r['y'] < water_level
            return r

        for name in ALL_LANDMARKS_UNDERWATER:
            is_foot = 'foot' in name or 'toe' in name or 'ankle' in name or 'heel' in name
            hist = [f[name] for f in self.position_history if name in f]
            if hist:
                smoothed[name] = _wavg(hist, power=1.5 if is_foot else 1.0)

        for side in ['left', 'right']:
            if self.hip_history[side]:
                smoothed[f'{side}_hip'] = _wavg(self.hip_history[side], power=1.5)

        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


# ============================================================================
# WATERLINE DETECTION — single-video (pose-based, full frame)
# ============================================================================

def _is_horizontal_coco(person_kps, person_scores, h, w):
    if person_scores[5] < 0.3 or person_scores[6] < 0.3 or person_scores[11] < 0.3 or person_scores[12] < 0.3:
        return False
    ls_y, rs_y = person_kps[5][1] / h, person_kps[6][1] / h
    lh_y, rh_y = person_kps[11][1] / h, person_kps[12][1] / h
    shoulder_y, hip_y = (ls_y + rs_y) / 2, (lh_y + rh_y) / 2
    ls_x, rs_x = person_kps[5][0] / w, person_kps[6][0] / w
    lh_x, rh_x = person_kps[11][0] / w, person_kps[12][0] / w
    body_width = max(abs(ls_x - rs_x), abs(lh_x - rh_x))
    return abs(shoulder_y - hip_y) < 0.25 or body_width > 0.25


def _has_water_below(hip_x_norm, hip_y_norm, frame, margin=40):
    """Lightweight, RELATIVE water check used only during waterline
    detection, where water_level isn't known yet. Distinguishes a real
    swimmer from a person on the pool deck near the tent-mask boundary —
    position bounds alone let exactly that kind of candidate through and
    dragged a real video's waterline up to 0.38 when the true value was
    ~0.73."""
    h, w = frame.shape[:2]
    x = int(np.clip(hip_x_norm, 0, 1) * w)
    y = int(np.clip(hip_y_norm, 0, 1) * h)
    y0, y1 = min(h, y + 10), min(h, y + 10 + margin)
    x0, x1 = max(0, x - margin), min(w, x + margin)
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return True
    region_i16 = region.astype(np.int16)
    b, r = region_i16[:, :, 0], region_i16[:, :, 2]
    return ((b - r) > 10).mean() > 0.30


def detect_waterline_from_poses(video_path, pose_tracker_fn):
    """Averages SHOULDER + HIP position (nose excluded — it sits above
    the true surface even in a good float, which was biasing the
    waterline too high) across enough horizontal-swimmer detections to
    find the waterline.

    Three-pass search instead of a hard 100-frame cap that gives up:
      1. First 100 frames, horizontal swimmers only.
      2. If that's not enough samples, extend to 400 frames.
      3. If STILL not enough, drop the horizontal requirement (relaxed
         fallback) rather than defaulting to a fixed guess.
    Each candidate is also validated (position bounds + real water color
    below them) so a person on the pool deck can't contaminate the
    average."""
    MIN_SAMPLES_NEEDED = 15
    MAX_SEARCH_FRAMES = 400

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Speed fix: previously, when Pass 2 or Pass 3 triggered, _scan
    # restarted from frame 0 and re-ran full pose inference on every frame
    # Pass 1 (and Pass 2) had ALREADY computed — same frame, same
    # preprocessing (_detect_with_retry is deterministic), same result,
    # just thrown away and recomputed. That's exactly the case on the
    # harder videos (the ones needing Pass 2/3 at all), which are also
    # the ones that already feel the slowest. Caching inference results
    # by frame number across passes makes Pass 2 reuse Pass 1's first 100
    # frames for free, and Pass 3 reuse everything Pass 1+2 already
    # computed — this changes NOTHING about which frames get scanned or
    # what candidates pass validation, only eliminates repeat inference
    # calls on frames already seen.
    _inference_cache = {}

    def _detect_with_retry(frame_num, frame):
        if frame_num in _inference_cache:
            return _inference_cache[frame_num]
        keypoints, scores = pose_tracker_fn(quick_enhance(frame))
        if keypoints is None or len(keypoints) == 0:
            strong = cv2.convertScaleAbs(frame, alpha=1.6, beta=35)
            keypoints, scores = pose_tracker_fn(strong)
        _inference_cache[frame_num] = (keypoints, scores)
        return keypoints, scores

    def _scan(max_frames, require_horizontal):
        heads, shoulders, hips = [], [], []
        max_search_frame = min(max_frames, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_num in range(max_search_frame):
            ret, frame = cap.read()
            if not ret:
                break
            keypoints, scores = _detect_with_retry(frame_num, frame)
            if keypoints is None or len(keypoints) == 0:
                continue
            for kps, sc in zip(keypoints, scores):
                if require_horizontal and not _is_horizontal_coco(kps, sc, h_frame, w_frame):
                    continue
                if sc[11] > 0.15 and sc[12] > 0.15:
                    cand_hip_y = (kps[11][1] + kps[12][1]) / 2 / h_frame
                    cand_hip_x = (kps[11][0] + kps[12][0]) / 2 / w_frame
                    if cand_hip_y < IGNORE_TOP_PERCENT or cand_hip_y < 0.35:
                        continue
                    if cand_hip_x < 0.15 or cand_hip_x > 0.85:
                        continue
                    if not _has_water_below(cand_hip_x, cand_hip_y, frame):
                        continue
                if not validate_pose_anatomy_coco(kps, sc, h_frame, w_frame):
                    continue
                if sc[0] > 0.5:
                    heads.append(kps[0][1] / h_frame)
                if sc[5] > 0.4 and sc[6] > 0.4:
                    shoulders.append((kps[5][1] + kps[6][1]) / 2 / h_frame)
                if sc[11] > 0.4 and sc[12] > 0.4:
                    hips.append((kps[11][1] + kps[12][1]) / 2 / h_frame)
            if len(shoulders) >= MIN_SAMPLES_NEEDED and len(hips) >= MIN_SAMPLES_NEEDED:
                break
        return heads, shoulders, hips

    head_positions, shoulder_positions, hip_positions = _scan(100, require_horizontal=True)

    if len(shoulder_positions) < MIN_SAMPLES_NEEDED or len(hip_positions) < MIN_SAMPLES_NEEDED:
        head_positions, shoulder_positions, hip_positions = _scan(MAX_SEARCH_FRAMES, require_horizontal=True)

    if len(shoulder_positions) < MIN_SAMPLES_NEEDED or len(hip_positions) < MIN_SAMPLES_NEEDED:
        head_positions, shoulder_positions, hip_positions = _scan(MAX_SEARCH_FRAMES, require_horizontal=False)

    cap.release()

    def _clean(values):
        if len(values) < 5:
            return None
        median, std = np.median(values), np.std(values)
        filtered = [v for v in values if abs(v - median) < 2 * std]
        return np.mean(filtered) if filtered else median

    shoulder_avg, hip_avg = _clean(shoulder_positions), _clean(hip_positions)
    signals = [v for v in [shoulder_avg, hip_avg] if v is not None]
    if signals:
        return float(np.mean(signals))
    head_avg = _clean(head_positions)
    if head_avg is not None:
        return head_avg
    return 0.70


def detect_waterline_above_walticam(frame_top, split_y):
    h, w = frame_top.shape[:2]
    frame_i16 = frame_top.astype(np.int16)
    b, r = frame_i16[:, :, 0], frame_i16[:, :, 2]
    blue_dominant = (b - r) > 15
    row_frac_blue = blue_dominant.mean(axis=1)
    window = max(3, int(h * 0.02))
    threshold = 0.5
    y, waterline_y = h - window, h - 1
    while y > 0:
        if row_frac_blue[y:y + window].mean() >= threshold:
            waterline_y = y
            y -= window
        else:
            break
    return float(np.clip(waterline_y / h, 0.5, 0.97))


def detect_waterline_below_walticam(frame_bottom):
    h, w = frame_bottom.shape[:2]
    gray = cv2.cvtColor(frame_bottom, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    search_end = int(h * 0.30)
    horizontal_sum = np.sum(edges[0:search_end, :], axis=1)
    if len(horizontal_sum) > 0 and np.max(horizontal_sum) > w * 0.20:
        return np.argmax(horizontal_sum) / h
    return 0.05


# ============================================================================
# VISUALIZATION (single-video)
# ============================================================================

def draw_frame_above(frame, best_person, water_level, frame_num, total_frames, tracker_status=None):
    viz = frame.copy()
    h, w = frame.shape[:2]
    water_y = int(water_level * h)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 0, 0), 6)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 255, 255), 3)
    cv2.putText(viz, f"WATERLINE: {water_level:.3f}", (10, water_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if best_person is not None:
        person_kps, person_scores = best_person
        for (i, j) in COCO_CONNECTIONS:
            if person_scores[i] > 0.2 and person_scores[j] > 0.2:
                pt1 = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))
                cv2.line(viz, pt1, pt2, (255, 255, 255), 4, cv2.LINE_AA)
        for idx in range(17):
            if person_scores[idx] > 0.2:
                x, y = int(person_kps[idx][0]), int(person_kps[idx][1])
                color = (0, 255, 0) if y < water_y else (0, 0, 255)
                cv2.circle(viz, (x, y), 9, color, -1)
                cv2.circle(viz, (x, y), 9, (255, 255, 255), 2)
    else:
        cv2.putText(viz, "No swimmer locked", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(viz, f"Frame: {frame_num}/{total_frames}", (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return viz


def draw_frame_underwater(frame, best_person, water_level, frame_num, total_frames):
    viz = frame.copy()
    h, w = frame.shape[:2]
    water_y = int(water_level * h)
    cv2.line(viz, (0, water_y), (w, water_y), (255, 200, 0), 2)

    if best_person is not None:
        person_kps, person_scores = best_person
        for (i, j) in COCO_CONNECTIONS:
            if person_scores[i] > 0.2 and person_scores[j] > 0.2:
                pt1 = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))
                cv2.line(viz, pt1, pt2, (255, 255, 255), 4, cv2.LINE_AA)
        for idx in range(17):
            if person_scores[idx] > 0.2:
                x, y = int(person_kps[idx][0]), int(person_kps[idx][1])
                cv2.circle(viz, (x, y), 9, (0, 200, 255), -1)
                cv2.circle(viz, (x, y), 9, (255, 255, 255), 2)
    else:
        cv2.putText(viz, "No swimmer locked", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(viz, f"Frame: {frame_num}/{total_frames}", (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return viz


# ============================================================================
# MAIN ENTRY POINTS — called by app.py
# ============================================================================

def process_video_above_water(input_path, output_path, water_level=None,
                               mode='performance', det_frequency=1,
                               max_duration=60, progress_callback=None,
                               pose_tracker=None):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = AboveWaterRTMPoseTracker(mode=mode, det_frequency=det_frequency, pose_tracker=pose_tracker)

    if water_level is None:
        water_level = detect_waterline_from_poses(str(input_path), tracker.pose_tracker)

    max_frames = min(int(max_duration * fps), total) if max_duration else total

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = _output_video_size(w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        best_person = tracker.process_frame(frame, frame_count, fps, water_level)
        annotated = draw_frame_above(frame, best_person, water_level, frame_count, max_frames)
        if (out_w, out_h) != (w, h):
            annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out.write(annotated)
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, max_frames)

    cap.release()
    out.release()

    df = tracker.get_dataframe()
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')

    return output_path, csv_path


def process_video_underwater(input_path, output_path, water_level=None,
                              mode='performance', det_frequency=1,
                              max_duration=60, progress_callback=None,
                              pose_tracker=None):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if water_level is None:
        water_level = 0.05  # underwater videos: waterline near top of frame by convention

    tracker = RTMPoseUnderwaterTracker(mode=mode, det_frequency=det_frequency, pose_tracker=pose_tracker)
    max_frames = min(int(max_duration * fps), total) if max_duration else total

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = _output_video_size(w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        best_person = tracker.process_frame(frame, frame_count, fps, water_level)
        annotated = draw_frame_underwater(frame, best_person, water_level, frame_count, max_frames)
        if (out_w, out_h) != (w, h):
            annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out.write(annotated)
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, max_frames)

    cap.release()
    out.release()

    df = tracker.get_dataframe()
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')

    return output_path, csv_path


# ============================================================================
# WALTICAM TRACKERS (Halpe26, split-frame) — the well-tested v2 logic
# ============================================================================

class WalticamAboveTracker:
    def __init__(self, mode='performance', det_frequency=1, pose_tracker=None):
        self.pose_tracker = pose_tracker if pose_tracker is not None else make_pose_tracker_halpe(mode, det_frequency)
        self.tracking_data = []
        self.position_history = []
        self.history_size = 4
        self.last_known = {}
        self.max_jump = 0.25
        self.hip_correction_ratio = 0.20
        self.lock = SwimmerLock()

    def _is_valid_candidate(self, kps, sc, hip, sub_frame):
        if hip['x'] < 0.10 or hip['x'] > 0.90:
            return False
        return is_water_relative(hip['x'], hip['y'], sub_frame)

    def process_frame(self, frame_top, frame_num, fps, water_level):
        h, w = frame_top.shape[:2]
        enhanced = quick_enhance(frame_top)
        keypoints, scores = self.pose_tracker(enhanced)

        frame_data = {
            'frame': frame_num, 'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4), 'tracking_locked': False,
        }

        best_idx = self.lock.select(keypoints, scores, h, w, self._is_valid_candidate, frame_top)
        best_person = (keypoints[best_idx], scores[best_idx]) if best_idx is not None else None

        if best_person is not None:
            person_kps, person_scores = best_person
            current = {}
            for hidx, name in HALPE26_TO_LANDMARKS.items():
                x_norm, y_norm = person_kps[hidx][0] / w, person_kps[hidx][1] / h
                conf = float(person_scores[hidx])
                resolved = resolve_landmark(name, x_norm, y_norm, conf, water_level)
                if resolved is not None:
                    current[name] = resolved

            for side in ['left', 'right']:
                hip, knee = f'{side}_hip', f'{side}_knee'
                if hip in current and knee in current:
                    r = self.hip_correction_ratio
                    current[hip] = {
                        'x': current[hip]['x'] + r * (current[knee]['x'] - current[hip]['x']),
                        'y': current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y']),
                        'z': 0.0, 'visibility': min(current[hip]['visibility'], current[knee]['visibility']),
                        'above_water': (current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y'])) < water_level,
                    }

            current = self._filter_jumps(current)
            self.position_history.append(current)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            smoothed = self._smooth(water_level)
            for name, pos in smoothed.items():
                frame_data[f'{name}_x'] = round(pos['x'], 4)
                frame_data[f'{name}_y'] = round(pos['y'], 4)
                frame_data[f'{name}_z'] = round(pos['z'], 4)
                frame_data[f'{name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{name}_above_water'] = pos['above_water']
            frame_data['tracking_locked'] = True
        else:
            if self.position_history:
                last = self.position_history[-1]
                for name, pos in last.items():
                    frame_data[f'{name}_x'] = round(pos['x'], 4)
                    frame_data[f'{name}_y'] = round(pos['y'], 4)
                    frame_data[f'{name}_z'] = round(pos['z'], 4)
                    frame_data[f'{name}_visibility'] = round(pos['visibility'] * 0.5, 4)
                    frame_data[f'{name}_above_water'] = pos['above_water']

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current):
        filtered = {}
        for name, pos in current.items():
            if name in self.last_known:
                last = self.last_known[name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    filtered[name] = {**pos, 'x': last['x'], 'y': last['y'], 'visibility': pos['visibility'] * 0.3}
                    continue
            filtered[name] = pos
            self.last_known[name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth(self, water_level):
        smoothed = {}
        for name in ALL_LANDMARKS_WALTICAM_ABOVE:
            hist = [f[name] for f in self.position_history if name in f]
            if hist:
                weights = [(i + 1) ** 1.0 for i in range(len(hist))]
                tw = sum(weights)
                smoothed[name] = {
                    'x': sum(p['x'] * w for p, w in zip(hist, weights)) / tw,
                    'y': sum(p['y'] * w for p, w in zip(hist, weights)) / tw,
                    'z': 0.0,
                    'visibility': sum(p['visibility'] * w for p, w in zip(hist, weights)) / tw,
                    'above_water': sum(p['y'] * w for p, w in zip(hist, weights)) / tw < water_level,
                }
        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


class WalticamBelowTracker:
    def __init__(self, mode='performance', det_frequency=1, pose_tracker=None):
        self.pose_tracker = pose_tracker if pose_tracker is not None else make_pose_tracker_halpe(mode, det_frequency)
        self.tracking_data = []
        self.position_history = []
        self.history_size = 4
        self.hip_history = {'left': [], 'right': []}
        self.hip_history_size = 6
        self.last_known = {}
        self.max_jump = 0.25
        self.hip_correction_ratio = 0.20
        self.lock = SwimmerLock()

    def _is_valid_candidate(self, kps, sc, hip, sub_frame):
        return not (hip['x'] < 0.10 or hip['x'] > 0.90)

    def process_frame(self, frame_bottom, frame_num, fps, water_level):
        h, w = frame_bottom.shape[:2]
        enhanced = enhance_underwater(frame_bottom)
        keypoints, scores = self.pose_tracker(enhanced)

        frame_data = {
            'frame': frame_num, 'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4), 'tracking_locked': False,
            'frames_since_detection': self.lock.frames_since_detection,
        }

        best_idx = self.lock.select(keypoints, scores, h, w, self._is_valid_candidate, frame_bottom)
        best_person = (keypoints[best_idx], scores[best_idx]) if best_idx is not None else None

        if best_person is not None:
            person_kps, person_scores = best_person
            current = {}
            for hidx, name in HALPE26_TO_LANDMARKS.items():
                x_norm, y_norm = person_kps[hidx][0] / w, person_kps[hidx][1] / h
                conf = float(person_scores[hidx])
                resolved = resolve_landmark(name, x_norm, y_norm, conf, water_level)
                if resolved is not None:
                    current[name] = resolved

            for side in ['left', 'right']:
                hip, knee = f'{side}_hip', f'{side}_knee'
                if hip in current and knee in current:
                    r = self.hip_correction_ratio
                    current[hip] = {
                        'x': current[hip]['x'] + r * (current[knee]['x'] - current[hip]['x']),
                        'y': current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y']),
                        'z': 0.0, 'visibility': min(current[hip]['visibility'], current[knee]['visibility']),
                        'above_water': (current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y'])) < water_level,
                    }

            current = self._filter_jumps(current)

            for side in ['left', 'right']:
                hip = f'{side}_hip'
                if hip in current:
                    self.hip_history[side].append(dict(current[hip]))
                    if len(self.hip_history[side]) > self.hip_history_size:
                        self.hip_history[side].pop(0)

            self.position_history.append(current)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            smoothed = self._smooth(water_level)
            for name, pos in smoothed.items():
                frame_data[f'{name}_x'] = round(pos['x'], 4)
                frame_data[f'{name}_y'] = round(pos['y'], 4)
                frame_data[f'{name}_z'] = round(pos['z'], 4)
                frame_data[f'{name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{name}_above_water'] = pos['above_water']
            frame_data['tracking_locked'] = True
        else:
            if self.position_history:
                last = self.position_history[-1]
                for name, pos in last.items():
                    frame_data[f'{name}_x'] = round(pos['x'], 4)
                    frame_data[f'{name}_y'] = round(pos['y'], 4)
                    frame_data[f'{name}_z'] = round(pos['z'], 4)
                    frame_data[f'{name}_visibility'] = round(pos['visibility'] * 0.5, 4)
                    frame_data[f'{name}_above_water'] = pos['above_water']

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current):
        filtered = {}
        for name, pos in current.items():
            if name in self.last_known:
                last = self.last_known[name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    filtered[name] = {**pos, 'x': last['x'], 'y': last['y'], 'visibility': pos['visibility'] * 0.3}
                    continue
            filtered[name] = pos
            self.last_known[name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth(self, water_level):
        smoothed = {}

        def _wavg(hist, power=1.0):
            weights = [(i + 1) ** power for i in range(len(hist))]
            tw = sum(weights)
            r = {
                'x': sum(p['x'] * w for p, w in zip(hist, weights)) / tw,
                'y': sum(p['y'] * w for p, w in zip(hist, weights)) / tw,
                'z': 0.0,
                'visibility': sum(p['visibility'] * w for p, w in zip(hist, weights)) / tw,
            }
            r['above_water'] = r['y'] < water_level
            return r

        for name in ALL_LANDMARKS_WALTICAM_BELOW:
            is_foot = 'foot' in name or 'toe' in name or 'ankle' in name or 'heel' in name
            hist = [f[name] for f in self.position_history if name in f]
            if hist:
                smoothed[name] = _wavg(hist, power=1.5 if is_foot else 1.0)

        for side in ['left', 'right']:
            if self.hip_history[side]:
                smoothed[f'{side}_hip'] = _wavg(self.hip_history[side], power=1.5)

        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


def draw_walticam_skeleton(viz, person_kps, person_scores, offset_y=0, water_y=None):
    if person_kps is None:
        return
    for (i, j) in HALPE_CONNECTIONS:
        if person_scores[i] > 0.2 and person_scores[j] > 0.2:
            pt1 = (int(person_kps[i][0]), int(person_kps[i][1]) + offset_y)
            pt2 = (int(person_kps[j][0]), int(person_kps[j][1]) + offset_y)
            cv2.line(viz, pt1, pt2, (255, 255, 255), 4, cv2.LINE_AA)
    for idx in [0, 5, 6, 11, 12, 13, 14, 15, 16, 18, 20, 21, 24, 25]:
        if person_scores[idx] > 0.2:
            x, y = int(person_kps[idx][0]), int(person_kps[idx][1]) + offset_y
            is_above = (y < water_y) if water_y is not None else True
            color = (0, 255, 0) if is_above else (0, 0, 255)
            cv2.circle(viz, (x, y), 9, color, -1)
            cv2.circle(viz, (x, y), 9, (255, 255, 255), 2)


def process_video_walticam(input_path, output_path, mode='performance',
                            det_frequency=1, max_duration=60, progress_callback=None,
                            above_pose_tracker=None, below_pose_tracker=None):
    """Split-screen WaltiCam video: top half = above-water, bottom half =
    underwater. Returns (video_path, above_csv_path, below_csv_path).

    above_pose_tracker / below_pose_tracker: pass pre-built ones to skip
    re-loading the model (see AboveWaterRTMPoseTracker's pose_tracker
    param docstring). Kept as TWO separate objects here rather than one
    shared instance — above/below calls are interleaved every single
    frame in the loop below (not sequential like the non-Walticam path),
    and reusing one live model object across rapidly alternating calls on
    unrelated image regions isn't something verifiable without the real
    rtmlib library available, so this stays on the cautious side."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    name = output_path.stem.replace('_walticam_tracking', '')

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    split_y = h // 2

    above_tracker = WalticamAboveTracker(mode=mode, det_frequency=det_frequency, pose_tracker=above_pose_tracker)
    below_tracker = WalticamBelowTracker(mode=mode, det_frequency=det_frequency, pose_tracker=below_pose_tracker)

    # Sample a handful of frames to detect each half's waterline once.
    sample_count = min(8, max(1, int(fps * 2)))
    sample_idxs = np.linspace(0, min(int(fps * 2), max(total - 1, 0)), sample_count).astype(int)
    above_samples, below_samples = [], []
    for fi in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, sframe = cap.read()
        if not ret:
            continue
        above_samples.append(detect_waterline_above_walticam(sframe[:split_y, :], split_y))
        below_samples.append(detect_waterline_below_walticam(sframe[split_y:, :]))
    wl_above = float(np.median(above_samples)) if above_samples else 0.50
    wl_below = float(np.median(below_samples)) if below_samples else 0.05

    max_frames = min(int(max_duration * fps), total) if max_duration else total

    out_w, out_h = _output_video_size(w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_top, frame_bottom = frame[:split_y, :], frame[split_y:, :]
        above_result = above_tracker.process_frame(frame_top, frame_count, fps, wl_above)
        below_result = below_tracker.process_frame(frame_bottom, frame_count, fps, wl_below)

        viz = frame.copy()
        cv2.line(viz, (0, split_y), (w, split_y), (0, 255, 255), 2)
        above_wl_y = int(wl_above * split_y)
        below_wl_y = split_y + int(wl_below * (h - split_y))
        cv2.line(viz, (0, above_wl_y), (w, above_wl_y), (255, 200, 0), 1)
        cv2.line(viz, (0, below_wl_y), (w, below_wl_y), (255, 200, 0), 1)

        if above_result is not None:
            draw_walticam_skeleton(viz, above_result[0], above_result[1], offset_y=0, water_y=above_wl_y)
        if below_result is not None:
            draw_walticam_skeleton(viz, below_result[0], below_result[1], offset_y=split_y, water_y=below_wl_y)

        if (out_w, out_h) != (w, h):
            viz = cv2.resize(viz, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out.write(viz)
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, max_frames)

    cap.release()
    out.release()

    above_df = _kalman_filter_dataframe(above_tracker.get_dataframe(), ALL_LANDMARKS_WALTICAM_ABOVE)
    below_df = _kalman_filter_dataframe(below_tracker.get_dataframe(), ALL_LANDMARKS_WALTICAM_BELOW)

    above_csv = output_path.parent / f"{name}_above_tracking_data.csv"
    below_csv = output_path.parent / f"{name}_below_tracking_data.csv"
    above_df.to_csv(above_csv, index=False, float_format='%.4f')
    below_df.to_csv(below_csv, index=False, float_format='%.4f')

    return output_path, above_csv, below_csv
