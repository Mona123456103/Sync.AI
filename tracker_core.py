#!/usr/bin/env python3
"""
BARRACUDA TRACKER CORE — Web App Backend
===========================================================================
Combines three RTMPose-based tracking modes for the Streamlit web app:

  1. ABOVE-WATER (full frame, tent masking) — for a standalone above-water
     video. Waterline auto-detected from head+shoulder+hip average.

  2. UNDERWATER (full frame, mid_spine synthesis) — for a standalone
     underwater video. Waterline auto-detected via edge/color + shoulder
     calibration.

  3. WALTICAM (split-screen) — for a single video where the top half is
     the above-water view and the bottom half is the underwater view.

All three use RTMPose-x via rtmlib for detection.
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
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def quick_enhance(frame):
    """Light enhancement for above-water footage."""
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=15)


def enhance_underwater(frame):
    """Underwater color correction + CLAHE + light sharpen"""
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

    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.addWeighted(sharpened, 0.6, enhanced, 0.4, 0)


def make_pose_tracker(mode='balanced', det_frequency=2):
    """Create a shared RTMPose PoseTracker instance."""
    from rtmlib import PoseTracker, Body
    return PoseTracker(
        Body,
        mode=mode,
        det_frequency=det_frequency,
        backend='onnxruntime',
        device='cpu',
        to_openpose=False,
    )


# ============================================================================
# SHARED: KALMAN FILTER
# ============================================================================

class ImprovedKalmanFilter1D:
    """1D Kalman filter with adaptive gains."""

    def __init__(self, process_var=0.001, measurement_var=0.05, outlier_threshold=0.15):
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
        R_adj = self.R * (2.0 - confidence)
        y = measurement - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + R_adj
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
        else:
            self.consecutive_predictions += 1
            return np.nan if self.consecutive_predictions > self.max_predictions else predicted


def apply_kalman_filter_to_csv(csv_path, landmarks):
    """Apply adaptive Kalman filter to a tracking CSV, for the given
    list of landmark names present in that CSV."""
    df = pd.read_csv(csv_path)

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

            filtered = []
            for _, row in df.iterrows():
                m = row[col] if not pd.isna(row[col]) else None
                c = row[vis_col] if vis_col in df.columns and not pd.isna(row[vis_col]) else 1.0
                if m is not None and kf.is_outlier(m):
                    m = None
                filtered.append(kf.filter(m, c))

            df[f'{joint}_{axis}_raw'] = df[col]
            df[col] = filtered

    output_path = Path(csv_path).parent / (Path(csv_path).stem + "_KALMAN.csv")
    df.to_csv(output_path, index=False, float_format='%.4f')
    return output_path


# ============================================================================
# MODE 1: ABOVE-WATER (full frame, tent masking)
# ============================================================================

def is_horizontal_position_from_kps(person_kps, person_scores, h, w):
    """Check if swimmer is in horizontal back-layout position."""
    if person_scores[5] < 0.3 or person_scores[6] < 0.3:
        return False
    if person_scores[11] < 0.3 or person_scores[12] < 0.3:
        return False

    ls_y = person_kps[5][1] / h
    rs_y = person_kps[6][1] / h
    lh_y = person_kps[11][1] / h
    rh_y = person_kps[12][1] / h

    ls_x = person_kps[5][0] / w
    rs_x = person_kps[6][0] / w
    lh_x = person_kps[11][0] / w
    rh_x = person_kps[12][0] / w

    shoulder_y = (ls_y + rs_y) / 2
    hip_y = (lh_y + rh_y) / 2
    height_diff = abs(shoulder_y - hip_y)
    body_width = max(abs(ls_x - rs_x), abs(lh_x - rh_x))

    return height_diff < 0.25 or body_width > 0.25


def detect_waterline_from_poses(video_path, pose_tracker_fn):
    """Waterline = average of head, shoulder, hip over first 100 frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    head_positions, hip_positions, shoulder_positions = [], [], []
    max_search_frame = min(100, total_frames)

    for frame_num in range(max_search_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        enhanced = quick_enhance(frame)
        keypoints, scores = pose_tracker_fn(enhanced)
        if keypoints is None or len(keypoints) == 0:
            continue
        for person_kps, person_scores in zip(keypoints, scores):
            if not is_horizontal_position_from_kps(person_kps, person_scores, h_frame, w_frame):
                continue
            if person_scores[0] > 0.5:
                head_positions.append(person_kps[0][1] / h_frame)
            if person_scores[5] > 0.4 and person_scores[6] > 0.4:
                shoulder_positions.append((person_kps[5][1] + person_kps[6][1]) / 2 / h_frame)
            if person_scores[11] > 0.4 and person_scores[12] > 0.4:
                hip_positions.append((person_kps[11][1] + person_kps[12][1]) / 2 / h_frame)
    cap.release()

    def _clean(values):
        if len(values) < 5:
            return None
        median = np.median(values)
        std = np.std(values)
        filtered = [v for v in values if abs(v - median) < 2 * std]
        return np.mean(filtered) if filtered else median

    head_avg = _clean(head_positions)
    shoulder_avg = _clean(shoulder_positions)
    hip_avg = _clean(hip_positions)

    signals = [s for s in [head_avg, shoulder_avg, hip_avg] if s is not None]
    return float(np.mean(signals)) if signals else 0.70


def is_in_water_lenient(hip_x_norm, hip_y_norm, water_level, frame):
    """Blue-water color validation to reject tents/decks."""
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
        pct = (np.sum(mask1 > 0) + np.sum(mask2 > 0)) / (below_region.shape[0] * below_region.shape[1])
        if pct < 0.15:
            return False

    sample_at_hip = hsv[max(0, hip_y - 20):min(h, hip_y + 20), max(0, hip_x - 40):min(w, hip_x + 40)]
    if sample_at_hip.size > 0:
        mask1_hip = cv2.inRange(sample_at_hip, water_lower1, water_upper1)
        mask2_hip = cv2.inRange(sample_at_hip, water_lower2, water_upper2)
        pct_hip = (np.sum(mask1_hip > 0) + np.sum(mask2_hip > 0)) / (sample_at_hip.shape[0] * sample_at_hip.shape[1])
        if pct_hip < 0.10:
            return False
    return True


def calculate_pose_size_coco(person_kps, person_scores, h, w):
    if person_scores[11] < 0.10 or person_scores[12] < 0.10:
        return None
    lh_x, lh_y = person_kps[11][0] / w, person_kps[11][1] / h
    rh_x, rh_y = person_kps[12][0] / w, person_kps[12][1] / h

    torso_height = torso_width = 0
    has_shoulders = False
    if person_scores[5] > 0.10 and person_scores[6] > 0.10:
        has_shoulders = True
        ls_y, rs_y = person_kps[5][1] / h, person_kps[6][1] / h
        hip_y = (lh_y + rh_y) / 2
        shoulder_y = (ls_y + rs_y) / 2
        torso_height = abs(hip_y - shoulder_y)
        ls_x = person_kps[5][0] / w
        hip_x = (lh_x + rh_x) / 2
        torso_width = abs(hip_x - ls_x)

    hip_width = abs(lh_x - rh_x)
    full_height = 0
    has_feet = False
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
    size = size / 6.5
    return {'size': size}


def validate_pose_anatomy_coco(person_kps, person_scores, h, w):
    if person_scores[5] > 0.10 and person_scores[11] > 0.10:
        if person_kps[5][1] / h > person_kps[11][1] / h + 0.35:
            return False
    if person_scores[6] > 0.10 and person_scores[12] > 0.10:
        if person_kps[6][1] / h > person_kps[12][1] / h + 0.35:
            return False
    if person_scores[11] > 0.10 and person_scores[12] > 0.10:
        hip_width = abs(person_kps[11][0] / w - person_kps[12][0] / w)
        if hip_width < 0.02 or hip_width > 0.60:
            return False
    return True


def select_best_swimmer_coco(all_keypoints, all_scores, water_level, frame, ignore_top_percent):
    """Select best swimmer, rejecting tent zone / edges / non-water areas."""
    if all_keypoints is None or len(all_keypoints) == 0:
        return None
    h, w = frame.shape[:2]

    if len(all_keypoints) == 1:
        person_kps, person_scores = all_keypoints[0], all_scores[0]
        if is_horizontal_position_from_kps(person_kps, person_scores, h, w):
            if person_scores[11] > 0.15 and person_scores[12] > 0.15:
                hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
                hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w
                if hip_y < ignore_top_percent or hip_y < 0.35:
                    return None
                if hip_x < 0.15 or hip_x > 0.85:
                    return None
                if abs(hip_x - 0.5) < 0.35 and is_in_water_lenient(hip_x, hip_y, water_level, frame):
                    return 0
        return None

    pose_sizes = []
    for i, (person_kps, person_scores) in enumerate(zip(all_keypoints, all_scores)):
        if person_scores[11] > 0.15 and person_scores[12] > 0.15:
            hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
            hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w
            if hip_y < ignore_top_percent or hip_y < 0.35:
                continue
            if hip_x < 0.15 or hip_x > 0.85:
                continue
        size_info = calculate_pose_size_coco(person_kps, person_scores, h, w)
        if size_info:
            pose_sizes.append((i, person_kps, person_scores, size_info))

    if not pose_sizes:
        return None
    pose_sizes.sort(key=lambda x: x[3]['size'], reverse=True)
    min_foreground_size = pose_sizes[0][3]['size'] * 0.60

    for idx, person_kps, person_scores, size_info in pose_sizes:
        if size_info['size'] < min_foreground_size or size_info['size'] < 0.15:
            continue
        if not is_horizontal_position_from_kps(person_kps, person_scores, h, w):
            continue
        if person_scores[11] < 0.15 or person_scores[12] < 0.15:
            continue
        hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
        hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w
        if hip_y < ignore_top_percent or hip_y < 0.45:
            continue
        if hip_x < 0.15 or hip_x > 0.85 or abs(hip_x - 0.5) > 0.45:
            continue
        if hip_y < water_level - 0.10:
            continue
        if not is_in_water_lenient(hip_x, hip_y, water_level, frame):
            continue
        if not validate_pose_anatomy_coco(person_kps, person_scores, h, w):
            continue
        visible_count = sum(1 for c in [0, 5, 6, 11, 12, 13, 14, 15, 16] if person_scores[c] > 0.15)
        if visible_count < 3:
            continue
        return idx
    return None


class AboveWaterRTMPoseTracker:
    """Above-water tracker: tent masking, swimmer locking, smoothing."""

    def __init__(self, mode='balanced', det_frequency=2, ignore_top_percent=0.35):
        self.pose_tracker = make_pose_tracker(mode, det_frequency)
        self.ignore_top_percent = ignore_top_percent
        self.tracking_data = []
        self.locked_swimmer = None
        self.frames_since_detection = 0
        self.max_frames_lost = 30
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
        # Periodic re-validation, to catch slow drift onto a wrong person
        # (e.g. a window reflection) that frame-to-frame proximity
        # matching alone can miss.
        self.relock_check_interval = 45   # ~1.5s at 30fps
        self.relock_drift_threshold = 0.15  # normalized frame units
        self._pending_relock_idx = None
        self._pending_relock_streak = 0

    def process_frame(self, frame, frame_num, fps, water_level):
        h, w = frame.shape[:2]
        frame_masked = frame.copy()
        mask_height = int(h * self.ignore_top_percent)
        frame_masked[0:mask_height, :] = 0
        enhanced = quick_enhance(frame_masked)
        keypoints, scores = self.pose_tracker(enhanced)

        best_person_idx, best_person = None, None
        if keypoints is not None and len(keypoints) > 0:
            if self.locked_swimmer is not None:
                best_person_idx = self._find_matching_swimmer(
                    keypoints, scores, self.locked_swimmer, h, w,
                    water_level=water_level, frame_for_water_check=frame_masked,
                )
                if best_person_idx is not None:
                    self.frames_since_detection = 0
                    self.locked_swimmer = self._get_swimmer_position(keypoints[best_person_idx], scores[best_person_idx], h, w)

                    # Periodic re-validation: even while "locked," check
                    # every `relock_check_interval` frames whether a fresh,
                    # fully-validated selection agrees with the current
                    # lock. If they've drifted far apart, trust the fresh
                    # selection instead — this catches slow drift onto a
                    # wrong person (e.g. a window reflection) that
                    # proximity-matching alone wouldn't catch, since each
                    # single-frame step can look "close enough."
                    self._frames_since_relock_check = getattr(self, '_frames_since_relock_check', 0) + 1
                    if self._frames_since_relock_check >= self.relock_check_interval:
                        self._frames_since_relock_check = 0
                        fresh_idx = select_best_swimmer_coco(
                            keypoints, scores, water_level, frame_masked, self.ignore_top_percent
                        )
                        if fresh_idx is not None and fresh_idx != best_person_idx:
                            fresh_pos = self._get_swimmer_position(keypoints[fresh_idx], scores[fresh_idx], h, w)
                            drift = np.sqrt(
                                (fresh_pos['x'] - self.locked_swimmer['x']) ** 2 +
                                (fresh_pos['y'] - self.locked_swimmer['y']) ** 2
                            )
                            if drift > self.relock_drift_threshold:
                                # Require the SAME fresh candidate to win
                                # two checks in a row before switching —
                                # guards against a one-off outlier (e.g. a
                                # passing reflection) hijacking a good lock.
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

        if best_person_idx is not None:
            best_person = (keypoints[best_person_idx], scores[best_person_idx])

        frame_data = {
            'frame': frame_num, 'time_seconds': round(frame_num / fps, 4),
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
                        'x': x_norm, 'y': y_norm, 'z': 0.0,
                        'visibility': conf, 'above_water': y_norm < water_level,
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
                    foot_key = f'{side}_foot_best'
                    self.foot_history.setdefault(foot_key, []).append({
                        'x': current_positions[best_foot]['x'], 'y': current_positions[best_foot]['y'],
                        'z': current_positions[best_foot]['z'], 'visibility': current_positions[best_foot]['visibility'],
                        'above_water': current_positions[best_foot]['above_water'], 'source': best_foot,
                    })
                    if len(self.foot_history[foot_key]) > self.foot_history_size:
                        self.foot_history[foot_key].pop(0)

            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                if hip_name in current_positions:
                    hip_key = f'{side}_hip_ultra'
                    self.hip_history.setdefault(hip_key, []).append(current_positions[hip_name].copy())
                    if len(self.hip_history[hip_key]) > self.hip_history_size:
                        self.hip_history[hip_key].pop(0)

            for side in ['left', 'right']:
                toe_name = f'{side}_foot_index'
                if toe_name in current_positions:
                    toe_key = f'{side}_toe_ultra'
                    self.toe_history.setdefault(toe_key, []).append(current_positions[toe_name].copy())
                    if len(self.toe_history[toe_key]) > self.toe_history_size:
                        self.toe_history[toe_key].pop(0)

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

        def _wavg(history, power):
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
                smoothed[joint_name] = _wavg(hist, 2.0 if is_foot else 1.0)

        for side in ['left', 'right']:
            foot_key = f'{side}_foot_best'
            if foot_key in self.foot_history and self.foot_history[foot_key]:
                smoothed[foot_key] = _wavg(self.foot_history[foot_key], 2.0)
        for side in ['left', 'right']:
            hip_key = f'{side}_hip_ultra'
            if hip_key in self.hip_history and self.hip_history[hip_key]:
                smoothed[f'{side}_hip'] = _wavg(self.hip_history[hip_key], 3.0)
        for side in ['left', 'right']:
            toe_key = f'{side}_toe_ultra'
            if toe_key in self.toe_history and self.toe_history[toe_key]:
                smoothed[f'{side}_foot_index'] = _wavg(self.toe_history[toe_key], 2.8)
        return smoothed

    def _get_swimmer_position(self, person_kps, person_scores, h, w):
        if person_scores[11] < 0.05 or person_scores[12] < 0.05:
            return None
        return {'x': (person_kps[11][0] + person_kps[12][0]) / 2 / w,
                'y': (person_kps[11][1] + person_kps[12][1]) / 2 / h}

    def _find_matching_swimmer(self, all_keypoints, all_scores, locked_position, h, w,
                                water_level=None, frame_for_water_check=None):
        """Find the detected person closest to the locked position — but
        ONLY among candidates that still pass the same sanity checks used
        for the initial lock (position bounds, anatomy, in-water). Without
        these checks, anything that drifts within `best_distance` of the
        last known hip position (a window reflection, a person on deck,
        etc.) can silently steal the lock and never be re-validated.

        FIX (was the root cause of "tracking two people"): the old version
        only checked proximity, with no re-validation at all once locked.
        """
        best_match, best_distance = None, 0.12  # tightened from 0.20
        for i, (person_kps, person_scores) in enumerate(zip(all_keypoints, all_scores)):
            if person_scores[11] < 0.10 or person_scores[12] < 0.10:
                continue
            hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
            hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w

            # Same bounds as the initial selection — reject anything in the
            # masked/edge zone regardless of how close it is to last frame.
            if hip_y < self.ignore_top_percent or hip_y < 0.35:
                continue
            if hip_x < 0.15 or hip_x > 0.85:
                continue

            # Anatomy sanity check (rejects e.g. a torso-only reflection
            # with implausible proportions).
            if not validate_pose_anatomy_coco(person_kps, person_scores, h, w):
                continue

            # Water check, when we have a frame to check against — a
            # window reflection is very unlikely to sit "in the water"
            # color-wise the way the real swimmer does.
            if water_level is not None and frame_for_water_check is not None:
                if not is_in_water_lenient(hip_x, hip_y, water_level, frame_for_water_check):
                    continue

            distance = np.sqrt((hip_x - locked_position['x'])**2 + (hip_y - locked_position['y'])**2)
            if distance < best_distance:
                best_distance, best_match = distance, i
        return best_match

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


def draw_frame_above(frame, best_person, water_level, frame_num, total_frames):
    """Visualization for above-water mode."""
    viz = frame.copy()
    h, w = frame.shape[:2]
    water_y = int(water_level * h)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 0, 0), 6)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 255, 255), 3)
    cv2.putText(viz, f"WATERLINE: {water_level:.3f}", (10, water_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(viz, f"WATERLINE: {water_level:.3f}", (10, water_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    overlay = viz.copy()
    cv2.rectangle(overlay, (0, water_y), (w, h), (255, 200, 100), -1)
    cv2.addWeighted(overlay, 0.15, viz, 0.85, 0, viz)
    cv2.rectangle(viz, (10, 10), (650, 100), (0, 0, 0), -1)

    if best_person is not None:
        person_kps, person_scores = best_person
        for (i, j) in COCO_CONNECTIONS:
            if person_scores[i] > 0.2 and person_scores[j] > 0.2:
                pt1 = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))
                thick = 4 if i >= 13 or j >= 13 else 3
                cv2.line(viz, pt1, pt2, (255, 255, 255), thick, cv2.LINE_AA)
        above = total_joints = 0
        for idx in range(17):
            min_vis = 0.2 if idx >= 15 else 0.3
            if person_scores[idx] > min_vis:
                x, y = int(person_kps[idx][0]), int(person_kps[idx][1])
                is_above = y < water_y
                color = (0, 255, 0) if is_above else (0, 0, 255)
                radius = 12 if idx >= 15 else 8
                cv2.circle(viz, (x, y), radius, color, -1)
                cv2.circle(viz, (x, y), radius, (255, 255, 255), 2)
                above += int(is_above)
                total_joints += 1
        pct = (above / total_joints * 100) if total_joints else 0
        cv2.putText(viz, f"Above: {above}/{total_joints} ({pct:.0f}%)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz, f"Below: {total_joints-above}/{total_joints} ({100-pct:.0f}%)", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(viz, "No horizontal swimmer detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ft = f"Frame: {frame_num}/{total_frames}"
    ts = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(viz, ft, (650 - ts[0] - 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return viz


def process_video_above_water(video_path, output_path, water_level=None,
                               mode='balanced', det_frequency=2,
                               max_duration=60, ignore_top_percent=0.35,
                               progress_callback=None):
    """Process a standalone above-water video. Returns (video_path, csv_path)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = AboveWaterRTMPoseTracker(mode=mode, det_frequency=det_frequency,
                                        ignore_top_percent=ignore_top_percent)

    if water_level is None:
        water_level = detect_waterline_from_poses(str(video_path), tracker.pose_tracker)

    max_frames = int(max_duration * fps) if max_duration else total

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        os.remove(output_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        best_person = tracker.process_frame(frame, frame_count, fps, water_level)
        annotated = draw_frame_above(frame, best_person, water_level, frame_count, total)
        out.write(annotated)
        frame_count += 1
        if progress_callback is not None:
            progress_callback(frame_count, min(max_frames, total))

    cap.release()
    out.release()

    df = tracker.get_dataframe()
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')

    return output_path, csv_path


# ============================================================================
# MODE 2: UNDERWATER (full frame, mid_spine synthesis)
# ============================================================================

def detect_underwater_waterline(video_path):
    """Detect waterline from underwater footage via edge/color gradient."""
    from scipy import signal
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    search_frames = min(200, total_frames)
    edge_candidates, color_candidates = [], []

    for frame_num in range(0, search_frames, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        search_end = int(h * 0.50)
        search_region = edges[0:search_end, :]
        horizontal_sum = np.sum(search_region, axis=1)
        if len(horizontal_sum) > 0 and np.max(horizontal_sum) > w * 0.20:
            edge_candidates.append(np.argmax(horizontal_sum) / h)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        avg_brightness = np.mean(value_channel[0:search_end, :], axis=1)
        if len(avg_brightness) > 10:
            win = min(11, len(avg_brightness) // 2 * 2 + 1)
            if win >= 5:
                smoothed = signal.savgol_filter(avg_brightness, window_length=win, polyorder=2)
                gradient = np.diff(smoothed)
                if len(gradient) > 0:
                    max_idx = np.argmax(gradient)
                    if gradient[max_idx] > 10:
                        color_candidates.append(max_idx / h)
    cap.release()

    all_c = []
    for cands in [edge_candidates, color_candidates]:
        if len(cands) >= 8:
            med, std = np.median(cands), np.std(cands)
            all_c.extend([c for c in cands if abs(c - med) < 2 * std])

    if len(all_c) >= 10:
        waterline = float(np.median(all_c))
        if waterline > 0.50:
            waterline = 0.25
    else:
        waterline = 0.25
    return waterline


def calibrate_waterline_to_shoulders(video_path, initial_waterline, pose_tracker_fn):
    """Refine waterline using shoulder level from the first 100 frames."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shoulder_ys = []

    for frame_num in range(0, min(100, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        enhanced = enhance_underwater(frame)
        keypoints, scores = pose_tracker_fn(enhanced)
        if keypoints is None or len(keypoints) == 0:
            continue
        for person_kps, person_scores in zip(keypoints, scores):
            ls_s, rs_s = person_scores[5], person_scores[6]
            if ls_s < 0.5 or rs_s < 0.5:
                continue
            shoulder_y = (person_kps[5][1] / h + person_kps[6][1] / h) / 2
            if 0.05 < shoulder_y < 0.50:
                shoulder_ys.append(shoulder_y)
    cap.release()

    if len(shoulder_ys) >= 5:
        return float(np.median(shoulder_ys))
    return initial_waterline


class RTMPoseUnderwaterTracker:
    """Underwater tracker: mid_spine synthesis, hip correction, smoothing."""

    def __init__(self, mode='balanced', det_frequency=2):
        self.pose_tracker = make_pose_tracker(mode, det_frequency)
        self.tracking_data = []
        self.position_history = []
        self.history_size = 4
        self.hip_history = {'left': [], 'right': []}
        self.hip_history_size = 6
        self.last_known = {}
        self.max_jump = 0.25
        self.frames_since_detection = 0
        self.locked_swimmer = None
        self.hip_correction_ratio = 0.20
        # FIX: these two were previously unused — process_frame picked
        # the highest-scoring detected person fresh on every single frame
        # with no locking/persistence at all, so a second person in frame
        # (or a momentary bad detection) could swap the tracked identity
        # frame-to-frame. That produced composite/split-looking skeletons
        # once _filter_jumps clamped some joints toward the old person's
        # position while others jumped straight to the new one.
        self.max_frames_lost = 15
        self._frames_since_relock_check = 0
        self.relock_check_interval = 45
        self.relock_drift_threshold = 0.15
        self._pending_relock_idx = None
        self._pending_relock_streak = 0

    def _select_best_underwater(self, keypoints, scores, h, w):
        """Fresh (unlocked) selection: highest confidence+hip-width score,
        among candidates that pass a basic anatomy sanity check."""
        best_idx, best_score = None, -1
        for i, (person_kps, person_scores) in enumerate(zip(keypoints, scores)):
            if not validate_pose_anatomy_coco(person_kps, person_scores, h, w):
                continue
            major = [5, 6, 11, 12, 13, 14, 15, 16]
            avg_conf = np.mean([person_scores[j] for j in major])
            hip_width = abs(person_kps[11][0] - person_kps[12][0]) / w
            total_score = avg_conf + hip_width * 2.0
            if total_score > best_score:
                best_score, best_idx = total_score, i
        return best_idx

    def _find_matching_underwater(self, keypoints, scores, locked_position, h, w):
        """Locked matching: closest hip to last known position, but only
        among candidates that still pass anatomy validation — mirrors the
        above-water tracker's fix for the same underlying issue."""
        best_match, best_distance = None, 0.12
        for i, (person_kps, person_scores) in enumerate(zip(keypoints, scores)):
            if person_scores[11] < 0.10 or person_scores[12] < 0.10:
                continue
            if not validate_pose_anatomy_coco(person_kps, person_scores, h, w):
                continue
            hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
            hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w
            distance = np.sqrt((hip_x - locked_position['x']) ** 2 + (hip_y - locked_position['y']) ** 2)
            if distance < best_distance:
                best_distance, best_match = distance, i
        return best_match

    def _hip_position(self, person_kps, person_scores, h, w):
        return {
            'x': (person_kps[11][0] + person_kps[12][0]) / 2 / w,
            'y': (person_kps[11][1] + person_kps[12][1]) / 2 / h,
        }

    def process_frame(self, frame, frame_num, fps, water_level):
        h, w = frame.shape[:2]
        enhanced = enhance_underwater(frame)
        keypoints, scores = self.pose_tracker(enhanced)

        frame_data = {
            'frame': frame_num, 'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4), 'tracking_locked': False,
            'frames_since_detection': self.frames_since_detection,
        }

        best_person, best_score = None, -1
        if keypoints is not None and len(keypoints) > 0:
            best_idx = None
            if self.locked_swimmer is not None:
                best_idx = self._find_matching_underwater(keypoints, scores, self.locked_swimmer, h, w)
                if best_idx is not None:
                    self.frames_since_detection = 0

                    # Periodic re-check, but require TWO consecutive
                    # disagreements before switching — a single one-off
                    # fresh-selection disagreement (e.g. a passing
                    # reflection) shouldn't be enough to hijack a good
                    # lock, only a sustained one.
                    self._frames_since_relock_check += 1
                    if self._frames_since_relock_check >= self.relock_check_interval:
                        self._frames_since_relock_check = 0
                        fresh_idx = self._select_best_underwater(keypoints, scores, h, w)
                        if fresh_idx is not None and fresh_idx != best_idx:
                            fresh_pos = self._hip_position(keypoints[fresh_idx], scores[fresh_idx], h, w)
                            cur_pos = self._hip_position(keypoints[best_idx], scores[best_idx], h, w)
                            drift = np.sqrt((fresh_pos['x'] - cur_pos['x']) ** 2 + (fresh_pos['y'] - cur_pos['y']) ** 2)
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
                person_kps, person_scores = keypoints[best_idx], scores[best_idx]
                self.locked_swimmer = self._hip_position(person_kps, person_scores, h, w)
                best_person = (person_kps, person_scores)
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_frames_lost:
                self.locked_swimmer = None

        if best_person is not None:
            person_kps, person_scores = best_person
            current_positions = {}
            for coco_idx, landmark_name in COCO_TO_LANDMARKS.items():
                conf = float(person_scores[coco_idx])
                if conf > 0.1:
                    current_positions[landmark_name] = {
                        'x': person_kps[coco_idx][0] / w, 'y': person_kps[coco_idx][1] / h,
                        'z': 0.0, 'visibility': conf,
                        'above_water': (person_kps[coco_idx][1] / h) < water_level,
                    }

            for side in ['left', 'right']:
                hip_name, knee_name = f'{side}_hip', f'{side}_knee'
                if hip_name in current_positions and knee_name in current_positions:
                    hip, knee = current_positions[hip_name], current_positions[knee_name]
                    r = self.hip_correction_ratio
                    current_positions[hip_name] = {
                        'x': hip['x'] + r * (knee['x'] - hip['x']), 'y': hip['y'] + r * (knee['y'] - hip['y']),
                        'z': hip['z'], 'visibility': min(hip['visibility'], knee['visibility']),
                        'above_water': (hip['y'] + r * (knee['y'] - hip['y'])) < water_level,
                    }

            # Mid-spine synthesis for back-curvature measurements
            if all(k in current_positions for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                ls, rs = current_positions['left_shoulder'], current_positions['right_shoulder']
                lh, rh = current_positions['left_hip'], current_positions['right_hip']
                shoulder_center_x = (ls['x'] + rs['x']) / 2
                shoulder_center_y = (ls['y'] + rs['y']) / 2
                hip_center_x = (lh['x'] + rh['x']) / 2
                hip_center_y = (lh['y'] + rh['y']) / 2
                mid_spine_x = (shoulder_center_x + hip_center_x) / 2
                mid_spine_y = (shoulder_center_y + hip_center_y) / 2
                mid_spine_vis = min(ls['visibility'], rs['visibility'], lh['visibility'], rh['visibility'])
                current_positions['mid_spine'] = {
                    'x': mid_spine_x, 'y': mid_spine_y, 'z': 0.0,
                    'visibility': mid_spine_vis, 'above_water': mid_spine_y < water_level,
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
                hip_name = f'{side}_hip'
                if hip_name in current_positions:
                    self.hip_history[side].append(current_positions[hip_name].copy())
                    if len(self.hip_history[side]) > self.hip_history_size:
                        self.hip_history[side].pop(0)

            self.position_history.append(current_positions)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            smoothed = self._smooth_all(water_level)
            for joint_name, pos in smoothed.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos['z'], 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']

            frame_data['tracking_locked'] = True
            self.frames_since_detection = 0
            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                if hip_name in current_positions:
                    self.locked_swimmer = {'x': current_positions[hip_name]['x'], 'y': current_positions[hip_name]['y']}
                    break
        elif self.position_history:
            last = self.position_history[-1]
            for joint_name, pos in last.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos['z'], 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'] * 0.5, 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']
            self.frames_since_detection += 1
        else:
            self.frames_since_detection += 1

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current_positions):
        filtered = {}
        for joint_name, pos in current_positions.items():
            if joint_name in self.last_known:
                last = self.last_known[joint_name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    filtered[joint_name] = {'x': last['x'], 'y': last['y'], 'z': pos['z'],
                                             'visibility': pos['visibility'] * 0.3, 'above_water': pos['above_water']}
                    if 'source' in pos:
                        filtered[joint_name]['source'] = pos['source']
                    continue
            filtered[joint_name] = pos
            self.last_known[joint_name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth_all(self, water_level):
        smoothed = {}

        def _wavg(history, power=1.0):
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

        for joint_name in ALL_LANDMARKS_UNDERWATER:
            is_foot = 'foot' in joint_name or 'ankle' in joint_name or 'heel' in joint_name
            hist = [f[joint_name] for f in self.position_history if joint_name in f]
            if hist:
                smoothed[joint_name] = _wavg(hist, power=1.5 if is_foot else 1.0)

        for side in ['left', 'right']:
            if self.hip_history[side]:
                smoothed[f'{side}_hip'] = _wavg(self.hip_history[side], power=1.5)
        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


def draw_frame_underwater(frame, best_person, water_level, frame_num, total_frames):
    """Visualization for underwater mode, including the synthesized mid_spine."""
    viz = frame.copy()
    h, w = frame.shape[:2]
    water_y = int(water_level * h)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 0, 0), 4)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 255, 255), 2)
    cv2.putText(viz, f"WATERLINE: {water_level:.3f}", (10, water_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.rectangle(viz, (10, 10), (550, 100), (0, 0, 0), -1)

    if best_person is not None:
        person_kps, person_scores = best_person
        for (i, j) in COCO_CONNECTIONS:
            if person_scores[i] > 0.2 and person_scores[j] > 0.2:
                pt1 = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))
                thick = 4 if i >= 13 or j >= 13 else 3
                cv2.line(viz, pt1, pt2, (255, 255, 255), thick, cv2.LINE_AA)

        if all(person_scores[i] > 0.2 for i in [5, 6, 11, 12]):
            sc = ((person_kps[5][0] + person_kps[6][0]) / 2, (person_kps[5][1] + person_kps[6][1]) / 2)
            hc = ((person_kps[11][0] + person_kps[12][0]) / 2, (person_kps[11][1] + person_kps[12][1]) / 2)
            mid = (int((sc[0] + hc[0]) / 2), int((sc[1] + hc[1]) / 2))
            cv2.line(viz, (int(sc[0]), int(sc[1])), mid, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.line(viz, mid, (int(hc[0]), int(hc[1])), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(viz, mid, 9, (255, 200, 0), -1)
            cv2.circle(viz, mid, 9, (255, 255, 255), 2)

        above = total_joints = 0
        for idx in range(17):
            if person_scores[idx] > 0.2:
                x, y = int(person_kps[idx][0]), int(person_kps[idx][1])
                is_above = y < water_y
                color = (0, 255, 0) if is_above else (0, 0, 255)
                radius = 12 if idx >= 15 else 8
                cv2.circle(viz, (x, y), radius, color, -1)
                cv2.circle(viz, (x, y), radius, (255, 255, 255), 2)
                above += int(is_above)
                total_joints += 1
        pct = (above / total_joints * 100) if total_joints else 0
        cv2.putText(viz, f"Above: {above}/{total_joints} ({pct:.0f}%)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz, f"Below: {total_joints-above}/{total_joints} ({100-pct:.0f}%)", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(viz, "No swimmer detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ft = f"Frame: {frame_num}/{total_frames}"
    ts = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(viz, ft, (550 - ts[0] - 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return viz


def process_video_underwater(video_path, output_path, water_level=None,
                              mode='balanced', det_frequency=2, max_duration=60,
                              progress_callback=None):
    """Process a standalone underwater video. Returns (video_path, csv_path)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = RTMPoseUnderwaterTracker(mode=mode, det_frequency=det_frequency)

    if water_level is None:
        water_level = detect_underwater_waterline(str(video_path))
        water_level = calibrate_waterline_to_shoulders(str(video_path), water_level, tracker.pose_tracker)

    max_frames = int(max_duration * fps) if max_duration else total

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        os.remove(output_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        best_person = tracker.process_frame(frame, frame_count, fps, water_level)
        annotated = draw_frame_underwater(frame, best_person, water_level, frame_count, total)
        out.write(annotated)
        frame_count += 1
        if progress_callback is not None:
            progress_callback(frame_count, min(max_frames, total))

    cap.release()
    out.release()

    df = tracker.get_dataframe()
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')

    return output_path, csv_path


# ============================================================================
# MODE 3: WALTICAM (split-screen — top = above, bottom = below)
# ============================================================================

def detect_waterline_below_split(frame_bottom):
    """Waterline within the bottom (underwater) half of a split frame."""
    h, w = frame_bottom.shape[:2]
    gray = cv2.cvtColor(frame_bottom, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    search_end = int(h * 0.30)
    search_region = edges[0:search_end, :]
    horizontal_sum = np.sum(search_region, axis=1)
    if len(horizontal_sum) > 0 and np.max(horizontal_sum) > w * 0.20:
        return np.argmax(horizontal_sum) / h
    return 0.05


class SplitAboveWaterTracker:
    """Lightweight RTMPose tracker for the top (above-water) half of a
    WaltiCam split-screen frame. No tent masking needed (already cropped)."""

    def __init__(self, mode='balanced', det_frequency=2):
        self.pose_tracker = make_pose_tracker(mode, det_frequency)
        self.tracking_data = []
        self.position_history = []
        self.history_size = 4
        self.last_known = {}
        self.max_jump = 0.25
        self.frames_since_detection = 0
        self.hip_correction_ratio = 0.20

    def process_frame(self, frame_top, frame_num, fps, water_level):
        h, w = frame_top.shape[:2]
        enhanced = quick_enhance(frame_top)
        keypoints, scores = self.pose_tracker(enhanced)

        frame_data = {'frame': frame_num, 'time_seconds': round(frame_num / fps, 4),
                       'water_level': round(water_level, 4), 'tracking_locked': False}

        best_person = self._select_best(keypoints, scores, w)
        if best_person is not None:
            person_kps, person_scores = best_person
            current = {}
            for coco_idx, name in COCO_TO_LANDMARKS.items():
                conf = float(person_scores[coco_idx])
                if conf > 0.1:
                    x_px, y_px = person_kps[coco_idx][0], person_kps[coco_idx][1]
                    current[name] = {'x': x_px / w, 'y': y_px / h, 'z': 0.0,
                                      'visibility': conf, 'above_water': (y_px / h) < water_level}

            for side in ['left', 'right']:
                hip, knee = f'{side}_hip', f'{side}_knee'
                if hip in current and knee in current:
                    r = self.hip_correction_ratio
                    current[hip] = {
                        'x': current[hip]['x'] + r * (current[knee]['x'] - current[hip]['x']),
                        'y': current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y']), 'z': 0.0,
                        'visibility': min(current[hip]['visibility'], current[knee]['visibility']),
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
            self.frames_since_detection = 0
        else:
            if self.position_history:
                last = self.position_history[-1]
                for name, pos in last.items():
                    frame_data[f'{name}_x'] = round(pos['x'], 4)
                    frame_data[f'{name}_y'] = round(pos['y'], 4)
                    frame_data[f'{name}_z'] = round(pos['z'], 4)
                    frame_data[f'{name}_visibility'] = round(pos['visibility'] * 0.5, 4)
                    frame_data[f'{name}_above_water'] = pos['above_water']
            self.frames_since_detection += 1

        self.tracking_data.append(frame_data)
        return best_person

    def _select_best(self, keypoints, scores, w):
        if keypoints is None or len(keypoints) == 0:
            return None
        best, best_score = None, -1
        for kps, sc in zip(keypoints, scores):
            major = [5, 6, 11, 12, 13, 14, 15, 16]
            avg_conf = np.mean([sc[j] for j in major])
            hip_width = abs(kps[11][0] - kps[12][0]) / w
            total = avg_conf + hip_width * 2.0
            if total > best_score:
                best_score, best = total, (kps, sc)
        return best

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
        for name in ALL_LANDMARKS_ABOVE[:9]:  # base COCO joints only (no synthesized feet on top half)
            hist = [f[name] for f in self.position_history if name in f]
            if hist:
                weights = [(i + 1) ** 1.0 for i in range(len(hist))]
                tw = sum(weights)
                smoothed[name] = {
                    'x': sum(p['x'] * w for p, w in zip(hist, weights)) / tw,
                    'y': sum(p['y'] * w for p, w in zip(hist, weights)) / tw,
                    'z': 0.0,
                    'visibility': sum(p['visibility'] * w for p, w in zip(hist, weights)) / tw,
                }
                smoothed[name]['above_water'] = smoothed[name]['y'] < water_level
        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


class SplitUnderwaterTracker:
    """Lightweight RTMPose tracker for the bottom (underwater) half of a
    WaltiCam split-screen frame."""

    def __init__(self, mode='balanced', det_frequency=2):
        self.pose_tracker = make_pose_tracker(mode, det_frequency)
        self.tracking_data = []
        self.position_history = []
        self.history_size = 4
        self.hip_history = {'left': [], 'right': []}
        self.hip_history_size = 6
        self.last_known = {}
        self.max_jump = 0.25
        self.frames_since_detection = 0
        self.hip_correction_ratio = 0.20

    def process_frame(self, frame_bottom, frame_num, fps, water_level):
        h, w = frame_bottom.shape[:2]
        enhanced = enhance_underwater(frame_bottom)
        keypoints, scores = self.pose_tracker(enhanced)

        frame_data = {'frame': frame_num, 'time_seconds': round(frame_num / fps, 4),
                       'water_level': round(water_level, 4), 'tracking_locked': False,
                       'frames_since_detection': self.frames_since_detection}

        best_person = self._select_best(keypoints, scores, w)
        if best_person is not None:
            person_kps, person_scores = best_person
            current = {}
            for coco_idx, name in COCO_TO_LANDMARKS.items():
                conf = float(person_scores[coco_idx])
                if conf > 0.1:
                    x_px, y_px = person_kps[coco_idx][0], person_kps[coco_idx][1]
                    current[name] = {'x': x_px / w, 'y': y_px / h, 'z': 0.0,
                                      'visibility': conf, 'above_water': (y_px / h) < water_level}

            for side in ['left', 'right']:
                hip, knee = f'{side}_hip', f'{side}_knee'
                if hip in current and knee in current:
                    r = self.hip_correction_ratio
                    current[hip] = {
                        'x': current[hip]['x'] + r * (current[knee]['x'] - current[hip]['x']),
                        'y': current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y']), 'z': 0.0,
                        'visibility': min(current[hip]['visibility'], current[knee]['visibility']),
                        'above_water': (current[hip]['y'] + r * (current[knee]['y'] - current[hip]['y'])) < water_level,
                    }

            for side in ['left', 'right']:
                ankle = f'{side}_ankle'
                if ankle in current:
                    a = current[ankle]
                    current[f'{side}_heel'] = {'x': a['x'], 'y': a['y'] - 0.005, 'z': 0.0,
                                                'visibility': a['visibility'] * 0.8, 'above_water': (a['y'] - 0.005) < water_level}
                    current[f'{side}_foot_index'] = {'x': a['x'], 'y': a['y'] + 0.005, 'z': 0.0,
                                                      'visibility': a['visibility'] * 0.8, 'above_water': (a['y'] + 0.005) < water_level}
                    current[f'{side}_foot_best'] = {'x': a['x'], 'y': a['y'], 'z': 0.0,
                                                     'visibility': a['visibility'], 'above_water': a['y'] < water_level}

            current = self._filter_jumps(current)

            for side in ['left', 'right']:
                hip = f'{side}_hip'
                if hip in current:
                    self.hip_history[side].append(current[hip].copy())
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
            self.frames_since_detection = 0
        else:
            if self.position_history:
                last = self.position_history[-1]
                for name, pos in last.items():
                    frame_data[f'{name}_x'] = round(pos['x'], 4)
                    frame_data[f'{name}_y'] = round(pos['y'], 4)
                    frame_data[f'{name}_z'] = round(pos['z'], 4)
                    frame_data[f'{name}_visibility'] = round(pos['visibility'] * 0.5, 4)
                    frame_data[f'{name}_above_water'] = pos['above_water']
            self.frames_since_detection += 1

        self.tracking_data.append(frame_data)
        return best_person

    def _select_best(self, keypoints, scores, w):
        if keypoints is None or len(keypoints) == 0:
            return None
        best, best_score = None, -1
        for kps, sc in zip(keypoints, scores):
            major = [5, 6, 11, 12, 13, 14, 15, 16]
            avg_conf = np.mean([sc[j] for j in major])
            hip_width = abs(kps[11][0] - kps[12][0]) / w
            total = avg_conf + hip_width * 2.0
            if total > best_score:
                best_score, best = total, (kps, sc)
        return best

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
            r = {'x': sum(p['x'] * w for p, w in zip(hist, weights)) / tw,
                 'y': sum(p['y'] * w for p, w in zip(hist, weights)) / tw, 'z': 0.0,
                 'visibility': sum(p['visibility'] * w for p, w in zip(hist, weights)) / tw}
            r['above_water'] = r['y'] < water_level
            return r

        for name in ALL_LANDMARKS_ABOVE:
            is_foot = 'foot' in name or 'ankle' in name or 'heel' in name
            hist = [f[name] for f in self.position_history if name in f]
            if hist:
                smoothed[name] = _wavg(hist, power=1.5 if is_foot else 1.0)
        for side in ['left', 'right']:
            if self.hip_history[side]:
                smoothed[f'{side}_hip'] = _wavg(self.hip_history[side], power=1.5)
        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


def draw_split_skeleton(viz, person_kps, person_scores, offset_y=0, water_y=None):
    if person_kps is None:
        return
    for (i, j) in COCO_CONNECTIONS:
        if person_scores[i] > 0.2 and person_scores[j] > 0.2:
            pt1 = (int(person_kps[i][0]), int(person_kps[i][1]) + offset_y)
            pt2 = (int(person_kps[j][0]), int(person_kps[j][1]) + offset_y)
            thick = 4 if i >= 13 or j >= 13 else 3
            cv2.line(viz, pt1, pt2, (255, 255, 255), thick, cv2.LINE_AA)
    for idx in range(17):
        if person_scores[idx] > 0.2:
            x = int(person_kps[idx][0])
            y = int(person_kps[idx][1]) + offset_y
            is_above = (y < water_y) if water_y is not None else True
            color = (0, 255, 0) if is_above else (0, 0, 255)
            radius = 10 if idx >= 15 else 7
            cv2.circle(viz, (x, y), radius, color, -1)
            cv2.circle(viz, (x, y), radius, (255, 255, 255), 2)


def process_video_walticam(video_path, output_path, mode='balanced', det_frequency=2,
                            max_duration=60, split_ratio=0.5, progress_callback=None):
    """
    Process a WaltiCam split-screen video (top=above, bottom=below).
    Returns (video_path, above_csv_path, below_csv_path).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    name = output_path.stem

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    split_y = int(h * split_ratio)
    max_frames = min(int(max_duration * fps), total) if max_duration else total

    above_tracker = SplitAboveWaterTracker(mode=mode, det_frequency=det_frequency)
    below_tracker = SplitUnderwaterTracker(mode=mode, det_frequency=det_frequency)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        frame_bottom = first_frame[split_y:, :]
        wl_above = 0.50
        wl_below = detect_waterline_below_split(frame_bottom)
    else:
        wl_above, wl_below = 0.50, 0.05

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        os.remove(output_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_top = frame[:split_y, :]
        frame_bottom = frame[split_y:, :]

        above_result = above_tracker.process_frame(frame_top, frame_count, fps, wl_above)
        below_result = below_tracker.process_frame(frame_bottom, frame_count, fps, wl_below)

        viz = frame.copy()
        cv2.line(viz, (0, split_y), (w, split_y), (0, 255, 255), 2)
        cv2.putText(viz, "ABOVE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(viz, "BELOW", (10, split_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        above_wl_y = int(wl_above * split_y)
        cv2.line(viz, (0, above_wl_y), (w, above_wl_y), (255, 200, 0), 1)
        below_wl_y = split_y + int(wl_below * (h - split_y))
        cv2.line(viz, (0, below_wl_y), (w, below_wl_y), (255, 200, 0), 1)

        if above_result is not None:
            draw_split_skeleton(viz, above_result[0], above_result[1], offset_y=0, water_y=above_wl_y)
        if below_result is not None:
            draw_split_skeleton(viz, below_result[0], below_result[1], offset_y=split_y, water_y=below_wl_y)

        out.write(viz)
        frame_count += 1
        if progress_callback is not None:
            progress_callback(frame_count, max_frames)

    cap.release()
    out.release()

    above_csv = output_path.parent / f"{name}_above_data.csv"
    below_csv = output_path.parent / f"{name}_below_data.csv"
    above_tracker.get_dataframe().to_csv(above_csv, index=False, float_format='%.4f')
    below_tracker.get_dataframe().to_csv(below_csv, index=False, float_format='%.4f')

    return output_path, above_csv, below_csv
