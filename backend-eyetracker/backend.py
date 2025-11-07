import asyncio
import base64
import json
import math
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.interpolate import RBFInterpolator                                       

# --- Configuration ---
PUPIL_THRESH_DEFAULT = 50
PUPIL_BLUR_DEFAULT = 3
GLINT_THRESH_DEFAULT = 240
GLINT_BLUR_DEFAULT = 9
RAY_HISTORY_DEFAULT = 100
SMOOTHING_FACTOR_DEFAULT = 0.12
DATA_DIR = os.path.join(os.path.expanduser("~"), ".eye_tracker")
os.makedirs(DATA_DIR, exist_ok=True)
CALIBRATION_FILE = os.path.join(DATA_DIR, "spherical_calibration_data.json")
SAMPLE_DURATION = 2.5


# --- Runtime tracking ---
PREVIEW_CLIENTS: Dict[int, WebSocket] = {}
PREVIEW_CLIENTS_LOCK: Optional[asyncio.Lock] = None


def get_preview_clients_lock() -> asyncio.Lock:
    global PREVIEW_CLIENTS_LOCK
    if PREVIEW_CLIENTS_LOCK is None:
        PREVIEW_CLIENTS_LOCK = asyncio.Lock()
    return PREVIEW_CLIENTS_LOCK


# --- Helper Functions ---
def apply_binary_threshold(image: np.ndarray, thresh: int) -> np.ndarray:
    _, thresholded_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image


def filter_contours_by_area_and_return_largest(contours, pixel_thresh: int, ratio_thresh: float):
    max_area = 0
    largest = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < pixel_thresh:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w / h, h / w)
        if aspect <= ratio_thresh and area > max_area:
            max_area = area
            largest = cnt
    return largest


def find_glint(gray: np.ndarray, glint_thresh_val: int, blur_val: int) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    _, glint_th = cv2.threshold(blurred, glint_thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(glint_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glint_center: Optional[Tuple[int, int]] = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            glint_center = (int(x), int(y))
            break
    return glint_center, glint_th


def smooth_gaze_vector(new_vector: np.ndarray, history: deque, alpha: float) -> np.ndarray:
    if len(history) == 0:
        return new_vector
    smoothed = alpha * new_vector + (1 - alpha) * history[-1]
    return smoothed


def compute_spherical_gaze_vector(
    pupil_x: int,
    pupil_y: int,
    center_x: int,
    center_y: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    dx = pupil_x - center_x
    dy = pupil_y - center_y

    norm_dx = dx / (frame_width / 2)
    norm_dy = dy / (frame_height / 2)

    distance = math.sqrt(norm_dx**2 + norm_dy**2)

    if distance > 0:
        sphere_factor = math.sin(distance * math.pi / 3)
        corrected_dx = norm_dx * sphere_factor / distance if distance > 0 else 0
        corrected_dy = norm_dy * sphere_factor / distance if distance > 0 else 0
        z_component = math.cos(distance * math.pi / 3) - 1
    else:
        corrected_dx = 0
        corrected_dy = 0
        z_component = 0

    return np.array([corrected_dx, corrected_dy, z_component])


def create_enhanced_calibration_points(screen_w: int, screen_h: int) -> List[Tuple[int, int]]:
    margin_x = int(screen_w * 0.08)
    margin_y = int(screen_h * 0.08)

    points = []
    points.append((screen_w // 2, screen_h // 2))
    points.append((margin_x, margin_y))
    points.append((screen_w - margin_x, margin_y))
    points.append((margin_x, screen_h - margin_y))
    points.append((screen_w - margin_x, screen_h - margin_y))
    points.append((screen_w // 2, margin_y))
    points.append((screen_w // 2, screen_h - margin_y))
    points.append((margin_x, screen_h // 2))
    points.append((screen_w - margin_x, screen_h // 2))
    points.append((int(screen_w * 0.25), int(screen_h * 0.25)))
    points.append((int(screen_w * 0.75), int(screen_h * 0.25)))
    points.append((int(screen_w * 0.25), int(screen_h * 0.75)))
    points.append((int(screen_w * 0.75), int(screen_h * 0.75)))
    return points


class EyeTracker:
    def __init__(self, calibration_file: str = CALIBRATION_FILE):
        self.cap: Optional[cv2.VideoCapture] = None
        self.source: Optional[str] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.PUPIL_THRESH = PUPIL_THRESH_DEFAULT
        self.PUPIL_BLUR = PUPIL_BLUR_DEFAULT
        self.GLINT_THRESH = GLINT_THRESH_DEFAULT
        self.GLINT_BLUR = GLINT_BLUR_DEFAULT
        self.RAY_HISTORY = RAY_HISTORY_DEFAULT
        self.max_observed_distance = 150
        self.SMOOTHING_FACTOR = SMOOTHING_FACTOR_DEFAULT

        self.gaze_vector_lock = threading.Lock()
        self.gaze_vector = np.zeros(3)
        self.raw_gaze_history: deque = deque(maxlen=15)
        self.calibrated_gaze_history: deque = deque(maxlen=8)

        self.latest_data: Optional[Dict] = None
        self.latest_timestamp = 0.0
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_pupil_mask: Optional[np.ndarray] = None
        self.latest_glint_mask: Optional[np.ndarray] = None

        self.crop_roi: Optional[Tuple[int, int, int, int]] = None
        self.fixed_center: Optional[Tuple[int, int]] = None

        self.rbf_interpolator_x: Optional[RBFInterpolator] = None
        self.rbf_interpolator_y: Optional[RBFInterpolator] = None
        self.calibration_points_3d: Optional[np.ndarray] = None
        self.calibration_points_screen: Optional[np.ndarray] = None

        self.calibration_samples: List[List[np.ndarray]] = []
        self.calibration_screen_targets: List[Tuple[int, int]] = []
        self.calibration_active = False
        self.calibration_file = calibration_file
        calibration_dir = os.path.dirname(os.path.abspath(self.calibration_file))
        os.makedirs(calibration_dir, exist_ok=True)

        self.load_calibration()

    # --- Lifecycle ---
    def start(self, source: str = "0"):
        if self.running:
            return

        try:
            source_index = int(source)
            cap = cv2.VideoCapture(source_index)
        except ValueError:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self.cap = cap
        self.source = source
        self.raw_gaze_history.clear()
        self.calibrated_gaze_history.clear()
        with self.lock:
            self.latest_data = None
            self.fixed_center = None
            self.latest_raw_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.source = None
        with self.lock:
            self.latest_raw_frame = None
            self.latest_pupil_mask = None
            self.latest_glint_mask = None

    def is_running(self) -> bool:
        return self.running

    # --- Runtime controls ---
    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]):
        with self.lock:
            self.crop_roi = roi

    def set_parameters(
        self,
        pupil_thresh: Optional[int] = None,
        pupil_blur: Optional[int] = None,
        glint_thresh: Optional[int] = None,
        glint_blur: Optional[int] = None,
        ray_history: Optional[int] = None,
        smoothing_factor: Optional[float] = None,
        sphere_radius: Optional[int] = None,
    ):
        with self.lock:
            if pupil_thresh is not None:
                self.PUPIL_THRESH = max(1, pupil_thresh)
            if pupil_blur is not None:
                blur = max(1, pupil_blur)
                if blur % 2 == 0:
                    blur += 1
                self.PUPIL_BLUR = blur
            if glint_thresh is not None:
                self.GLINT_THRESH = max(1, glint_thresh)
            if glint_blur is not None:
                blur = max(1, glint_blur)
                if blur % 2 == 0:
                    blur += 1
                self.GLINT_BLUR = blur
            if ray_history is not None:
                self.RAY_HISTORY = max(10, ray_history)
            if smoothing_factor is not None:
                self.SMOOTHING_FACTOR = max(0.01, min(0.5, smoothing_factor))
            if sphere_radius is not None:
                self.max_observed_distance = max(50, sphere_radius)

    def get_parameters(self) -> Dict[str, float]:
        with self.lock:
            return {
                "pupil_thresh": self.PUPIL_THRESH,
                "pupil_blur": self.PUPIL_BLUR,
                "glint_thresh": self.GLINT_THRESH,
                "glint_blur": self.GLINT_BLUR,
                "ray_history": self.RAY_HISTORY,
                "smoothing_factor": self.SMOOTHING_FACTOR,
                "sphere_radius": self.max_observed_distance,
            }

    # --- Calibration persistence ---
    def load_calibration(self) -> bool:
        if not os.path.exists(self.calibration_file):
            return False
        try:
            with open(self.calibration_file, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
            self.calibration_points_3d = np.array(cal_data["calibration_points_3d"])
            self.calibration_points_screen = np.array(cal_data["calibration_points_screen"])
            self.rbf_interpolator_x = RBFInterpolator(
                self.calibration_points_3d,
                self.calibration_points_screen[:, 0],
                kernel="thin_plate_spline",
                smoothing=0.001,
            )
            self.rbf_interpolator_y = RBFInterpolator(
                self.calibration_points_3d,
                self.calibration_points_screen[:, 1],
                kernel="thin_plate_spline",
                smoothing=0.001,
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load calibration: {exc}")
            self.rbf_interpolator_x = None
            self.rbf_interpolator_y = None
            return False

    def save_calibration(self, points_3d: np.ndarray, points_screen: np.ndarray, screen_w: int, screen_h: int):
        cal_data = {
            "calibration_points_3d": points_3d.tolist(),
            "calibration_points_screen": points_screen.tolist(),
            "screen_width": screen_w,
            "screen_height": screen_h,
            "timestamp": time.time(),
            "method": "spherical_rbf",
        }
        with open(self.calibration_file, "w", encoding="utf-8") as f:
            json.dump(cal_data, f, indent=2)

    # --- Calibration workflow ---
    def start_calibration(self, screen_w: int, screen_h: int):
        self.calibration_samples = []
        self.calibration_screen_targets = []
        self.calibration_active = True
        return create_enhanced_calibration_points(screen_w, screen_h)

    def capture_calibration_point(self, point_index: int, screen_x: int, screen_y: int) -> Dict[str, float]:
        if not self.running:
            raise RuntimeError("Tracker must be running before calibration")
        if not self.calibration_active:
            raise RuntimeError("Calibration has not been started")

        samples: List[np.ndarray] = []
        deadline = time.time() + SAMPLE_DURATION
        while time.time() < deadline:
            gaze = self.get_current_gaze_vector()
            if np.linalg.norm(gaze) > 0.005:
                samples.append(gaze.copy())
            time.sleep(0.02)

        if len(samples) < 10:
            raise RuntimeError("Insufficient data collected for calibration point")

        if point_index >= len(self.calibration_samples):
            self.calibration_samples.append(samples)
            self.calibration_screen_targets.append((screen_x, screen_y))
        else:
            self.calibration_samples[point_index] = samples
            self.calibration_screen_targets[point_index] = (screen_x, screen_y)

        return {
            "point_index": point_index,
            "samples_collected": len(samples),
            "screen_target": {
                "x": screen_x,
                "y": screen_y,
            },
        }

    def finalize_calibration(self, screen_w: int, screen_h: int) -> Dict[str, float]:
        if not self.calibration_active:
            raise RuntimeError("Calibration has not been started")
        if len(self.calibration_samples) < 8:
            raise RuntimeError("At least 8 calibration points are required")

        valid_gaze_vectors = []
        valid_screen_points = []
        for samples, screen_point in zip(self.calibration_samples, self.calibration_screen_targets):
            if samples and len(samples) >= 10:
                samples_array = np.array(samples)
                median_gaze_vector = np.median(samples_array, axis=0)
                valid_gaze_vectors.append(median_gaze_vector)
                valid_screen_points.append(screen_point)

        if len(valid_gaze_vectors) < 8:
            raise RuntimeError("Not enough valid calibration samples collected")

        self.calibration_points_3d = np.array(valid_gaze_vectors)
        self.calibration_points_screen = np.array(valid_screen_points)

        self.rbf_interpolator_x = RBFInterpolator(
            self.calibration_points_3d,
            self.calibration_points_screen[:, 0],
            kernel="thin_plate_spline",
            smoothing=0.001,
        )
        self.rbf_interpolator_y = RBFInterpolator(
            self.calibration_points_3d,
            self.calibration_points_screen[:, 1],
            kernel="thin_plate_spline",
            smoothing=0.001,
        )

        self.save_calibration(self.calibration_points_3d, self.calibration_points_screen, screen_w, screen_h)

        self.calibration_active = False

        return {
            "points_used": len(valid_gaze_vectors),
            "screen_width": screen_w,
            "screen_height": screen_h,
        }

    # --- Data access ---
    def get_current_gaze_vector(self) -> np.ndarray:
        with self.gaze_vector_lock:
            return self.gaze_vector.copy()

    def get_calibrated_screen_position(self) -> Optional[Tuple[int, int]]:
        if self.rbf_interpolator_x is None or self.rbf_interpolator_y is None:
            return None

        gaze = self.get_current_gaze_vector()
        if np.linalg.norm(gaze) < 0.005:
            return None

        try:
            px = self.rbf_interpolator_x(gaze.reshape(1, -1))[0]
            py = self.rbf_interpolator_y(gaze.reshape(1, -1))[0]

            self.calibrated_gaze_history.append((px, py))
            if len(self.calibrated_gaze_history) > 1:
                weights = np.exp(-np.arange(len(self.calibrated_gaze_history))[::-1] * 0.3)
                weights /= weights.sum()
                smooth_x = np.average([p[0] for p in self.calibrated_gaze_history], weights=weights)
                smooth_y = np.average([p[1] for p in self.calibrated_gaze_history], weights=weights)
                return int(smooth_x), int(smooth_y)
            return int(px), int(py)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Calibration error: {exc}")
            return None

    def get_latest_data(self) -> Optional[Dict]:
        with self.lock:
            if self.latest_data is None:
                return None
            return json.loads(json.dumps(self.latest_data))

    def get_preview_frame(self) -> Optional[Dict[str, object]]:
        with self.lock:
            if self.latest_raw_frame is None:
                return None
            frame = self.latest_raw_frame.copy()
            roi = self.crop_roi
            data_snapshot = json.loads(json.dumps(self.latest_data)) if self.latest_data is not None else None
            pupil_mask = self.latest_pupil_mask.copy() if self.latest_pupil_mask is not None else None
            glint_mask = self.latest_glint_mask.copy() if self.latest_glint_mask is not None else None
        if frame.size == 0:
            return None
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pupil_mask_payload = None
        glint_mask_payload = None

        if data_snapshot is not None:
            pupil_center = data_snapshot.get("pupil_center")
            glint_center = data_snapshot.get("glint_center")

            offset_x = roi[0] if roi else 0
            offset_y = roi[1] if roi else 0

            if pupil_center:
                px = int(pupil_center[0] + offset_x)
                py = int(pupil_center[1] + offset_y)
                if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                    cv2.circle(frame, (px, py), 10, (0, 200, 255), 2)
                    cv2.circle(frame, (px, py), 3, (0, 200, 255), -1)

            if glint_center:
                gx = int(glint_center[0] + offset_x)
                gy = int(glint_center[1] + offset_y)
                if 0 <= gx < frame.shape[1] and 0 <= gy < frame.shape[0]:
                    cv2.circle(frame, (gx, gy), 6, (255, 255, 255), 2)
                    cv2.circle(frame, (gx, gy), 1, (255, 255, 255), -1)

        overlay_margin = 14
        if pupil_mask is not None and pupil_mask.size > 0:
            overlay_w = max(80, frame.shape[1] // 4)
            overlay_h = max(60, frame.shape[0] // 4)
            pupil_resized = cv2.resize(pupil_mask, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            pupil_color = cv2.applyColorMap(pupil_resized, cv2.COLORMAP_INFERNO)
            x0 = overlay_margin
            y0 = frame.shape[0] - overlay_h - overlay_margin
            region = frame[y0 : y0 + overlay_h, x0 : x0 + overlay_w]
            if region.shape[0] == overlay_h and region.shape[1] == overlay_w:
                cv2.addWeighted(pupil_color, 0.65, region, 0.35, 0, region)
                cv2.rectangle(frame, (x0, y0), (x0 + overlay_w, y0 + overlay_h), (255, 215, 0), 1)
                cv2.putText(
                    frame,
                    "Pupil mask",
                    (x0 + 6, y0 + overlay_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 225, 180),
                    1,
                    cv2.LINE_AA,
                )

            success, buffer = cv2.imencode(".png", pupil_mask)
            if success:
                pupil_mask_payload = {
                    "data": base64.b64encode(buffer).decode("ascii"),
                    "width": int(pupil_mask.shape[1]),
                    "height": int(pupil_mask.shape[0]),
                }

        if glint_mask is not None and glint_mask.size > 0:
            overlay_w = max(60, frame.shape[1] // 5)
            overlay_h = max(45, frame.shape[0] // 5)
            glint_resized = cv2.resize(glint_mask, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            glint_color = cv2.applyColorMap(glint_resized, cv2.COLORMAP_OCEAN)
            x1 = frame.shape[1] - overlay_w - overlay_margin
            y1 = overlay_margin
            region = frame[y1 : y1 + overlay_h, x1 : x1 + overlay_w]
            if region.shape[0] == overlay_h and region.shape[1] == overlay_w:
                cv2.addWeighted(glint_color, 0.6, region, 0.4, 0, region)
                cv2.rectangle(frame, (x1, y1), (x1 + overlay_w, y1 + overlay_h), (180, 220, 255), 1)
                cv2.putText(
                    frame,
                    "Glint mask",
                    (x1 + 6, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 230, 255),
                    1,
                    cv2.LINE_AA,
                )

            success, buffer = cv2.imencode(".png", glint_mask)
            if success:
                glint_mask_payload = {
                    "data": base64.b64encode(buffer).decode("ascii"),
                    "width": int(glint_mask.shape[1]),
                    "height": int(glint_mask.shape[0]),
                }
        return {
            "frame": frame,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "roi": roi,
            "pupil_mask": pupil_mask_payload,
            "glint_mask": glint_mask_payload,
        }

    # --- Internal processing ---
    def _loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            raw_frame = frame.copy()

            roi = self.crop_roi
            if roi is not None:
                x, y, w, h = roi
                frame_h, frame_w = frame.shape[:2]
                if 0 <= x < frame_w and 0 <= y < frame_h:
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)
                    if w > 0 and h > 0:
                        frame = frame[y : y + h, x : x + w]

            processed, pupil_mask, glint_mask = self._process_frame(frame)

            with self.lock:
                self.latest_data = processed
                self.latest_timestamp = processed["timestamp"]
                self.latest_raw_frame = raw_frame
                self.latest_pupil_mask = pupil_mask
                self.latest_glint_mask = glint_mask

        self.running = False

    def _process_frame(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        with self.lock:
            pupil_blur = self.PUPIL_BLUR
            pupil_thresh = self.PUPIL_THRESH
            glint_thresh = self.GLINT_THRESH
            glint_blur = self.GLINT_BLUR
            smoothing_factor = self.SMOOTHING_FACTOR

        if self.fixed_center is None or self.fixed_center[0] >= width or self.fixed_center[1] >= height:
            self.fixed_center = (width // 2, height // 2)

        blurred_pupil = cv2.GaussianBlur(gray, (pupil_blur, pupil_blur), 0)
        thresholded_pupil = apply_binary_threshold(blurred_pupil, pupil_thresh)
        kernel = np.ones((7, 7), np.uint8)
        closed_pupil = cv2.morphologyEx(thresholded_pupil, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pupil_center = None
        normalized_pupil = None
        distance = None
        angle_deg = None
        gaze_vector = None

        largest = filter_contours_by_area_and_return_largest(contours, 100, 3)
        if largest is not None and len(largest) >= 5:
            (x, y), _ = cv2.minEnclosingCircle(largest)
            pupil_center = (int(x), int(y))
            normalized_pupil = {
                "x": float(x) / max(1, width),
                "y": float(y) / max(1, height),
            }

            current_gaze_direction = compute_spherical_gaze_vector(
                pupil_center[0],
                pupil_center[1],
                self.fixed_center[0],
                self.fixed_center[1],
                width,
                height,
            )

            smoothed_gaze = smooth_gaze_vector(current_gaze_direction, self.raw_gaze_history, smoothing_factor)
            self.raw_gaze_history.append(smoothed_gaze)

            with self.gaze_vector_lock:
                self.gaze_vector = smoothed_gaze

            gaze_vector = smoothed_gaze.tolist()

            dx = pupil_center[0] - self.fixed_center[0]
            dy = pupil_center[1] - self.fixed_center[1]
            distance = float(math.sqrt(dx**2 + dy**2))
            angle_deg = math.degrees(math.atan2(dy, dx))
        else:
            gaze_vector = self.get_current_gaze_vector().tolist()

        glint_center, glint_mask = find_glint(gray, glint_thresh, glint_blur)

        calibrated_position = self.get_calibrated_screen_position()

        return (
            {
                "timestamp": time.time(),
                "pupil_center": pupil_center,
                "glint_center": glint_center,
                "gaze_vector": gaze_vector,
                "distance_px": distance,
                "angle_deg": angle_deg,
                "calibrated_position": {
                    "x": calibrated_position[0],
                    "y": calibrated_position[1],
                }
                if calibrated_position
                else None,
                "frame_size": {"width": width, "height": height},
                "roi": self.crop_roi,
                "normalized_pupil": normalized_pupil,
                "parameters": self.get_parameters(),
                "calibration_loaded": self.rbf_interpolator_x is not None,
            },
            closed_pupil.copy(),
            glint_mask.copy(),
        )


tracker = EyeTracker()


# --- API Schemas ---
class StartRequest(BaseModel):
    source: str = Field("0", description="Video source index or path")


class ROIRequest(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ParametersRequest(BaseModel):
    pupil_thresh: Optional[int] = None
    pupil_blur: Optional[int] = None
    glint_thresh: Optional[int] = None
    glint_blur: Optional[int] = None
    ray_history: Optional[int] = None
    smoothing_factor: Optional[float] = None
    sphere_radius: Optional[int] = None


class CalibrationStartRequest(BaseModel):
    screen_width: int
    screen_height: int


class CalibrationCaptureRequest(BaseModel):
    point_index: int
    screen_x: int
    screen_y: int


class CalibrationFinalizeRequest(BaseModel):
    screen_width: int
    screen_height: int


app = FastAPI(title="Eye Tracker Backend", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status():
    return {
        "running": tracker.is_running(),
        "source": tracker.source,
        "roi": tracker.crop_roi,
        "parameters": tracker.get_parameters(),
        "calibration_loaded": tracker.rbf_interpolator_x is not None,
    }


@app.post("/api/start")
def start_tracker(request: StartRequest):
    restart = False
    if tracker.is_running():
        if tracker.source == request.source:
            return {"status": "already-running", "source": request.source}
        tracker.stop()
        restart = True
    try:
        tracker.start(request.source)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "restarted" if restart else "started", "source": request.source}


@app.post("/api/stop")
def stop_tracker():
    if not tracker.is_running():
        raise HTTPException(status_code=400, detail="Tracker is not running")
    tracker.stop()
    return {"status": "stopped"}


@app.get("/api/gaze")
def get_gaze_data():
    data = tracker.get_latest_data()
    if data is None:
        raise HTTPException(status_code=503, detail="No gaze data available")
    return data


@app.put("/api/roi")
def set_roi(request: ROIRequest):
    tracker.set_roi((request.x, request.y, request.width, request.height))
    return {"status": "roi-updated", "roi": tracker.crop_roi}


@app.delete("/api/roi")
def clear_roi():
    tracker.set_roi(None)
    return {"status": "roi-cleared"}


@app.put("/api/parameters")
def update_parameters(request: ParametersRequest):
    tracker.set_parameters(**request.dict(exclude_unset=True))
    return {"status": "parameters-updated", "parameters": tracker.get_parameters()}


@app.post("/api/calibration/start")
def calibration_start(request: CalibrationStartRequest):
    points = tracker.start_calibration(request.screen_width, request.screen_height)
    return {"status": "calibration-started", "points": points}


@app.post("/api/calibration/capture")
async def calibration_capture(request: CalibrationCaptureRequest):
    try:
        result = await asyncio.to_thread(
            tracker.capture_calibration_point, request.point_index, request.screen_x, request.screen_y
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "calibration-point-captured", **result}


@app.post("/api/calibration/finish")
async def calibration_finish(request: CalibrationFinalizeRequest):
    try:
        result = await asyncio.to_thread(tracker.finalize_calibration, request.screen_width, request.screen_height)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "calibration-complete", **result}


@app.get("/api/calibration/state")
def calibration_state():
    return {
        "active": tracker.calibration_active,
        "points_collected": len(tracker.calibration_samples),
    }


@app.websocket("/ws/preview")
async def preview_feed(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    async with get_preview_clients_lock():
        PREVIEW_CLIENTS[client_id] = websocket
    try:
        while True:
            frame_info = tracker.get_preview_frame()
            if frame_info is None:
                await asyncio.sleep(0.1)
                continue

            frame = frame_info["frame"]
            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not success:
                await asyncio.sleep(0.1)
                continue

            payload = {
                "width": frame_info["width"],
                "height": frame_info["height"],
                "roi": frame_info["roi"],
                "timestamp": time.time(),
                "data": base64.b64encode(buffer).decode("ascii"),
                "pupil_mask": frame_info.get("pupil_mask"),
                "glint_mask": frame_info.get("glint_mask"),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1 / 15)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Preview websocket error: {exc}")
    finally:
        async with get_preview_clients_lock():
            PREVIEW_CLIENTS.pop(client_id, None)
        try:
            await websocket.close()
        except Exception:  # pragma: no cover - defensive
            pass


@app.on_event("shutdown")
def shutdown_event():
    if tracker.is_running():
        tracker.stop()
