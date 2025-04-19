import collections
from typing import Dict, Deque
import socket
import threading
import struct
import logging
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from sort.sort import Sort

# ─── Configuration ─────────────────────────────────────────────────────────────
HOST = '127.0.0.1'
COLOR_PORT = 9999
DEPTH_PORT = 9998
MODEL_PATH = "yolo11n.pt"

# Add these constants in the configuration section
MIN_DETECTION_CONFIDENCE = 0.7    # Minimum confidence threshold for detections
MAX_TRACKING_AGE = 120  # Maximum number of frames to keep track of objects
MIN_HITS = 1   # Minimum number of detection hits needed to start tracking
IOU_THRESHOLD = 0.5  # Intersection over a Union threshold

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Globals ───────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
color_frame = None
depth_frame = None
depth_lock = threading.Lock()

# Add these after the globals section
# Buffer for temporal smoothing
tracked_depths: Dict[int, Deque[float]] = collections.defaultdict(
    lambda: collections.deque(maxlen=5)
)
tracked_positions: Dict[int, Deque[tuple]] = collections.defaultdict(
    lambda: collections.deque(maxlen=3)
)

# Initialize SORT tracker with optimized parameters
tracker = Sort(
    max_age=MAX_TRACKING_AGE,
    min_hits=MIN_HITS,
    iou_threshold=IOU_THRESHOLD
)

frame_counter = 0

# ─── Color Receiver ────────────────────────────────────────────────────────────
def receive_color():
    global color_frame
    with socket.socket() as s:
        s.bind((HOST, COLOR_PORT))
        s.listen(1)
        logger.info(f"Listening for COLOR on {HOST}:{COLOR_PORT}")
        conn, addr = s.accept()
        logger.info(f"COLOR client connected from {addr}")
        with conn:
            while True:
                raw = conn.recv(4)
                if not raw:
                    logger.warning("COLOR connection closed by sender")
                    break
                size = struct.unpack('>I', raw)[0]
                logger.debug(f"Expecting COLOR frame of {size} bytes")
                buf = b''

                while len(buf) < size:
                    chunk = conn.recv(size - len(buf))
                    if not chunk:
                        logger.error("Color frame truncated")
                        break
                    buf += chunk

                try:
                    img = Image.open(BytesIO(buf))
                    color_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    logger.debug(f"Received COLOR frame: shape={color_frame.shape}")
                except Exception as ex:
                    logger.exception(f"Failed to decode COLOR frame: {ex}")

# ─── Depth Receiver ────────────────────────────────────────────────────────────
def receive_depth():
    global depth_frame
    with socket.socket() as s:
        s.bind((HOST, DEPTH_PORT))
        s.listen(1)
        logger.info(f"Listening for DEPTH on {HOST}:{DEPTH_PORT}")
        conn, addr = s.accept()
        logger.info(f"DEPTH client connected from {addr}")
        with conn:
            while True:
                raw = conn.recv(4)
                if not raw:
                    logger.warning("DEPTH connection closed by sender")
                    break
                size = struct.unpack('>I', raw)[0]
                logger.debug(f"Expecting DEPTH packet of {size} bytes")
                buf = b''

                while len(buf) < size:
                    chunk = conn.recv(size - len(buf))
                    if not chunk:
                        logger.error("Depth packet truncated")
                        break
                    buf += chunk

                try:
                    arr = np.frombuffer(buf, dtype=np.uint16).reshape((480, 640))
                    with depth_lock:
                        depth_frame = arr
                    logger.debug("Received DEPTH frame")
                except Exception as ex:
                    logger.exception(f"Failed to decode DEPTH frame: {ex}")

def get_reliable_depth(depth_img, cx: int, cy: int, window_size: int = 5) -> float:
    """Get a more reliable depth measurement using neighborhood averaging."""
    half_size = window_size // 2
    depth_window = depth_img[
        max(0, cy - half_size):min(depth_img.shape[0], cy + half_size + 1),
        max(0, cx - half_size):min(depth_img.shape[1], cx + half_size + 1)
    ]

    # Filter out zero values and get median
    valid_depths = depth_window[depth_window > 0]
    if len(valid_depths) > 0:
        return np.median(valid_depths) * 0.1  # Convert to cm
    return 0.0

def smooth_position(track_id: int, x: float, y: float) -> tuple:
    """Apply position smoothing using moving average."""
    tracked_positions[track_id].append((x, y))
    if len(tracked_positions[track_id]) >= 2:
        positions = np.array(tracked_positions[track_id])
        return tuple(np.mean(positions, axis=0))
    return x, y

def overlay_distance(color_img, depth_img):
    results = model(color_img, conf=MIN_DETECTION_CONFIDENCE)
    out = color_img.copy()

    detection_boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf.item()

            if len([x1, y1, x2, y2]) == 4 and conf >= MIN_DETECTION_CONFIDENCE:
                cls = int(box.cls)
                label = model.names[cls]
                if label == "person":
                    detection_boxes.append([x1, y1, x2, y2, conf])

    # Ensure that tracked_objects is initialized even if no detection_boxes exist
    tracked_objects = tracker.update(np.array(detection_boxes)) if detection_boxes else []

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)

        # Apply position smoothing
        cx, cy = smooth_position(track_id, (x1 + x2) // 2, (y1 + y2) // 2)

        # Get reliable depth measurement
        d = get_reliable_depth(depth_img, int(cx), int(cy))

        # Apply temporal smoothing to depth
        tracked_depths[track_id].append(d)
        smoothed_depth = np.median(tracked_depths[track_id])

        if smoothed_depth > 0:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{smoothed_depth:.0f}cm",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (124, 0, 255), 2)

    return out


def display_loop():
    logger.info("Starting display loop")
    while True:
        if color_frame is not None and depth_frame is not None:
            try:
                frame = overlay_distance(color_frame, depth_frame)
                cv2.imshow("YOLOv8 + Kinect Distance + SORT Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received, closing")
                    break
            except Exception as ex:
                logger.exception(f"Error in display loop: {ex}")
    cv2.destroyAllWindows()

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting threads")
    threading.Thread(target=receive_color, daemon=True, name="ColorThread").start()
    threading.Thread(target=receive_depth, daemon=True, name="DepthThread").start()
    display_loop()
    logger.info("Program exiting")