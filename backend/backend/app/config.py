import os
import socket
import sys

# Older versions saved download copies here; removed at startup (services/privacy.py). Nothing writes here now.
LEGACY_TEMP_DIR = "temp_images"

MAX_IMAGE_SIZE = 15360

# QR download copies are held in memory only (services/download_store.py)
DOWNLOAD_TTL_SECONDS = int(os.getenv("DOWNLOAD_TTL_SECONDS", "600"))
DOWNLOAD_GRACE_SECONDS = int(os.getenv("DOWNLOAD_GRACE_SECONDS", "120"))  # after the visitor's session ends

# Visitor business records, never images (services/visitor_store.py)
VISITOR_DATA_ENABLED = os.getenv("VISITOR_DATA_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
VISITOR_DB_PATH = os.getenv("VISITOR_DB_PATH", os.path.join("data", "visitors.db"))
VISITOR_DEFAULT_COUNTRY_CODE = os.getenv("VISITOR_DEFAULT_COUNTRY_CODE", "91")  # for 10-digit numbers typed without one

# Phase 1: Silhouette Verification & Session Security
# Calibrated for real-world webcam lighting & natural garment drape variations
SILHOUETTE_VERIFY_THRESHOLD = float(os.getenv("SILHOUETTE_VERIFY_THRESHOLD", "0.60"))
SILHOUETTE_MAX_RETRIES = int(os.getenv("SILHOUETTE_MAX_RETRIES", "1"))
# Total Gemini generation attempts per try-on request (raw-photo rejections, transient API errors,
# and the silhouette strict retry all count toward this budget).
GENERATION_MAX_ATTEMPTS = int(os.getenv("GENERATION_MAX_ATTEMPTS", "3"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "900"))  # 15 minutes default
# Output resolution of try-on renders: 1K, 2K or 4K. See tools/gemini_benchmark.py for the latency and cost of each.
TRYON_IMAGE_SIZE = os.getenv("TRYON_IMAGE_SIZE", "4K").strip()

# Phase 2: 2–3 Meter Capture Distance & Scale Calibration
CAPTURE_DISTANCE_MIN_METERS = float(os.getenv("CAPTURE_DISTANCE_MIN_METERS", "2.0"))
CAPTURE_DISTANCE_MAX_METERS = float(os.getenv("CAPTURE_DISTANCE_MAX_METERS", "3.0"))
CAPTURE_DISTANCE_OPTIMAL_METERS = float(os.getenv("CAPTURE_DISTANCE_OPTIMAL_METERS", "2.5"))
# Real-world scale factor (cm per normalized frame height unit) calibrated for 2.5m distance
# At 2.5m, standard 1.70m human occupies ~0.62 of vertical frame height: 170 / 0.62 ≈ 274 cm/unit
CALIBRATED_SCALE_FACTOR_2_3M = float(os.getenv("CALIBRATED_SCALE_FACTOR_2_3M", "274.0"))
CALIBRATED_SCALE_FACTOR_5_6M = CALIBRATED_SCALE_FACTOR_2_3M  # Backward compatibility alias

# Informational cm display (/api/extract-profile body_measurements) — never used for sizing.
# Camera focal length in pixels at a 1080px-tall capture; scaled linearly for other capture heights.
# Per-kiosk: run backend/tools/calibrate_focal_length.py with a person of known height standing at
# MEASUREMENT_CALIBRATION_DISTANCE_M. Default is derived from CALIBRATED_SCALE_FACTOR_2_3M
# (1080px spanning 274cm at 2.5m) so both estimates agree until the kiosk is calibrated.
MEASUREMENT_REFERENCE_HEIGHT_PX = 1080
MEASUREMENT_CALIBRATION_DISTANCE_M = float(os.getenv("MEASUREMENT_CALIBRATION_DISTANCE_M", str(CAPTURE_DISTANCE_OPTIMAL_METERS)))
MEASUREMENT_FOCAL_LENGTH_PX = float(os.getenv(
    "MEASUREMENT_FOCAL_LENGTH_PX",
    str(round(MEASUREMENT_REFERENCE_HEIGHT_PX * MEASUREMENT_CALIBRATION_DISTANCE_M * 100 / CALIBRATED_SCALE_FACTOR_2_3M, 1)),
))
# Calibration/testing aids — both OFF by default and must stay off on a production kiosk
# (they write visitor photos / measurements to disk). Paths are relative to the backend's working folder.
# CALIBRATION_CAPTURE_SAVE: every /api/extract-profile photo is also saved as PNG for calibrate_focal_length.py.
CALIBRATION_CAPTURE_SAVE = os.getenv("CALIBRATION_CAPTURE_SAVE", "false").strip().lower() in ("1", "true", "yes", "on")
CALIBRATION_CAPTURE_DIR = "calibration_captures"
# MEASUREMENT_CORRECTION_LOG: visitor measurement corrections are appended to a CSV (holds session ids —
# delete the file when testing ends).
MEASUREMENT_CORRECTION_LOG = os.getenv("MEASUREMENT_CORRECTION_LOG", "false").strip().lower() in ("1", "true", "yes", "on")
MEASUREMENT_CORRECTION_LOG_PATH = os.path.join("logs", "measurement_corrections.csv")

# Gemini cost monitoring: one CSV row per API call (token counts, USD estimate, 8-char session prefix, size).
# On by default — every size change is a full generation, so real per-visitor cost needs watching. No images or
# personal data are written.
GEMINI_COST_LOG = os.getenv("GEMINI_COST_LOG", "true").strip().lower() in ("1", "true", "yes", "on")
GEMINI_COST_LOG_PATH = os.path.join("logs", "gemini_costs.csv")

# Try-on latency: one CSV row per /api/tryon request with the time spent in each pipeline step (see
# services/request_timing.py). On by default; no images or personal data are written.
TRYON_TIMING_LOG = os.getenv("TRYON_TIMING_LOG", "true").strip().lower() in ("1", "true", "yes", "on")
TRYON_TIMING_LOG_PATH = os.path.join("logs", "tryon_timings.csv")

# Full activity log: all backend output and frontend activity, one file per day (see services/app_log.py). On by default.
APP_LOG = os.getenv("APP_LOG", "true").strip().lower() in ("1", "true", "yes", "on")
APP_LOG_DIR = "logs"
# Visitor body data printed to the console stays out of the file unless this is on. Testing only: must stay off on a
# production kiosk, like the calibration aids above.
APP_LOG_PERSONAL_DATA = os.getenv("APP_LOG_PERSONAL_DATA", "false").strip().lower() in ("1", "true", "yes", "on")

# YOLO Person Detection Configuration
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "assets/yolox_s.onnx")  # YOLOX-S, Apache 2.0 (OpenCV model zoo)
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
# Gate /api/tryon on YOLO person detection before any Gemini call. When false the pipeline runs exactly as before.
YOLO_VALIDATION_ENABLED = os.getenv("YOLO_VALIDATION_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# YOLO + MediaPipe body cross-validation before try-on generation (runs only when YOLO validation produced a result)
BODY_VALIDATION_ENABLED = os.getenv("BODY_VALIDATION_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
# Margin (fraction of YOLO box width/height, each side) within which the MediaPipe landmarks must fall
BODY_BBOX_TOLERANCE = float(os.getenv("BODY_BBOX_TOLERANCE", "0.20"))
# Max horizontal offset of the MediaPipe torso centre from the YOLO box centre (fraction of box width)
BODY_CENTER_TOLERANCE = float(os.getenv("BODY_CENTER_TOLERANCE", "0.25"))
# Minimum MediaPipe visibility (0-1) for each required landmark
BODY_MIN_LANDMARK_VISIBILITY = float(os.getenv("BODY_MIN_LANDMARK_VISIBILITY", "0.50"))
# Landmarks that must be visible and inside the frame for a usable try-on capture
BODY_REQUIRED_LANDMARKS = os.getenv("BODY_REQUIRED_LANDMARKS", "nose,left_shoulder,right_shoulder,left_hip,right_hip")



def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def get_visitor_data_key() -> str | None:
    """Fernet key encrypting visitor names and phone numbers. Losing it makes them unreadable: back it up apart from the database."""
    return os.getenv("VISITOR_DATA_KEY") or None


def get_local_ip() -> str:
    """Retrieve the primary local LAN/Wi-Fi IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # 8.8.8.8 UDP connect does not send packets, but prompts the OS routing table to identify the primary local interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


def get_public_base_url() -> str:
    """
    Returns the base URL used in QR codes for mobile download.
    - If PUBLIC_BASE_URL is explicitly configured to a non-localhost URL (e.g. ngrok or custom domain), use it.
    - Otherwise, automatically resolve to this PC's LAN IP address so phones on the same Wi-Fi can open it.
    """
    env_url = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    # If user provided a specific non-localhost URL (e.g. ngrok, public domain, or manual IP)
    if env_url and not any(h in env_url.lower() for h in ["localhost", "127.0.0.1", "0.0.0.0"]):
        return env_url

    local_ip = get_local_ip()
    port = os.getenv("PORT", "8888")
    if env_url and ":" in env_url.replace("http://", "").replace("https://", ""):
        parsed_port = env_url.split(":")[-1].split("/")[0]
        if parsed_port.isdigit():
            port = parsed_port

    return f"http://{local_ip}:{port}"


def get_dist_dir() -> str:
    """Resolve the built frontend directory (works in dev, packaged app, and repo layout)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    frontend_dist = os.path.abspath(os.path.join(project_root, "frontend", "dist"))
    frontend_dist_app = os.path.abspath(os.path.join(project_root, "frontend", "dist-app"))
    root_dist = os.path.abspath(os.path.join(project_root, "dist"))

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundled_dist = os.path.join(sys._MEIPASS, "dist")
        if os.path.exists(bundled_dist):
            return bundled_dist

    for candidate in (frontend_dist_app, frontend_dist, root_dist):
        if os.path.exists(candidate):
            return candidate

    return frontend_dist_app
