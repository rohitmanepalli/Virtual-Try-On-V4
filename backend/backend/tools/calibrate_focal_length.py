"""
One-time per-kiosk calibration of MEASUREMENT_FOCAL_LENGTH_PX (informational cm display only).

1. Have a person of known height stand exactly at MEASUREMENT_CALIBRATION_DISTANCE_M (default 2.5m,
   measured from camera lens to their feet), full body in frame, facing the camera, arms at sides.
2. Set CALIBRATION_CAPTURE_SAVE=1 in backend/.env, restart run_app.bat, and take a normal kiosk capture.
   The exact photo the pipeline measured is saved to backend/calibration_captures/ (path printed in the
   console as [CALIBRATION CAPTURE]).
3. Run from the backend folder:
       venv\\Scripts\\python.exe tools\\calibrate_focal_length.py calibration_captures\\capture_....png --known-height-cm 175
   (or --known-shoulder-cm if feet cannot be in frame; less reliable, since a tape measurement rarely
   matches MediaPipe's shoulder-joint landmarks exactly)
4. Put the printed MEASUREMENT_FOCAL_LENGTH_PX=... line in backend/.env, remove CALIBRATION_CAPTURE_SAVE,
   restart, and delete calibration_captures/ (it holds visitor photos).

Pinhole model: focal_px = size_px × distance / real_size. Measured cm scale as 1 / focal_px, so the
calibrated value is current_focal × measured_cm / known_cm.
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import MEASUREMENT_CALIBRATION_DISTANCE_M, MEASUREMENT_FOCAL_LENGTH_PX  # noqa: E402
from app.services.body_profile import body_profile_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", help="Capture photo of the calibration person")
    ref = parser.add_mutually_exclusive_group(required=True)
    ref.add_argument("--known-height-cm", type=float, help="Person's true height in cm (preferred)")
    ref.add_argument("--known-shoulder-cm", type=float, help="Person's true shoulder width in cm")
    args = parser.parse_args()

    frames = body_profile_service.extract_frames(Image.open(args.photo).convert("RGB"))
    m = body_profile_service.measure_body(frames)

    if args.known_height_cm:
        if not m.height or m.height.source != "silhouette":
            print("[ERROR] Full body (head to feet) not detected in frame; cannot calibrate from height.")
            return 1
        measured, known, label = m.height.value_cm, args.known_height_cm, "height"
    else:
        if not m.shoulder_width:
            print("[ERROR] Shoulders not detected; cannot calibrate from shoulder width.")
            return 1
        measured, known, label = m.shoulder_width.value_cm, args.known_shoulder_cm, "shoulder width"

    focal = MEASUREMENT_FOCAL_LENGTH_PX * measured / known
    print(f"Calibration distance: {MEASUREMENT_CALIBRATION_DISTANCE_M} m")
    print(f"Current focal length: {MEASUREMENT_FOCAL_LENGTH_PX} px -> measured {label} {measured} cm (true {known} cm)")
    print(f"MEASUREMENT_FOCAL_LENGTH_PX={focal:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
