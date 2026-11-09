"""
Keeps visitor images from outliving their use. The pipeline works in memory and writes no image files; this module clears
what sits in memory once it expires (even when no further request arrives), and removes files left by older versions.
"""
import os
import threading
import time

from app.config import CALIBRATION_CAPTURE_SAVE, LEGACY_TEMP_DIR
from app.services import capture_cache, download_store, gemini_client
from app.services.session_manager import session_manager

SWEEP_INTERVAL_SECONDS = 60


def sweep_once() -> None:
    session_manager.cleanup_expired()
    capture_cache.sweep()
    download_store.sweep()
    if not session_manager.active_count():
        # Nobody is mid-visit: nothing can still need a capture or a visitor description
        capture_cache.clear()
        gemini_client.clear_user_appearance()


def _sweep_forever() -> None:
    while True:
        time.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            sweep_once()
        except Exception as err:
            print(f"[PRIVACY WARNING] Sweep failed: {type(err).__name__}")


def remove_legacy_temp_images() -> int:
    """Older versions saved each try-on's download copy here. Removes only those .jpg files, then the empty folder."""
    if not os.path.isdir(LEGACY_TEMP_DIR):
        return 0
    removed = 0
    for name in os.listdir(LEGACY_TEMP_DIR):
        path = os.path.join(LEGACY_TEMP_DIR, name)
        if name.lower().endswith(".jpg") and os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as err:
                print(f"[PRIVACY WARNING] Could not remove an old try-on image: {type(err).__name__}")
    try:
        os.rmdir(LEGACY_TEMP_DIR)
    except OSError:
        pass  # Holds something else: left alone
    return removed


def start() -> None:
    """Call once at startup."""
    removed = remove_legacy_temp_images()
    if removed:
        print(f"[PRIVACY] Removed {removed} try-on image(s) left by an older version.", flush=True)
    if CALIBRATION_CAPTURE_SAVE:
        print("[PRIVACY WARNING] CALIBRATION_CAPTURE_SAVE is on: visitor photos are being written to disk. "
              "Turn it off outside camera calibration.", flush=True)
    threading.Thread(target=_sweep_forever, daemon=True).start()
