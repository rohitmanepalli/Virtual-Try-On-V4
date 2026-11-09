"""
The QR code's download copies, in memory only: nothing is written to disk. A copy lasts DOWNLOAD_TTL_SECONDS, or
DOWNLOAD_GRACE_SECONDS after its session ends (long enough to finish saving on the phone after the visitor taps Home).
A server restart loses them, and the app must run as a single process for the phone to reach the copy.
"""
import threading
import time
from collections import OrderedDict
from typing import Optional

from app.config import DOWNLOAD_GRACE_SECONDS, DOWNLOAD_TTL_SECONDS

MAX_COPIES = 10  # each is a few MB

_lock = threading.Lock()
_copies: "OrderedDict[str, dict]" = OrderedDict()


def put(image_id: str, jpeg: bytes, session_id: Optional[str] = None) -> None:
    with _lock:
        _copies[image_id] = {"jpeg": jpeg, "session": session_id, "expires": time.monotonic() + DOWNLOAD_TTL_SECONDS}
        while len(_copies) > MAX_COPIES:
            _copies.popitem(last=False)


def get(image_id: str) -> Optional[bytes]:
    with _lock:
        copy = _copies.get(image_id)
        if copy is None:
            return None
        if copy["expires"] <= time.monotonic():
            del _copies[image_id]
            return None
        return copy["jpeg"]


def end_session(session_id: str) -> None:
    latest = time.monotonic() + DOWNLOAD_GRACE_SECONDS
    with _lock:
        for copy in _copies.values():
            if copy["session"] == session_id:
                copy["expires"] = min(copy["expires"], latest)


def sweep() -> None:
    now = time.monotonic()
    with _lock:
        for image_id in [i for i, c in _copies.items() if c["expires"] <= now]:
            del _copies[image_id]


def clear() -> None:
    with _lock:
        _copies.clear()
