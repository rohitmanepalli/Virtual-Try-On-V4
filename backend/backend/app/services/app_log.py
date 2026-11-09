"""
Full activity log: everything the backend prints or logs, plus what the kiosk frontend reports, is appended to
logs/app_<YYYY-MM-DD>.log while still showing on the console.

The file is opened per write rather than held open, so uvicorn's reload parent and worker can share it on Windows and
it can be moved or deleted while the app runs. Log failures are swallowed: a full disk must not break a try-on.
"""
import logging
import os
import re
import sys
import threading
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from app.config import APP_LOG, APP_LOG_DIR, APP_LOG_PERSONAL_DATA
from app.services.request_timing import current_id

MAX_LINE_CHARS = 4000
# /api/client-logs is unauthenticated, so it must not be able to fill the disk
MAX_CLIENT_LOG_BYTES = 200 * 1024 * 1024
WITHHELD = "[withheld: personal data]"

# Output holding visitor body data. Only the label is kept (see config.APP_LOG_PERSONAL_DATA).
_PERSONAL = re.compile(
    r"Raw body_profile in payload"
    r"|\[SESSION RETRIEVED PROFILE\]"
    r"|\[SIZE REC(?: ERROR)?\]"
    r"|\[MEASUREMENT CORRECTION\]"
    r"|\[BODY PROFILE EXTRACTED\]"
    r"|Extracted on-the-fly BodyProfile"
    r"|Baseline fit landmarks recorded"
    r"|\[USER APPEARANCE\] Profile created"  # gender, skin tone, glasses, build and wrists, read from the photo
    r"|\[COVERAGE\] Capture:"  # skin colour and wrist accessories
    r"|\[SILHOUETTE VERIFY\]"  # body volume
    r"|\[FIT CHECK\]"  # body widths
    r"|\[MIDSECTION WARNING\]"
    r"|\[API DEBUG\] Gemini text part"  # the model's own words about the visitor
)
_BODY_CHECK = re.compile(r" \| Body vs [^|]*")
_SECRET = re.compile(r"(GEMINI_API_KEY: |[?&]key=)[^\s|&]+")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ENCODED_IMAGE = re.compile(r"[A-Za-z0-9+/=_-]{200,}")

_write_lock = threading.Lock()


def _log_path() -> str:
    return os.path.join(APP_LOG_DIR, f"app_{datetime.now():%Y-%m-%d}.log")


def _clean(text: str) -> str:
    """Applied to a whole print/log call, so a multi-line dump is withheld as one unit."""
    text = _ANSI.sub("", text)
    if not APP_LOG_PERSONAL_DATA:
        personal = _PERSONAL.search(text)
        if personal:
            return f"{personal.group(0)} {WITHHELD}"
        text = _BODY_CHECK.sub("", text)
    return _ENCODED_IMAGE.sub("[encoded data]", _SECRET.sub(r"\1[hidden]", text))


def _append(lines: Iterable[str]) -> None:
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    text = "".join(f"{stamp} {line[:MAX_LINE_CHARS]}\n" for line in lines)
    if not text:
        return
    try:
        with _write_lock:
            os.makedirs(APP_LOG_DIR, exist_ok=True)
            with open(_log_path(), "a", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
    except OSError:
        pass


class _Tee:
    """Console stream that also sends each completed print or log call to the log file."""

    def __init__(self, stream):
        self._stream = stream
        self._local = threading.local()  # print() makes separate writes, which must not mix across threads

    def write(self, text: str) -> int:
        buffered = getattr(self._local, "buffer", "") + text
        if text.endswith("\n") or len(buffered) > 65536:
            self._local.buffer = ""
            request_id = current_id()
            prefix = f"[req {request_id}] " if request_id else ""
            _append(prefix + line for line in _clean(buffered).splitlines() if line.strip())
        else:
            self._local.buffer = buffered
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging() -> None:
    """Call once at startup, before other app modules are imported, so their output is captured too."""
    if not APP_LOG or isinstance(sys.stdout, _Tee):
        return
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    # Without this, logger.info() calls in the services are dropped
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s [%(name)s] %(message)s")
    # uvicorn's reloader watches the folder holding logs/: each "change detected" line would itself be a change, forever
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


def capture_uvicorn_logs() -> None:
    """Call from lifespan. uvicorn may have bound its handlers to the raw console before setup_logging ran."""
    if not APP_LOG:
        return
    for name in ("uvicorn", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler.stream, _Tee):
                handler.setStream(_Tee(handler.stream))
    # Log uploads are a request every few seconds and say nothing about what the visitor did
    logging.getLogger("uvicorn.access").addFilter(lambda record: "/api/client-logs" not in record.getMessage())


def write_client_entries(client_id: str, entries: List[Tuple[str, str, Optional[str]]]) -> None:
    """Appends (level, message, client time) entries sent by the frontend. All three fields are untrusted."""
    if not APP_LOG:
        return
    try:
        if os.path.getsize(_log_path()) > MAX_CLIENT_LOG_BYTES:
            return
    except OSError:
        pass  # No log file yet today
    client = re.sub(r"[^\w-]", "", client_id)[:16] or "unknown"
    lines = []
    for level, message, client_time in entries:
        level_tag = re.sub(r"[^a-zA-Z]", "", level)[:8].upper() or "LOG"
        time_tag = re.sub(r"[^\d:.]", "", client_time or "")[:12]
        # Flattened to one line so a client can't forge entries with embedded newlines
        lines.append(f"[CLIENT {client} {time_tag}] {level_tag} {' | '.join(_clean(message).splitlines())}")
    _append(lines)
