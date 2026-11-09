"""
Per-request step timings for /api/tryon: one CSV row per request (logs/tryon_timings.csv), so real kiosk latency can
be read from the log instead of re-running a test. Each Gemini call's own duration is in logs/gemini_costs.csv.

Columns, in seconds, summed over every generation attempt of the request:
  decode_s         decoding the capture and garment uploads
  yolo_s           YOLO person detection
  mediapipe_s      every MediaPipe Pose run in the request (capture and renders)
  garment_s        garment analysis and hash (catalog garments are instant)
  appearance_s     visitor appearance analysis: a Gemini call unless the capture was prepared in advance
  checks_before_s  the rest of the customer preprocessing: body validation logic, portrait crop, face mask, size logic,
                   session cache lookup, sleeve coverage read, face check
  gemini_request_s Gemini request preparation: client and prompt
  gemini_image_s   the image generation calls (upload, generation and download: the API does not separate them)
  face_restore_s   restoring the captured face (or the first render's face) onto each render
  checks_after_s   framing, background, silhouette, skeleton, fit and coverage checks on each render
  finalize_s       final JPEG, watermarked download copy and QR code
  other_s          everything else: response decoding, retry decisions, response building
request_id appears on this row, on the request's rows in logs/gemini_costs.csv and on its lines in logs/app_<date>.log.
attempt_results lists one result per generation attempt: ok, or the first check it failed (raw_photo, silhouette,
coverage, fit), or error / blocked when the call itself failed. tools/tryon_stats.py summarises these logs.
"""
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from app.config import TRYON_TIMING_LOG, TRYON_TIMING_LOG_PATH
from app.services.csv_log import append_row

STEPS = ["decode", "yolo", "mediapipe", "garment", "appearance", "checks_before", "gemini_request", "gemini_image",
         "face_restore", "checks_after", "finalize"]
_FIELDS = (["timestamp", "request_id", "session", "size", "fit_category", "outcome", "attempts", "attempt_results",
            "total_s"] + [f"{name}_s" for name in STEPS] + ["other_s"])


class RequestTiming:
    """Step durations of one request. Time is attributed either by lap() (the time since the previous lap) or by a
    step() block; lap() leaves out the step() blocks inside its span, so nothing is counted twice."""

    def __init__(self, session_id: Optional[str]):
        self.request_id = uuid.uuid4().hex[:8]
        self.session = (session_id or "")[:8]
        self.started = time.perf_counter()
        self.steps: Dict[str, float] = dict.fromkeys(STEPS, 0.0)
        self.attempt_results: List[str] = []
        self._mark = self.started
        self._in_steps = 0.0

    def lap(self, name: Optional[str]) -> None:
        """Attributes the time since the previous lap to a step (None: leave it in other_s)."""
        now = time.perf_counter()
        if name is not None:
            self.steps[name] += max(0.0, now - self._mark - self._in_steps)
        self._mark, self._in_steps = now, 0.0


_current: ContextVar[Optional[RequestTiming]] = ContextVar("tryon_request_timing", default=None)


def begin(session_id: Optional[str]) -> RequestTiming:
    timing = RequestTiming(session_id)
    _current.set(timing)
    return timing


def current_id() -> Optional[str]:
    """The running request's id, for correlating logs (None outside a timed request)."""
    timing = _current.get()
    return timing.request_id if timing is not None else None


def lap(name: Optional[str]) -> None:
    timing = _current.get()
    if timing is not None:
        timing.lap(name)


@contextmanager
def step(name: str) -> Iterator[None]:
    """Adds the block's duration to the current request's step (no-op outside a timed request)."""
    timing = _current.get()
    started = time.perf_counter()
    try:
        yield
    finally:
        if timing is not None:
            elapsed = time.perf_counter() - started
            timing.steps[name] += elapsed
            timing._in_steps += elapsed


def attempt_result(result: str) -> None:
    timing = _current.get()
    if timing is not None:
        timing.attempt_results.append(result)


def finish(timing: RequestTiming, outcome: str, size: Optional[str], fit_category: Optional[str]) -> Dict[str, Any]:
    """Prints the request's breakdown and appends its row to logs/tryon_timings.csv; returns the row."""
    total = time.perf_counter() - timing.started
    other = max(0.0, total - sum(timing.steps.values()))
    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "request_id": timing.request_id,
        "session": timing.session, "size": size or "",
        "fit_category": fit_category or "", "outcome": outcome, "attempts": len(timing.attempt_results),
        "attempt_results": ";".join(timing.attempt_results), "total_s": f"{total:.2f}",
        **{f"{name}_s": f"{secs:.2f}" for name, secs in timing.steps.items()}, "other_s": f"{other:.2f}",
    }
    print(f"[TRYON TIMING] {outcome} in {total:.1f}s | " + " | ".join(
        f"{name} {secs:.1f}s" for name, secs in [*timing.steps.items(), ("other", other)] if secs >= 0.05)
        + (f" | attempts: {row['attempt_results']}" if timing.attempt_results else ""), flush=True)
    if TRYON_TIMING_LOG:
        try:
            append_row(TRYON_TIMING_LOG_PATH, _FIELDS, row)
        except Exception as err:
            print(f"[TRYON TIMING WARNING] Could not append timing log: {err}")
    _current.set(None)
    return row
