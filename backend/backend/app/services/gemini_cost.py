"""
Per-call Gemini token usage, cost and duration logging, so the real cost of per-size regeneration can be monitored.

Prices: https://ai.google.dev/gemini-api/docs/pricing (paid tier, standard), checked 2026-09-29.
A 4K output image is 2520 tokens ≈ $0.151 from gemini-3.1-flash-image, 2000 tokens ≈ $0.24 from gemini-3-pro-image.
"""
from contextvars import ContextVar
from datetime import datetime
from typing import Any, List, Optional

from app.config import GEMINI_COST_LOG, GEMINI_COST_LOG_PATH
from app.services.csv_log import append_row
from app.services.request_timing import current_id

FLASH_IMAGE_MODEL = "gemini-3.1-flash-image"
PRO_IMAGE_MODEL = "gemini-3-pro-image"
TRYON_MODEL = FLASH_IMAGE_MODEL
ANALYSIS_MODEL = "gemini-2.5-flash"

# USD per 1M tokens: input (text/image), text + thinking output, image output
PRICES_PER_M = {
    FLASH_IMAGE_MODEL: {"input": 0.50, "text_out": 3.00, "image_out": 60.00},
    PRO_IMAGE_MODEL: {"input": 2.00, "text_out": 12.00, "image_out": 120.00},
    ANALYSIS_MODEL: {"input": 0.30, "text_out": 2.50, "image_out": 0.0},
}

_FIELDS = [
    "timestamp", "kind", "model", "session", "size", "fit_category", "size_offset", "attempt",
    "input_tokens", "text_out_tokens", "image_out_tokens", "thinking_tokens", "cost_usd", "duration_s", "request_id",
]

# Costs of the calls made while handling the current request (one /api/tryon = one size), see begin_cost_tracking
_request_costs: ContextVar[Optional[List[float]]] = ContextVar("gemini_request_costs", default=None)


def begin_cost_tracking() -> List[float]:
    """Starts collecting this request's call costs; the returned list fills as calls are recorded."""
    costs: List[float] = []
    _request_costs.set(costs)
    return costs


def current_costs() -> List[float]:
    """The calls priced so far in this request."""
    return list(_request_costs.get() or [])


def _image_tokens(details: Any) -> Optional[int]:
    """Image-modality output tokens from candidates_tokens_details; None when the API gave no breakdown."""
    if not details:
        return None
    total = 0
    for d in details:
        modality = getattr(d, "modality", None)
        if str(getattr(modality, "name", modality)).upper().endswith("IMAGE"):
            total += getattr(d, "token_count", 0) or 0
    return total


def record_usage(
    kind: str,
    model: str,
    response: Any,
    session_id: Optional[str] = None,
    size: Optional[str] = None,
    attempt: Optional[int] = None,
    fit_category: Optional[str] = None,
    size_offset: Optional[int] = None,
    duration_s: Optional[float] = None,
) -> Optional[float]:
    """Logs one API call's tokens, USD estimate and duration (seconds); returns the cost (None when usage or pricing is
    unavailable)."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        print(f"[GEMINI COST] {kind} {model}: no usage metadata in response")
        return None
    prices = PRICES_PER_M.get(model)
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    out_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thinking = getattr(usage, "thoughts_token_count", 0) or 0
    image_out = _image_tokens(getattr(usage, "candidates_tokens_details", None))
    if image_out is None:  # no modality breakdown: an image model's output is the image
        image_out = out_tokens if prices and prices["image_out"] > 0 else 0
    text_out = max(0, out_tokens - image_out)

    cost = None
    if prices:
        cost = (input_tokens * prices["input"] + (text_out + thinking) * prices["text_out"]
                + image_out * prices["image_out"]) / 1_000_000
    session = (session_id or "")[:8]
    print(
        f"[GEMINI COST] {kind} | {model} | in {input_tokens} tok, out text {text_out} / image {image_out}, "
        f"thinking {thinking} | {'$%.4f' % cost if cost is not None else 'unpriced model'}"
        f"{f' | {duration_s:.1f}s' if duration_s is not None else ''}"
        f"{f' | session {session}' if session else ''}{f' | size {size}' if size else ''}"
        f"{f' | fit {fit_category}' if fit_category else ''}{f' | offset {size_offset:+d}' if size_offset is not None else ''}"
        f"{f' | attempt {attempt}' if attempt else ''}"
    )
    tracked = _request_costs.get()
    if tracked is not None and cost is not None:
        tracked.append(cost)
    if GEMINI_COST_LOG:
        try:
            append_row(GEMINI_COST_LOG_PATH, _FIELDS, {
                "timestamp": datetime.now().isoformat(timespec="seconds"), "kind": kind, "model": model,
                "session": session, "size": size or "", "fit_category": fit_category or "",
                "size_offset": size_offset if size_offset is not None else "", "attempt": attempt or "",
                "input_tokens": input_tokens, "text_out_tokens": text_out, "image_out_tokens": image_out,
                "thinking_tokens": thinking, "cost_usd": f"{cost:.5f}" if cost is not None else "",
                "duration_s": f"{duration_s:.2f}" if duration_s is not None else "",
                "request_id": current_id() or "",
            })
        except Exception as err:
            print(f"[GEMINI COST WARNING] Could not append cost log: {err}")
    return cost
