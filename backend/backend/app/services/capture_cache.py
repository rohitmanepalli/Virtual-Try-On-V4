"""
Short-lived, memory-only cache of the work done on one capture (never generated images), so a new garment, a size
change, a retry or the background preparation does not repeat it. Keyed by the capture's exact pixels plus
PREPROCESS_VERSION: bump the version when validation or portrait framing changes.

Holds visitor biometrics, so like a session it expires after SESSION_TTL_SECONDS, is never written to disk, and is
cleared when a session is purged.
"""
import hashlib
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from PIL import Image

from app.config import SESSION_TTL_SECONDS

PREPROCESS_VERSION = "v1"
MAX_ENTRIES = 8


@dataclass
class PreparedCapture:
    """Only a capture that passed validation is stored: a failed one raises and is checked afresh next time."""
    portrait: Image.Image
    person_meta: Optional[Dict[str, Any]]
    coverage: Optional[Any] = None
    has_face: bool = False


class _NamedLock:
    __slots__ = ("lock", "__weakref__")

    def __init__(self):
        self.lock = threading.Lock()


_locks: "weakref.WeakValueDictionary[str, _NamedLock]" = weakref.WeakValueDictionary()
_locks_guard = threading.Lock()
_entries: "OrderedDict[str, tuple]" = OrderedDict()
_entries_guard = threading.Lock()


@contextmanager
def keyed_lock(name: str) -> Iterator[None]:
    """One holder per name, so duplicate work waits for the first and then finds its result. Freed when unused."""
    with _locks_guard:
        named = _locks.get(name)
        if named is None:
            named = _locks[name] = _NamedLock()
    with named.lock:
        yield


def key_for(image: Image.Image) -> str:
    digest = hashlib.sha256(f"{PREPROCESS_VERSION}|{image.mode}|{image.size}|".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def get(key: str) -> Optional[PreparedCapture]:
    now = time.monotonic()
    with _entries_guard:
        entry = _entries.get(key)
        if entry is None:
            return None
        if entry[0] <= now:
            del _entries[key]
            return None
        _entries.move_to_end(key)
        return entry[1]


def put(key: str, prepared: PreparedCapture) -> None:
    with _entries_guard:
        _entries[key] = (time.monotonic() + SESSION_TTL_SECONDS, prepared)
        _entries.move_to_end(key)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)


def sweep() -> None:
    """Drops expired entries even when no request arrives to look them up."""
    now = time.monotonic()
    with _entries_guard:
        for key in [k for k, entry in _entries.items() if entry[0] <= now]:
            del _entries[key]


def clear() -> None:
    with _entries_guard:
        _entries.clear()
