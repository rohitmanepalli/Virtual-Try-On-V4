import json
import threading
import time
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from app.config import SESSION_TTL_SECONDS
from app.models import BodyMeasurements, BodyProfile


RENDER_CACHE_VERSION = "v5_clean_no_mask"


class SessionRecord:
    """
    Ephemeral session record holding session-scoped biometric data and last good render.
    The Fernet encryption key is strictly scoped to this session instance and destroyed
    upon session termination, ensuring true zero-retention biometric privacy.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._key: Optional[bytes] = Fernet.generate_key()
        self._cipher: Optional[Fernet] = Fernet(self._key)
        self._encrypted_profile: Optional[bytes] = None
        self._encrypted_measurements: Optional[bytes] = None
        self._encrypted_last_good_render: Optional[bytes] = None
        self._encrypted_canonical_render: Optional[bytes] = None
        self._encrypted_face_mask: Optional[bytes] = None
        self._encrypted_baseline_landmarks: Optional[bytes] = None
        self.baseline_fit_state: Optional[str] = None
        self.last_dress_hash: Optional[str] = None
        self.canonical_dress_hash: Optional[str] = None
        self.size_renders_cache: Dict[str, str] = {}
        # Garment-silhouette metrics of each cached size render (same keys), for cross-size verification
        self.size_render_metrics: Dict[str, Dict[str, Any]] = {}
        # Same keys: "won't fit" sizes whose render was not believable (message + illustration served instead)
        self.size_illustrations: Dict[str, Dict[str, Any]] = {}
        # First served render of this capture (any garment): its face is the visitor's face on every later size when
        # the capture face is too small to restore. Keyed to the capture so a retake never inherits an old face.
        self._encrypted_identity_render: Optional[bytes] = None
        self.identity_capture_hash: Optional[str] = None
        self.render_cache_version: str = RENDER_CACHE_VERSION
        self.created_at = time.time()
        self.last_accessed_at = time.time()

    def _validate_cache_version(self) -> None:
        """Purges any cached renders from previous versions to prevent mask artifacts from resurfacing."""
        if getattr(self, "render_cache_version", None) != RENDER_CACHE_VERSION:
            self._encrypted_canonical_render = None
            self.canonical_dress_hash = None
            self._encrypted_last_good_render = None
            self.size_renders_cache.clear()
            self.size_render_metrics.clear()
            self.size_illustrations.clear()
            self.render_cache_version = RENDER_CACHE_VERSION

    def set_body_profile(self, profile: BodyProfile) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        payload = profile.model_dump_json().encode("utf-8")
        self._encrypted_profile = self._cipher.encrypt(payload)
        self.last_accessed_at = time.time()

    def get_body_profile(self) -> Optional[BodyProfile]:
        if not self._cipher or not self._encrypted_profile:
            return None
        self.last_accessed_at = time.time()
        decrypted = self._cipher.decrypt(self._encrypted_profile)
        data = json.loads(decrypted.decode("utf-8"))
        return BodyProfile(**data)

    def set_body_measurements(self, measurements: BodyMeasurements) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        self._encrypted_measurements = self._cipher.encrypt(measurements.model_dump_json().encode("utf-8"))
        self.last_accessed_at = time.time()

    def get_body_measurements(self) -> Optional[BodyMeasurements]:
        if not self._cipher or not self._encrypted_measurements:
            return None
        self.last_accessed_at = time.time()
        return BodyMeasurements(**json.loads(self._cipher.decrypt(self._encrypted_measurements).decode("utf-8")))

    def invalidate_outfit_cache(self, new_dress_hash: Optional[str] = None) -> None:
        """
        Invalidates garment-specific caches (size renders, canonical render, last good render,
        baseline landmarks) when switching garments or forcing refresh.
        Strictly preserves the visitor's biometric body profile, calibration data, and face mask.
        """
        self._encrypted_canonical_render = None
        self.canonical_dress_hash = None
        self._encrypted_last_good_render = None
        self._encrypted_baseline_landmarks = None
        self.baseline_fit_state = None
        self.size_renders_cache.clear()
        self.size_render_metrics.clear()
        self.size_illustrations.clear()
        self.render_cache_version = RENDER_CACHE_VERSION
        if new_dress_hash:
            self.last_dress_hash = new_dress_hash
        self.last_accessed_at = time.time()

    def set_last_good_render(self, render_bytes_or_base64: str, dress_hash: Optional[str] = None) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        payload = render_bytes_or_base64.encode("utf-8")
        self._encrypted_last_good_render = self._cipher.encrypt(payload)
        self.render_cache_version = RENDER_CACHE_VERSION
        if dress_hash and self.last_dress_hash != dress_hash:
            self.invalidate_outfit_cache(new_dress_hash=dress_hash)
        elif dress_hash:
            self.last_dress_hash = dress_hash
        self.last_accessed_at = time.time()

    def get_last_good_render(self, dress_hash: Optional[str] = None) -> Optional[str]:
        self._validate_cache_version()
        if not self._cipher or not self._encrypted_last_good_render:
            return None
        if dress_hash and self.last_dress_hash and self.last_dress_hash != dress_hash:
            return None
        self.last_accessed_at = time.time()
        decrypted = self._cipher.decrypt(self._encrypted_last_good_render)
        return decrypted.decode("utf-8")

    def set_canonical_render(self, render_b64: str, dress_hash: Optional[str] = None) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        payload = render_b64.encode("utf-8")
        self._encrypted_canonical_render = self._cipher.encrypt(payload)
        self.canonical_dress_hash = dress_hash or self.last_dress_hash
        self.render_cache_version = RENDER_CACHE_VERSION
        self.last_accessed_at = time.time()

    def get_canonical_render(self, dress_hash: Optional[str] = None) -> Optional[str]:
        self._validate_cache_version()
        if not self._cipher or not self._encrypted_canonical_render:
            return None
        if dress_hash and self.canonical_dress_hash and self.canonical_dress_hash != dress_hash:
            return None
        self.last_accessed_at = time.time()
        decrypted = self._cipher.decrypt(self._encrypted_canonical_render)
        return decrypted.decode("utf-8")

    def set_size_render(self, size_label: str, render_b64: str, dress_hash: Optional[str] = None,
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        target_hash = dress_hash or self.last_dress_hash or "default"
        key = f"{target_hash}:{size_label.upper().strip()}"
        self.size_renders_cache[key] = render_b64
        if metrics is not None:
            self.size_render_metrics[key] = metrics
        self.render_cache_version = RENDER_CACHE_VERSION
        self.last_accessed_at = time.time()

    def get_size_render(self, size_label: str, dress_hash: Optional[str] = None) -> Optional[str]:
        self._validate_cache_version()
        target_hash = dress_hash or self.last_dress_hash
        if not target_hash:
            return None
        key = f"{target_hash}:{size_label.upper().strip()}"
        self.last_accessed_at = time.time()
        return self.size_renders_cache.get(key)

    def get_size_render_metrics(self, dress_hash: str) -> Dict[str, Dict[str, Any]]:
        """{size: metrics} for every cached render of this garment that has silhouette metrics."""
        self._validate_cache_version()
        prefix = f"{dress_hash}:"
        return {k[len(prefix):]: v for k, v in self.size_render_metrics.items() if k.startswith(prefix)}

    def set_size_illustration(self, size_label: str, info: Dict[str, Any], dress_hash: Optional[str] = None) -> None:
        target_hash = dress_hash or self.last_dress_hash or "default"
        self.size_illustrations[f"{target_hash}:{size_label.upper().strip()}"] = info
        self.last_accessed_at = time.time()

    def get_size_illustration(self, size_label: str, dress_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self._validate_cache_version()
        target_hash = dress_hash or self.last_dress_hash
        if not target_hash:
            return None
        return self.size_illustrations.get(f"{target_hash}:{size_label.upper().strip()}")

    def set_identity_render(self, render_b64: str, capture_hash: str) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        self._encrypted_identity_render = self._cipher.encrypt(render_b64.encode("utf-8"))
        self.identity_capture_hash = capture_hash
        self.last_accessed_at = time.time()

    def get_identity_render(self, capture_hash: str) -> Optional[str]:
        if not self._cipher or not self._encrypted_identity_render or self.identity_capture_hash != capture_hash:
            return None
        self.last_accessed_at = time.time()
        return self._cipher.decrypt(self._encrypted_identity_render).decode("utf-8")

    def set_face_mask(self, mask: Any) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        import numpy as np
        arr = np.ascontiguousarray(mask, dtype=np.float32)
        meta = json.dumps({"shape": list(arr.shape)}).encode("utf-8")
        payload = meta + b":::RAW:::" + arr.tobytes()
        self._encrypted_face_mask = self._cipher.encrypt(payload)
        self.last_accessed_at = time.time()

    def get_face_mask(self) -> Optional[Any]:
        if not self._cipher or not self._encrypted_face_mask:
            return None
        import numpy as np
        self.last_accessed_at = time.time()
        decrypted = self._cipher.decrypt(self._encrypted_face_mask)
        parts = decrypted.split(b":::RAW:::", 1)
        if len(parts) == 2:
            meta = json.loads(parts[0].decode("utf-8"))
            return np.frombuffer(parts[1], dtype=np.float32).reshape(meta["shape"]).copy()
        return None

    def set_baseline_landmarks(self, landmarks: Dict[str, float], fit_state: Optional[str] = None) -> None:
        if not self._cipher:
            raise RuntimeError("Session has already been purged or expired.")
        payload = json.dumps(landmarks).encode("utf-8")
        self._encrypted_baseline_landmarks = self._cipher.encrypt(payload)
        self.baseline_fit_state = fit_state
        self.last_accessed_at = time.time()

    def get_baseline_landmarks(self) -> Optional[Dict[str, float]]:
        if not self._cipher or not self._encrypted_baseline_landmarks:
            return None
        self.last_accessed_at = time.time()
        decrypted = self._cipher.decrypt(self._encrypted_baseline_landmarks)
        return json.loads(decrypted.decode("utf-8"))

    def clear_baseline_landmarks(self) -> None:
        self._encrypted_baseline_landmarks = None
        self.baseline_fit_state = None

    def purge(self) -> None:
        """
        Cryptographic erasure: overwrite and destroy the session-specific encryption key
        and all encrypted biometric buffers from memory.
        """
        if self._encrypted_profile:
            buf = bytearray(self._encrypted_profile)
            for i in range(len(buf)):
                buf[i] = 0
            self._encrypted_profile = None

        if self._encrypted_measurements:
            buf = bytearray(self._encrypted_measurements)
            for i in range(len(buf)):
                buf[i] = 0
            self._encrypted_measurements = None

        if self._encrypted_last_good_render:
            buf = bytearray(self._encrypted_last_good_render)
            for i in range(len(buf)):
                buf[i] = 0
            self._encrypted_last_good_render = None

        if self._encrypted_face_mask:
            buf = bytearray(self._encrypted_face_mask)
            for i in range(len(buf)):
                buf[i] = 0
            self._encrypted_face_mask = None

        if self._encrypted_baseline_landmarks:
            buf = bytearray(self._encrypted_baseline_landmarks)
            for i in range(len(buf)):
                buf[i] = 0
            self._encrypted_baseline_landmarks = None
        self.baseline_fit_state = None

        self._encrypted_identity_render = None
        self.identity_capture_hash = None
        self._encrypted_canonical_render = None
        self.size_renders_cache.clear()
        self.size_render_metrics.clear()
        self.size_illustrations.clear()
        self.last_dress_hash = None

        if self._key:
            buf = bytearray(self._key)
            for i in range(len(buf)):
                buf[i] = 0
            self._key = None

        self._cipher = None


class SessionManager:
    """
    In-memory session registry with zero disk retention.
    Manages session lifecycle and automatic TTL eviction.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.RLock()  # Requests and the background sweep run on different threads

    def get_or_create_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            self.cleanup_expired()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionRecord(session_id)
                print(f"[SESSION SECURITY] Created new session with ephemeral Fernet key: {session_id[:8]}...")
            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            self.cleanup_expired()
            return self._sessions.get(session_id)

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def purge_session(self, session_id: str) -> bool:
        """
        Immediately purges session biometrics and destroys its unique encryption key.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.purge()
            print(f"[SESSION SECURITY] Purged session & destroyed encryption key for session: {session_id[:8]}...")
            return True
        return False

    def cleanup_expired(self) -> int:
        """Purges any sessions that have exceeded the TTL window."""
        now = time.time()
        with self._lock:
            expired_ids = [
                sid for sid, s in self._sessions.items()
                if (now - s.last_accessed_at) > SESSION_TTL_SECONDS
            ]
            for sid in expired_ids:
                self.purge_session(sid)
        return len(expired_ids)


# Global singleton in-memory manager
session_manager = SessionManager()
