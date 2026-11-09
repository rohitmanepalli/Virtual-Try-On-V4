"""
Visitor business records in a local SQLite file (data/visitors.db). Text and numbers only: no column can hold an image, and
every value is length-limited before it is written.

Every try-on is recorded anonymously. Name, phone and gender are saved only when the visitor gives them and consents, and
gender is only ever what the visitor chose, never what the appearance analysis inferred from their photo. Name and phone are
encrypted with VISITOR_DATA_KEY; without that key they are refused rather than stored readable. A phone number identifies a
returning visitor through a keyed hash, so it can be matched without being decrypted.

There is deliberately no HTTP route that reads this data: use tools/visitor_data.py on this PC.
"""
import hashlib
import hmac
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional

from cryptography.fernet import Fernet

from app.config import VISITOR_DATA_ENABLED, VISITOR_DB_PATH, VISITOR_DEFAULT_COUNTRY_CODE, get_visitor_data_key

GENDERS = ("female", "male", "non_binary", "prefer_not_to_say")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS visitors (
    visitor_id      TEXT PRIMARY KEY,
    name_enc        TEXT,
    phone_enc       TEXT,
    phone_lookup    TEXT UNIQUE,
    gender          TEXT CHECK (gender IS NULL OR gender IN ('female', 'male', 'non_binary', 'prefer_not_to_say')),
    consent_version TEXT NOT NULL,
    consent_at      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tryon_sessions (
    session_id       TEXT PRIMARY KEY,
    visitor_id       TEXT REFERENCES visitors (visitor_id) ON DELETE SET NULL,
    started_at       TEXT NOT NULL,
    last_activity_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS garments (
    garment_id        TEXT PRIMARY KEY,
    clothing_category TEXT,
    garment_name      TEXT
);
CREATE TABLE IF NOT EXISTS tryon_transactions (
    transaction_id   TEXT PRIMARY KEY,
    request_id       TEXT,
    session_id       TEXT REFERENCES tryon_sessions (session_id) ON DELETE SET NULL,
    garment_id       TEXT REFERENCES garments (garment_id),
    selected_size    TEXT,
    recommended_size TEXT,
    fit_assessment   TEXT,
    outcome          TEXT NOT NULL,
    attempts         INTEGER,
    duration_s       REAL,
    cost_usd         REAL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transactions_session ON tryon_transactions (session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_visitor ON tryon_sessions (visitor_id);
"""


class VisitorKeyMissing(Exception):
    """VISITOR_DATA_KEY is not set, so personal details cannot be stored encrypted."""


class InvalidPhone(ValueError):
    pass


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    folder = os.path.dirname(VISITOR_DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    connection = sqlite3.connect(VISITOR_DB_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        with connection:
            yield connection
    finally:
        connection.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _text(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    return " ".join(str(value).split())[:limit] or None


def _session_key(session_id: Optional[str]) -> Optional[str]:
    return session_id if session_id and re.fullmatch(r"[\w-]{1,80}", session_id) else None


def normalize_phone(phone: str) -> str:
    """Digits with country code, so the same number always matches. Raises InvalidPhone."""
    compact = re.sub(r"[\s()-]", "", phone)
    if not re.fullmatch(r"\+?\d{7,15}", compact):
        raise InvalidPhone("Please enter a valid phone number.")
    if compact.startswith("+"):
        return compact[1:]
    return VISITOR_DEFAULT_COUNTRY_CODE + compact if len(compact) == 10 else compact


def _cipher() -> Fernet:
    key = get_visitor_data_key()
    if not key:
        raise VisitorKeyMissing()
    return Fernet(key)


def _phone_lookup(phone: str) -> str:
    return hmac.new(get_visitor_data_key().encode(), phone.encode(), hashlib.sha256).hexdigest()


def _ensure_session(db: sqlite3.Connection, session_id: str) -> None:
    now = _now()
    db.execute("INSERT INTO tryon_sessions (session_id, started_at, last_activity_at) VALUES (?, ?, ?) "
               "ON CONFLICT (session_id) DO UPDATE SET last_activity_at = excluded.last_activity_at",
               (session_id, now, now))


def save_visitor(session_id: str, consent_version: str, name: Optional[str] = None, phone: Optional[str] = None,
                 gender: Optional[str] = None) -> str:
    """
    Saves consented details and links this session (and its try-ons) to the visitor. The same phone number reaches the
    same visitor_id on a later visit. Call only after the visitor has consented. Returns the visitor_id.
    """
    session = _session_key(session_id)
    if session is None:
        raise ValueError("invalid session")
    cipher = _cipher()
    name = _text(name, 100)
    normalized = normalize_phone(phone) if phone and phone.strip() else None
    gender = gender if gender in GENDERS else None
    consent = _text(consent_version, 40) or "unspecified"
    now = _now()
    with _connect() as db:
        _ensure_session(db, session)
        lookup = _phone_lookup(normalized) if normalized else None
        row = db.execute("SELECT visitor_id FROM visitors WHERE phone_lookup = ?", (lookup,)).fetchone() if lookup else None
        if row is None:
            row = db.execute("SELECT visitor_id FROM tryon_sessions WHERE session_id = ?", (session,)).fetchone()
        visitor_id = row["visitor_id"] if row is not None and row["visitor_id"] else None
        name_enc = cipher.encrypt(name.encode()).decode() if name else None
        phone_enc = cipher.encrypt(normalized.encode()).decode() if normalized else None
        if visitor_id is None:
            visitor_id = uuid.uuid4().hex
            db.execute("INSERT INTO visitors (visitor_id, name_enc, phone_enc, phone_lookup, gender, consent_version, "
                       "consent_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (visitor_id, name_enc, phone_enc, lookup, gender, consent, now, now, now))
        else:
            # A detail left blank this time keeps what the visitor gave before
            db.execute("UPDATE visitors SET name_enc = COALESCE(?, name_enc), phone_enc = COALESCE(?, phone_enc), "
                       "phone_lookup = COALESCE(?, phone_lookup), gender = COALESCE(?, gender), consent_version = ?, "
                       "consent_at = ?, updated_at = ? WHERE visitor_id = ?",
                       (name_enc, phone_enc, lookup, gender, consent, now, now, visitor_id))
        db.execute("UPDATE tryon_sessions SET visitor_id = ? WHERE session_id = ?", (visitor_id, session))
    return visitor_id


def record_tryon(session_id: Optional[str], request_id: Optional[str], outcome: str, garment_id: Optional[str] = None,
                 category: Optional[str] = None, garment_name: Optional[str] = None, selected_size: Optional[str] = None,
                 recommended_size: Optional[str] = None, fit_assessment: Optional[str] = None,
                 attempts: Optional[int] = None, duration_s: Optional[float] = None,
                 cost_usd: Optional[float] = None) -> None:
    """One row per /api/tryon request. A failure here is only logged: record keeping must never break a try-on."""
    if not VISITOR_DATA_ENABLED:
        return
    try:
        session = _session_key(session_id)
        garment = garment_id if garment_id and re.fullmatch(r"[\w-]{1,64}", garment_id) else None
        with _connect() as db:
            if session:
                _ensure_session(db, session)
            if garment:
                db.execute("INSERT INTO garments (garment_id, clothing_category, garment_name) VALUES (?, ?, ?) "
                           "ON CONFLICT (garment_id) DO UPDATE SET clothing_category = excluded.clothing_category, "
                           "garment_name = excluded.garment_name", (garment, _text(category, 60), _text(garment_name, 120)))
            db.execute("INSERT INTO tryon_transactions (transaction_id, request_id, session_id, garment_id, selected_size, "
                       "recommended_size, fit_assessment, outcome, attempts, duration_s, cost_usd, created_at) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (uuid.uuid4().hex, _text(request_id, 16), session, garment, _text(selected_size, 8),
                        _text(recommended_size, 8), _text(fit_assessment, 24), _text(outcome, 24) or "unknown",
                        attempts, duration_s, cost_usd, _now()))
    except Exception as err:
        print(f"[VISITOR DATA WARNING] Try-on not recorded: {type(err).__name__}")


def export_rows() -> List[Dict[str, object]]:
    """Every try-on with its visitor's decrypted details, for tools/visitor_data.py."""
    cipher = _cipher() if get_visitor_data_key() else None

    def reveal(value: Optional[str]) -> str:
        if not value:
            return ""
        return cipher.decrypt(value.encode()).decode() if cipher else "[VISITOR_DATA_KEY not set]"

    with _connect() as db:
        rows = db.execute(
            "SELECT t.created_at, t.request_id, s.session_id, s.started_at AS visit_started_at, v.visitor_id, v.name_enc, "
            "v.phone_enc, v.gender, v.consent_version, v.consent_at, g.garment_id, g.clothing_category, g.garment_name, "
            "t.selected_size, t.recommended_size, t.fit_assessment, t.outcome, t.attempts, t.duration_s, t.cost_usd "
            "FROM tryon_transactions t LEFT JOIN tryon_sessions s ON s.session_id = t.session_id "
            "LEFT JOIN visitors v ON v.visitor_id = s.visitor_id LEFT JOIN garments g ON g.garment_id = t.garment_id "
            "ORDER BY t.created_at").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["name"], item["phone"] = reveal(item.pop("name_enc")), reveal(item.pop("phone_enc"))
        result.append(item)
    return result


def delete_visitor_by_phone(phone: str) -> bool:
    """A visitor's deletion request: removes their details. Their try-ons stay as anonymous business records."""
    if not get_visitor_data_key():
        raise VisitorKeyMissing()
    with _connect() as db:
        return db.execute("DELETE FROM visitors WHERE phone_lookup = ?", (_phone_lookup(normalize_phone(phone)),)).rowcount > 0


def purge_older_than(days: int) -> Dict[str, int]:
    """Retention: removes visitors not updated, and try-ons not made, in the last `days` days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _connect() as db:
        visitors = db.execute("DELETE FROM visitors WHERE updated_at < ?", (cutoff,)).rowcount
        transactions = db.execute("DELETE FROM tryon_transactions WHERE created_at < ?", (cutoff,)).rowcount
        db.execute("DELETE FROM tryon_sessions WHERE session_id NOT IN "
                   "(SELECT session_id FROM tryon_transactions WHERE session_id IS NOT NULL) AND last_activity_at < ?", (cutoff,))
    return {"visitors": visitors, "transactions": transactions}
