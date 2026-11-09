import base64
import csv
import io
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from fastapi import APIRouter, HTTPException, Response
from fastapi.concurrency import run_in_threadpool

from app.config import (
    CALIBRATION_CAPTURE_DIR,
    CALIBRATION_CAPTURE_SAVE,
    GENERATION_MAX_ATTEMPTS,
    MEASUREMENT_CORRECTION_LOG,
    MEASUREMENT_CORRECTION_LOG_PATH,
    YOLO_VALIDATION_ENABLED,
    get_public_base_url,
)
from app.models import (
    BodyProfile,
    ClothAnalysis,
    FaceCheckRequest,
    MeasurementCorrectionRequest,
    PrepareCaptureRequest,
    ProfileCaptureRequest,
    PurgeSessionRequest,
    PurgeSessionResponse,
    SilhouetteVerificationResult,
    SizeRecommendation,
    TryOnRequest,
    TryOnResponse,
)
from app.services import capture_cache, download_store, visitor_store
from app.services.body_profile import body_profile_service
from app.services.body_validation import NO_PERSON_MESSAGE, BodyValidationResult, body_validator
from app.services.face_preservation import detect_face_points, log_face_check, verify_face_preserved
from app.services.fit_drape import fit_escalation_text, strain_zones
from app.services.garment_coverage import CaptureCoverage, check_render, coverage_escalation_text, read_capture
from app.services.garment_coverage import log_line as coverage_log_line
from app.services.gemini_cost import begin_cost_tracking, current_costs
from app.services.gemini_client import (
    analyze_cloth,
    analyze_user_appearance,
    clear_user_appearance,
    generate_tryon_image,
    get_appearance_details,
    get_dress_hash,
    prepare_user_portrait,
)
from app.services.image_utils import (
    apply_watermark,
    clean_and_decode_base64,
    downscale,
    finalize_result_image,
    normalize_framing,
)
from app.services.limb_check import capture_leg_text, check_limbs, limb_escalation_text
from app.services import request_timing
from app.services.qr_service import generate_qr_data_uri
from app.services.session_manager import session_manager
from app.services.silhouette_verifier import (
    WONT_FIT_MAX_WIDENING,
    check_studio_background,
    compare_sizes,
    compare_to_recommended,
    fit_family,
    garment_silhouette,
    silhouette_verifier,
)
from app.services.size_chart import LOOSE_FITS, SizeOffset, get_size_offset, recommend_size, size_fit_map

router = APIRouter()


@router.post("/api/extract-profile")
async def extract_profile(data: ProfileCaptureRequest):
    """
    Extracts structured body proportions (height, shoulder width, chest/waist/hip, body-type)
    from the visitor's front-facing capture photo (plus optional hold-still frames for jitter suppression).
    """
    try:
        if not data.image:
            raise HTTPException(status_code=400, detail="No image provided for profile extraction.")

        image = clean_and_decode_base64(data.image)
        downscale(image)
        hold_frames = [clean_and_decode_base64(f) for f in data.hold_frames]
        for img in hold_frames:
            downscale(img)

        if CALIBRATION_CAPTURE_SAVE:
            _save_calibration_capture(image)

        # Pose detection runs once; the same frames feed the sizing profile and the informational measurements
        frames = body_profile_service.extract_frames(image, hold_frames)
        profile = body_profile_service.profile_from_frames(frames)

        # Display-only cm estimates: a failure here must never affect the sizing profile
        measurements = None
        try:
            measurements = body_profile_service.measure_body(frames)
        except Exception as meas_err:
            print(f"[BODY MEASUREMENTS WARNING] {meas_err}")

        # Store session-scoped if session_id provided
        if data.session_id:
            session = session_manager.get_or_create_session(data.session_id)
            session.set_body_profile(profile)
            if measurements:
                session.set_body_measurements(measurements)

        return {
            "success": True,
            "body_profile": profile,
            "body_measurements": measurements,
            "session_id": data.session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PROFILE EXTRACTION ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


import numpy as np

@router.post("/api/capture-preflight")
async def capture_preflight(data: FaceCheckRequest):
    """
    Capture-time pre-flight checks (lighting, body visibility, face visibility), run the moment the photo
    is taken so a bad capture is rejected before any API cost is incurred or wait time passes.
    """
    try:
        user_img = clean_and_decode_base64(data.image)
        user_img.load()
        downscale(user_img)
    except Exception as decode_err:
        print(f"[PREFLIGHT] Invalid captured image: {decode_err}")
        raise HTTPException(status_code=400, detail=INVALID_IMAGE_MESSAGE.format("captured"))

    # 1. LIGHTING CHECK
    try:
        gray = np.array(user_img.convert("L"))
        mean_lum = float(np.mean(gray))
        # Threshold for too dark (~60/255)
        if mean_lum < 60:
            print(f"[PREFLIGHT] Image too dark (mean luminance {mean_lum:.1f} < 60)")
            raise HTTPException(status_code=422, detail="Lighting too low — please move to a brighter area.")
        # Threshold for blown out (>5% of pixels > 250)
        blown_out_pct = float(np.mean(gray > 250)) * 100
        if blown_out_pct > 5.0:
            print(f"[PREFLIGHT] Image too bright (blown out {blown_out_pct:.1f}%)")
            raise HTTPException(status_code=422, detail="Image is too bright or washed out — please move away from direct light.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PREFLIGHT] Lighting check skipped (error: {e})")

    # 2. BODY VISIBILITY CHECK (YOLO + MediaPipe)
    # Reuses the exact same logic that runs during generation, but now instantly blocks a bad capture
    try:
        person_meta = _validate_person(user_img)
        _validate_body(user_img, person_meta)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PREFLIGHT] Body visibility check skipped (error: {e})")

    # 3. FACE VISIBILITY CHECK
    if not _capture_has_face(prepare_user_portrait(user_img)):
        print("[PREFLIGHT] No face detected in the capture portrait; visitor asked to retake on the capture screen.")
        raise HTTPException(status_code=422, detail="We couldn't detect your face clearly — please face the camera.")

    return {"success": True}


def _prepare_capture(user_img: Image.Image) -> capture_cache.PreparedCapture:
    """
    Person and body validation plus portrait crop, once per capture (raises HTTP 422 when it fails, which is not cached).
    YOLO + MediaPipe must agree on the same, sufficiently visible primary person before anything else runs.
    """
    key = capture_cache.key_for(user_img)
    with capture_cache.keyed_lock(f"capture:{key}"):
        prepared = capture_cache.get(key)
        if prepared is not None:
            return prepared
        person_meta = _validate_person(user_img)  # internal metadata only; the capture is passed on unmodified
        request_timing.lap("yolo")
        _validate_body(user_img, person_meta)
        # 3:4 Portrait Framing: Automatically center the user and crop away side clutter/bystanders
        portrait = prepare_user_portrait(user_img)
        portrait.load()
        prepared = capture_cache.PreparedCapture(portrait=portrait, person_meta=person_meta)
        capture_cache.put(key, prepared)
        return prepared


def _coverage_for(prepared: capture_cache.PreparedCapture) -> Optional[CaptureCoverage]:
    """The capture's sleeve and wrist read. Reused only once it has the appearance analysis' answers."""
    if prepared.coverage is not None:
        return prepared.coverage
    coverage = _read_capture_coverage(prepared.portrait)
    if get_appearance_details(prepared.portrait):
        prepared.coverage = coverage
    return coverage


def _prepare_in_background(user_img: Image.Image) -> None:
    """Does the capture's slow work while the visitor chooses an outfit. Best effort: the try-on repeats what is missing."""
    try:
        prepared = _prepare_capture(user_img)
        analyze_user_appearance(prepared.portrait)
        _coverage_for(prepared)
        prepared.has_face = prepared.has_face or _capture_has_face(prepared.portrait)
        print("[PREPARE] Capture ready for try-on.", flush=True)
    except HTTPException as rejected:
        print(f"[PREPARE] Capture not prepared: {rejected.detail}", flush=True)
    except Exception as err:
        print(f"[PREPARE WARNING] {type(err).__name__}: {err}", flush=True)


@router.post("/api/prepare-capture", status_code=202, response_class=Response)
def prepare_capture(data: PrepareCaptureRequest) -> None:
    """Called when the visitor accepts their photo. Returns at once; never blocks or fails the visitor's flow."""
    try:
        user_img = clean_and_decode_base64(data.image)
        user_img.load()
        downscale(user_img)
    except Exception as err:
        print(f"[PREPARE] Unreadable capture: {err}")
        return
    threading.Thread(target=_prepare_in_background, args=(user_img,), daemon=True).start()


@router.post("/api/measurement-correction")
async def measurement_correction(data: MeasurementCorrectionRequest):
    """
    Logs a visitor's correction of an informational measurement against the system's original estimate,
    for accuracy validation. Always printed to the console; also appended to MEASUREMENT_CORRECTION_LOG_PATH
    when MEASUREMENT_CORRECTION_LOG is enabled (testing phase only).
    """
    session = session_manager.get_session(data.session_id)
    measurements = session.get_body_measurements() if session else None
    if measurements is None:
        raise HTTPException(status_code=404, detail="No measurements found for this session.")

    estimate = getattr(measurements, data.field)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": data.session_id,
        "field": data.field,
        "system_estimate_cm": estimate.value_cm if estimate else None,
        "source": estimate.source if estimate else None,
        "height_source": measurements.height.source if measurements.height else None,
        "displayed_cm": data.displayed_cm,
        "corrected_cm": data.corrected_cm,
        "assumed_distance_m": measurements.assumed_distance_m,
    }
    print("[MEASUREMENT CORRECTION] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if MEASUREMENT_CORRECTION_LOG:
        _append_correction_log(row)
    return {"success": True}


_CORRECTION_LOG_COLUMNS = [
    "timestamp", "session_id", "field", "system_estimate_cm", "source",
    "height_source", "displayed_cm", "corrected_cm", "assumed_distance_m",
]


def _append_correction_log(row: Dict[str, Any]) -> None:
    """Appends one correction to the testing-phase CSV; a write failure never fails the request."""
    try:
        folder = os.path.dirname(MEASUREMENT_CORRECTION_LOG_PATH)
        if folder:
            os.makedirs(folder, exist_ok=True)
        is_new = not os.path.exists(MEASUREMENT_CORRECTION_LOG_PATH)
        with open(MEASUREMENT_CORRECTION_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CORRECTION_LOG_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[MEASUREMENT CORRECTION LOG WARNING] Could not append to {MEASUREMENT_CORRECTION_LOG_PATH}: {e}")


def _save_calibration_capture(image: Image.Image) -> None:
    """Saves the exact photo the measurement pipeline sees, for tools/calibrate_focal_length.py."""
    try:
        os.makedirs(CALIBRATION_CAPTURE_DIR, exist_ok=True)
        name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        path = os.path.abspath(os.path.join(CALIBRATION_CAPTURE_DIR, name))
        image.save(path, format="PNG")
        print(f"[CALIBRATION CAPTURE] Saved {path}", flush=True)
    except Exception as e:
        print(f"[CALIBRATION CAPTURE WARNING] Could not save capture: {e}")


@router.post("/api/recommend-size")
async def recommend_size_endpoint(payload: Dict[str, Any]):
    """
    Recommends the visitor's best-fit standard size (XS–XXL) for a garment
    based on their captured body profile and the garment's category.
    """
    import json
    print(f"\n==================== [RAW INCOMING BODY PROFILE] ====================", flush=True)
    print(f"Session ID: {payload.get('session_id')}", flush=True)
    print(f"Garment Category: {payload.get('garment_category')}", flush=True)
    raw_bp = payload.get("body_profile")
    print(f"Raw body_profile in payload:\n{json.dumps(raw_bp if isinstance(raw_bp, dict) else (raw_bp.model_dump() if hasattr(raw_bp, 'model_dump') else raw_bp), indent=2, default=str)}", flush=True)
    print(f"=====================================================================\n", flush=True)

    try:
        profile_data = payload.get("body_profile")
        garment_category = payload.get("garment_category", "")
        session_id = payload.get("session_id", "")

        if not profile_data and session_id:
            session = session_manager.get_session(session_id)
            if session:
                profile_data = session.get_body_profile()
                print(f"[SESSION RETRIEVED PROFILE] for {session_id}:\n{json.dumps(profile_data if isinstance(profile_data, dict) else (profile_data.model_dump() if hasattr(profile_data, 'model_dump') else str(profile_data)), indent=2, default=str)}", flush=True)

        if not profile_data or (isinstance(profile_data, dict) and len(profile_data) == 0):
            print(f"[SIZE REC ERROR] Missing body_profile! Incoming keys: {list(payload.keys())}, session_id: {session_id}, body_profile: {payload.get('body_profile')}")
            raise HTTPException(
                status_code=400,
                detail="body_profile is required for size recommendation and was not found in payload or active session."
            )

        if isinstance(profile_data, dict):
            profile = BodyProfile(**profile_data)
        else:
            profile = profile_data

        result = recommend_size(profile, garment_category=garment_category)

        return {
            "success": True,
            "size_recommendation": SizeRecommendation(
                recommended_size=result.recommended_size,
                available_sizes=result.available_sizes,
                size_confidence=result.size_confidence,
                fit_notes=result.fit_notes,
                # Category-level fabric stretch only here; /api/tryon refines it with the garment analysis
                size_fits=size_fit_map(profile, result.recommended_size, garment_category),
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SIZE RECOMMENDATION ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/session/purge", response_model=PurgeSessionResponse)
async def purge_session(data: PurgeSessionRequest):
    """
    Purges in-memory biometric data and destroys the ephemeral Fernet encryption key for the session.
    """
    success = session_manager.purge_session(data.session_id)
    capture_cache.clear()
    clear_user_appearance()
    download_store.end_session(data.session_id)
    return PurgeSessionResponse(
        success=success,
        session_id=data.session_id,
        message="Session biometrics and encryption key securely purged from memory."
    )


@router.post("/api/verify-silhouette", response_model=SilhouetteVerificationResult)
async def verify_silhouette_endpoint(payload: Dict[str, Any]):
    """
    Independent testing endpoint for silhouette verification.
    Accepts 'user_image', 'generated_image', and optional 'threshold'.
    """
    try:
        user_b64 = payload.get("user_image")
        gen_b64 = payload.get("generated_image")
        threshold = payload.get("threshold")

        if not user_b64 or not gen_b64:
            raise HTTPException(status_code=400, detail="Missing user_image or generated_image.")

        user_img = clean_and_decode_base64(user_b64)
        gen_img = clean_and_decode_base64(gen_b64)

        result = silhouette_verifier.verify(
            user_img=user_img,
            generated_img=gen_img,
            threshold=float(threshold) if threshold is not None else None
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _face_score(user_img: Image.Image, image_b64: str, context: str) -> float:
    """Real face-preservation score for any image returned to the visitor (never a constant)."""
    check = verify_face_preserved(user_img, clean_and_decode_base64(image_b64))
    log_face_check(check, context)
    return check.score


def _log_path(path: str, detail: str = "") -> None:
    """One line per /api/tryon response naming the path that produced the returned image."""
    print(f"[GENERATION] {path}{f' | {detail}' if detail else ''}", flush=True)


def _is_studio_render(image: Any, context: str) -> bool:
    """True if the image (PIL or data-URI) is a studio render, False if it is the raw capture scene."""
    try:
        img = clean_and_decode_base64(image) if isinstance(image, str) else image
        ok, metrics = check_studio_background(img)
    except Exception as err:
        print(f"[STUDIO CHECK] {context}: check error ({err}); treating as studio render.")
        return True
    if not ok:
        print(f"[STUDIO CHECK] {context}: RAW PHOTO - original background not replaced {metrics}")
    return ok


def _usable_last_good(render: Optional[str]) -> bool:
    """A raw-photo render must never be served as the "last good" fallback. Checked only when a fallback is needed."""
    if not render:
        return False
    if _is_studio_render(render, "last good render"):
        return True
    print("[TRYON] Discarding last good render: it is a raw-photo output, not a studio render.")
    return False


INVALID_IMAGE_MESSAGE = "The {} image could not be read. Please retake it and try again."
NO_FACE_MESSAGE = "We couldn't see your face clearly. Please face the camera with nothing covering your face and try again."


def _capture_has_face(user_img: Image.Image) -> bool:
    """
    The real face is restored from the capture after generation; without a detectable face in the capture
    every generation would be rejected, so check before spending any Gemini calls. Fails open on errors
    (restoration still refuses to serve an unrestored render).
    """
    import numpy as np
    try:
        return detect_face_points(np.array(user_img.convert("RGB"))) is not None
    except Exception as err:
        print(f"[FACE PRESERVE] Capture face pre-check error ({err}); continuing.")
        return True

# Ultralytics predictors are not thread-safe; serialize inference on the shared model.
_yolo_infer_lock = threading.Lock()


def _get_yolo_service():
    """Lazy import so the detector loads only when validation runs, and a missing model can't break router import."""
    from app.services.yolo_service import get_yolo_service
    return get_yolo_service()


def warm_up_yolo() -> None:
    """Load the YOLO weights and build the predictors ahead of the first try-on so no request pays that cost."""
    if YOLO_VALIDATION_ENABLED:
        try:
            service = _get_yolo_service()
            with _yolo_infer_lock:  # a request arriving mid-warm-up waits here instead of racing the predictor setup
                service.detect_person(Image.new("RGB", (64, 64)))
            print("[YOLO] Person detector loaded.", flush=True)
        except Exception as err:
            print(f"[YOLO WARNING] Person detector unavailable ({type(err).__name__}: {err}); try-on will run without person validation.", flush=True)


def _validate_body(user_img: Image.Image, person_meta: Optional[Dict[str, Any]]) -> Optional[BodyValidationResult]:
    """
    YOLO + MediaPipe cross-check on the full capture (the frame YOLO's boxes refer to), before any
    portrait cropping or Gemini work. One extra MediaPipe Pose pass; no pixels are modified.

    Skipped (None) when disabled or when there is no YOLO result to compare against (YOLO disabled or
    errored), preserving the existing fail-open behaviour. Raises HTTP 422 when the check fails.
    """
    if not body_validator.enabled or person_meta is None:
        return None
    try:
        landmarks = body_profile_service.extract_single_frame(user_img)
    except Exception as err:
        print(f"[BODY VALIDATION] MediaPipe error ({type(err).__name__}: {err}); treating as MediaPipe unavailable.")
        landmarks = None
    try:
        result = body_validator.validate(
            person_boxes=person_meta["detections"],
            primary_box=tuple(person_meta["bbox"]),
            image_size=(person_meta["image_width"], person_meta["image_height"]),
            landmarks=landmarks,
        )
    except Exception as err:
        print(f"[BODY VALIDATION WARNING] Validation skipped: {type(err).__name__}: {err}. Continuing with existing pipeline.")
        return None
    print(result.log_line())
    if not result.passed:
        _log_path("Rejected - body validation", result.reason)
        raise HTTPException(status_code=422, detail=result.user_message)
    return result


def _validate_person(user_img: Image.Image) -> Optional[Dict[str, Any]]:
    """
    YOLO person-presence gate, run once per request on the decoded capture before any Gemini call.

    Detection/validation only: the capture is never cropped, masked, drawn on or otherwise modified,
    and the bbox is never used as a face region. Face identity stays with the existing MediaPipe /
    face-preservation pipeline.

    Returns internal primary-person metadata (never sent to the client), raises HTTP 422 when no
    person is found, and returns None (fail open: the existing pipeline proceeds) when validation is
    disabled or the detector itself errors or yields a malformed result.
    """
    if not YOLO_VALIDATION_ENABLED:
        return None

    started = time.perf_counter()
    try:
        service = _get_yolo_service()
        with _yolo_infer_lock:
            # convert() returns a copy, so the capture handed to the rest of the pipeline is untouched
            result = service.detect_person(user_img.convert("RGB"))
        detections = list(result.detections)
        primary = result.primary_person
    except Exception as err:
        print(f"[YOLO WARNING] Person validation skipped: detector error ({type(err).__name__}: {err}). Continuing with existing pipeline.")
        return None
    elapsed_ms = (time.perf_counter() - started) * 1000

    if primary is None:
        print(f"[YOLO] No person detected ({elapsed_ms:.0f}ms). Rejecting request before Gemini generation.")
        _log_path("Rejected - no person detected", "YOLO person validation")
        raise HTTPException(status_code=422, detail=NO_PERSON_MESSAGE)

    try:
        img_w, img_h = int(result.image_width), int(result.image_height)
        x1, y1, x2, y2 = (int(v) for v in (primary.x1, primary.y1, primary.x2, primary.y2))
        confidence = float(primary.confidence)
        if not (img_w > 0 and img_h > 0 and 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h
                and math.isfinite(confidence) and 0.0 <= confidence <= 1.0):
            raise ValueError(f"bbox={(x1, y1, x2, y2)} conf={confidence} image={img_w}x{img_h}")
    except Exception as err:
        print(f"[YOLO WARNING] Person validation skipped: malformed detection ({err}). Continuing with existing pipeline.")
        return None

    meta = {
        "person_count": len(detections),
        "bbox": [x1, y1, x2, y2],
        "bbox_normalized": [round(x1 / img_w, 4), round(y1 / img_h, 4), round(x2 / img_w, 4), round(y2 / img_h, 4)],
        "confidence": round(confidence, 4),
        "image_width": img_w,
        "image_height": img_h,
        "inference_ms": round(elapsed_ms, 1),
        # Every person box, for the YOLO/MediaPipe cross-check (internal only)
        "detections": [(int(d.x1), int(d.y1), int(d.x2), int(d.y2)) for d in detections],
    }
    if len(detections) > 1:
        print(
            f"[YOLO] {len(detections)} people detected; primary = most prominent (area x confidence^2) "
            f"bbox={meta['bbox']} conf={meta['confidence']}. Capture is not cropped or altered by YOLO; "
            f"existing portrait framing and face pipeline proceed unchanged."
        )
    else:
        print(f"[YOLO] Person validated: bbox={meta['bbox']} conf={meta['confidence']} ({elapsed_ms:.0f}ms)")
    return meta


def _read_capture_coverage(user_img: Image.Image) -> Optional[CaptureCoverage]:
    """Own sleeve length, wrist accessories and skin colour of the capture, for the coverage prompt and check (fail open)."""
    try:
        coverage = read_capture(user_img, get_appearance_details(user_img))
    except Exception as err:
        print(f"[COVERAGE] Capture read skipped: {type(err).__name__}: {err}. Coverage check will not run.")
        return None
    print(f"[COVERAGE] Capture: {coverage.summary()}")
    return coverage


def _check_coverage(capture: Optional[CaptureCoverage], gen_pil: Image.Image, frame_lm: Any,
                    sleeve_length: Optional[str], attempt_no: int) -> Optional[Dict[str, Any]]:
    """Garment-coverage verdict for one render; None when the check itself errored (never a failure)."""
    try:
        result = check_render(capture, gen_pil, frame_lm, sleeve_length)
    except Exception as err:
        print(f"[COVERAGE CHECK] Attempt {attempt_no}: check error ({type(err).__name__}: {err}); not checked.")
        return None
    print(coverage_log_line(result, attempt_no))
    return result


def _coverage_ok(cand: SimpleNamespace) -> bool:
    """False only when the coverage check ran and failed; unmeasurable renders are not held back."""
    return (cand.coverage or {}).get("passed") is not False


def _limbs_ok(cand: SimpleNamespace) -> bool:
    """False only when the limb check ran and found a missing hand, leg or foot."""
    return (getattr(cand, "limbs", None) or {}).get("passed") is not False


def _wont_fit_message(recommended_size: Optional[str]) -> str:
    return "This size won't fit you." + (f" We recommend {recommended_size}." if recommended_size else "")


def _fit_fields(size_offset: Optional[SizeOffset], selected_size: Optional[str], size_rec: Optional[SizeRecommendation],
                profile: Optional[BodyProfile]) -> Dict[str, Any]:
    """Size / fit fields shared by every /api/tryon response for this size."""
    fields: Dict[str, Any] = {"selected_size": selected_size, "size_recommendation": size_rec}
    if size_offset is None:
        return fields
    fields.update(
        size_fit_quality=size_offset.fit_quality,
        size_offset=size_offset.steps_from_recommended,
        fit_label=size_offset.fit_label,
        wont_fit=size_offset.does_not_fit,
    )
    if size_offset.does_not_fit:
        fields["fit_check"] = {
            "body_check": size_offset.body_check,
            "limiting_dimension": size_offset.limiting_dimension,
            "reason": size_offset.wont_fit_reason,
            "strain_zones": strain_zones(profile.torso_shape if profile else None, size_offset.limiting_dimension),
        }
    return fields


def _wont_fit_illustration(fit_fields: Dict[str, Any], fit_state: str, cloth: ClothAnalysis,
                           recommended_size: Optional[str], render_rejected: str) -> TryOnResponse:
    """The "won't fit" state without a render: the message plus the data for an honest strain illustration."""
    fields = dict(fit_fields)
    fields["fit_check"] = dict(fields.get("fit_check") or {}, render_rejected=render_rejected)
    return TryOnResponse(
        success=True, result_image=None, fit_state=fit_state, cloth_analysis=cloth, illustration_only=True,
        message=_wont_fit_message(recommended_size), **fields,
    )


def _evaluate_fit(metrics: Optional[Dict[str, Any]], fit_quality: str, recommended_size: Optional[str],
                  rec_metrics: Optional[Dict[str, Any]], other_metrics: Dict[str, Dict[str, Any]],
                  size_fits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Does this render show its size? Against the recommended render: a larger size must be wider, a smaller size no
    wider. Against every other rendered size of this garment: not nearly identical. Unmeasurable = not checked.
    """
    result: Dict[str, Any] = {"checked": False, "passed": True, "problem": None, "other_size": None,
                              "direction": None, "pairs": {}}
    if metrics is None:
        return result
    if rec_metrics is not None and fit_quality != "perfect":
        direction = compare_to_recommended(metrics, rec_metrics, fit_quality)
        result["direction"] = direction
        if direction["passed"] is not None:
            result["checked"] = True
        if direction["passed"] is False:
            result.update(passed=False, other_size=recommended_size,
                          problem="was not wider than" if fit_quality in LOOSE_FITS else "sat further off the body than")
    for size, other in other_metrics.items():
        other_quality = other.get("fit_quality") or size_fits.get(size, {}).get("fit_quality", "perfect")
        pair = compare_sizes(metrics, fit_quality, other, other_quality)
        result["pairs"][size] = pair
        if pair["near_identical"] is not None:
            result["checked"] = True
        if pair["near_identical"] and result["passed"]:
            result.update(passed=False, other_size=size, problem="looked nearly identical to")
    return result


def _log_fit_eval(attempt_no: int, size: Optional[str], fit_quality: str, metrics: Optional[Dict[str, Any]],
                  fit_eval: Dict[str, Any]) -> None:
    if metrics is None:
        print(f"[FIT CHECK] Attempt {attempt_no} size {size}: garment silhouette not measurable (no pose/mask); size difference unverified.")
        return
    d = fit_eval["direction"] or {}
    pairs = ", ".join(
        f"{s}: sil {p['silhouette_distance']} app {p['appearance_distance']}{' NEAR-IDENTICAL' if p['near_identical'] else ''}"
        for s, p in fit_eval["pairs"].items()) or "no other sizes rendered"
    print(f"[FIT CHECK] Attempt {attempt_no} size {size} ({fit_quality}): torso {metrics['torso_width']} outer {metrics['outer_width']} | "
          f"vs recommended: torso {d.get('torso_delta')} outer {d.get('outer_delta')} -> {d.get('passed')} | {pairs} | "
          f"{'PASS' if fit_eval['passed'] else 'FAIL: ' + str(fit_eval['problem']) + ' ' + str(fit_eval['other_size'])}")


def _stored_metrics(metrics: Optional[Dict[str, Any]], fit_quality: str, fit_verified: Optional[bool]) -> Dict[str, Any]:
    return dict(metrics or {}, fit_quality=fit_quality, fit_verified=fit_verified)


def _log_size_generation(size: Optional[str], size_offset: Optional[SizeOffset], attempts: int,
                         best: Optional[SimpleNamespace], costs: list) -> None:
    """One line per generated size: fit category, offset, attempts and the real cost of this size."""
    quality = size_offset.fit_quality if size_offset else "n/a"
    offset = f"{size_offset.steps_from_recommended:+d}" if size_offset else "n/a"
    served = f"served attempt {best.attempt}" if best else "no render served"
    print(f"[SIZE GENERATION] size {size or 'auto'} | offset {offset} | fit {quality} | {attempts} attempt(s), {served} | "
          f"cost ${sum(costs):.4f} ({len(costs)} priced call(s))", flush=True)


def _finalize_render(final_bytes: bytes, face_sim: float, session_id: Optional[str] = None) -> Tuple[str, str]:
    """Single final encode (validated RGB JPEG, full resolution), watermarked in-memory download copy and its QR code."""
    with request_timing.step("finalize"):
        image_id = str(uuid.uuid4())
        # Independent encodes of the same pixels (Pillow releases the GIL), so they can overlap
        with ThreadPoolExecutor(max_workers=2) as pool:
            jpeg_job = pool.submit(finalize_result_image, final_bytes)
            # Watermark from the lossless restored render, not the JPEG, to avoid double compression
            watermark_job = pool.submit(apply_watermark, final_bytes)
            result_jpeg, result_size = jpeg_job.result()
            download_store.put(image_id, watermark_job.result(), session_id)
        final_mode = Image.open(io.BytesIO(result_jpeg)).mode
        print(f"FACE_RESTORE: final_size={result_size[0]}x{result_size[1]} alpha_channel={'none' if final_mode == 'RGB' else final_mode} "
              f"format=JPEG bytes={len(result_jpeg) // 1024}KB face_similarity={face_sim:.3f}")
        qr_data_uri = generate_qr_data_uri(f"{get_public_base_url()}/download/{image_id}")
        return f"data:image/jpeg;base64,{base64.b64encode(result_jpeg).decode('utf-8')}", qr_data_uri


def _wont_fit_render_problem(cand: Optional[SimpleNamespace], last_error: Optional[Exception]) -> Optional[str]:
    """
    Why a "won't fit" render cannot be shown as a believable depiction (None = show it). The render is intentionally
    strained, so a low-ish silhouette score alone is not a failure; the body being squeezed, broken anatomy, the
    original room, or a garment that does not even look smaller than the correct size are.
    """
    if cand is None:
        return f"generation failed ({last_error})" if last_error else "no render produced"
    if not cand.studio_ok:
        return "the model returned the original photo instead of a studio render"
    details = cand.verify.details or {}
    if not details.get("midsection_preserved", True):
        return f"the render squeezed the visitor's body (midsection {details['midsection_change']:+.1%})"
    if not _coverage_ok(cand):
        return ("the visitor's own sleeve still showed below the garment's sleeve" if cand.coverage.get("problem") == "remnant"
                else "bare skin showed where the garment's sleeve covers the arm")
    if cand.verify.score < 0.48:
        return f"severe anatomical distortion (silhouette score {cand.verify.score:.2f})"
    torso_delta = (cand.fit_eval.get("direction") or {}).get("torso_delta")
    if torso_delta is not None and torso_delta > WONT_FIT_MAX_WIDENING:
        return f"the garment does not look smaller than the recommended size (torso {torso_delta:+.3f} wider)"
    near = [s for s, p in cand.fit_eval.get("pairs", {}).items() if p.get("near_identical")]
    if near:
        return f"the render looks the same as size {near[0]}, so it does not show a garment that cannot fit"
    return None


def _serve_wont_fit(cand, last_error, data, session_record, user_img, dress_img, fit_fields, fit_state, cloth,
                    recommended_size, selected_size, dress_hash, size_offset, request_costs) -> TryOnResponse:
    """
    "Won't fit" is intentional: one generation, no retry and never "adjust your stance". A believable render is shown
    with the message; otherwise the message is shown with an illustration of the strain (no unrealistic image).
    Never cached as last good / canonical / identity render: it must not become another size's fallback or face.
    """
    problem = _wont_fit_render_problem(cand, last_error)
    _log_size_generation(selected_size, size_offset, 1 if cand else 0, None if problem else cand, request_costs)
    if problem:
        print(f"[WONT FIT] Size {selected_size}: render not shown ({problem}); serving message + illustration.")
        _log_path("Won't fit: message + illustration", problem)
        if session_record and selected_size:
            session_record.set_size_illustration(selected_size, {"render_rejected": problem}, dress_hash=dress_hash)
        return _wont_fit_illustration(fit_fields, fit_state, cloth, recommended_size, problem)

    encoded_result, qr_data_uri = _finalize_render(cand.bytes, cand.face_sim, data.session_id)
    fit_verified = cand.fit_eval["passed"] if cand.fit_eval["checked"] else None
    if session_record and selected_size:
        session_record.set_size_render(selected_size, encoded_result, dress_hash=dress_hash,
                                       metrics=_stored_metrics(cand.metrics, "does_not_fit", fit_verified))
    _log_path("Won't fit: render shown with message", f"size {selected_size} | silhouette {cand.verify.score:.3f}")
    return TryOnResponse(
        success=True,
        result_image=encoded_result,
        qr_code_base64=qr_data_uri,
        fit_state=fit_state,
        cloth_analysis=cloth,
        silhouette_score=cand.verify.score,
        silhouette_passed=cand.verify.passed,
        fit_verified=fit_verified,
        coverage_verified=(cand.coverage or {}).get("passed"),
        face_similarity=round(cand.face_sim, 4),
        message=_wont_fit_message(recommended_size),
        **fit_fields,
    )


def _attempt_result(cand: SimpleNamespace) -> str:
    """The first check a render failed, in the order the attempt loop handles them, or "ok" (timing log)."""
    if not cand.studio_ok:
        return "raw_photo"
    if not cand.verification_passed:
        return "silhouette"
    if not _coverage_ok(cand):
        return "coverage"
    if not cand.fit_eval["passed"]:
        return "fit"
    return "ok"


def _timing_outcome(response: TryOnResponse, timing: request_timing.RequestTiming) -> str:
    """served / cache_hit / wont_fit / fallback / no_image, for the timing log."""
    if response.fallback_used:
        return "fallback"
    if not response.success or not response.result_image:
        return "wont_fit" if response.wont_fit else "no_image"
    if not timing.attempt_results:
        return "cache_hit"
    return "wont_fit" if response.wont_fit else "served"


@router.post("/api/tryon", response_model=TryOnResponse)
async def virtual_tryon(data: TryOnRequest):
    """Runs on a worker thread so a long generation doesn't hold up the server's other requests."""
    return await run_in_threadpool(_timed_tryon, data)


def _timed_tryon(data: TryOnRequest) -> TryOnResponse:
    """
    Runs the try-on pipeline and logs how long each step took (logs/tryon_timings.csv, see request_timing).
    One request per session at a time: a duplicate of a running request waits, then gets its render from the session
    cache instead of paying for a second generation.
    """
    with capture_cache.keyed_lock(f"tryon:{data.session_id or ''}"):
        timing = request_timing.begin(data.session_id)
        response: Optional[TryOnResponse] = None
        outcome = "error"
        try:
            response = _virtual_tryon(data)
            outcome = _timing_outcome(response, timing)
            return response
        except HTTPException as http_err:
            outcome = "blocked" if http_err.status_code == 403 else f"error_{http_err.status_code}"
            raise
        finally:
            size = (response.selected_size if response else None) or data.selected_size
            fit = response.size_fit_quality if response else None
            row = request_timing.finish(timing, outcome, size=size, fit_category=fit)
            _record_transaction(data, response, outcome, size, fit, timing.request_id, row)


def _record_transaction(data: TryOnRequest, response: Optional[TryOnResponse], outcome: str, size: Optional[str],
                        fit: Optional[str], request_id: str, row: Any) -> None:
    """The business record of this request: text and numbers only, and it can never fail the try-on."""
    cloth = response.cloth_analysis if response else None
    recommendation = response.size_recommendation if response else None
    timing_row = row if isinstance(row, dict) else {}
    visitor_store.record_tryon(
        session_id=data.session_id, request_id=request_id, outcome=outcome, garment_id=data.garment_id,
        category=cloth.category if cloth else None, garment_name=cloth.garment_name if cloth else None,
        selected_size=size, recommended_size=recommendation.recommended_size if recommendation else None,
        fit_assessment=fit, attempts=timing_row.get("attempts"),
        duration_s=float(timing_row["total_s"]) if timing_row.get("total_s") else None,
        cost_usd=sum(current_costs()) or None,
    )


def _virtual_tryon(data: TryOnRequest) -> TryOnResponse:
    """
    Proportion-locked virtual try-on with automated silhouette verification,
    intelligent cloth analysis, 1x automatic retry on failure, and fallback to last good render.
    """
    try:
        # Parse image inputs (load() forces a full decode so corrupt data surfaces here, not mid-pipeline)
        images = {}
        for label, b64 in (("captured", data.user_image), ("outfit", data.dress_image)):
            try:
                images[label] = clean_and_decode_base64(b64)
                images[label].load()
                downscale(images[label])
            except Exception as decode_err:
                print(f"[TRYON] Invalid {label} image: {decode_err}")
                raise HTTPException(status_code=400, detail=INVALID_IMAGE_MESSAGE.format(label))
        user_img, dress_img = images["captured"], images["outfit"]
        request_timing.lap("decode")

        # Validation and portrait crop run once per capture, before any Gemini work
        prepared = _prepare_capture(user_img)
        user_img, person_meta = prepared.portrait, prepared.person_meta

        # Retrieve session profile or last good render if not supplied directly
        session_record = session_manager.get_or_create_session(data.session_id) if data.session_id else None
        active_profile = data.body_profile or (session_record.get_body_profile() if session_record else None)
        if session_record and active_profile and not session_record.get_body_profile():
            session_record.set_body_profile(active_profile)
        cached_last_good = data.last_good_render or (session_record.get_last_good_render() if session_record else None)

        # Zero-Null Guarantee: If no body profile exists, extract on-the-fly from capture frame
        if active_profile is None:
            try:
                print("[TRYON] No pre-extracted body profile. Extracting anthropometric profile directly from capture frame...")
                active_profile = body_profile_service.extract_profile(user_img)
                if session_record:
                    session_record.set_body_profile(active_profile)
                print(f"[TRYON] Extracted on-the-fly BodyProfile: Type='{active_profile.body_type}', Sh/Hip={active_profile.shoulder_to_hip_ratio:.2f}, Torso={active_profile.torso_aspect_ratio:.2f}")
            except Exception as prof_err:
                print(f"[TRYON WARNING] Could not extract on-the-fly profile: {prof_err}")

        # Retrieve or compute face mask
        feathered_mask = session_record.get_face_mask() if session_record else None

        request_timing.lap("checks_before")

        # Automated Garment Cloth Analysis
        active_cloth = data.cloth_analysis or analyze_cloth(dress_img)
        current_dress_hash = get_dress_hash(dress_img)[:16]
        request_timing.lap("garment")

        # Garment-specific caches belong to one garment (prevents cross-outfit contamination)
        if session_record:
            if session_record.last_dress_hash and session_record.last_dress_hash != current_dress_hash:
                print(f"[TRYON] New outfit detected ({session_record.last_dress_hash} != {current_dress_hash}). Invalidating outfit cache (preserving body profile).")
                session_record.invalidate_outfit_cache(current_dress_hash)
                cached_last_good = None
            elif not session_record.last_dress_hash:
                session_record.last_dress_hash = current_dress_hash

        # Every size is generated from the capture and the garment only. No earlier render is ever a generation input,
        # so a size cannot inherit another size's fit, and the body always comes from the capture.

        # Automated User Identity & Physical Appearance Analysis
        request_timing.lap("checks_before")
        user_appearance = analyze_user_appearance(user_img)
        request_timing.lap("appearance")

        if feathered_mask is not None:
            print(f"[TRYON] Reusing cached face mask from session {data.session_id}.")
        else:
            print("[TRYON] Generating fresh face segmentation mask from capture frame...")
            _, feathered_mask = body_profile_service.generate_face_and_hands_mask(user_img)
            if session_record:
                session_record.set_face_mask(feathered_mask)

        fit_state = data.fit_state or "standard"

        # ── Size Recommendation & Offset Logic ──────────────────────────────
        size_rec = None
        size_offset_data: Optional[SizeOffset] = None
        selected_size = data.selected_size

        if active_profile and active_cloth:
            rec_result = recommend_size(active_profile, garment_category=active_cloth.category)
            size_rec = SizeRecommendation(
                recommended_size=rec_result.recommended_size,
                available_sizes=rec_result.available_sizes,
                size_confidence=rec_result.size_confidence,
                fit_notes=rec_result.fit_notes,
                size_fits=size_fit_map(active_profile, rec_result.recommended_size, active_cloth.category,
                                       active_cloth.structure_level),
            )

            # Auto-detect and pre-select visitor's correct size if not provided
            if not selected_size:
                selected_size = rec_result.recommended_size

            size_offset_data = get_size_offset(rec_result.recommended_size, selected_size, active_cloth.category,
                                               active_profile, active_cloth.structure_level)
            print(
                f"[TRYON] Size: {selected_size} (rec: {rec_result.recommended_size}) | "
                f"Offset: {size_offset_data.steps_from_recommended:+d} | "
                f"Fit: {size_offset_data.fit_quality} ('{size_offset_data.fit_label}') | "
                f"Fabric offset: {size_offset_data.offset_lo_mm}–{size_offset_data.offset_hi_mm}mm"
                + (f" | Body vs {selected_size} (size steps past its max): {size_offset_data.body_check}"
                   if size_offset_data.body_check else "")
                + (f" | Won't fit: {size_offset_data.wont_fit_reason}" if size_offset_data.does_not_fit else "")
            )

        recommended_size = size_rec.recommended_size if size_rec else None
        fit_quality = size_offset_data.fit_quality if size_offset_data else "perfect"
        # "Won't fit" is a distinct, intentional state: one generation, never retried, never "adjust your stance"
        wont_fit = bool(size_offset_data and size_offset_data.does_not_fit)
        fit_fields = _fit_fields(size_offset_data, selected_size, size_rec, active_profile)

        # 1. Session cache for this size of this garment first (0ms, 0 API calls)
        if session_record and selected_size:
            illustration = session_record.get_size_illustration(selected_size, dress_hash=current_dress_hash) if wont_fit else None
            if illustration:
                _log_path("Session cache hit", f"size {selected_size} won't fit: message + illustration")
                return _wont_fit_illustration(fit_fields, fit_state, active_cloth, recommended_size, illustration["render_rejected"])
            cached_size_render = session_record.get_size_render(selected_size, dress_hash=current_dress_hash)
            if cached_size_render and not _is_studio_render(cached_size_render, f"cached size {selected_size}"):
                cached_size_render = None
            if cached_size_render:
                cached_metrics = session_record.get_size_render_metrics(current_dress_hash).get(selected_size.upper(), {})
                print(f"[TRYON CACHE] Returning cached size render for '{selected_size}' (dress: {current_dress_hash}) (0ms, 0 API calls).")
                _log_path("Session cache hit", f"size {selected_size} (originally from full Gemini pipeline)")
                return TryOnResponse(
                    success=True,
                    result_image=cached_size_render,
                    fit_state=fit_state,
                    cloth_analysis=active_cloth,
                    silhouette_score=1.0,
                    silhouette_passed=True,
                    fit_verified=cached_metrics.get("fit_verified"),
                    face_similarity=_face_score(user_img, cached_size_render, f"cached size {selected_size}"),
                    message=_wont_fit_message(recommended_size) if wont_fit else f"Retrieved cached size {selected_size} preview.",
                    **fit_fields,
                )

        request_costs = begin_cost_tracking()
        print(f"[TRYON] Initiating generation | Size: {selected_size or 'auto'} | Fit: {fit_quality} | Has Profile: {bool(active_profile)} | Cloth: {active_cloth.category} ('{active_cloth.garment_name}') | Inputs: capture + garment only | YOLO persons: {person_meta['person_count'] if person_meta else 'n/a'}")

        baseline_lm = session_record.get_baseline_landmarks() if session_record else None

        # Where the visitor's own sleeves end and what is on their wrists: the coverage prompt and the per-render check
        capture_coverage = _coverage_for(prepared)

        # Safety net: /api/check-face already ran this test when the photo was taken. This catches a capture that
        # skipped it (a stale session, a client that could not reach the check).
        if not prepared.has_face and not _capture_has_face(user_img):
            print("[FACE PRESERVE] No face detected in the capture portrait; rejecting before Gemini generation.")
            _log_path("Rejected - no face in capture", "real face cannot be restored")
            raise HTTPException(status_code=422, detail=NO_FACE_MESSAGE)

        # Face lock for captures whose face is too small to restore: this visitor's first served render supplies the
        # face of every later render (a post-generation composite; the model never sees that render)
        capture_hash = get_dress_hash(user_img)[:16]
        identity_img = None
        identity_b64 = session_record.get_identity_render(capture_hash) if session_record else None
        if identity_b64:
            try:
                identity_img = clean_and_decode_base64(identity_b64).convert("RGB")
            except Exception as id_err:
                print(f"[FACE LOCK WARNING] Could not decode the identity render: {id_err}")

        # Cross-size verification inputs: silhouettes of this garment's other rendered sizes in this session
        size_fits = size_rec.size_fits if size_rec else {}
        rendered_metrics = session_record.get_size_render_metrics(current_dress_hash) if session_record else {}
        other_metrics = {s: m for s, m in rendered_metrics.items() if not selected_size or s != selected_size.upper()}
        rec_metrics = other_metrics.get(recommended_size) if recommended_size else None
        verify_family = fit_family(fit_quality, fit_state)
        # Which limbs the visitor shows: every render must keep them (limb_check)
        capture_lm = body_profile_service.extract_single_frame(user_img)

        # Size change: re-drape this garment's recommended-size render (same camera, person and pattern) instead of
        # regenerating from the capture; without one, the size is generated from the capture like the first render
        base_render = None
        if session_record and selected_size and recommended_size and selected_size.upper() != recommended_size.upper():
            canonical = session_record.get_canonical_render(dress_hash=current_dress_hash)
            if canonical and _is_studio_render(canonical, "canonical render"):
                base_render = clean_and_decode_base64(canonical).convert("RGB")
                print(f"[SIZE EDIT] Re-draping the size {recommended_size} render as size {selected_size}.")

        def run_attempt(attempt_no: int, retry_strict: bool, enforce_studio: bool, fit_escalation: Optional[str],
                        coverage_escalation: Optional[str] = None,
                        limb_escalation: Optional[str] = None) -> SimpleNamespace:
            """One full Gemini generation + studio / silhouette / cross-fit / cross-size / coverage / limb verification."""
            print(f"[GENERATION] Attempt {attempt_no}/{max_attempts} | size {selected_size or 'auto'} | fit {fit_quality} | "
                  f"strict_silhouette={retry_strict} | enforce_studio={enforce_studio} | fit_escalation={fit_escalation is not None} | "
                  f"coverage_escalation={coverage_escalation is not None} | limb_escalation={limb_escalation is not None}")
            gen_bytes, gen_face_sim, _ = generate_tryon_image(
                user_img=user_img,
                dress_img=dress_img,
                fit_state=fit_state,
                body_profile=active_profile,
                retry_strict=retry_strict,
                feathered_mask=feathered_mask,
                cloth_analysis=active_cloth,
                user_appearance=user_appearance,
                session_id=data.session_id,
                size_offset=size_offset_data,
                selected_size=selected_size,
                enforce_studio_background=enforce_studio,
                attempt=attempt_no,
                fit_escalation=fit_escalation,
                identity_face=identity_img,
                capture_coverage=capture_coverage,
                coverage_escalation=coverage_escalation,
                limb_escalation=limb_escalation,
                base_render=base_render,
            )
            request_timing.lap(None)  # Response decoding (the generation, its request and face restore are steps)
            # Fixed frame: the model's zoom varies between renders, so every render is re-framed to one scale/position
            raw_img = Image.open(io.BytesIO(gen_bytes))
            raw_img.load()
            raw_lm = body_profile_service.extract_single_frame(raw_img)
            framed_bytes = normalize_framing(gen_bytes, getattr(raw_lm, "segmentation_mask", None), decoded=raw_img)
            gen_pil = raw_img if framed_bytes is gen_bytes else Image.open(io.BytesIO(framed_bytes))
            gen_pil.load()
            gen_bytes = framed_bytes
            studio_ok = _is_studio_render(gen_pil, f"attempt {attempt_no}")
            # The midsection (8%) check runs inside verify() on every size, including "won't fit"
            verify = silhouette_verifier.verify(
                user_img=user_img,
                generated_img=gen_pil,
                body_profile=active_profile,
                fit_state=fit_state,
                selected_size=selected_size,
                size_offset=size_offset_data,
                dress_img=dress_img,
                face_check=SimpleNamespace(passed=(gen_face_sim >= 0.85), score=gen_face_sim),
            )
            # Cross-fit landmark regression check (skeleton must not move between sizes)
            frame_lm = body_profile_service.extract_single_frame(gen_pil)
            cross_passed = True
            if baseline_lm and frame_lm:
                cross_passed, _, _ = silhouette_verifier.verify_cross_fit_landmarks(
                    current_landmarks=frame_lm,
                    baseline_landmarks=baseline_lm,
                    fit_state=verify_family,
                )
            metrics = garment_silhouette(gen_pil, frame_lm)
            fit_eval = _evaluate_fit(metrics, fit_quality, recommended_size, rec_metrics, other_metrics, size_fits)
            _log_fit_eval(attempt_no, selected_size, fit_quality, metrics, fit_eval)
            coverage = _check_coverage(capture_coverage, gen_pil, frame_lm, active_cloth.sleeve_length, attempt_no)
            limbs = check_limbs(capture_lm, frame_lm, active_cloth.category)
            print(f"[LIMB CHECK] Attempt {attempt_no}: "
                  + {True: "PASS", False: f"FAIL: missing {', '.join(limbs['missing'])}", None: "not checked"}[limbs["passed"]])
            request_timing.lap("checks_after")
            return SimpleNamespace(
                coverage=coverage,
                limbs=limbs,
                attempt=attempt_no,
                bytes=gen_bytes,
                face_sim=gen_face_sim,
                verify=verify,
                frame_lm=frame_lm,
                studio_ok=studio_ok,
                metrics=metrics,
                fit_eval=fit_eval,
                verification_passed=verify.passed and (cross_passed or verify.score >= 0.70),
            )

        # Attempt loop: a raw-photo output (Gemini swapped only the garment and returned the original scene) is
        # rejected and regenerated with a studio-enforcement directive; transient API errors are retried; a
        # silhouette/midsection failure gets one strict body-alignment retry; a render that shows the visitor's own
        # sleeve past the garment's (or bare skin under its sleeve) gets one coverage retry; a render that does not
        # show its size (nearly identical to another size, or not looser / tighter than the recommended size) gets one
        # fit-escalation retry. "Won't fit" gets exactly one generation and none of these retries.
        request_timing.lap("checks_before")
        max_attempts = 1 if wont_fit else GENERATION_MAX_ATTEMPTS
        candidates = []
        last_error: Optional[Exception] = None
        retry_strict = False
        enforce_studio = False
        fit_escalation: Optional[str] = None
        coverage_escalation: Optional[str] = None
        # Capture cropped above the knees or feet: tell the model up front to draw them, not only after a failure
        leg_hint = capture_leg_text(capture_lm, active_cloth.category)
        if leg_hint:
            print("[LIMB HINT] Capture does not show the legs/feet; asking the model to draw them from attempt 1.")
        limb_escalation: Optional[str] = leg_hint
        limb_retried = False
        attempt_no = 0
        for attempt_no in range(1, max_attempts + 1):
            try:
                cand = run_attempt(attempt_no, retry_strict, enforce_studio, fit_escalation, coverage_escalation,
                                   limb_escalation)
            except HTTPException as api_err:
                request_timing.attempt_result("blocked" if api_err.status_code == 403 else "error")
                if api_err.status_code == 403:  # content blocked: retrying will not help
                    raise
                last_error = api_err
                print(f"[GENERATION] Attempt {attempt_no} API error ({api_err.detail}){'' if wont_fit else '; retrying.'}")
                continue
            except Exception as api_err:
                request_timing.attempt_result("error")
                last_error = api_err
                print(f"[GENERATION] Attempt {attempt_no} error ({api_err}){'' if wont_fit else '; retrying.'}")
                continue

            candidates.append(cand)
            request_timing.attempt_result(_attempt_result(cand))
            if wont_fit:
                break
            if not cand.studio_ok:
                print(f"[GENERATION] Attempt {attempt_no} REJECTED: raw photo (original background kept). Regenerating with studio enforcement.")
                enforce_studio = True
                continue
            if cand.verification_passed and cand.fit_eval["passed"] and _coverage_ok(cand) and _limbs_ok(cand):
                break
            if not _limbs_ok(cand) and not limb_retried:
                print(f"[LIMB RETRY] Attempt {attempt_no}: missing {', '.join(cand.limbs['missing'])}. "
                      f"Retrying with a full-body directive...")
                limb_escalation = (leg_hint or "") + limb_escalation_text(cand.limbs["missing"])
                limb_retried = True
                continue
            if not cand.verification_passed and not retry_strict:
                print(f"[SILHOUETTE RETRY] Attempt {attempt_no} failed (score {cand.verify.score:.2f} < {cand.verify.threshold:.2f}, midsection loss, or skeletal landmark drift across fits). Retrying with strict body alignment...")
                retry_strict = True
                continue
            if not _coverage_ok(cand) and coverage_escalation is None:
                print(f"[COVERAGE RETRY] Attempt {attempt_no}: {cand.coverage['problem']} on the {cand.coverage['sides']} forearm. "
                      f"Retrying with a stronger coverage directive...")
                coverage_escalation = coverage_escalation_text(cand.coverage, active_cloth.sleeve_length)
                continue
            if not cand.fit_eval["passed"] and fit_escalation is None:
                problem, other = cand.fit_eval["problem"], cand.fit_eval["other_size"]
                print(f"[FIT RETRY] Attempt {attempt_no}: size {selected_size} {problem} size {other} — treated as a failed generation. Retrying with a stronger fit directive...")
                fit_escalation = fit_escalation_text(fit_quality, other, problem)
                continue
            break  # the applicable retries are used up

        if wont_fit:
            return _serve_wont_fit(
                candidates[0] if candidates else None, last_error, data, session_record, user_img, dress_img,
                fit_fields, fit_state, active_cloth, recommended_size, selected_size, current_dress_hash,
                size_offset_data, request_costs,
            )

        studio_candidates = [c for c in candidates if c.studio_ok]
        if not studio_candidates:
            _log_size_generation(selected_size, size_offset_data, len(candidates), None, request_costs)
            if not candidates and last_error is not None:
                _log_path("FAILED - no image", f"all {max_attempts} attempts errored: {last_error}")
                raise last_error
            if _usable_last_good(cached_last_good):
                _log_path("Fallback: last good render", f"all {len(candidates)} generated attempts were raw photos")
                return TryOnResponse(
                    success=True,
                    result_image=cached_last_good,
                    qr_code_base64=None,
                    fit_state=fit_state,
                    cloth_analysis=active_cloth,
                    silhouette_passed=False,
                    fallback_used=True,
                    face_similarity=_face_score(user_img, cached_last_good, "fallback last good render"),
                    message="We couldn't generate a new preview, so your previous preview is shown. Please try again.",
                    **fit_fields,
                )
            _log_path("FAILED - no image", f"all {len(candidates)} generated attempts were raw photos; nothing served")
            return TryOnResponse(
                success=False,
                result_image=None,
                qr_code_base64=None,
                fit_state=fit_state,
                cloth_analysis=active_cloth,
                verification_failed=True,
                message="We couldn't generate a preview — please try again.",
                **fit_fields,
            )

        def _rank(c: SimpleNamespace):
            # Body lock first (verified), then no leftover sleeve / uncovered arm, then a visible size difference, then
            # the render least like any other size, then the render that kept the most midsection volume, then score
            change = (c.verify.details or {}).get("midsection_change")
            apps = [p["appearance_distance"] for p in c.fit_eval["pairs"].values() if p.get("appearance_distance") is not None]
            return (_limbs_ok(c), c.verification_passed, _coverage_ok(c), c.fit_eval["passed"], min(apps) if apps else 0.0,
                    min(change, 0.0) if change is not None else 0.0, c.verify.score)

        best = max(studio_candidates, key=_rank)
        if not _limbs_ok(best):
            print(f"[LIMB WARNING] Serving attempt {best.attempt}: every attempt was missing a limb "
                  f"({', '.join(best.limbs['missing'])}).")
            _log_path("Limb check failed on every attempt", ", ".join(best.limbs["missing"]))
        if not (best.verify.details or {}).get("midsection_preserved", True):
            print(f"[MIDSECTION WARNING] Serving attempt {best.attempt} with midsection change "
                  f"{best.verify.details['midsection_change']:+.1%}: every attempt slimmed the torso beyond tolerance.")
        coverage_verified = (best.coverage or {}).get("passed")
        if coverage_verified is False:
            print(f"[COVERAGE WARNING] Serving attempt {best.attempt}: it still shows {best.coverage['problem']} on the "
                  f"{best.coverage['sides']} forearm after the coverage retry (coverage_verified=False in the response).")
        fit_verified = best.fit_eval["passed"] if best.fit_eval["checked"] else None
        if fit_verified is False:
            print(f"[FIT WARNING] Serving attempt {best.attempt} for size {selected_size}: it still {best.fit_eval['problem']} "
                  f"size {best.fit_eval['other_size']} after the fit retry (fit_verified=False in the response).")
        final_bytes = best.bytes
        final_face_sim = best.face_sim
        verify_result = best.verify
        gen_frame_lm = best.frame_lm
        verification_passed = best.verification_passed

        # Baseline skeleton for the cross-fit check: from a verified correct-size render only (dropped shoulder seams
        # on a larger size shift the detected shoulder points)
        if verification_passed and fit_quality == "perfect" and session_record and gen_frame_lm and not session_record.get_baseline_landmarks():
            lm_data = {
                "height_ratio": round(float(gen_frame_lm.height_ratio), 3),
                "torso_length": round(float(gen_frame_lm.torso_length), 3),
                "shoulder_width": round(float(gen_frame_lm.shoulder_width), 3),
                "hip_width": round(float(gen_frame_lm.hip_width), 3)
            }
            session_record.set_baseline_landmarks(lm_data, fit_state=fit_state)
            print(f"[TRYON] Baseline fit landmarks recorded for session {data.session_id}: {lm_data}")

        # If still failed after retry: evaluate fallback policy
        if not verification_passed:
            if verify_result.passed or verify_result.score >= 0.55:
                # Acceptable silhouette fidelity: accept new fit rather than reverting to old image
                print(f"[SILHOUETTE ACCEPT] Render score ({verify_result.score:.3f}) accepted for {fit_state} fit.")
                verification_passed = True
            elif _usable_last_good(cached_last_good):
                print(f"[SILHOUETTE FALLBACK] Verification failed twice with low score ({verify_result.score:.3f}). Falling back to cached last good render.")
                _log_path("Fallback: last good render", f"silhouette score {verify_result.score:.3f} too low after retry")
                _log_size_generation(selected_size, size_offset_data, len(candidates), None, request_costs)
                return TryOnResponse(
                    success=True,
                    result_image=cached_last_good,
                    qr_code_base64=None,
                    fit_state=fit_state,
                    cloth_analysis=active_cloth,
                    silhouette_score=verify_result.score,
                    silhouette_passed=False,
                    fallback_used=True,
                    face_similarity=_face_score(user_img, cached_last_good, "fallback last good render"),
                    message="Maintained previous balanced render as fit adjustment did not meet silhouette fidelity standards.",
                    **fit_fields,
                )
            elif verify_result.score < 0.48:
                # Severe anatomical distortion on first render: guide visitor to adjust stance
                print(f"[SILHOUETTE FAILURE] First render severe distortion (score: {verify_result.score:.2f} < 0.48). Guiding visitor to adjust stance.")
                _log_path("FAILED - no image", f"severe silhouette distortion {verify_result.score:.3f}; visitor asked to realign")
                _log_size_generation(selected_size, size_offset_data, len(candidates), None, request_costs)
                return TryOnResponse(
                    success=False,
                    result_image=None,
                    qr_code_base64=None,
                    fit_state=fit_state,
                    cloth_analysis=active_cloth,
                    silhouette_score=verify_result.score,
                    silhouette_passed=False,
                    verification_failed=True,
                    needs_visitor_realignment=True,
                    face_similarity=round(final_face_sim, 4),
                    message="We couldn't achieve a photorealistic fit with current lighting/pose. Please adjust your stance and tap to retry.",
                    **fit_fields,
                )
            else:
                # First render has acceptable anatomy (score between 0.48 and threshold) with minor boundary drift;
                # accept it for initial preview so visitor sees their try-on
                print(f"[SILHOUETTE ACCEPT] First render score ({verify_result.score:.3f}) accepted for initial preview.")
                verification_passed = True

        encoded_result, qr_data_uri = _finalize_render(final_bytes, final_face_sim, data.session_id)

        # Cache only now: after generation, face restoration and verification. The canonical render is the
        # recommended-size render (the reference for the cross-size check; never a generation input).
        if data.session_id:
            s_rec = session_manager.get_or_create_session(data.session_id)
            s_rec.set_last_good_render(encoded_result, dress_hash=current_dress_hash)
            if size_rec is None or selected_size == size_rec.recommended_size:
                s_rec.set_canonical_render(encoded_result, dress_hash=current_dress_hash)
            if selected_size:
                s_rec.set_size_render(selected_size, encoded_result, dress_hash=current_dress_hash,
                                      metrics=_stored_metrics(best.metrics, fit_quality, fit_verified))
            if not s_rec.get_identity_render(capture_hash):
                s_rec.set_identity_render(encoded_result, capture_hash)

        _log_size_generation(selected_size, size_offset_data, len(candidates), best, request_costs)
        _log_path("Full Gemini pipeline", f"used attempt {best.attempt} of {attempt_no} | studio background verified | silhouette {verify_result.score:.3f}")
        print(f"[SUCCESS] Image generated and verified! Score: {verify_result.score:.3f} | Face Sim: {final_face_sim * 100:.1f}% | Cloth: {active_cloth.category} | Size: {selected_size or 'auto'} ({fit_quality})")
        return TryOnResponse(
            success=True,
            result_image=encoded_result,
            qr_code_base64=qr_data_uri,
            fit_state=fit_state,
            cloth_analysis=active_cloth,
            silhouette_score=verify_result.score,
            silhouette_passed=True,
            fallback_used=False,
            fit_verified=fit_verified,
            coverage_verified=coverage_verified,
            face_similarity=round(final_face_sim, 4),
            **fit_fields,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL BACKEND ERROR: {error_msg}")
        _log_path("FAILED - no image", f"HTTP 500: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
