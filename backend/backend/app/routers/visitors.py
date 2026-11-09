from fastapi import APIRouter, HTTPException

from app.models import VisitorDetailsRequest, VisitorDetailsResponse
from app.services import visitor_store

router = APIRouter()


@router.post("/api/visitors", response_model=VisitorDetailsResponse)
def save_visitor_details(data: VisitorDetailsRequest) -> VisitorDetailsResponse:
    """
    Saves details the visitor chose to give, only with consent. Write-only: the response never says what is already stored
    or whether a phone number is known, and no route reads this data back. The details themselves are never logged.
    """
    if not data.consent or not (data.name or data.phone or data.gender):
        return VisitorDetailsResponse(saved=False)
    try:
        visitor_store.save_visitor(data.session_id, data.consent_version, data.name, data.phone, data.gender)
    except visitor_store.InvalidPhone as err:
        raise HTTPException(status_code=422, detail=str(err))
    except visitor_store.VisitorKeyMissing:
        print("[VISITOR DATA WARNING] VISITOR_DATA_KEY is not set: visitor details were not saved.", flush=True)
        raise HTTPException(status_code=503, detail="Your details could not be saved right now.")
    print("[VISITOR DATA] Consented details saved.", flush=True)
    return VisitorDetailsResponse(saved=True)
