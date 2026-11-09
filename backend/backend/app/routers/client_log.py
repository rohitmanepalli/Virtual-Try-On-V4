from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.app_log import write_client_entries

router = APIRouter()


class ClientLogEntry(BaseModel):
    level: str = "log"
    message: str = ""
    time: Optional[str] = None


class ClientLogBatch(BaseModel):
    client_id: str = ""
    entries: List[ClientLogEntry] = Field(default_factory=list, max_length=200)


@router.post("/api/client-logs", status_code=204)
def client_logs(batch: ClientLogBatch) -> None:
    """The kiosk frontend's activity (screens, button taps, console output, errors), for logs/app_<date>.log."""
    write_client_entries(batch.client_id, [(e.level, e.message, e.time) for e in batch.entries])
