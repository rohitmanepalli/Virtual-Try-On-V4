import os
from string import Template

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from app.services import download_store

router = APIRouter()

# The phone's browser must not keep its own copy of a visitor's image beyond the save they asked for
_NO_STORE = {"Cache-Control": "no-store"}

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "download.html")
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    _DOWNLOAD_PAGE_TEMPLATE = Template(f.read())


@router.get("/download/{image_id}", response_class=HTMLResponse)
async def download_page(image_id: str):
    if download_store.get(image_id) is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")

    html_content = _DOWNLOAD_PAGE_TEMPLATE.substitute(image_id=image_id)
    return HTMLResponse(content=html_content, headers=_NO_STORE)


@router.get("/api/image/{image_id}")
async def serve_image(image_id: str, download: bool = False):
    jpeg = download_store.get(image_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    headers = dict(_NO_STORE)
    if download:
        headers["Content-Disposition"] = f'attachment; filename="virtual-tryon-{image_id[:8]}.jpg"'
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)
