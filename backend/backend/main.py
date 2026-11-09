import os
import sys

from dotenv import load_dotenv

# Load .env before app modules read their configuration
load_dotenv()

from app.services.app_log import capture_uvicorn_logs, setup_logging

# Before the other app imports, so anything they print is captured
setup_logging()

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_dist_dir, get_local_ip, get_public_base_url
from app.routers import beauty, client_log, download, tryon, visitors
from app.services import privacy


@asynccontextmanager
async def lifespan(_app: FastAPI):
    capture_uvicorn_logs()
    privacy.start()
    # Load the person detector in the background so the first try-on doesn't wait for it
    threading.Thread(target=tryon.warm_up_yolo, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tryon.router)
app.include_router(download.router)
app.include_router(beauty.router)
app.include_router(client_log.router)
app.include_router(visitors.router)

# Serve the built frontend from /
dist_dir = get_dist_dir()
if os.path.exists(dist_dir):
    print(f"[STATICFILES] Mounting static directory from: {dist_dir}")
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
else:
    print(f"[STATICFILES] Warning: Static directory not found at: {dist_dir}")

if __name__ == "__main__":
    import threading
    import time
    import webbrowser

    import uvicorn

    # Must match the port uvicorn listens on below
    PORT = 8888
    frontend_url = os.getenv("FRONTEND_URL", f"http://localhost:{PORT}")

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(frontend_url)

    local_ip = get_local_ip()
    public_url = get_public_base_url()
    print("[STARTUP] Starting virtual trial room application...")
    print(f"[NETWORK] PC Interface: {frontend_url}")
    print(f"[NETWORK] Mobile / QR Code Download Base: {public_url}")
    print(f"[NETWORK] Ensure your mobile phone is connected to the same Wi-Fi network ({local_ip})")
    threading.Thread(target=open_browser, daemon=True).start()

    # Auto-reload while developing; a PyInstaller build runs the app object directly
    is_dev = not getattr(sys, "frozen", False)
    if is_dev:
        uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=PORT)
