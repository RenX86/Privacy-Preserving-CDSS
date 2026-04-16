import os
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from app.api.router import router

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("cdss")

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Privacy Preserving CDSS",
    description="A Hybrid RAG pipeline for clinical genomic queries"
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.include_router(router)

# ─── Track 2 UI Setup ─────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend" / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return (BASE_DIR / "frontend" / "templates" / "index.html").read_text(encoding="utf-8")

@app.on_event("startup")
def startup_event():
    from app.db.pool import _get_pool
    _get_pool()
    log.info("Connection pool pre-warmed on startup")


if __name__=="__main__":
    uvicorn.run("app.main:app", port=5656, reload=True)
