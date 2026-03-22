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
    description="A Hybrid RAG pipeline for clinical genomic queries",
    version="1.0.0"
)

app.include_router(router)

if __name__=="__main__":
    uvicorn.run("app.main:app", port=5656, reload=True)
