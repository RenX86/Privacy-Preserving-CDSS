import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from app.api.router import router

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Privacy Preserving CDSS",
    description="A Hybrid RAG pipeline for clinical genomic queries",
    version="1.0.0"
)

app.include_router(router)

if __name__=="__main__":
    uvicorn.run ("app.main:app", port=5656, reload= True)
