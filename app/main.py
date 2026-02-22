import os
from pathlib import Path
from fastapi import FastAPI
import uvicorn

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Privacy=Preserving CDSS",
    descrition="A Hybrid RAG pipeline for clinical genomic queries",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "online", "service": "CDSS API is running"}

if __name__=="__main__":
    uvicorn.run ("app.main:app", port=5656, reload= True)