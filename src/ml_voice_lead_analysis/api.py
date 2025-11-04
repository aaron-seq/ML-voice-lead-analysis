from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pathlib import Path
from cachetools import TTLCache

from ml_voice_lead_analysis.config import settings

api_app = FastAPI(default_response_class=ORJSONResponse, title="Voice Lead Analysis API (Bootstrap)")

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TTLCache(maxsize=1024, ttl=settings.cache_ttl_seconds)

@api_app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_ttl": settings.cache_ttl_seconds,
        "audio_features_enabled": settings.enable_audio_features,
    }

@api_app.post("/analyze")
def analyze(transcript_text: str):
    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="transcript_text is required")
    key = f"k::{transcript_text[:100]}"
    if key in cache:
        return cache[key]
    # trivial bootstrap scoring
    lower = transcript_text.lower()
    label = "Hot" if "pricing" in lower or "interested" in lower else ("Warm" if "call back" in lower else "Cold")
    payload = {"label": label, "confidence": 0.75}
    cache[key] = payload
    return payload
