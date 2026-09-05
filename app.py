"""HTTP API for the song recommender.

Run locally:      uvicorn app:app --reload
In the container: uvicorn app:app --host 0.0.0.0 --port 7860

Routes:
    GET  /health            is the server up, and which model is loaded
    GET  /songs?q=mask      search the vocab for exact spellings
    POST /recommend         {"songs": [10 names], "k": 5} -> top-k next songs
    GET  /docs              interactive page FastAPI generates for the routes above
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from checkpoint import load_bundle
from recommender import Recommender

# Runs once, when uvicorn imports this file. If the bundle is missing or broken
# this raises, the process exits, and the host reports a failed start. That is
# the intent: never serve a half-loaded or randomly initialised model.
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "artifacts")
model, vocab, meta = load_bundle(BUNDLE_DIR)
rec = Recommender(model, vocab)

app = FastAPI(
    title="Transformer Song Recommender",
    description="Next-song prediction from a Transformer built from scratch in NumPy.",
)


class RecommendRequest(BaseModel):
    """Shape of the JSON body for POST /recommend. FastAPI validates against this
    before the route runs; a wrong shape gets an automatic 422 reply."""
    songs: list[str] = Field(description="Song names, in playlist order. At least 10.")
    k: int = Field(default=10, ge=1, le=50, description="How many recommendations to return.")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vocab_size": meta["vocab_size"],
        "num_parameters": meta.get("num_parameters"),
        "context_length": meta["context_length"],
        "best_ndcg_at_10": meta.get("best_ndcg_at_10"),
        "trained_at": meta.get("trained_at"),
        "git_sha": meta.get("git_sha"),
    }


@app.get("/songs")
def songs(q: str = Query(min_length=1, description="Substring to search for"),
          limit: int = Query(default=20, ge=1, le=100)):
    return {"matches": rec.search(q, limit)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        picks = rec.recommend(req.songs, req.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"recommendations": [{"song": name, "prob": prob} for name, prob in picks]}
