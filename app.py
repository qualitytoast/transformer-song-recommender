"""HTTP API for the song recommender.

Run locally:      uvicorn app:app --reload
In the container: uvicorn app:app --host 0.0.0.0 --port 7860

Routes:
    GET  /                  the demo page (static/index.html)
    GET  /health            is the server up, and which model is loaded
    GET  /songs?q=mask      search the vocab for exact spellings
    POST /recommend         {"songs": [10 names], "k": 5} -> top-k next songs
    GET  /docs              interactive API console FastAPI generates for the routes above
"""
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from checkpoint import load_bundle
from recommender import Recommender

# Log to stdout, which the host captures and shows in its dashboard — no log
# service needed. Every request records what came in, what went out and how long
# it took, so latency is a measured number and a bad request leaves a trace.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("recommender")

# Runs once, when uvicorn imports this file. If the bundle is missing or broken
# this raises, the process exits, and the host reports a failed start. That is
# the intent: never serve a half-loaded or randomly initialised model.
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "artifacts")

# Resolved against this file, not the working directory, so the page is found
# wherever uvicorn happens to be started from.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Load model filled with pre-existing weights, vocab, and metadata.
_t0 = time.perf_counter()
model, vocab, meta = load_bundle(BUNDLE_DIR)

# Create Recommender instance w/ filled model and vocab. This is the object that handles every request.
rec = Recommender(model, vocab)
log.info("loaded %s: %d songs, %d params, trained %s (git %s) in %.2fs",
         BUNDLE_DIR, meta["vocab_size"], meta.get("num_parameters", -1),
         meta.get("trained_at"), meta.get("git_sha"), time.perf_counter() - _t0)

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
    """The demo page: a playlist, a search box, and the model's picks. Falls back
    to the API docs if the file is missing, so the API still works without it."""
    index = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index):
        return RedirectResponse(url="/docs")
    return FileResponse(index, media_type="text/html")


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
    t0 = time.perf_counter()
    matches = rec.search(q, limit)
    log.info("search q=%r -> %d matches, %.1f ms",
             q, len(matches), (time.perf_counter() - t0) * 1000)
    return {"matches": matches}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    t0 = time.perf_counter()
    try:
        picks = rec.recommend(req.songs, req.k)
    except ValueError as e:
        # Rejected input is a warning, not an error: the server is fine, the
        # request broke a rule. Logged so a caller's repeated failures are visible.
        log.warning("recommend rejected (%d songs, k=%d): %s", len(req.songs), req.k, e)
        raise HTTPException(status_code=400, detail=str(e))
    ms = (time.perf_counter() - t0) * 1000
    top = f"{picks[0][0]!r} (p={picks[0][1]:.3f})" if picks else "none"
    log.info("recommend %d songs, k=%d -> top=%s, %.1f ms", len(req.songs), req.k, top, ms)
    return {"recommendations": [{"song": name, "prob": prob} for name, prob in picks]}
