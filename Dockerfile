# Serving image: the FastAPI app with a trained bundle baked in.
#
#   Build:  docker build -t song-recommender .
#   Run:    docker run --rm -p 7860:7860 song-recommender
#   Then:   curl localhost:7860/health
#
# Needs artifacts/ (weights.npz, vocab.json, metadata.json) next to this file.
# That folder is committed to the repo, so the host builds this straight from a
# git clone with nothing to fetch.
# The gradient-check image lives in Dockerfile.check.
FROM python:3.12-slim

# Run as a non-root user, which several hosts require (Hugging Face pins uid 1000).
# Created before WORKDIR so /app is owned by it and Python can write __pycache__ there.
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user:user /app

# Dependencies first, as their own layer, so they are cached across code edits.
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

USER user

# Only the serving files. Nothing training-only, no dataset.
COPY --chown=user:user engine.py model.py checkpoint.py recommender.py app.py ./
COPY --chown=user:user static/ ./static/
COPY --chown=user:user artifacts/ ./artifacts/

# Listen on $PORT so the image is host-agnostic: Render and Cloud Run both inject
# their own. 7860 is the default when nothing sets it (and what CI checks).
ENV PORT=7860
EXPOSE 7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
