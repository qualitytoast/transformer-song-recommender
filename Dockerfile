# Serving image: the FastAPI app with a trained bundle baked in.
#
#   Build:  docker build -t song-recommender .
#   Run:    docker run --rm -p 7860:7860 song-recommender
#   Then:   curl localhost:7860/health
#
# Needs artifacts/ (weights.npz, vocab.json, metadata.json) next to this file.
# A real one comes from `python train.py`. CI has no trained weights, so it
# fakes one first with `python -m tests.fake_bundle artifacts`.
# The gradient-check image lives in Dockerfile.check.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, as their own layer, so they are cached across code edits.
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Run as a non-root user. Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
USER user

# Only the serving files. Nothing training-only, no dataset.
COPY --chown=user:user engine.py model.py checkpoint.py recommender.py app.py ./
COPY --chown=user:user artifacts/ ./artifacts/

# 7860 is the port Hugging Face Spaces forwards to. Other hosts set PORT themselves.
ENV PORT=7860
EXPOSE 7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
