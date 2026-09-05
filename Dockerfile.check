# Pinned slim base -> small, reproducible image
FROM python:3.12-slim

WORKDIR /app

# Install deps FIRST as their own layer, so they're cached and only
# reinstalled when requirements.txt changes (not on every code edit).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the source
COPY *.py ./

# Default command verifies the autograd — needs NO dataset, so `docker run <image>`
# instantly proves the env + math work on a clean machine.
# To train, mount the dataset as a volume (it's too big to bake into the image):
#   docker run --rm -v /path/to/data:/app/data song-recommender python train.py
CMD ["python", "test_gradients.py"]