"""Build a small fake bundle for tests and CI: 20 made-up songs, a tiny randomly
initialised model, and the same three files a real training run writes.

    python -m tests.fake_bundle artifacts     # CI runs this before building the serving image

No dataset needed. The model has never been trained, so its recommendations are
noise, but every code path (save, load, validate, rank, serve) is the real one.
"""
import sys

import numpy as np

from checkpoint import save_bundle
from config import Config
from model import SongRecommender

SONGS = [f"Song {i}" for i in range(20)]


def make_fake_bundle(out_dir, embed_dim=8, num_layers=1, context_length=10, seed=0):
    """Write a bundle into out_dir. Returns (model, vocab, cfg)."""
    np.random.seed(seed)
    cfg = Config(embed_dim=embed_dim, num_layers=num_layers, context_length=context_length)
    model = SongRecommender(vocab_size=len(SONGS), embed_dim=embed_dim,
                            context_length=context_length, num_layers=num_layers)
    save_bundle(model, SONGS, cfg, out_dir, best_ndcg=0.123)
    return model, SONGS, cfg


if __name__ == "__main__":
    make_fake_bundle(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
