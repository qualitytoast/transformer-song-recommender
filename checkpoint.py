"""Save and load a trained SongRecommender as a self-contained "bundle" folder.

A bundle is three files in one directory (default: artifacts/):

    weights.npz     one array per parameter, keyed by its attribute path,
                    e.g. "blocks.0.attention.W_query.W" -> shape (64, 64)
    vocab.json      list of song names; the index in the list IS the song ID,
                    which IS the row in the embedding table
    metadata.json   the sizes needed to rebuild the model (vocab_size, embed_dim,
                    num_layers, context_length) plus provenance (best NDCG, date,
                    git sha, the full training Config)

Why names instead of np.savez's default arr_0, arr_1, ...: positional loading
silently misaligns the moment the model's parameter order changes (e.g. adding a
layer). Loading by name lets us fail loudly and say exactly which array is
missing or mis-shaped.

This module imports only numpy, engine, and model, so the serving API can use it
without pulling in matplotlib or anything else that is training-only.
"""
import json
import os
import subprocess
import time
from dataclasses import asdict

import numpy as np

import engine
from model import SongRecommender

FORMAT_VERSION = 1
WEIGHTS_FILE = "weights.npz"
VOCAB_FILE = "vocab.json"
META_FILE = "metadata.json"

# Walk through each param in the model, creating (name, tensor) pairs per param to allow for param verification
# in loading and param name, tensor pairs for saving
def named_parameters(module, prefix=""):
    """Walk a Module exactly like Module.parameters() in engine.py, but return
    (name, tensor) pairs, where name is the attribute path used to reach the tensor.

    For SongRecommender this yields names such as:
        embedding.weight
        position_embedding.weight
        blocks.0.attention.W_query.W
        blocks.0.norm1.gamma
        matchmaker.B
    """
    out = []
    for attr, val in module.__dict__.items():
        if isinstance(val, engine.Tensor):
            out.append((prefix + attr, val))
        elif isinstance(val, engine.Module):
            out.extend(named_parameters(val, prefix + attr + "."))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, engine.Module):
                    out.extend(named_parameters(item, f"{prefix}{attr}.{i}."))
                elif isinstance(item, engine.Tensor):
                    out.append((f"{prefix}{attr}.{i}", item))
    return out


def save_bundle(model, id_to_track, cfg, out_dir, best_ndcg=None):
    """Write weights.npz, vocab.json and metadata.json into out_dir.

    train.py calls this whenever validation NDCG improves, so out_dir always
    holds the best checkpoint of the run.

    id_to_track may be the dict {id: name} that data.py returns, or a plain list
    where the index is the ID.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Weights, one named array per parameter. ** gives each array its name as the key.
    named = named_parameters(model)
    np.savez(os.path.join(out_dir, WEIGHTS_FILE), **{name: p.data for name, p in named})

    # Vocab as a list: position == song ID == embedding row.
    if isinstance(id_to_track, dict):
        vocab = [id_to_track[i] for i in range(len(id_to_track))]
    else:
        vocab = list(id_to_track)
    with open(os.path.join(out_dir, VOCAB_FILE), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Metadata: enough to rebuild the model with no dataset, plus where it came from.
    meta = {
        "format_version": FORMAT_VERSION,
        "vocab_size": len(vocab),
        "embed_dim": cfg.embed_dim,
        "num_layers": cfg.num_layers,
        "context_length": cfg.context_length,
        "num_parameters": int(sum(p.data.size for _, p in named)),
        "best_ndcg_at_10": best_ndcg,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "config": asdict(cfg),
    }
    with open(os.path.join(out_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SYSTEM] Bundle saved to {out_dir}/ "
          f"({meta['num_parameters']:,} params, vocab {len(vocab):,})")


def load_weights(model, weights_path):
    """Fill an already-built model from weights.npz. Strict on purpose: every
    parameter the model has must exist in the file under its name with the same
    shape, or this raises. train.py uses this to resume a run."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file at {weights_path}")

    saved = np.load(weights_path)

    # Verify all parameters are present and shaped correctly, then copy the data into the model
    for name, p in named_parameters(model):
        if name not in saved.files:
            raise KeyError(f"{weights_path} has no array named '{name}'")
        arr = saved[name]
        if arr.shape != p.data.shape:
            raise ValueError(f"'{name}': saved shape {arr.shape}, model expects {p.data.shape}")
        p.data = arr.astype(engine.DEFAULT_DTYPE)
        p.grad = np.zeros_like(p.data)


def read_metadata(bundle_dir):
    """metadata.json as a dict. Raises if missing or written by a different format version."""
    path = os.path.join(bundle_dir, META_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle at '{bundle_dir}' is missing {META_FILE}")
    with open(path) as f:
        meta = json.load(f)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Bundle format version {meta.get('format_version')}, "
                         f"this code expects {FORMAT_VERSION}")
    return meta


def read_vocab(bundle_dir):
    """vocab.json as a list where index == song ID. Raises if missing."""
    path = os.path.join(bundle_dir, VOCAB_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle at '{bundle_dir}' is missing {VOCAB_FILE}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_bundle(bundle_dir):
    """Rebuild a model from a bundle with no dataset needed.

    Reads metadata.json, builds an empty SongRecommender of exactly the saved
    shape, fills it from weights.npz, loads vocab.json and cross-checks it.
    Raises on any missing file, unknown format version, or mismatch.

    Returns (model, vocab, metadata). This is what the API calls once at startup.
    """
    meta = read_metadata(bundle_dir)
    vocab = read_vocab(bundle_dir)
    if len(vocab) != meta["vocab_size"]:
        raise ValueError(f"vocab.json has {len(vocab)} songs, metadata says {meta['vocab_size']}")

    model = SongRecommender(vocab_size=meta["vocab_size"],
                            embed_dim=meta["embed_dim"],
                            context_length=meta["context_length"],
                            num_layers=meta["num_layers"])
    load_weights(model, os.path.join(bundle_dir, WEIGHTS_FILE))
    return model, vocab, meta


def _git_sha():
    """Short git commit hash, or None if git is unavailable (e.g. inside the container)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),  # the repo, not the caller's cwd
        ).strip()
    except Exception:
        return None
