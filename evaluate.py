"""Score a trained bundle on the held-out split.

Training prints NDCG@10 once per epoch and then the number is gone. This script
answers "what does the bundle on disk actually score?" at any time, without
retraining — the check you want before publishing a model, and the way to
compare two bundles on equal terms.

    python evaluate.py                       # score artifacts/ on the held-out split
    python evaluate.py --bundle old_run/     # score a different bundle
    python evaluate.py --baseline            # also score an untrained model, for context

Needs the dataset, because the held-out split is rebuilt from it. The split is
reproducible: the same seed reproduces the same shuffle, so these are the same
sequences the model never trained on.
"""
import argparse

import numpy as np

import engine
from checkpoint import load_bundle
from config import Config
from data import load_spotify_data, tokenize_and_slice
from metrics import gains_at_k, hits_at_k
from model import SongRecommender

BATCH = 512  # full held-out logits would be n x 33770 floats; score in chunks instead


def score(model, X, Y, ks=(1, 5, 10), ndcg_k=10):
    """NDCG@ndcg_k and hit rate at each k, computed in batches."""
    gains, hits = [], {k: [] for k in ks}
    for start in range(0, len(X), BATCH):
        xb, yb = X[start:start + BATCH], Y[start:start + BATCH]
        logits = model(xb).data
        gains.append(gains_at_k(logits, yb, ndcg_k))
        for k in ks:
            hits[k].append(hits_at_k(logits, yb, k))
    return (float(np.mean(np.concatenate(gains))),
            {k: float(np.mean(np.concatenate(v))) for k, v in hits.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=None, help="bundle folder (default: Config.bundle_dir)")
    ap.add_argument("--baseline", action="store_true",
                    help="also score an untrained model, to show the improvement over chance")
    ap.add_argument("--k", type=int, default=10, help="k for NDCG@k (default: 10)")
    args = ap.parse_args()

    cfg = Config()
    bundle_dir = args.bundle or cfg.bundle_dir

    # Seed before touching the data: the split must match the one training held out.
    np.random.seed(cfg.seed)
    engine.TRAINING = False  # no dropout; evaluation must be deterministic

    raw = load_spotify_data(cfg.data_folder, max_playlists=cfg.max_playlists)
    _, _, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(
        raw, cfg.context_length, test_split=cfg.test_split, min_freq=cfg.min_freq)

    model, vocab, meta = load_bundle(bundle_dir)

    # The held-out IDs come from the data's vocab; the model's rows come from the bundle's.
    # Scoring one against the other would produce a meaningless number, so refuse.
    if [id_to_track[i] for i in range(vocab_size)] != vocab:
        raise ValueError(f"'{cfg.data_folder}' produces a different vocab than the bundle in "
                         f"'{bundle_dir}' was trained on. Cannot evaluate.")

    val_n = min(cfg.val_size, len(X_test))
    print(f"\n=== EVALUATING {bundle_dir} ===")
    print(f"trained:      {meta.get('trained_at')}  (git {meta.get('git_sha')})")
    print(f"best NDCG@10 recorded during training: {meta.get('best_ndcg_at_10')}")
    print(f"vocab {len(vocab):,} songs | held-out {len(X_test):,} sequences "
          f"| validation subset {val_n:,}")

    print(f"\n{'split':<24}{'NDCG@' + str(args.k):>10}{'hit@1':>9}{'hit@5':>9}{'hit@10':>9}")
    for label, X, Y in [("validation subset", X_test[:val_n], Y_test[:val_n]),
                        ("full held-out", X_test, Y_test)]:
        ndcg, hits = score(model, X, Y, ndcg_k=args.k)
        print(f"{label:<24}{ndcg:>10.4f}{hits[1]:>9.3f}{hits[5]:>9.3f}{hits[10]:>9.3f}")

    if args.baseline:
        # A fresh model with the same shape, never trained: the "what does chance look like"
        # number that makes the trained score interpretable.
        np.random.seed(0)
        untrained = SongRecommender(vocab_size=vocab_size, embed_dim=meta["embed_dim"],
                                    context_length=meta["context_length"],
                                    num_layers=meta["num_layers"])
        ndcg_u, hits_u = score(untrained, X_test, Y_test, ndcg_k=args.k)
        ndcg_t, _ = score(model, X_test, Y_test, ndcg_k=args.k)
        print(f"{'untrained baseline':<24}{ndcg_u:>10.5f}{hits_u[1]:>9.3f}"
              f"{hits_u[5]:>9.3f}{hits_u[10]:>9.3f}")
        print(f"\ntrained model is {ndcg_t / ndcg_u:.0f}x better than an untrained one "
              f"on the full held-out split")


if __name__ == "__main__":
    main()
