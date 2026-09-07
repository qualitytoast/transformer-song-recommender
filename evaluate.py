"""Score a trained bundle on the held-out split.

Training prints NDCG@10 once per epoch and then the number is gone. This script
answers "what does the bundle on disk actually score?" at any time, without
retraining — the check you want before publishing a model, and the way to
compare two bundles on equal terms.

    python evaluate.py                       # score artifacts/ on the held-out split
    python evaluate.py --bundle old_run/     # score a different bundle
    python evaluate.py --baseline            # also score an untrained model
    python evaluate.py --popularity          # also score a most-popular-songs baseline
    python evaluate.py --baseline --popularity --k 5

The two baselines answer different questions. An untrained model shows the score
of pure chance, which is the floor. Popularity shows the score of the obvious
non-ML solution, which is the bar the Transformer actually has to clear to have
been worth building.

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


def score(logits_fn, X, Y, ks=(1, 5, 10), ndcg_k=10):
    """NDCG@ndcg_k and hit rate at each k, computed in batches.

    logits_fn takes a batch of sequences and returns (batch, vocab_size) scores.
    Anything that can produce scores works, which is how the model and the
    baselines below are measured by identical code.
    """
    gains, hits = [], {k: [] for k in ks}
    for start in range(0, len(X), BATCH):
        xb, yb = X[start:start + BATCH], Y[start:start + BATCH]
        logits = logits_fn(xb)
        gains.append(gains_at_k(logits, yb, ndcg_k))
        for k in ks:
            hits[k].append(hits_at_k(logits, yb, k))
    return (float(np.mean(np.concatenate(gains))),
            {k: float(np.mean(np.concatenate(v))) for k, v in hits.items()})


def popularity_scores(Y_train, vocab_size):
    """A fixed ranking of every song by how often it is the next song in training.

    This is the "no machine learning" solution: ignore the playlist entirely and
    always suggest the songs that come next most often overall. It is a much
    harder bar than an untrained model, because popular songs genuinely do follow
    lots of things.

    Ties are broken deterministically (by song ID) so every song gets a distinct
    score. That matters: raw counts tie thousands of songs at 1, and the ranking
    convention in metrics.py resolves ties in the target's favour, which would
    flatter this baseline for a reason that has nothing to do with its quality.
    """
    counts = np.bincount(np.asarray(Y_train, dtype=int), minlength=vocab_size)
    order = np.lexsort((np.arange(vocab_size), -counts))  # count desc, then ID asc
    scores = np.empty(vocab_size, dtype=np.float64)
    scores[order] = -np.arange(vocab_size, dtype=np.float64)  # most popular scores highest
    return scores, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=None, help="bundle folder (default: Config.bundle_dir)")
    ap.add_argument("--baseline", action="store_true",
                    help="also score an untrained model (the floor: pure chance)")
    ap.add_argument("--popularity", action="store_true",
                    help="also score a most-popular-songs baseline (the bar: no-ML solution)")
    ap.add_argument("--k", type=int, default=10, help="k for NDCG@k (default: 10)")
    args = ap.parse_args()

    cfg = Config()
    bundle_dir = args.bundle or cfg.bundle_dir

    # Seed before touching the data: the split must match the one training held out.
    np.random.seed(cfg.seed)
    engine.TRAINING = False  # no dropout; evaluation must be deterministic

    raw = load_spotify_data(cfg.data_folder, max_playlists=cfg.max_playlists)
    _, Y_train, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(
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

    header = f"\n{'split':<26}{'NDCG@' + str(args.k):>10}{'hit@1':>9}{'hit@5':>9}{'hit@10':>9}"
    print(header)

    def row(label, logits_fn, X, Y, ndcg_fmt="{:>10.4f}"):
        ndcg, hits = score(logits_fn, X, Y, ndcg_k=args.k)
        print(f"{label:<26}" + ndcg_fmt.format(ndcg) +
              f"{hits[1]:>9.3f}{hits[5]:>9.3f}{hits[10]:>9.3f}")
        return ndcg

    model_logits = lambda xb: model(xb).data
    row("validation subset", model_logits, X_test[:val_n], Y_test[:val_n])
    trained_ndcg = row("full held-out", model_logits, X_test, Y_test)

    if args.popularity:
        # Counted from TRAINING data only — this baseline never sees held-out playlists.
        pop, counts = popularity_scores(Y_train, vocab_size)
        pop_logits = lambda xb: np.broadcast_to(pop, (len(xb), vocab_size))
        pop_ndcg = row("most-popular baseline", pop_logits, X_test, Y_test)
        top = np.argsort(-counts)[:3]
        print(f"\n  most popular next-songs in training: "
              + ", ".join(f"{id_to_track[int(i)]} ({counts[i]}x)" for i in top))
        if trained_ndcg > pop_ndcg:
            print(f"  model beats most-popular by {trained_ndcg / pop_ndcg:.2f}x "
                  f"({trained_ndcg:.4f} vs {pop_ndcg:.4f})")
        else:
            print(f"  model LOSES to most-popular ({trained_ndcg:.4f} vs {pop_ndcg:.4f}) — "
                  f"the Transformer is not yet earning its complexity on this split")

    if args.baseline:
        # A fresh model with the same shape, never trained: the score of pure chance.
        np.random.seed(0)
        untrained = SongRecommender(vocab_size=vocab_size, embed_dim=meta["embed_dim"],
                                    context_length=meta["context_length"],
                                    num_layers=meta["num_layers"])
        rand_ndcg = row("untrained baseline", lambda xb: untrained(xb).data,
                        X_test, Y_test, ndcg_fmt="{:>10.5f}")
        print(f"\n  model beats an untrained one by {trained_ndcg / rand_ndcg:.0f}x")


if __name__ == "__main__":
    main()
