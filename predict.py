import numpy as np
import engine
from config import Config
from data import load_spotify_data, tokenize_and_slice
from checkpoint import load_bundle # From checkpoint, not train: keeps matplotlib out of the import chain

def show_sample_predictions(cfg=None, n=5, k=5):
    cfg = cfg or Config()
    np.random.seed(cfg.seed) # Reproduce the SAME vocab + held-out split as training
    engine.TRAINING = False # Inference: full deterministic model, no dropout

    raw = load_spotify_data(cfg.data_folder, max_playlists=cfg.max_playlists)
    _, _, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(
        raw, cfg.context_length, test_split=cfg.test_split, min_freq=cfg.min_freq)

    # Load AFTER the split. Building a model draws from np.random, and doing that before
    # tokenize_and_slice would change its shuffle and leak training playlists into "held-out".
    model, vocab, meta = load_bundle(cfg.bundle_dir) # The best-NDCG bundle from training

    # X_test holds IDs from the data's vocab; the model's rows come from the bundle's vocab.
    # They must be the same vocab or the demo is meaningless.
    if [id_to_track[i] for i in range(vocab_size)] != vocab:
        raise ValueError(f"'{cfg.data_folder}' produces a different vocab than the bundle in "
                         f"'{cfg.bundle_dir}' was trained on.")

    X, Y_true = X_test[:n], Y_test[:n]
    logits = model(X).data # (n, vocab_size)

    print("\n=== SAMPLE PREDICTIONS (held-out playlists) ===")
    for i in range(n):
        context = " -> ".join(id_to_track[int(s)] for s in X[i])
        top_k = np.argsort(-logits[i])[:k] # Indices of the k highest scores
        actual_id = int(Y_true[i])
        hit = "HIT" if actual_id in top_k else "miss"
        print(f"\n--- Example {i + 1} ---")
        print(f"Context: {context}")
        for rank, sid in enumerate(top_k, 1):
            print(f"  Pred {rank}: {id_to_track[int(sid)]}")
        print(f"Actual next: {id_to_track[actual_id]}   [{hit}]")


if __name__ == "__main__":
    show_sample_predictions()
