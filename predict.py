import numpy as np
from config import Config
from data import load_spotify_data, tokenize_and_slice
from model import SongRecommender
from train import load_model # Reuse — importing train won't run it (it's guarded by __main__)


def show_sample_predictions(cfg=None, n=5, k=5):
    cfg = cfg or Config()
    np.random.seed(cfg.seed) # Reproduce the SAME vocab + held-out split as training

    raw = load_spotify_data(cfg.data_folder, max_playlists=cfg.max_playlists)
    _, _, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(
        raw, cfg.context_length, test_split=cfg.test_split, min_freq=cfg.min_freq)

    model = SongRecommender(vocab_size=vocab_size, embed_dim=cfg.embed_dim,
                            context_length=cfg.context_length, num_layers=cfg.num_layers)
    load_model(model, cfg.weight_file) # The best-NDCG checkpoint from training

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
