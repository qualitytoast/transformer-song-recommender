import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # non-interactive backend: lets training run headless / unattended
import matplotlib.pyplot as plt

import engine
from engine import SGD
from model import SongRecommender, cross_entropy_loss
from data import load_spotify_data, tokenize_and_slice
from config import Config
from tracking import ExperimentTracker
from checkpoint import save_bundle, load_weights, read_vocab, read_metadata, WEIGHTS_FILE
from metrics import ndcg_at_k

def train_transformer(cfg=None):
    cfg = cfg or Config()
    tracker = ExperimentTracker(cfg)
    print("\n--- TRANSFORMER INITIALIZATION ---")
    print(f"Config: {cfg}")

    # Reproducibility: seed ONCE, up front, so weight init + every shuffle is deterministic.
    np.random.seed(cfg.seed)

    raw_playlists = load_spotify_data(cfg.data_folder, max_playlists=cfg.max_playlists)

    X_train, Y_train, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(
        raw_playlists, cfg.context_length, test_split=cfg.test_split, min_freq=cfg.min_freq)

    model = SongRecommender(vocab_size=vocab_size, embed_dim=cfg.embed_dim, context_length=cfg.context_length,
                            num_layers=cfg.num_layers, dropout_rate=cfg.dropout_rate)
    optimizer = SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Resuming is explicit (Config.resume), never "a weights file happened to be there".
    # best_ndcg starts from the bundle's metadata, so the first epoch after a resume only
    # overwrites the bundle if it genuinely beats the previous best.
    best_ndcg = -1.0
    if cfg.resume:
        if read_vocab(cfg.bundle_dir) != [id_to_track[i] for i in range(vocab_size)]:
            raise ValueError(f"Cannot resume: '{cfg.data_folder}' produces a different vocab than "
                             f"the bundle in '{cfg.bundle_dir}' was trained on.")
        load_weights(model, os.path.join(cfg.bundle_dir, WEIGHTS_FILE))
        best_ndcg = read_metadata(cfg.bundle_dir).get("best_ndcg_at_10") or -1.0
        print(f"\n[SYSTEM] Resumed from {cfg.bundle_dir}/ (best NDCG@10 so far: {best_ndcg:.4f})")
    else:
        print("\n[SYSTEM] Starting from scratch.")
    
    # One-Hot Encode the Train targets, converts the target song IDs into one-hot vectors
    # The y-side prep that leads to loss calculation
    # Make a zero matrix (num_examples, vocab_size) — one row per example, one column per song.
    # For each example i, set the column at Y_train[i] (the correct song's ID) to 1.0.
    num_train_samples = len(Y_train)

    # Grab a small static validation subset from the vault to keep the evaluation fast
    # Helps measure generalization (if the model is learning not just memorizing) during training using
    # a small 500 X_inputs with 500 corresponding Y_targets
    val_size = min(cfg.val_size, len(X_test))
    X_val = X_test[:val_size]
    Y_val_raw = Y_test[:val_size]
    Y_val_onehot = np.zeros((val_size, vocab_size), dtype=np.float32)
    for i in range(val_size):
        Y_val_onehot[i, Y_val_raw[i]] = 1.0

    print("\n--- TRAINING & VALIDATION ---")
    epochs = cfg.epochs # Num. of full passes over the training data to do
    batch_size = cfg.batch_size # How many examples to process per gradient update
    
    # Logging Arrays to record loss of each epoch
    # Train_loss tells us the model is learning, val_loss tells us if its generalizing or memorizing
    train_loss_history = []
    val_loss_history = []
    ndcg_history = []

    # Everything is staged
    # data        → X_train, Y_train_onehot, X_val, Y_val_onehot, vocab_size, id_to_track
    # model       → SongRecommender (random or loaded weights)
    # optimizer   → SGD over all model parameters
    # config      → epochs=30, batch_size=32
    # logging     → train/val loss histories
    
    # Best-checkpoint saving: save_bundle only when validation NDCG beats best_ndcg (set above), so
    # disk only ever holds the best model. best_ndcg is -1 for a fresh run, or the bundle's value on resume.
    # Patience-based stopping, if NDCG@10 fails to improve for "patience" consecutive epochs, stop training
    patience = cfg.patience
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Each epoch uses the exact same training set (X_train)
        # Shuffle to make each epoch's batch different. Makes sure our model doesn't learn order-specific patterns
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        
        epoch_loss = 0.0 # Accumulator for the total loss this epoch
        
        # TRAINING BATCHES
        engine.TRAINING = True # Dropout ON for training
        for start_idx in range(0, num_train_samples, batch_size):
            # Slice data into mini-batches of 32
            end_idx = min(start_idx + batch_size, num_train_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            # Build the one-hot for this batch -> (batch_size, vocab_size)
            Y_batch = Y_shuffled[start_idx:end_idx]
            Y_batch_onehot = np.zeros((len(Y_batch), vocab_size), dtype=np.float32)
            Y_batch_onehot[np.arange(len(Y_batch)), Y_batch] = 1.0
            
            # Forward pass, run the batch through the model
            # Embedding -> transformer blocks -> last position -> matchmaker -> logits
            logits = model(X_batch)
            # Calculate loss
            loss = cross_entropy_loss(logits, Y_batch_onehot)

            # Divergence guard: a NaN/inf loss would otherwise poison every weight on the next step
            # and keep going silently. Fail here, with the epoch and batch, instead.
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"Loss became {loss.data} at epoch {epoch}, "
                                         f"batch {start_idx // batch_size}. Training diverged.")
            
            # Clear old gradients from last training run
            optimizer.zero_grad()

            # Backward pass
            # Topo sort autograd graph (start with loss, go all the way down to first Tensors -> run _backward in reverse, filling in .grad for every weight
            loss.backward()

            # Update the weights with the new gradients
            optimizer.step()
            
            # Accumulate the loss, weighted by the batch's actual size. Multiplying by batch size means each example contributes equally
            # even if the last batch is smaller, creates a true per-examples average not a per-batch one
            epoch_loss += loss.data * (end_idx - start_idx)
        
        # Divide the accumulated loss by the total num. of examples -> average training loss for this epoch. Record it for the learning curve
        avg_train_loss = epoch_loss / num_train_samples
        train_loss_history.append(avg_train_loss)
        
        # VALIDATION CHECK (Testing the Vault data)
        engine.TRAINING = False # Dropout OFF for testing
        # We run the forward pass, but we skip .backward() so it cannot learn from this!
        val_logits = model(X_val)
        val_loss_node = cross_entropy_loss(val_logits, Y_val_onehot)
        val_loss_history.append(val_loss_node.data)
        
        # Compute validation accuracy on test data using NDCG@10.
        # Rank the correct song against the whole catalog, then score it by 1/log2(1 + rank)
        # rank = 1 + num. of songs scored strictly higher than the correct song
        # so the top few answers score similarly and it tapers off. See metrics.py for the math.
        # evaluate.py calls this same function, so the training metric and the reported
        # metric can never drift apart.
        K = 10
        val_ndcg = ndcg_at_k(val_logits.data, Y_val_raw, K)
        ndcg_history.append(val_ndcg)
        tracker.log_epoch(epoch,
                          train_loss=float(avg_train_loss),
                          val_loss=float(val_loss_node.data),
                          ndcg=float(val_ndcg))
        
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_node.data:.4f} | NDCG@10: {val_ndcg:.4f}")
        
        # Only save on model improvement
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            epochs_without_improvement = 0
            save_bundle(model, id_to_track, cfg, cfg.bundle_dir, best_ndcg=float(val_ndcg))
        elif epoch >= cfg.min_epochs:
            epochs_without_improvement += 1
            print(f"[EARLY STOP] No NDCG improvement for {epochs_without_improvement}/{patience} "
                  f"(best NDCG@10: {best_ndcg:.4f})")
            if epochs_without_improvement >= patience:
                print(f"[EARLY STOP] Stopping at epoch {epoch}; best NDCG@10: {best_ndcg:.4f}")
                break

    # Plots training loss vs. validation loss over epochs
    # Both decreasing together = healthy learning, Train ↓ but val ↑ → overfitting (memorizing NOT learning)
    print("\nTraining complete. Generating Loss Curve...")
    plt.style.use('dark_background') # Looks better for terminal-based developers
    plt.plot(train_loss_history, label="Training Loss (Memorization)", color='cyan')
    plt.plot(val_loss_history, label="Validation Loss (True Understanding)", color='magenta')
    plt.title("Transformer Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig("loss_curve.png", dpi=120, bbox_inches="tight")
    print(f"[SYSTEM] Loss curve saved to loss_curve.png")
    print(f"[SYSTEM] Best NDCG@10 over {epochs} epochs: {max(ndcg_history):.4f}")
    tracker.finish(best_ndcg=float(max(ndcg_history)), epochs_run=len(ndcg_history))

if __name__ == "__main__":
    train_transformer()