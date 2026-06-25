import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # non-interactive backend: lets training run headless / unattended
import matplotlib.pyplot as plt

from engine import SGD
from model import SongRecommender, cross_entropy_loss
from data import load_spotify_data, tokenize_and_slice

def save_model(model, filename="transformer_weights.npz"):
    """Extracts raw numpy arrays from the model's Tensors and saves to disk."""
    weights = [p.data for p in model.parameters()]
    np.savez(filename, *weights)
    print(f"\n[SYSTEM] Weights successfully saved to {filename}")

def load_model(model, filename="transformer_weights.npz"):
    """Overwrites the random initialization with saved weights if they exist."""
    if os.path.exists(filename):
        loaded = np.load(filename)
        params = model.parameters()
        for i, p in enumerate(params):
            p.data = loaded[f'arr_{i}']
        print(f"\n[SYSTEM] Previous weights loaded from {filename}. Resuming training...")
    else:
        print(f"\n[SYSTEM] No saved weights found at {filename}. Starting from scratch.")

def train_transformer():
    print("\n--- TRANSFORMER INITIALIZATION ---")
    DATA_FOLDER = "data"
    CONTEXT_LENGTH = 10 # Input window size "Predict the 11th song from 10"
    EMBED_DIM = 64 # Feature-vector size for each song
    WEIGHT_FILE = "spotify_transformer_weights.npz" # Where we save/load trained weights
    
    raw_playlists = load_spotify_data(DATA_FOLDER, max_playlists=5000)
    
    # Notice the new X_test and Y_test variables
    X_train, Y_train, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(raw_playlists, CONTEXT_LENGTH)
    
    model = SongRecommender(vocab_size=vocab_size, embed_dim=EMBED_DIM, context_length = CONTEXT_LENGTH, num_layers=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    
    load_model(model, WEIGHT_FILE) # Loads saved weights if it exists
    
    # One-Hot Encode the Train targets, converts the target song IDs into one-hot vectors
    # The y-side prep that leads to loss calculation
    # Make a zero matrix (num_examples, vocab_size) — one row per example, one column per song.
    # For each example i, set the column at Y_train[i] (the correct song's ID) to 1.0.
    num_train_samples = len(Y_train)

    # Grab a small static validation subset from the vault to keep the evaluation fast
    # Helps measure generalization (if the model is learning not just memorizing) during training using
    # a small 500 X_inputs with 500 corresponding Y_targets
    val_size = min(500, len(X_test))
    X_val = X_test[:val_size]
    Y_val_raw = Y_test[:val_size]
    Y_val_onehot = np.zeros((val_size, vocab_size), dtype=np.float32)
    for i in range(val_size):
        Y_val_onehot[i, Y_val_raw[i]] = 1.0

    print("\n--- TRAINING & VALIDATION ---")
    epochs = 30 # Num. of full passes over the training data to do
    batch_size = 32 # How many examples to process per gradient update
    
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
    
    # Best-checkpoint saving, only call save_model when validation loss improves on the best val_loss seen so far, disk only holds the best model
    # Patience-based stopping, if NDCG@10 fails to improve for "patience" consecutive epochs, stop training
    # best_val_loss = float('inf')
    best_ndcg = -1.0
    patience = 10
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
        # We run the forward pass, but we skip .backward() so it cannot learn from this!
        val_logits = model(X_val)
        val_loss_node = cross_entropy_loss(val_logits, Y_val_onehot)
        val_loss_history.append(val_loss_node.data)
        
        # Compute validation accuracy on test data using NDCG@10
        # Take the logit the model outputted, find the correct song, and find num. of songs the model put above that song
        # Calculate rank, rank = 1 + num. of songs scored strictly higher than the correct song
        # Give a final score given that rank, score = 1/log2(1 + rank), so ideal score is 1/log2(2) = 1. 
        # Use a log function since log is a smoother curve (top 3 answers are given similar scores, then a gradual dropoff). Normalize (divide by 1, but this is done trivially)
        K = 10
        true_scores = val_logits.data[np.arange(val_size), Y_val_raw][:, None] # Grab the score the model gave the correct song, per example
        ranks = np.sum(val_logits.data > true_scores, axis=1) + 1 # Find how many score outscored the correct song, add 1 to get rank
        gains = 1.0 / np.log2(ranks + 1)
        gains[ranks > K] = 0.0 # Not in top-K, set to 0
        val_ndcg = np.mean(gains) # Average score across all validation examples (across all (X_train, Y_target) pairs)
        ndcg_history.append(val_ndcg)
        
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_node.data:.4f} | NDCG@10: {val_ndcg:.4f}")
        
        # Only save on model improvement
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            epochs_without_improvement = 0
            save_model(model, WEIGHT_FILE)
        else:
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


if __name__ == "__main__":
    train_transformer()