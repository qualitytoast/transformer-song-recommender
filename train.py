import os
import numpy as np
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
    optimizer = SGD(model.parameters(), lr=0.5)
    
    load_model(model, WEIGHT_FILE) # Loads saved weights if it exists
    
    # One-Hot Encode the Train targets, converts the target song IDs into one-hot vectors
    # The y-side prep that leads to loss calculation
    # Make a zero matrix (num_examples, vocab_size) — one row per example, one column per song.
    # For each example i, set the column at Y_train[i] (the correct song's ID) to 1.0.
    num_train_samples = len(Y_train)
    Y_train_onehot = np.zeros((num_train_samples, vocab_size), dtype=np.float32)
    for i in range(num_train_samples):
        Y_train_onehot[i, Y_train[i]] = 1.0

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

    # Everything is staged
    # data        → X_train, Y_train_onehot, X_val, Y_val_onehot, vocab_size, id_to_track
    # model       → SongRecommender (random or loaded weights)
    # optimizer   → SGD over all model parameters
    # config      → epochs=30, batch_size=32
    # logging     → train/val loss histories
    for epoch in range(epochs):
        # Each epoch uses the exact same training set (X_train)
        # Shuffle to make each epoch's batch different. Makes sure our model doesn't learn order-specific patterns
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled_onehot = Y_train_onehot[indices]
        
        epoch_loss = 0.0 # Accumulator for the total loss this epoch
        
        # TRAINING BATCHES
        for start_idx in range(0, num_train_samples, batch_size):
            # Slice data into mini-batches of 32
            end_idx = min(start_idx + batch_size, num_train_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch_onehot = Y_shuffled_onehot[start_idx:end_idx]
            
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

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_node.data:.4f} | NDCG@10: {val_ndcg:.4f}")
        
        save_model(model, WEIGHT_FILE)

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
    plt.show()

if __name__ == "__main__":
    train_transformer()