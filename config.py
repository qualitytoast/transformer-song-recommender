from dataclasses import dataclass

@dataclass
class Config:
    """Single source of truth for every hyperparameter and run setting."""
    # Reproducibility
    seed: int = 42

    # Data
    data_folder: str = "data"
    max_playlists: int = 5000
    min_freq: int = 2 # Drop songs appearing fewer than this many times
    context_length: int = 10 # Input window size ("predict the 11th from 10")
    test_split: float = 0.1 # Fraction of playlists held out for evaluation (save 10% of playlists for testing)
    val_size: int = 500 # Validation examples used for the loss / NDCG check

    # Model
    embed_dim: int = 64 # Size of feature-vectors
    num_layers: int = 2

    # Training
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.05
    patience: int = 10 # Early-stopping patience (epochs w/o NDCG improvement)

    # Artifacts
    weight_file: str = "spotify_transformer_weights.npz"