"""Wraps a loaded SongRecommender so callers deal in song names, not integer IDs.

Built once at server startup (see app.py) and reused for every request. Holds
the model, the two name<->ID lookup tables, and the input rules:

    - at least context_length songs (10); longer inputs keep the last 10,
      which is exactly what a training window is
    - every name must be in the vocab, spelled exactly as the model knows it
    - the input songs are never recommended back
"""
import numpy as np

import engine


class Recommender:
    def __init__(self, model, vocab):
        engine.TRAINING = False  # dropout off: same input, same output, every time
        self.model = model
        self.id_to_track = vocab  # list: vocab[1234] is the name of song 1234
        self.track_to_id = {name: i for i, name in enumerate(vocab)}
        self._lower = [name.lower() for name in vocab]  # for case-insensitive search
        # Read the window size off the model rather than hardcoding 10.
        self.context_length = model.position_embedding.weight.data.shape[0]

    def search(self, query, limit=20):
        """Case-insensitive substring search over the vocab. Returns names spelled
        exactly as the model knows them, so they can be passed to recommend()."""
        q = query.lower()
        hits = [self.id_to_track[i] for i, name in enumerate(self._lower) if q in name]
        return hits[:limit]

    def recommend(self, songs, k=10):
        """songs: list of song names. Returns up to k (name, probability) pairs, best
        first. Fewer than k only if the vocab has fewer than k songs outside the input.
        Raises ValueError with a readable message on bad input."""
        # Length rule.
        if len(songs) < self.context_length:
            raise ValueError(f"Need at least {self.context_length} songs, got {len(songs)}")
        songs = songs[-self.context_length:]

        # Names -> IDs. Report every unknown name at once, not just the first.
        unknown = [s for s in songs if s not in self.track_to_id]
        if unknown:
            raise ValueError(f"Unknown songs: {unknown}")
        ids = [self.track_to_id[s] for s in songs]

        # Forward pass. Same call as train.py, but a batch of one: X is (1, 10).
        X = np.array([ids])
        logits = self.model(X).data[0]  # (vocab_size,) one score per song

        # Scores -> probabilities (numerically stable softmax).
        shifted = logits - logits.max()
        exps = np.exp(shifted)
        probs = exps / exps.sum()

        # Rank every song, skip the ones already in the playlist, keep the top k.
        # Looking at k + len(ids) candidates is enough to step over every input.
        exclude = set(ids)
        ranked = np.argsort(-probs)[: k + len(ids)]
        top = [int(i) for i in ranked if int(i) not in exclude][:k]

        # IDs -> names.
        return [(self.id_to_track[i], float(probs[i])) for i in top]
