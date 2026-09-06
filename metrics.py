"""Ranking metrics, shared by train.py (per-epoch validation) and evaluate.py
(scoring a finished bundle), so the two can never drift apart.

Every function takes raw logits of shape (n_examples, vocab_size) — one row per
example, one column per song in the catalog — and the true next-song IDs, shape
(n_examples,). Nothing here needs the autograd Tensor; pass `.data`.

Imports only numpy, so evaluate.py and the tests can use it without matplotlib.
"""
import numpy as np


def target_ranks(logits, targets):
    """Rank of each true song among all songs, 1 = the model's top pick.

    rank = 1 + the number of songs scored strictly higher than the correct one.
    Ties therefore resolve in the true song's favour, which is the same
    convention the training loop has always used.
    """
    targets = np.asarray(targets)
    true_scores = logits[np.arange(len(targets)), targets][:, None]
    return np.sum(logits > true_scores, axis=1) + 1


def gains_at_k(logits, targets, k=10):
    """Per-example NDCG gain.

    A song at rank r scores 1/log2(1 + r), so rank 1 scores 1/log2(2) = 1.0 and
    the score decays smoothly with rank — the top few answers are treated as
    nearly as good as each other, then it tapers. Anything past rank k scores 0.

    There is exactly one relevant song per example, so the ideal DCG is 1 and
    normalising is a division by 1 — that is why NDCG here is just the gain.
    """
    ranks = target_ranks(logits, targets)
    gains = 1.0 / np.log2(ranks + 1)
    gains[ranks > k] = 0.0
    return gains


def ndcg_at_k(logits, targets, k=10):
    """Mean NDCG@k over the batch. This is the number training early-stops on."""
    return float(np.mean(gains_at_k(logits, targets, k)))


def hits_at_k(logits, targets, k=10):
    """Per-example bool: did the true song land in the top k?

    Reported alongside NDCG because it is easier to read — "the right song is in
    the top 10 about 6% of the time" says something NDCG's 0.033 does not.
    """
    return target_ranks(logits, targets) <= k
