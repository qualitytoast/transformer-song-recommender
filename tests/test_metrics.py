"""Metrics are the numbers the README quotes, so they get tested on cases whose
answers can be worked out by hand."""
import numpy as np

from metrics import gains_at_k, hits_at_k, ndcg_at_k, target_ranks


def test_rank_one_when_target_scores_highest():
    logits = np.array([[5.0, 1.0, 0.0]])
    assert target_ranks(logits, [0]) == [1]


def test_rank_counts_songs_scored_higher():
    logits = np.array([[1.0, 5.0, 3.0]])  # song 0 is beaten by songs 1 and 2
    assert target_ranks(logits, [0]) == [3]


def test_ties_favour_the_target():
    logits = np.array([[1.0, 1.0, 1.0]])  # nothing scores strictly higher
    assert target_ranks(logits, [2]) == [1]


def test_perfect_ranking_scores_one():
    logits = np.array([[9.0, 1.0], [1.0, 9.0]])
    assert ndcg_at_k(logits, [0, 1]) == 1.0


def test_gain_decays_with_rank():
    logits = np.array([[3.0, 2.0, 1.0]])
    assert gains_at_k(logits, [0])[0] == 1.0                       # rank 1 -> 1/log2(2)
    assert np.isclose(gains_at_k(logits, [1])[0], 1 / np.log2(3))  # rank 2
    assert np.isclose(gains_at_k(logits, [2])[0], 1 / np.log2(4))  # rank 3


def test_beyond_k_scores_zero():
    logits = np.array([np.arange(20, 0, -1, dtype=float)])  # song 0 best ... song 19 worst
    assert gains_at_k(logits, [15], k=10)[0] == 0.0
    assert gains_at_k(logits, [15], k=20)[0] > 0.0


def test_ndcg_is_the_mean_of_gains():
    rng = np.random.RandomState(0)
    logits, targets = rng.randn(7, 30), rng.randint(0, 30, 7)
    assert np.isclose(ndcg_at_k(logits, targets), np.mean(gains_at_k(logits, targets)))


def test_hits_match_ranks():
    rng = np.random.RandomState(1)
    logits, targets = rng.randn(20, 50), rng.randint(0, 50, 20)
    assert np.array_equal(hits_at_k(logits, targets, 5), target_ranks(logits, targets) <= 5)


def test_hit_rate_is_one_when_every_target_is_top():
    logits = np.zeros((4, 10))
    logits[np.arange(4), [3, 1, 4, 0]] = 1.0
    assert hits_at_k(logits, [3, 1, 4, 0], k=1).all()
