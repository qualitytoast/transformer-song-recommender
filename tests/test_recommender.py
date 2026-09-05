import pytest

import engine
from recommender import Recommender


@pytest.fixture(scope="module")
def rec(bundle):
    return Recommender(bundle.model, bundle.vocab)


def test_constructor_turns_dropout_off(bundle):
    engine.TRAINING = True
    Recommender(bundle.model, bundle.vocab)
    assert engine.TRAINING is False


def test_context_length_comes_from_model(rec, bundle):
    assert rec.context_length == bundle.cfg.context_length == 10


def test_nine_songs_raise(rec, ten_songs):
    with pytest.raises(ValueError, match="at least 10"):
        rec.recommend(ten_songs[:9])


def test_unknown_name_raises_and_names_it(rec, ten_songs):
    with pytest.raises(ValueError) as e:
        rec.recommend(ten_songs[:8] + ["Not A Song", "Also Fake"])
    assert "Not A Song" in str(e.value)
    assert "Also Fake" in str(e.value)


def test_twelve_songs_use_the_last_ten(rec, bundle, ten_songs):
    twelve = bundle.vocab[10:12] + ten_songs
    assert rec.recommend(twelve, k=5) == rec.recommend(ten_songs, k=5)


def test_inputs_never_appear_in_output(rec, ten_songs):
    picks = rec.recommend(ten_songs, k=20)  # ask for more than exist outside the input
    names = [name for name, _ in picks]
    assert not set(names) & set(ten_songs)
    assert len(names) == 10  # 20 in vocab minus the 10 inputs


def test_returns_k_results_best_first(rec, ten_songs):
    picks = rec.recommend(ten_songs, k=5)
    assert len(picks) == 5
    probs = [p for _, p in picks]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 < p <= 1.0 for p in probs)


def test_deterministic(rec, ten_songs):
    assert rec.recommend(ten_songs, k=5) == rec.recommend(ten_songs, k=5)


def test_search_is_case_insensitive_substring(rec):
    assert rec.search("SONG 1") == ["Song 1"] + [f"Song 1{d}" for d in range(10)]


def test_search_honours_limit(rec):
    assert rec.search("song", limit=3) == ["Song 0", "Song 1", "Song 2"]
