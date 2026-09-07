"""Tests for the data pipeline.

This is the part of the project that fails quietly. If the sliding window is off
by one, nothing crashes: training still runs, loss still falls, and the model
just learns the wrong task. So these tests use playlists small enough to work out
the correct answer by hand, and assert the exact windows and targets.
"""
import json

import numpy as np
import pytest

from data import load_spotify_data, tokenize_and_slice


def fresh(playlists):
    """tokenize_and_slice shuffles its argument in place, so hand it a copy."""
    return [list(p) for p in playlists]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

def test_songs_below_min_freq_are_dropped():
    # A x3, B x2, C x1 -> C is too rare to learn anything from
    playlists = [["A", "B", "C", "A", "B", "A"]]
    *_, vocab_size, id_to_track = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    assert vocab_size == 2
    assert set(id_to_track.values()) == {"A", "B"}


def test_min_freq_one_keeps_everything():
    playlists = [["A", "B", "C", "D"]]
    *_, vocab_size, _ = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=1)
    assert vocab_size == 4


def test_ids_are_contiguous_and_map_is_the_inverse():
    playlists = [["x", "y", "z"], ["z", "y", "x"]]
    *_, vocab_size, id_to_track = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    assert sorted(id_to_track) == list(range(vocab_size)), "IDs are 0..n-1 with no gaps"
    assert len(set(id_to_track.values())) == vocab_size, "no song appears twice"


def test_ids_follow_first_appearance_order():
    # Vocab is built before the shuffle, so IDs are deterministic run to run.
    playlists = [["first", "second", "first", "second"]]
    *_, id_to_track = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    assert id_to_track[0] == "first"
    assert id_to_track[1] == "second"


def test_vocab_covers_held_out_playlists_too():
    # Built over ALL playlists before the split, so a song only ever seen in the
    # held-out set still has an ID. Otherwise evaluation would hit a KeyError.
    np.random.seed(0)
    playlists = [["a", "b", "c", "d"] for _ in range(9)] + [["z", "z", "z", "z"]]
    *_, vocab_size, id_to_track = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.1, min_freq=2)
    assert "z" in id_to_track.values()
    assert vocab_size == 5


# --------------------------------------------------------------------------
# Sliding window: the off-by-one surface
# --------------------------------------------------------------------------

def test_windows_are_consecutive_and_target_is_the_next_song():
    # One playlist s0..s4 (duplicated so every song clears min_freq=2).
    # context_length=2 over 5 songs gives exactly 3 examples:
    #   [s0,s1]->s2   [s1,s2]->s3   [s2,s3]->s4
    playlists = [["s0", "s1", "s2", "s3", "s4"]] * 2
    X, Y, *_ = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    assert len(X) == 6, "3 windows per playlist, 2 playlists"
    np.testing.assert_array_equal(X[:3], [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(Y[:3], [2, 3, 4])


def test_window_count_matches_the_formula():
    # len(playlist) - context_length windows per playlist.
    playlists = [[f"s{i}" for i in range(10)]] * 2
    for cl in (2, 3, 5):
        X, _, *_ = tokenize_and_slice(
            fresh(playlists), context_length=cl, test_split=0.0, min_freq=2)
        assert len(X) == 2 * (10 - cl), f"context_length={cl}"
        assert X.shape[1] == cl


def test_playlist_shorter_than_the_window_yields_nothing():
    playlists = [["a", "b"]] * 2  # 2 songs, window of 2 -> no room for a target
    X, Y, *_ = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    assert len(X) == 0 and len(Y) == 0


def test_windows_touching_a_rare_song_are_skipped():
    # "rare" appears once, so it has no ID. Every window that overlaps it, as
    # input or as target, must vanish rather than be silently mangled.
    with_rare = ["a", "b", "rare", "c", "d"]     # all 3 of its windows touch "rare"
    clean_1 = ["a", "b", "c", "d", "a"]          # 3 windows
    clean_2 = ["b", "c", "d", "a", "b"]          # 3 windows
    X, _, *_ = tokenize_and_slice(
        fresh([with_rare, clean_1, clean_2]), context_length=2, test_split=0.0, min_freq=2)
    assert len(X) == 6, "only the two clean playlists contribute"


def test_a_rare_target_alone_is_enough_to_skip():
    # Window is fine, but the song being predicted is out of vocab.
    playlists = [["a", "b", "solo"], ["a", "b", "a", "b"]]
    X, Y, *_ = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    # playlist 1: [a,b]->solo is skipped. playlist 2: [a,b]->a, [b,a]->b.
    assert len(X) == 2
    assert "solo" not in [str(v) for v in Y]


def test_every_id_in_the_output_is_a_real_vocab_id():
    np.random.seed(0)
    playlists = [[f"s{i % 7}" for i in range(20)] for _ in range(5)]
    X, Y, X_te, Y_te, vocab_size, _ = tokenize_and_slice(
        fresh(playlists), context_length=3, test_split=0.2, min_freq=2)
    for arr in (X, Y, X_te, Y_te):
        if len(arr):
            assert arr.min() >= 0 and arr.max() < vocab_size


# --------------------------------------------------------------------------
# Train / held-out split
# --------------------------------------------------------------------------

def test_split_divides_playlists_by_the_requested_fraction():
    np.random.seed(0)
    playlists = [[f"s{i % 5}" for i in range(6)] for _ in range(10)]
    X, _, X_te, _, *_ = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.2, min_freq=2)
    # 10 playlists -> 8 train / 2 held out, each yielding 6 - 2 = 4 windows
    assert len(X) == 8 * 4
    assert len(X_te) == 2 * 4


def test_test_split_zero_holds_nothing_out():
    playlists = [["a", "b", "c", "d"]] * 4
    X, _, X_te, _, *_ = tokenize_and_slice(
        fresh(playlists), context_length=2, test_split=0.0, min_freq=2)
    assert len(X) > 0 and len(X_te) == 0


def test_split_is_reproducible_under_the_same_seed():
    playlists = [[f"p{p}s{i}" for i in range(5)] for p in range(10)] * 2
    np.random.seed(42)
    a = tokenize_and_slice(fresh(playlists), context_length=2, test_split=0.3, min_freq=2)
    np.random.seed(42)
    b = tokenize_and_slice(fresh(playlists), context_length=2, test_split=0.3, min_freq=2)
    np.testing.assert_array_equal(a[2], b[2])  # X_test identical
    np.testing.assert_array_equal(a[3], b[3])  # Y_test identical


# --------------------------------------------------------------------------
# Reading the Spotify JSON
# --------------------------------------------------------------------------

def write_slice(folder, name, playlists):
    """One file in the Million Playlist Dataset's format."""
    body = {"info": {"slice": name},
            "playlists": [{"tracks": [{"track_name": t} for t in p]} for p in playlists]}
    (folder / name).write_text(json.dumps(body))


def test_reads_tracks_and_drops_playlists_of_three_or_fewer(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-999.json",
                [["t0", "t1", "t2", "t3", "t4"],  # 5 tracks, kept
                 ["u0", "u1", "u2"],              # 3 tracks, dropped
                 ["v0", "v1", "v2", "v3"]])       # 4 tracks, kept
    out = load_spotify_data(str(tmp_path), max_playlists=10)
    assert len(out) == 2
    assert out[0] == ["t0", "t1", "t2", "t3", "t4"]
    assert out[1] == ["v0", "v1", "v2", "v3"]


def test_stops_at_max_playlists(tmp_path):
    write_slice(tmp_path, "mpd.slice.0-999.json",
                [[f"s{i}"] * 5 for i in range(50)])
    assert len(load_spotify_data(str(tmp_path), max_playlists=7)) == 7


def test_reads_files_in_sorted_order(tmp_path):
    # File order decides song IDs, so it must not depend on filesystem order.
    write_slice(tmp_path, "mpd.slice.1000-1999.json", [["second"] * 5])
    write_slice(tmp_path, "mpd.slice.0-999.json", [["first"] * 5])
    out = load_spotify_data(str(tmp_path), max_playlists=10)
    assert out[0][0] == "first", "the 0-999 slice must be read first"


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_spotify_data(str(tmp_path / "not_here"), max_playlists=10)


def test_folder_without_json_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("no data here")
    with pytest.raises(FileNotFoundError):
        load_spotify_data(str(tmp_path), max_playlists=10)
