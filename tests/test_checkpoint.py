import json
import os
import shutil

import numpy as np
import pytest

import checkpoint
import engine
from checkpoint import load_bundle, load_weights, named_parameters, save_bundle
from model import SongRecommender


def test_named_parameters_matches_parameters(bundle):
    named = named_parameters(bundle.model)
    plain = bundle.model.parameters()
    assert len(named) == len(plain)
    assert all(a is b for (_, a), b in zip(named, plain)), "same tensors, same order"
    names = [n for n, _ in named]
    assert len(set(names)) == len(names), "names are unique"
    assert names[:2] == ["embedding.weight", "position_embedding.weight"]
    assert "blocks.0.attention.W_query.W" in names
    assert names[-1] == "matchmaker.B"


def test_bundle_has_three_files(bundle):
    assert sorted(os.listdir(bundle.dir)) == ["metadata.json", "vocab.json", "weights.npz"]


def test_round_trip_restores_everything(bundle):
    model2, vocab2, meta = load_bundle(bundle.dir)
    assert vocab2 == bundle.vocab
    assert meta["vocab_size"] == len(bundle.vocab)
    assert meta["embed_dim"] == bundle.cfg.embed_dim
    assert meta["context_length"] == bundle.cfg.context_length
    assert meta["best_ndcg_at_10"] == 0.123
    assert meta["config"]["seed"] == bundle.cfg.seed
    for (n1, p1), (n2, p2) in zip(named_parameters(bundle.model), named_parameters(model2)):
        assert n1 == n2
        assert np.array_equal(p1.data, p2.data), n1


def test_round_trip_gives_identical_forward_pass(bundle):
    model2, _, _ = load_bundle(bundle.dir)
    engine.TRAINING = False
    X = np.arange(bundle.cfg.context_length)[None, :]
    assert np.allclose(bundle.model(X).data, model2(X).data)


def test_missing_weights_file_raises(bundle):
    with pytest.raises(FileNotFoundError):
        load_weights(bundle.model, os.path.join(bundle.dir, "nope.npz"))


def test_missing_bundle_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(str(tmp_path / "does_not_exist"))


def test_wrong_shape_raises_and_names_the_array(bundle):
    wrong = SongRecommender(vocab_size=len(bundle.vocab), embed_dim=bundle.cfg.embed_dim * 2,
                            context_length=bundle.cfg.context_length, num_layers=bundle.cfg.num_layers)
    with pytest.raises(ValueError, match="embedding.weight"):
        load_weights(wrong, os.path.join(bundle.dir, checkpoint.WEIGHTS_FILE))


def test_missing_array_raises_and_names_it(bundle):
    bigger = SongRecommender(vocab_size=len(bundle.vocab), embed_dim=bundle.cfg.embed_dim,
                             context_length=bundle.cfg.context_length, num_layers=bundle.cfg.num_layers + 1)
    with pytest.raises(KeyError, match="blocks.1"):
        load_weights(bigger, os.path.join(bundle.dir, checkpoint.WEIGHTS_FILE))


def test_format_version_mismatch_raises(bundle, tmp_path):
    d = str(tmp_path / "b")
    shutil.copytree(bundle.dir, d)
    meta_path = os.path.join(d, checkpoint.META_FILE)
    with open(meta_path) as f:
        meta = json.load(f)
    meta["format_version"] = 999
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="format version"):
        load_bundle(d)


def test_vocab_length_mismatch_raises(bundle, tmp_path):
    d = str(tmp_path / "b")
    shutil.copytree(bundle.dir, d)
    with open(os.path.join(d, checkpoint.VOCAB_FILE), "w") as f:
        json.dump(bundle.vocab[:-1], f)
    with pytest.raises(ValueError, match="vocab.json"):
        load_bundle(d)


def test_save_accepts_dict_or_list(bundle, tmp_path):
    save_bundle(bundle.model, dict(enumerate(bundle.vocab)), bundle.cfg, str(tmp_path / "d"))
    save_bundle(bundle.model, bundle.vocab, bundle.cfg, str(tmp_path / "l"))
    assert load_bundle(str(tmp_path / "d"))[1] == bundle.vocab
    assert load_bundle(str(tmp_path / "l"))[1] == bundle.vocab
