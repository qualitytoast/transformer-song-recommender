"""Shared fixtures. One fake bundle is built per test session and reused everywhere."""
from types import SimpleNamespace

import pytest

from tests.fake_bundle import make_fake_bundle


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """A tiny bundle in a temp folder: 20 made-up songs, 1 layer, random weights.
    Exposes .dir, .model, .vocab, .cfg."""
    out_dir = str(tmp_path_factory.mktemp("bundle"))
    model, vocab, cfg = make_fake_bundle(out_dir)
    return SimpleNamespace(dir=out_dir, model=model, vocab=vocab, cfg=cfg)


@pytest.fixture(scope="session")
def ten_songs(bundle):
    """A valid input: the first context_length (10) songs of the vocab."""
    return bundle.vocab[:bundle.cfg.context_length]
