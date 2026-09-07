import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(bundle):
    # app.py loads the bundle when imported, so point it at the fake one first.
    os.environ["BUNDLE_DIR"] = bundle.dir
    import app
    return TestClient(app.app)


def test_health_reports_metadata(client, bundle):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] == len(bundle.vocab)
    assert body["context_length"] == 10
    assert body["best_ndcg_at_10"] == 0.123
    assert "num_parameters" in body
    assert "git_sha" in body


def test_songs_search(client):
    r = client.get("/songs", params={"q": "song 1", "limit": 2})
    assert r.status_code == 200
    assert r.json() == {"matches": ["Song 1", "Song 10"]}


def test_songs_requires_query(client):
    assert client.get("/songs").status_code == 422


def test_recommend_good_input_gives_200(client, ten_songs):
    r = client.post("/recommend", json={"songs": ten_songs, "k": 3})
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert len(recs) == 3
    assert set(recs[0]) == {"song", "prob"}
    assert not {x["song"] for x in recs} & set(ten_songs)


def test_recommend_too_short_gives_400(client, ten_songs):
    r = client.post("/recommend", json={"songs": ten_songs[:9]})
    assert r.status_code == 400
    assert "at least 10" in r.json()["detail"]


def test_recommend_unknown_song_gives_400_and_names_it(client, ten_songs):
    r = client.post("/recommend", json={"songs": ten_songs[:9] + ["Zzz"]})
    assert r.status_code == 400
    assert "Zzz" in r.json()["detail"]


def test_recommend_wrong_json_shape_gives_422(client):
    assert client.post("/recommend", json={"songs": "not a list"}).status_code == 422


def test_recommend_k_out_of_range_gives_422(client, ten_songs):
    assert client.post("/recommend", json={"songs": ten_songs, "k": 500}).status_code == 422


def test_root_serves_the_demo_page(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<title>Transformer Song Recommender</title>" in body
    # The page is useless if it cannot reach the API, so check it wires to both routes.
    assert "/recommend" in body and "/songs" in body


def test_demo_page_preloads_a_ten_song_example(client):
    # One click from a result: an empty playlist would ask visitors to invent ten
    # song names the model happens to know, which loses most of them.
    body = client.get("/").text
    example = body.split("const EXAMPLE = [")[1].split("];")[0]
    assert example.count('"') // 2 == 10
