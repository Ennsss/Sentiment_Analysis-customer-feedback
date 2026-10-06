import io
import pytest
from app import create_app


class StubEngine:
    def metadata(self):
        return {"loaded": True}

    def analyze(self, reviews):
        return [
            {
                "text": text,
                "label": "positive",
                "scores": {"positive": 0.8, "negative": 0.2},
            }
            for text in reviews
        ]


@pytest.fixture
def client():
    app = create_app(StubEngine())
    app.config["TESTING"] = True
    return app.test_client()


def test_home_and_single_review(client):
    assert client.get("/").status_code == 200
    response = client.post("/api/analyze", json={"reviews": [" Excellent camera "]})
    assert response.status_code == 200
    assert response.json["results"][0]["text"] == "Excellent camera"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    "reviews",
    [[], [""], ["  "], [None], [123], ["x" * 5001], ["ok"] * 101, "not-a-list"],
)
def test_bad_inputs(client, reviews):
    assert client.post("/api/analyze", json={"reviews": reviews}).status_code == 400


def test_csv_with_quotes_and_newline(client):
    response = client.post(
        "/api/analyze",
        data={
            "file": (
                io.BytesIO(
                    b'review\n"good camera, bad battery"\n"line one\nline two"\n'
                ),
                "reviews.csv",
            )
        },
    )
    assert response.status_code == 200
    assert len(response.json["results"]) == 2
    assert response.json["results"][1]["text"] == "line one\nline two"


def test_invalid_csv_column_and_encoding(client):
    for content in [b"wrong\nvalue", b"\xff\xfe"]:
        assert (
            client.post(
                "/api/analyze", data={"file": (io.BytesIO(content), "reviews.csv")}
            ).status_code
            == 400
        )


def test_upload_limit(client):
    response = client.post(
        "/api/analyze", data={"file": (io.BytesIO(b"x" * 1100000), "big.csv")}
    )
    assert response.status_code == 413


def test_missing_model_returns_error_not_fake_predictions():
    class BrokenEngine(StubEngine):
        def analyze(self, reviews):
            raise RuntimeError("model unavailable")

    client = create_app(BrokenEngine()).test_client()
    response = client.post("/api/analyze", json={"reviews": ["hello"]})
    assert response.status_code == 503 and "results" not in response.json
