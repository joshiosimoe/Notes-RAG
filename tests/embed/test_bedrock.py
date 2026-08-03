import json

import pytest

from notes_rag.embed.bedrock import TitanEmbedder


class StubBedrockClient:
    """Records calls and returns a fixed embedding."""

    def __init__(self, dimensions: int = 4) -> None:
        self.calls: list[dict] = []
        self.dimensions = dimensions

    def invoke_model(self, *, modelId: str, body: str):
        self.calls.append({"modelId": modelId, "body": json.loads(body)})
        payload = json.dumps({"embedding": [0.5] * self.dimensions})
        return {"body": _FakeStream(payload)}


class _FakeStream:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode()


def test_returns_one_vector_per_text():
    client = StubBedrockClient()
    embedder = TitanEmbedder(dimensions=4, client=client)
    assert embedder.embed(["a", "b"]) == [[0.5] * 4, [0.5] * 4]


def test_calls_the_titan_v2_model_id():
    client = StubBedrockClient()
    TitanEmbedder(dimensions=4, client=client).embed(["a"])
    assert client.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"


def test_requests_the_configured_dimensions_and_normalisation():
    client = StubBedrockClient()
    TitanEmbedder(dimensions=4, client=client).embed(["a"])
    body = client.calls[0]["body"]
    assert body["dimensions"] == 4
    assert body["normalize"] is True
    assert body["inputText"] == "a"


def test_empty_input_makes_no_calls():
    client = StubBedrockClient()
    assert TitanEmbedder(dimensions=4, client=client).embed([]) == []
    assert client.calls == []


def test_raises_when_response_dimensions_disagree():
    client = StubBedrockClient(dimensions=3)
    with pytest.raises(ValueError, match="dimensions"):
        TitanEmbedder(dimensions=4, client=client).embed(["a"])


@pytest.mark.integration
def test_real_titan_call_returns_1024_dimensions():
    """Confirms spec §10 item 1: listing is not entitlement.

    Run explicitly: pytest -m integration tests/embed/test_bedrock.py
    """
    vectors = TitanEmbedder().embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
