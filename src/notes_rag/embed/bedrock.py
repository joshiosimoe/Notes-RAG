"""Amazon Titan Text Embeddings v2 via Bedrock.

Titan is not an Anthropic model, so the Bedrock Anthropic use-case form does not
gate it — but listing a model is not the same as being entitled to it. The
integration test in tests/embed/test_bedrock.py is the entitlement check
(spec §10 item 1); run it before relying on this class.
"""

import json
from collections.abc import Sequence

MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = "us-east-2"


class TitanEmbedder:
    def __init__(
        self,
        *,
        region: str = DEFAULT_REGION,
        dimensions: int = 1024,
        client=None,
    ) -> None:
        self.dimensions = dimensions
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Titan v2 accepts one inputText per InvokeModel call. Incremental
        # embedding means a typical indexer run sends only a handful.
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        response = self._client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": True,
                }
            ),
        )
        vector = json.loads(response["body"].read())["embedding"]
        if len(vector) != self.dimensions:
            raise ValueError(f"Titan returned {len(vector)} dimensions, expected {self.dimensions}")
        return vector
