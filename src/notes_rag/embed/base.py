"""Embedding interface. Keeps Bedrock out of every unit test."""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input text, in the same order."""
