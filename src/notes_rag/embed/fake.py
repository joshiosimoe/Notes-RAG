"""Deterministic embedder for tests. No network, stable across processes."""

import hashlib
import math
from collections.abc import Sequence


class FakeEmbedder:
    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        # Expand the digest until it covers the requested dimensions, so the
        # same text always maps to the same point regardless of dimension count.
        raw = b""
        counter = 0
        while len(raw) < self.dimensions:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        components = [byte / 255.0 - 0.5 for byte in raw[: self.dimensions]]
        magnitude = math.sqrt(sum(value**2 for value in components)) or 1.0
        return [value / magnitude for value in components]
