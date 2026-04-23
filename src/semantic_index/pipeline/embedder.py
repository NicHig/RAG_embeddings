from __future__ import annotations

from typing import Protocol

from openai import OpenAI


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    def __init__(
        self, *, api_key: str, model: str, dimensions: int | None = None
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


class DeterministicEmbedder:
    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text)
            vectors.append(
                [float(((seed + idx * 17) % 1000) / 1000.0) for idx in range(self.dimensions)]
            )
        return vectors
