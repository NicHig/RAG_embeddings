from __future__ import annotations

import tiktoken


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model: str) -> int:
    encoding = _encoding_for_model(model)
    return len(encoding.encode(text))


def split_text_by_tokens(
    *,
    text: str,
    model: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    encoding = _encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    overlap = min(overlap_tokens, max_tokens // 4)
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = encoding.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(tokens):
            break
        start = max(0, end - overlap)
    return chunks
