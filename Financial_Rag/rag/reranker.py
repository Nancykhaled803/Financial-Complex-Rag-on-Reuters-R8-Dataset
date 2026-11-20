import torch
from sentence_transformers import CrossEncoder

from rag.config import RERANKER_MODEL

_reranker = None


def _get_reranker(model_name: str) -> CrossEncoder:
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(model_name, max_length=512, device=device)
        # print("Reranker loaded successfully.")
        # print("Device:", device)
    return _reranker


def rerank(
    semantic_query: str,
    candidates: list[dict],
    categories: list[str] | None = None,
    model_name: str | None = None,
    batch_size: int = 16,
) -> list[dict]:
    """Score RRF candidates with BAAI/bge-reranker-base.

    Multiple categories are sorted inside each category so a comparison
    query keeps documents from every requested category.
    """
    if not candidates:
        return []

    model = _get_reranker(model_name or RERANKER_MODEL)
    pairs = [(semantic_query, item["document"]) for item in candidates]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    scored = []
    for item, score in zip(candidates, scores):
        result = item.copy()
        result["rerank_score"] = float(score)
        scored.append(result)

    category_filters = list(dict.fromkeys(categories or []))
    if len(category_filters) > 1:
        ordered = []
        for category in category_filters:
            group = [
                item
                for item in scored
                if item["metadata"].get("category") == category
            ]
            ordered.extend(
                sorted(group, key=lambda item: item["rerank_score"], reverse=True)
            )
        return ordered

    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored
