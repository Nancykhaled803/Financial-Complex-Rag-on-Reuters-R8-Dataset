import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", str(text).lower())


@dataclass
class BM25Index:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi

    def scores(self, keywords: list[str], entities: list[str] | None = None):
        merged_terms = []
        seen = set()
        for term in [*(entities or []), *keywords]:
            cleaned = str(term).strip()
            key = cleaned.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged_terms.append(cleaned)
        query_tokens = tokenize(" ".join(merged_terms))
        return self.bm25.get_scores(query_tokens)


def embed_bm25(collection) -> BM25Index:
    """Build a BM25 index over the documents already stored in Chroma."""
    stored = collection.get(include=["documents", "metadatas"])
    documents = stored["documents"] or []
    # print(f"Building BM25 index over {len(documents)} documents")
    return BM25Index(
        ids=list(stored["ids"]),
        documents=list(documents),
        metadatas=list(stored["metadatas"]),
        bm25=BM25Okapi([tokenize(document) for document in documents]),
    )
