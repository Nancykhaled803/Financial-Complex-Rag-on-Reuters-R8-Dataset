from collections import defaultdict

from rag.embed_bm25 import BM25Index
from rag.query_planning import QueryPlan

RRF_K = 60
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _as_dict(query_plan: QueryPlan | dict) -> dict:
    if isinstance(query_plan, QueryPlan):
        return query_plan.model_dump()
    return query_plan


def rrf(
    collection,
    bm25_index: BM25Index,
    query_plan: QueryPlan | dict,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Fuse BGE similarity search and BM25 with reciprocal rank fusion.

    Retrieval is run per category from the query plan. Every unique fused
    candidate is returned; the reranker chooses the final order.
    """
    plan = _as_dict(query_plan)
    semantic_query = plan["semantic_query"]
    keywords = plan["keywords"]
    entities = plan.get("entities", [])
    categories = list(dict.fromkeys(plan["category_filters"]))

    if len(categories) > 1:
        top_k_per_retriever = 4
    else:
        top_k_per_retriever = 5

    query_for_embedding = DENSE_QUERY_PREFIX + semantic_query
    bm25_scores = bm25_index.scores(keywords, entities)
    category_groups = categories or ["__ALL__"]

    document_lookup = dict(zip(bm25_index.ids, bm25_index.documents))
    metadata_lookup = dict(zip(bm25_index.ids, bm25_index.metadatas))
    final_results = []

    for category in category_groups:
        query_kwargs = {
            "query_texts": [query_for_embedding],
            "n_results": top_k_per_retriever,
            "include": ["documents", "metadatas", "distances"],
        }
        if category != "__ALL__":
            query_kwargs["where"] = {"category": category}

        dense_results = collection.query(**query_kwargs)
        dense_ids = dense_results["ids"][0]

        if category == "__ALL__":
            eligible_indices = list(range(len(bm25_index.ids)))
        else:
            eligible_indices = [
                i
                for i, meta in enumerate(bm25_index.metadatas)
                if meta.get("category") == category
            ]

        bm25_ranked_indices = sorted(
            eligible_indices,
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:top_k_per_retriever]
        bm25_ids = [bm25_index.ids[i] for i in bm25_ranked_indices]

        rrf_scores = defaultdict(float)
        dense_ranks = {}
        bm25_ranks = {}
        for rank, doc_id in enumerate(dense_ids, start=1):
            rrf_scores[doc_id] += 1 / (rrf_k + rank)
            dense_ranks[doc_id] = rank
        for rank, doc_id in enumerate(bm25_ids, start=1):
            rrf_scores[doc_id] += 1 / (rrf_k + rank)
            bm25_ranks[doc_id] = rank

        for doc_id in sorted(rrf_scores, key=rrf_scores.get, reverse=True):
            final_results.append(
                {
                    "document_id": doc_id,
                    "document": document_lookup[doc_id],
                    "metadata": metadata_lookup[doc_id],
                    "rrf_score": rrf_scores[doc_id],
                    "dense_rank": dense_ranks.get(doc_id),
                    "bm25_rank": bm25_ranks.get(doc_id),
                }
            )

    # print("Intent:", plan["intent"])
    # print("Categories:", categories or "No category filters")
    # print("Unique candidates after RRF:", len(final_results))
    return final_results
