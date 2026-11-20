import json

from groq import Groq

from rag.config import GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT

ANSWER_SYSTEM_PROMPT = """
You are a Financial News and Market Intelligence Assistant.

Your task is to answer the user's question using ONLY
the provided Reuters news articles.

Rules:
1. Use only information explicitly supported by the articles.
2. Do not introduce external knowledge or assumptions.
3. Never invent dates, figures, prices, or quantities.
4. If a number or detail is missing, clearly mention that.
5. Support factual claims with the exact article IDs.
6. Never create or modify article IDs.
7. If the articles do not provide enough evidence,
   clearly mention this limitation.
8. For comparison questions, organize the answer into
   separate sections for each topic, followed by a comparison.
9. Answer in the same language as the user's question.

Be accurate, evidence-based, and avoid unsupported conclusions.
"""


def _select_context(reranked_results: list[dict], categories: list[str], top_k: int) -> list[dict]:
    if not categories:
        ranked = sorted(reranked_results, key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k]

    selected = []
    for category in categories:
        group = [
            item
            for item in reranked_results
            if item["metadata"].get("category") == category
        ]
        group = sorted(group, key=lambda item: item["rerank_score"], reverse=True)
        selected.extend(group[:top_k])
    return selected


def query_llm(
    user_query: str,
    reranked_results: list[dict],
    categories: list[str] | None = None,
    top_k: int = 4,
    model_name: str | None = None,
    temperature: float = 0.2,
) -> str:
    """Answer the user question from the top reranked articles."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is empty. In Kaggle Secrets, name the secret "
            "GROQ_API_KEY or 'groq api key'."
        )

    category_filters = list(dict.fromkeys(categories or []))
    context_results = _select_context(reranked_results, category_filters, top_k)
    articles_context = []
    for item in context_results:
        metadata = item["metadata"]
        articles_context.append(
            {
                "article_id": item["document_id"],
                "category": metadata.get("category", "unknown"),
                "article_text": metadata.get("original_text", item["document"]),
            }
        )

    context = json.dumps(articles_context, ensure_ascii=False, indent=2)
    user_prompt = f"""
User Question:
{user_query}

Retrieved Reuters Articles:
{context}

Please provide a complete and well-structured answer.

Use this structure when relevant:
- Overview
- First topic
- Second topic
- Comparison
- Key takeaways

Cite factual claims using the exact article IDs provided.
Do not invent missing information or numerical values.
"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model_name or GROQ_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=4000,
        reasoning_effort=GROQ_REASONING_EFFORT,
    )
    answer = response.choices[0].message.content
    # print("Number of context articles:", len(context_results))
    # print("Model:", model_name or GROQ_MODEL)
    # print("\nFinal Answer:\n")
    print(answer)
    return answer
