import json

from groq import Groq
from pydantic import BaseModel, ConfigDict

from rag.config import GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT

PLANNER_INSTRUCTIONS = """
You are a Financial News Query Planner.

Your task is to understand user questions and
convert them into structured search plans.

Available Reuters R8 categories:
{categories}

Instructions:

1. Identify the user's intent.
2. Extract mentioned entities.
3. Identify financial topics.
4. Map topics to relevant R8 categories.
5. Rewrite the question for semantic search.
6. Extract keywords for BM25 search.

Rules:
- Do not answer the user's question.
- Never invent entities.
- Use an empty list if no entities exist.
- Use an empty category_filters list if uncertain.
- Preserve the original meaning of the question.

For keywords:
- Extract meaningful financial terms only.
- Exclude generic words such as:
  what, is, the, recent, change, tell, about.
- Include relevant synonyms when useful.
- Avoid unnecessary duplicate keywords.

TOPIC EXTRACTION RULES:

- Topics should represent the actual financial
  subject, not the dataset category.
- Keep topics descriptive and specific.

Examples:
Category: crude
Topic: crude oil prices

Category: acq
Topic: company acquisitions

KEYWORD RULES:

- Extract specific financial keywords from the query.
- Include relevant synonyms when useful.
- For acquisitions, consider:
  acquisition, merger, takeover, buyout.
- Avoid generic words such as:
  corporate, recent, change, news.

Allowed intents:
news_retrieval
comparison
impact_analysis
summarization
"""


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str
    entities: list[str]
    topics: list[str]
    category_filters: list[str]
    semantic_query: str
    keywords: list[str]


def _groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is empty. In Kaggle Secrets, name the secret "
            "GROQ_API_KEY or 'groq api key'."
        )
    return Groq(api_key=GROQ_API_KEY)


def query_planning(
    user_query: str,
    categories: list[str],
    model_name: str | None = None,
    temperature: float = 0,
) -> QueryPlan:
    """Turn a user question into a structured search plan."""
    client = _groq_client()
    response = client.chat.completions.create(
        model=model_name or GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": PLANNER_INSTRUCTIONS.format(categories=categories),
            },
            {"role": "user", "content": user_query},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "query_plan",
                "strict": True,
                "schema": QueryPlan.model_json_schema(),
            },
        },
        temperature=temperature,
        reasoning_effort=GROQ_REASONING_EFFORT,
    )
    plan = QueryPlan.model_validate_json(response.choices[0].message.content)
    # print(json.dumps(plan.model_dump(), indent=4))
    return plan
