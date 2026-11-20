import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def ask_question() -> str:
    preset = os.environ.get("USER_QUERY", "").strip()
    if preset:
        print("Question:", preset, flush=True)
        return preset
    print("Enter your question:", flush=True)
    question = input().strip()
    if not question:
        raise SystemExit("No question entered.")
    return question


if __name__ == "__main__":
    user_query = ask_question()

if Path("/kaggle/working").is_dir():
    print("Checking packages...", flush=True)
    missing = []
    for module_name in ("chromadb", "groq", "rank_bm25", "dotenv", "sentence_transformers"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        print("Installing missing packages...", flush=True)
        requirements = ROOT / "requirements.txt"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
        )

print("Loading libraries...", flush=True)
from rag import embed_bge, embed_bm25, get_categories, load_data, query_llm, query_planning, rerank, rrf


def run(user_query: str) -> str:
    print("Step 1 started: load data")
    df = load_data()
    categories = get_categories(df)

    print("Step 2 started: embed with BGE")
    collection = embed_bge(df)

    print("Step 3 started: embed with BM25")
    bm25_index = embed_bm25(collection)

    print("Step 4 started: query planning")
    plan = query_planning(user_query, categories)

    print("Step 5 started: similarity search (RRF)")
    candidates = rrf(collection, bm25_index, plan)

    print("Step 6 started: reranker")
    reranked = rerank(plan.semantic_query, candidates, plan.category_filters)

    print("Step 7 started: query LLM")
    return query_llm(user_query, reranked, plan.category_filters)


if __name__ == "__main__":
    run(user_query)
