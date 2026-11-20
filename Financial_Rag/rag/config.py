import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ON_KAGGLE = Path("/kaggle/working").is_dir()


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


KAGGLE_SECRET_LABELS = (
    "GROQ_API_KEY",
    "groq api key",
    "groq_api_key",
    "Groq API Key",
    "GROQ API KEY",
)


def _groq_api_key() -> str:
    key = env("GROQ_API_KEY")
    if key or not ON_KAGGLE:
        return key

    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    errors = []
    for label in KAGGLE_SECRET_LABELS:
        try:
            secret = client.get_secret(label).strip()
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        if secret:
            # print(f"Using Kaggle secret: {label}")
            return secret

    # print("Could not read a Groq key from Kaggle secrets.")
    # for error in errors:
    #     print(" ", error)
    return ""


if ON_KAGGLE:
    # Keep model downloads on the writable session disk.
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/kaggle/working/hf_cache")
    CHROMA_PATH = "/kaggle/working/chroma_db"
else:
    CHROMA_PATH = env("CHROMA_PATH") or str(ROOT / "chroma_db")

DATA_PATH = env("DATA_PATH")
CHROMA_COLLECTION = env("CHROMA_COLLECTION") or "reuters_r8"

GROQ_API_KEY = _groq_api_key()
GROQ_MODEL = env("GROQ_MODEL") or "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = env("GROQ_REASONING_EFFORT") or "medium"

BGE_MODEL = env("BGE_MODEL") or "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = env("RERANKER_MODEL") or "BAAI/bge-reranker-base"
