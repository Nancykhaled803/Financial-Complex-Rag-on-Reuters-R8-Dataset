# Financial RAG

Hybrid search over the Reuters R8 news articles. The script loads the data, embeds it with BAAI/bge-base-en-v1.5, builds a BM25 index, plans the question, fuses both searches with reciprocal rank fusion, reranks the candidates, and answers with a Groq model.

The script asks for the question after it starts. Do not put the question on the run command.

## Run on Kaggle

1. Upload this folder (`main.py`, `rag/`, and `requirements.txt`) as a Kaggle dataset. Do not upload `.env`.
2. Attach that dataset and the financial dataset that contains `r8-train-all-terms.txt` and `r8-test-all-terms.txt`.
3. In the notebook settings, turn **Internet** on. A GPU accelerator is optional and makes embedding and reranking faster.
4. Open **Add-ons → Secrets**. Add a secret whose label is exactly `GROQ_API_KEY`, paste the Groq key into the value, and check the box beside it so this notebook can use it.

Run this cell. It passes the secret into the script, then the script asks for the question.

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")
%run /your-kaggle-DS-Directory/Financial_Rag/main.py
```

When it prints `Enter your question:`, type one question and press Enter. For example: `Compare grain and crude oil market developments`.

The first run can stay quiet for a minute while it loads PyTorch and Chroma. It then prints `Loading libraries...` and the seven step lines.

The first run downloads the embedding model and builds the Chroma database in `/kaggle/working/chroma_db`. A later run in the same session reuses that database. `/kaggle/working` is cleared when the session ends.

You should see seven lines, one per step, and then the model answer.

## Run locally

From the `Financial_Rag` folder:

```bash
pip install -r requirements.txt
python main.py
```

No code change is required. The script uses Kaggle paths only when `/kaggle/working` exists. On your machine it reads `.env` and writes the database to `./chroma_db`.

Create a file named `.env` in the `Financial_Rag` folder, next to `main.py`. The Groq key is required. The other lines already match this project and can stay as they are.

```
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_REASONING_EFFORT=medium

DATA_PATH=
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=reuters_r8

BGE_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

Leave `DATA_PATH` empty to use a clean file named `reuters_clean.csv` or `reuters_clean.parquet` inside this folder or a `data` folder next to it. If that file is missing, the script looks for `r8-train-all-terms.txt` and `r8-test-all-terms.txt` and cleans them in memory. Set `DATA_PATH` only when the clean file lives somewhere else.

A CUDA GPU is used automatically when PyTorch can see one. Otherwise the same script runs on CPU.
