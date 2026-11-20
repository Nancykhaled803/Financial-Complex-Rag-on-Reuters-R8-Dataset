import hashlib

import chromadb
import torch
from chromadb.utils import embedding_functions

from rag.config import BGE_MODEL, CHROMA_COLLECTION, CHROMA_PATH


def generate_content_hash(text: str) -> str:
    normalized = " ".join(str(text).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def embed_bge(
    df,
    chroma_path: str | None = None,
    collection_name: str | None = None,
    model_name: str | None = None,
    batch_size: int = 500,
):
    """Embed the clean articles with BGE and store them in Chroma.

    If the collection already has documents, it is reused and nothing is re-embedded.
    """
    path = chroma_path or CHROMA_PATH
    name = collection_name or CHROMA_COLLECTION
    model = model_name or BGE_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Embedding using device: {device}")

    client = chromadb.PersistentClient(path=path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model,
        device=device,
        normalize_embeddings=True,
    )
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0:
        # print(f"Loaded existing Chroma collection: {name}")
        # print(f"Existing documents: {existing_count}")
        # print("Skipping indexing step.")
        return collection

    chroma_df = df.reset_index(drop=True).copy()
    chroma_df["document_id"] = chroma_df["original_text"].apply(generate_content_hash)
    chroma_df["category"] = chroma_df["label"]
    chroma_df = chroma_df.drop_duplicates(subset=["document_id"], keep="first")

    documents = chroma_df["text"].astype(str).tolist()
    ids = chroma_df["document_id"].tolist()
    metadatas = chroma_df[
        ["category", "source", "split", "document_id", "original_text"]
    ].to_dict(orient="records")

    # print(f"Collection is empty. Indexing {len(ids)} documents...")
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        # print(f"Indexed {min(end, len(ids))}/{len(ids)} documents")

    # print(f"Chroma collection ready. Total documents: {collection.count()}")
    return collection
