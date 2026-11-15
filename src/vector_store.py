
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@staticmethod
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_documents(query, index, chunks, embedding_model, top_k=4):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            **chunks[idx],
            "score": float(score)
        })

    return results
