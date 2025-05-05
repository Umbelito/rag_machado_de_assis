import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import re


def split_text(text, max_words=150):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    chunk = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if count + len(words) > max_words:
            chunks.append(" ".join(chunk))
            chunk = words
            count = len(words)
        else:
            chunk.extend(words)
            count += len(words)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


def load_documents(base_dir, categorias):
    docs = []
    metadata = {}
    for categoria in categorias:
        cat_path = Path(base_dir) / categoria
        for txt_file in cat_path.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                partes = split_text(content)
                for i, trecho in enumerate(partes):
                    docs.append(trecho)
                    metadata[len(metadata)] = {
                        "arquivo": str(txt_file.name),
                        "categoria": categoria,
                        "trecho": trecho[:200]  # para debug rápido
                    }
    return docs, metadata


def create_embeddings(docs, model):
    print("\n🔍 Gerando embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    return np.array(embeddings)


def search_similar_chunks(query, index, metadata, embedding_model_name, top_k):
    model = SentenceTransformer(embedding_model_name)
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    trechos = [metadata[idx]["trecho"] for idx in indices[0]]
    return trechos
