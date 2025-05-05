# machado_rag.py

import os
import json
import faiss
import numpy as np
import requests
from utils import load_documents, create_embeddings, search_similar_chunks
from sentence_transformers import SentenceTransformer

# CONFIGURACOES
DATA_DIR = "/home/umbelito/IPCC/obras/raw/txt"
CATEGORIAS = ["romance", "cronica"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss_index.bin"
METADATA_PATH = "embeddings/metadata.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "tinyllama"
TOP_K = 5


def load_or_create_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("\n✅ Índice FAISS encontrado. Carregando...")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    print("\n🔧 Construindo índice FAISS...")
    docs, metadata = load_documents(DATA_DIR, CATEGORIAS)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = create_embeddings(docs, model)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("index", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return index, metadata


def montar_prompt(pergunta, trechos):
    contexto = "\n".join(trechos)
    return f"Contexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"


def gerar_resposta(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return response.json().get("response", "[Erro ao gerar resposta]")


if __name__ == "__main__":
    index, metadata = load_or_create_index()

    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair'): ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        trechos = search_similar_chunks(pergunta, index, metadata, EMBEDDING_MODEL, TOP_K)
        prompt = montar_prompt(pergunta, trechos)
        resposta = gerar_resposta(prompt)

        print("\n🧠 Resposta gerada:\n")
        print(resposta)
        print("\n" + "-" * 50)
