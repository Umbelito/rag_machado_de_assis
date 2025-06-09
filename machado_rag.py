# machado_rag.py - Teste Comparativo de LLMs

import os
import json
import faiss
import numpy as np
import requests
import time
from utils import load_documents, create_embeddings, search_similar_chunks
from sentence_transformers import SentenceTransformer

# CONFIGURACOES
DATA_DIR = "/home/umbelito/rag_machado_de_assis/obras/raw/txt"
CATEGORIAS = ["romance", "cronica"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss_index.bin"
METADATA_PATH = "embeddings/metadata.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

# MODELOS PARA COMPARAÇÃO
LLM_3B_MODEL = "llama2"  # LLM de 3B parâmetros (sem RAG)
LLM_1B_MODEL = "tinyllama"  # LLM de 1B parâmetros (com RAG)
TOP_K = 5


def load_or_create_index():
    """Carrega ou cria o índice FAISS para RAG"""
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


def montar_prompt_rag(pergunta, trechos):
    """Monta prompt com contexto para RAG"""
    contexto = "\n".join(trechos)
    return f"""Contexto sobre obras de Machado de Assis:
{contexto}

Pergunta: {pergunta}

Responda baseado no contexto fornecido:"""


def montar_prompt_simples(pergunta):
    """Monta prompt simples sem contexto para geração pura"""
    return f"""Você é um especialista em literatura brasileira, especialmente na obra de Machado de Assis.

Pergunta: {pergunta}

Responda com base no seu conhecimento sobre Machado de Assis:"""


def gerar_resposta(prompt, model):
    """Gera resposta usando a LLM especificada"""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "[Erro ao gerar resposta]")
    except Exception as e:
        return f"[Erro na comunicação com o modelo {model}: {str(e)}]"


def salvar_resultados(pergunta, resposta_3b, resposta_1b, tempo_3b, tempo_1b):
    """Salva os resultados da comparação em arquivo JSON"""
    resultado = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pergunta": pergunta,
        "llm_3b_sem_rag": {
            "modelo": LLM_3B_MODEL,
            "resposta": resposta_3b,
            "tempo": tempo_3b
        },
        "llm_1b_com_rag": {
            "modelo": LLM_1B_MODEL,
            "resposta": resposta_1b,
            "tempo": tempo_1b
        }
    }
    
    os.makedirs("resultados", exist_ok=True)
    arquivo_resultado = f"resultados/comparacao_{int(time.time())}.json"
    
    with open(arquivo_resultado, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Resultados salvos em: {arquivo_resultado}")


if __name__ == "__main__":
    print("🔬 TESTE COMPARATIVO: LLM 3B vs LLM 1B + RAG")
    print("=" * 60)
    print(f"📖 LLM de 3B parâmetros: {LLM_3B_MODEL} (SEM RAG)")
    print(f"🔍 LLM de 1B parâmetros: {LLM_1B_MODEL} (COM RAG)")
    print("=" * 60)
    
    # Carrega o índice FAISS para RAG
    index, metadata = load_or_create_index()
    
    while True:
        pergunta = input("\nDigite sua pergunta sobre Machado de Assis (ou 'sair'): ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        
        if not pergunta.strip():
            print("❌ Por favor, digite uma pergunta válida.")
            continue
        
        # Executar comparação e capturar tempos
        print(f"\n🤔 Pergunta: {pergunta}")
        print("=" * 80)
        
        # 1. LLM de 3B sem RAG (geração pura)
        print("\n📚 LLM de 3B parâmetros (SEM RAG):")
        print("-" * 50)
        prompt_3b = montar_prompt_simples(pergunta)
        start_time = time.time()
        resposta_3b = gerar_resposta(prompt_3b, LLM_3B_MODEL)
        tempo_3b = time.time() - start_time
        print(f"⏱️  Tempo de resposta: {tempo_3b:.2f}s")
        print(f"💬 Resposta:\n{resposta_3b}")
        
        # 2. LLM de 1B com RAG
        print("\n🔍 LLM de 1B parâmetros (COM RAG):")
        print("-" * 50)
        trechos = search_similar_chunks(pergunta, index, metadata, EMBEDDING_MODEL, TOP_K)
        prompt_1b = montar_prompt_rag(pergunta, trechos)
        start_time = time.time()
        resposta_1b = gerar_resposta(prompt_1b, LLM_1B_MODEL)
        tempo_1b = time.time() - start_time
        print(f"⏱️  Tempo de resposta: {tempo_1b:.2f}s")
        print(f"📖 Trechos relevantes encontrados: {len(trechos)}")
        print(f"💬 Resposta:\n{resposta_1b}")
        
        # 3. Comparação
        print("\n📊 COMPARAÇÃO:")
        print("-" * 50)
        print(f"🕐 Tempo 3B (sem RAG): {tempo_3b:.2f}s")
        print(f"🕐 Tempo 1B (com RAG): {tempo_1b:.2f}s")
        print(f"⚡ Diferença: {abs(tempo_3b - tempo_1b):.2f}s")
        
        if tempo_1b < tempo_3b:
            print("🏆 LLM de 1B com RAG foi mais rápida!")
        else:
            print("🏆 LLM de 3B sem RAG foi mais rápida!")
        
        print("\n" + "=" * 80)
        
        salvar = input("\n💾 Deseja salvar os resultados? (s/n): ")
        if salvar.lower() in ["s", "sim", "y", "yes"]:
            salvar_resultados(pergunta, resposta_3b, resposta_1b, tempo_3b, tempo_1b)
    
    print("\n👋 Teste finalizado!")
