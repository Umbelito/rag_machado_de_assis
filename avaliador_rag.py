# avaliador_rag.py - Comparação entre LLM 3B vs LLM 1B + RAG

import pandas as pd
import requests
import time
import json
import os
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
from utils import load_documents, create_embeddings, search_similar_chunks

# CONFIGURAÇÕES
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

def calcular_bertscore(respostas_geradas, referencias):
    """Calcula BERTScore entre respostas geradas e referências"""
    try:
        P, R, F1 = bertscore(respostas_geradas, referencias, lang="pt")
        return F1.tolist()
    except Exception as e:
        print(f"⚠️  Erro ao calcular BERTScore: {e}")
        return [0.0] * len(respostas_geradas)

def comparar_llms(perguntas, respostas_referencia):
    """Compara LLM de 3B sem RAG vs LLM de 1B com RAG"""
    
    print("🔬 INICIANDO COMPARAÇÃO: LLM 3B vs LLM 1B + RAG")
    print("=" * 60)
    
    # Carregar índice FAISS
    index, metadata = load_or_create_index()
    
    resultados = []
    
    for i, (pergunta, referencia) in enumerate(tqdm(zip(perguntas, respostas_referencia), 
                                                   total=len(perguntas), 
                                                   desc="Comparando LLMs")):
        
        print(f"\n📝 Pergunta {i+1}/{len(perguntas)}: {pergunta[:100]}...")
        
        # 1. LLM de 3B sem RAG (geração pura)
        prompt_3b = montar_prompt_simples(pergunta)
        start_time = time.time()
        resposta_3b = gerar_resposta(prompt_3b, LLM_3B_MODEL)
        tempo_3b = time.time() - start_time
        
        # 2. LLM de 1B com RAG
        trechos = search_similar_chunks(pergunta, index, metadata, EMBEDDING_MODEL, TOP_K)
        prompt_1b = montar_prompt_rag(pergunta, trechos)
        start_time = time.time()
        resposta_1b = gerar_resposta(prompt_1b, LLM_1B_MODEL)
        tempo_1b = time.time() - start_time
        
        # Salvar resultado
        resultado = {
            "pergunta": pergunta,
            "resposta_referencia": referencia,
            "llm_3b_sem_rag": {
                "modelo": LLM_3B_MODEL,
                "resposta": resposta_3b,
                "tempo": tempo_3b,
                "prompt": prompt_3b
            },
            "llm_1b_com_rag": {
                "modelo": LLM_1B_MODEL,
                "resposta": resposta_1b,
                "tempo": tempo_1b,
                "prompt": prompt_1b,
                "trechos_encontrados": len(trechos),
                "trechos": trechos
            }
        }
        
        resultados.append(resultado)
        
        # Mostrar progresso
        print(f"   ⏱️  3B (sem RAG): {tempo_3b:.2f}s")
        print(f"   ⏱️  1B (com RAG): {tempo_1b:.2f}s")
        print(f"   📊 Diferença: {abs(tempo_3b - tempo_1b):.2f}s")
    
    return resultados

def avaliar_resultados(resultados):
    """Avalia os resultados usando BERTScore e outras métricas"""
    
    print("\n📊 AVALIANDO RESULTADOS...")
    print("=" * 50)
    
    # Extrair dados para análise
    perguntas = [r["pergunta"] for r in resultados]
    referencias = [r["resposta_referencia"] for r in resultados]
    respostas_3b = [r["llm_3b_sem_rag"]["resposta"] for r in resultados]
    respostas_1b = [r["llm_1b_com_rag"]["resposta"] for r in resultados]
    tempos_3b = [r["llm_3b_sem_rag"]["tempo"] for r in resultados]
    tempos_1b = [r["llm_1b_com_rag"]["tempo"] for r in resultados]
    
    # Calcular BERTScore
    print("🔍 Calculando BERTScore...")
    bertscore_3b = calcular_bertscore(respostas_3b, referencias)
    bertscore_1b = calcular_bertscore(respostas_1b, referencias)
    
    # Estatísticas de tempo
    media_tempo_3b = np.mean(tempos_3b)
    media_tempo_1b = np.mean(tempos_1b)
    std_tempo_3b = np.std(tempos_3b)
    std_tempo_1b = np.std(tempos_1b)
    
    # Estatísticas BERTScore
    media_bertscore_3b = np.mean(bertscore_3b)
    media_bertscore_1b = np.mean(bertscore_1b)
    std_bertscore_3b = np.std(bertscore_3b)
    std_bertscore_1b = np.std(bertscore_1b)
    
    # Adicionar métricas aos resultados
    for i, resultado in enumerate(resultados):
        resultado["llm_3b_sem_rag"]["bertscore"] = bertscore_3b[i]
        resultado["llm_1b_com_rag"]["bertscore"] = bertscore_1b[i]
    
    # Relatório final
    print("\n📈 RELATÓRIO DE COMPARAÇÃO")
    print("=" * 60)
    
    print(f"\n🤖 LLM de 3B parâmetros (SEM RAG):")
    print(f"   Modelo: {LLM_3B_MODEL}")
    print(f"   Tempo médio: {media_tempo_3b:.2f}s ± {std_tempo_3b:.2f}s")
    print(f"   BERTScore médio: {media_bertscore_3b:.4f} ± {std_bertscore_3b:.4f}")
    
    print(f"\n🔍 LLM de 1B parâmetros (COM RAG):")
    print(f"   Modelo: {LLM_1B_MODEL}")
    print(f"   Tempo médio: {media_tempo_1b:.2f}s ± {std_tempo_1b:.2f}s")
    print(f"   BERTScore médio: {media_bertscore_1b:.4f} ± {std_bertscore_1b:.4f}")
    
    print(f"\n⚡ COMPARAÇÃO:")
    print(f"   Diferença de tempo: {abs(media_tempo_3b - media_tempo_1b):.2f}s")
    print(f"   Diferença de BERTScore: {abs(media_bertscore_3b - media_bertscore_1b):.4f}")
    
    if media_tempo_1b < media_tempo_3b:
        print(f"   🏆 LLM 1B com RAG é {((media_tempo_3b - media_tempo_1b)/media_tempo_3b)*100:.1f}% mais rápida!")
    else:
        print(f"   🏆 LLM 3B sem RAG é {((media_tempo_1b - media_tempo_3b)/media_tempo_1b)*100:.1f}% mais rápida!")
    
    if media_bertscore_1b > media_bertscore_3b:
        print(f"   🏆 LLM 1B com RAG tem BERTScore {((media_bertscore_1b - media_bertscore_3b)/media_bertscore_3b)*100:.1f}% melhor!")
    else:
        print(f"   🏆 LLM 3B sem RAG tem BERTScore {((media_bertscore_3b - media_bertscore_1b)/media_bertscore_1b)*100:.1f}% melhor!")
    
    return resultados

def salvar_resultados(resultados):
    """Salva os resultados em formato JSON e CSV"""
    
    # Salvar JSON completo
    os.makedirs("resultados", exist_ok=True)
    timestamp = int(time.time())
    arquivo_json = f"resultados/comparacao_completa_{timestamp}.json"
    
    with open(arquivo_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    
    # Criar DataFrame para CSV
    dados_csv = []
    for resultado in resultados:
        dados_csv.append({
            "pergunta": resultado["pergunta"],
            "resposta_referencia": resultado["resposta_referencia"],
            "modelo_3b": resultado["llm_3b_sem_rag"]["modelo"],
            "resposta_3b": resultado["llm_3b_sem_rag"]["resposta"],
            "tempo_3b": resultado["llm_3b_sem_rag"]["tempo"],
            "bertscore_3b": resultado["llm_3b_sem_rag"]["bertscore"],
            "modelo_1b": resultado["llm_1b_com_rag"]["modelo"],
            "resposta_1b": resultado["llm_1b_com_rag"]["resposta"],
            "tempo_1b": resultado["llm_1b_com_rag"]["tempo"],
            "bertscore_1b": resultado["llm_1b_com_rag"]["bertscore"],
            "trechos_encontrados": resultado["llm_1b_com_rag"]["trechos_encontrados"]
        })
    
    df = pd.DataFrame(dados_csv)
    arquivo_csv = f"resultados/comparacao_resumida_{timestamp}.csv"
    df.to_csv(arquivo_csv, index=False, encoding="utf-8")
    
    print(f"\n💾 Resultados salvos:")
    print(f"   JSON completo: {arquivo_json}")
    print(f"   CSV resumido: {arquivo_csv}")

def criar_perguntas_teste():
    """Cria um conjunto de perguntas de teste sobre Machado de Assis"""
    perguntas = [
        "Quem é o protagonista de Dom Casmurro?",
        "Qual é o tema principal de Memórias Póstumas de Brás Cubas?",
        "Como Machado de Assis retrata a sociedade brasileira do século XIX?",
        "Quem é Capitu na obra Dom Casmurro?",
        "Qual é o estilo literário de Machado de Assis?",
        "O que significa o título 'Dom Casmurro'?",
        "Como é narrada a história em Memórias Póstumas de Brás Cubas?",
        "Quais são as principais características dos personagens de Machado de Assis?",
        "Qual é o contexto histórico das obras de Machado de Assis?",
        "Como Machado de Assis aborda o tema do ciúme em Dom Casmurro?"
    ]
    
    respostas_referencia = [
        "Bento Santiago, também conhecido como Dom Casmurro, é o protagonista e narrador da obra.",
        "O tema principal é a crítica à sociedade brasileira do século XIX através da narração póstuma de Brás Cubas.",
        "Machado de Assis retrata a sociedade brasileira do século XIX de forma crítica, mostrando as desigualdades sociais e a hipocrisia da elite.",
        "Capitu é a esposa de Bento Santiago e uma das personagens mais enigmáticas da literatura brasileira.",
        "Machado de Assis é conhecido pelo realismo psicológico e pela ironia refinada em suas obras.",
        "O título 'Dom Casmurro' se refere ao apelido dado a Bento Santiago, que significa pessoa fechada, introspectiva.",
        "A história é narrada em primeira pessoa por Brás Cubas, que já está morto, criando uma perspectiva única.",
        "Os personagens de Machado de Assis são complexos, com profundidade psicológica e características realistas.",
        "As obras de Machado de Assis se passam no Brasil do século XIX, durante o Segundo Império.",
        "O ciúme é retratado como um sentimento destrutivo que leva Bento Santiago a questionar a fidelidade de Capitu."
    ]
    
    return perguntas, respostas_referencia

def main():
    """Função principal"""
    print("🔬 AVALIADOR COMPARATIVO: LLM 3B vs LLM 1B + RAG")
    print("=" * 60)
    
    # Criar perguntas de teste
    perguntas, respostas_referencia = criar_perguntas_teste()
    
    print(f"📝 Total de perguntas de teste: {len(perguntas)}")
    
    # Executar comparação
    resultados = comparar_llms(perguntas, respostas_referencia)
    
    # Avaliar resultados
    resultados_avaliados = avaliar_resultados(resultados)
    
    # Salvar resultados
    salvar_resultados(resultados_avaliados)
    
    print("\n🎉 Avaliação comparativa concluída!")

if __name__ == "__main__":
    main()
