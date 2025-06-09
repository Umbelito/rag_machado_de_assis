import pandas as pd
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

def calcular_bertscore(respostas_geradas, referencias):
    P, R, F1 = bertscore(respostas_geradas, referencias, lang="pt")
    return F1.tolist()

def preparar_faiss_index(contextos):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(contextos, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, model

def calcular_recall_at_k(perguntas, referencias, index, model, contextos, k=5):
    recalls = []
    for pergunta, referencia in tqdm(zip(perguntas, referencias), total=len(perguntas), desc="Calculando Recall@K"):
        query_embed = model.encode([referencia])
        D, I = index.search(query_embed, k)
        recuperados = [contextos[i] for i in I[0]]
        found = any(referencia in r for r in recuperados)
        recalls.append(1 if found else 0)
    return recalls

def avaliar_rag(path_csv):
    df = pd.read_csv(path_csv)

    bert_scores = calcular_bertscore(
        df['resposta_gerada'].tolist(),
        df['resposta_referencia'].tolist()
    )

    index, embeddings, model = preparar_faiss_index(df['contexto_recuperado'].tolist())

    recall_k = calcular_recall_at_k(
        df['pergunta'].tolist(),
        df['resposta_referencia'].tolist(),
        index,
        model,
        df['contexto_recuperado'].tolist(),
        k=5
    )

    df['bertscore_f1'] = bert_scores
    df['recall@5'] = recall_k

    media_bertscore = sum(bert_scores) / len(bert_scores)
    media_recall = sum(recall_k) / len(recall_k)

    print("\n📝 Resultados Médios:")
    print(f"→ Média BERTScore (F1): {media_bertscore:.4f}")
    print(f"→ Média Recall@5:       {media_recall:.2%}")

    df.to_csv("avaliacao_com_resultados.csv", index=False)
    print("\n✅ Avaliação salva em: avaliacao_com_resultados.csv")

if __name__ == "__main__":
    avaliar_rag("avaliacao_contexto_vazio.csv")
