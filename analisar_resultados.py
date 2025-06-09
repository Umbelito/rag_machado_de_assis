#!/usr/bin/env python3
"""
Script para analisar os resultados das comparações entre LLMs
"""

import os
import json
import glob
from datetime import datetime
import pandas as pd

def carregar_resultados():
    """Carrega todos os resultados salvos"""
    resultados = []
    
    if not os.path.exists("resultados"):
        print("❌ Pasta 'resultados' não encontrada!")
        return resultados
    
    arquivos = glob.glob("resultados/comparacao_*.json")
    
    if not arquivos:
        print("❌ Nenhum arquivo de resultado encontrado!")
        return resultados
    
    print(f"📁 Encontrados {len(arquivos)} arquivos de resultado")
    
    for arquivo in arquivos:
        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                resultado = json.load(f)
                resultados.append(resultado)
        except Exception as e:
            print(f"❌ Erro ao carregar {arquivo}: {str(e)}")
    
    return resultados

def analisar_tempos(resultados):
    """Analisa os tempos de resposta"""
    print("\n⏱️  ANÁLISE DE TEMPOS DE RESPOSTA")
    print("=" * 50)
    
    if not resultados:
        print("❌ Nenhum resultado para analisar")
        return
    
    tempos_3b = []
    tempos_1b = []
    
    for resultado in resultados:
        tempo_3b = resultado.get("llm_3b_sem_rag", {}).get("tempo", 0)
        tempo_1b = resultado.get("llm_1b_com_rag", {}).get("tempo", 0)
        
        if tempo_3b > 0 and tempo_1b > 0:
            tempos_3b.append(tempo_3b)
            tempos_1b.append(tempo_1b)
    
    if not tempos_3b:
        print("❌ Nenhum dado de tempo válido encontrado")
        return
    
    print(f"📊 Total de comparações válidas: {len(tempos_3b)}")
    print(f"\n🕐 LLM 3B (sem RAG):")
    print(f"   Média: {sum(tempos_3b)/len(tempos_3b):.2f}s")
    print(f"   Mínimo: {min(tempos_3b):.2f}s")
    print(f"   Máximo: {max(tempos_3b):.2f}s")
    
    print(f"\n🕐 LLM 1B (com RAG):")
    print(f"   Média: {sum(tempos_1b)/len(tempos_1b):.2f}s")
    print(f"   Mínimo: {min(tempos_1b):.2f}s")
    print(f"   Máximo: {max(tempos_1b):.2f}s")
    
    # Comparação
    media_3b = sum(tempos_3b)/len(tempos_3b)
    media_1b = sum(tempos_1b)/len(tempos_1b)
    
    print(f"\n⚡ COMPARAÇÃO:")
    print(f"   Diferença média: {abs(media_3b - media_1b):.2f}s")
    
    if media_1b < media_3b:
        print(f"   🏆 LLM 1B com RAG é {((media_3b - media_1b)/media_3b)*100:.1f}% mais rápida!")
    else:
        print(f"   🏆 LLM 3B sem RAG é {((media_1b - media_3b)/media_1b)*100:.1f}% mais rápida!")

def analisar_perguntas(resultados):
    """Analisa as perguntas realizadas"""
    print("\n🤔 ANÁLISE DAS PERGUNTAS")
    print("=" * 50)
    
    if not resultados:
        print("❌ Nenhum resultado para analisar")
        return
    
    perguntas = [r.get("pergunta", "") for r in resultados if r.get("pergunta")]
    
    print(f"📝 Total de perguntas: {len(perguntas)}")
    
    # Categorizar perguntas
    categorias = {
        "personagens": 0,
        "enredo": 0,
        "estilo": 0,
        "contexto": 0,
        "outros": 0
    }
    
    palavras_chave = {
        "personagens": ["quem", "personagem", "protagonista", "antagonista", "nome"],
        "enredo": ["história", "enredo", "acontece", "final", "começo", "trama"],
        "estilo": ["estilo", "escreve", "linguagem", "técnica", "literário"],
        "contexto": ["época", "século", "histórico", "social", "brasil", "rio"]
    }
    
    for pergunta in perguntas:
        pergunta_lower = pergunta.lower()
        categorizada = False
        
        for categoria, palavras in palavras_chave.items():
            if any(palavra in pergunta_lower for palavra in palavras):
                categorias[categoria] += 1
                categorizada = True
                break
        
        if not categorizada:
            categorias["outros"] += 1
    
    print("\n📊 Categorias de perguntas:")
    for categoria, count in categorias.items():
        if count > 0:
            porcentagem = (count / len(perguntas)) * 100
            print(f"   {categoria.title()}: {count} ({porcentagem:.1f}%)")

def exportar_para_csv(resultados):
    """Exporta os resultados para CSV"""
    print("\n📊 EXPORTANDO PARA CSV")
    print("=" * 50)
    
    if not resultados:
        print("❌ Nenhum resultado para exportar")
        return
    
    dados = []
    
    for resultado in resultados:
        dados.append({
            "timestamp": resultado.get("timestamp", ""),
            "pergunta": resultado.get("pergunta", ""),
            "modelo_3b": resultado.get("llm_3b_sem_rag", {}).get("modelo", ""),
            "resposta_3b": resultado.get("llm_3b_sem_rag", {}).get("resposta", ""),
            "tempo_3b": resultado.get("llm_3b_sem_rag", {}).get("tempo", 0),
            "modelo_1b": resultado.get("llm_1b_com_rag", {}).get("modelo", ""),
            "resposta_1b": resultado.get("llm_1b_com_rag", {}).get("resposta", ""),
            "tempo_1b": resultado.get("llm_1b_com_rag", {}).get("tempo", 0)
        })
    
    df = pd.DataFrame(dados)
    
    # Criar pasta se não existir
    os.makedirs("analises", exist_ok=True)
    
    # Salvar CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_csv = f"analises/resultados_comparacao_{timestamp}.csv"
    
    df.to_csv(arquivo_csv, index=False, encoding="utf-8")
    print(f"✅ Resultados exportados para: {arquivo_csv}")
    
    return arquivo_csv

def mostrar_estatisticas_gerais(resultados):
    """Mostra estatísticas gerais dos resultados"""
    print("\n📈 ESTATÍSTICAS GERAIS")
    print("=" * 50)
    
    if not resultados:
        print("❌ Nenhum resultado para analisar")
        return
    
    print(f"📅 Período de testes:")
    timestamps = [r.get("timestamp", "") for r in resultados if r.get("timestamp")]
    
    if timestamps:
        timestamps.sort()
        print(f"   Primeiro teste: {timestamps[0]}")
        print(f"   Último teste: {timestamps[-1]}")
    
    print(f"\n📊 Total de comparações: {len(resultados)}")
    
    # Contar modelos usados
    modelos_3b = set()
    modelos_1b = set()
    
    for resultado in resultados:
        modelo_3b = resultado.get("llm_3b_sem_rag", {}).get("modelo", "")
        modelo_1b = resultado.get("llm_1b_com_rag", {}).get("modelo", "")
        
        if modelo_3b:
            modelos_3b.add(modelo_3b)
        if modelo_1b:
            modelos_1b.add(modelo_1b)
    
    print(f"\n🤖 Modelos utilizados:")
    print(f"   LLM 3B: {', '.join(modelos_3b) if modelos_3b else 'Nenhum'}")
    print(f"   LLM 1B: {', '.join(modelos_1b) if modelos_1b else 'Nenhum'}")

def main():
    """Função principal"""
    print("📊 ANALISADOR DE RESULTADOS - TESTE COMPARATIVO LLMs")
    print("=" * 60)
    
    # Carregar resultados
    resultados = carregar_resultados()
    
    if not resultados:
        print("\n❌ Nenhum resultado encontrado para análise.")
        print("💡 Execute primeiro o script machado_rag.py para gerar resultados.")
        return
    
    # Análises
    mostrar_estatisticas_gerais(resultados)
    analisar_tempos(resultados)
    analisar_perguntas(resultados)
    
    # Exportar para CSV
    arquivo_csv = exportar_para_csv(resultados)
    
    print(f"\n🎉 Análise concluída!")
    if arquivo_csv:
        print(f"📄 Dados detalhados disponíveis em: {arquivo_csv}")

if __name__ == "__main__":
    main() 