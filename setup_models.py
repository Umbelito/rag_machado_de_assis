#!/usr/bin/env python3
"""
Script para configurar os modelos necessários para o teste comparativo
"""

import subprocess
import sys
import time

def run_command(command, description):
    """Executa um comando e mostra o progresso"""
    print(f"\n🔄 {description}...")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} concluído com sucesso!")
            return True
        else:
            print(f"❌ Erro ao {description.lower()}:")
            print(f"Erro: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exceção ao {description.lower()}: {str(e)}")
        return False

def check_ollama():
    """Verifica se o Ollama está instalado e rodando"""
    print("🔍 Verificando se o Ollama está instalado...")
    
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama encontrado: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama não encontrado no PATH")
            return False
    except FileNotFoundError:
        print("❌ Ollama não está instalado")
        return False

def check_ollama_server():
    """Verifica se o servidor Ollama está rodando"""
    print("🔍 Verificando se o servidor Ollama está rodando...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor Ollama está rodando!")
            return True
        else:
            print("❌ Servidor Ollama não está respondendo corretamente")
            return False
    except Exception as e:
        print(f"❌ Erro ao conectar com servidor Ollama: {str(e)}")
        return False

def install_models():
    """Instala os modelos necessários"""
    models = [
        ("llama2:3b", "LLM de 3B parâmetros (sem RAG)"),
        ("tinyllama", "LLM de 1B parâmetros (com RAG)")
    ]
    
    print("\n📦 Instalando modelos necessários...")
    print("=" * 60)
    
    for model, description in models:
        print(f"\n📚 {description}")
        print(f"Modelo: {model}")
        
        success = run_command(f"ollama pull {model}", f"Baixando {model}")
        
        if success:
            print(f"✅ {model} instalado com sucesso!")
        else:
            print(f"❌ Falha ao instalar {model}")
            return False
        
        # Pequena pausa entre downloads
        time.sleep(1)
    
    return True

def list_installed_models():
    """Lista os modelos instalados"""
    print("\n📋 Modelos instalados:")
    print("-" * 30)
    
    try:
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Erro ao listar modelos")
    except Exception as e:
        print(f"❌ Erro: {str(e)}")

def main():
    """Função principal"""
    print("🔧 CONFIGURAÇÃO DOS MODELOS PARA TESTE COMPARATIVO")
    print("=" * 60)
    
    # Verificar Ollama
    if not check_ollama():
        print("\n❌ Ollama não está instalado.")
        print("📥 Instale o Ollama em: https://ollama.ai/")
        return False
    
    # Verificar servidor
    if not check_ollama_server():
        print("\n❌ Servidor Ollama não está rodando.")
        print("🚀 Inicie o servidor com: ollama serve")
        return False
    
    # Instalar modelos
    if not install_models():
        print("\n❌ Falha na instalação dos modelos.")
        return False
    
    # Listar modelos
    list_installed_models()
    
    print("\n🎉 Configuração concluída!")
    print("✅ Você pode agora executar: python machado_rag.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 