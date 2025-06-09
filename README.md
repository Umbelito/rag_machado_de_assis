# Teste Comparativo: LLM 3B vs LLM 1B + RAG

Este projeto implementa um teste comparativo entre duas abordagens de LLMs para responder perguntas sobre obras de Machado de Assis:

## 🎯 Objetivo

Comparar o desempenho de:
- **LLM de 3B parâmetros** (sem RAG) - geração pura baseada no conhecimento pré-treinado
- **LLM de 1B parâmetros** (com RAG) - geração baseada em contexto recuperado de um banco vetorial

## 🏗️ Arquitetura

### LLM de 3B (sem RAG)
- Modelo: `llama2:3b`
- Abordagem: Geração pura sem contexto adicional
- Prompt: Instrução simples para responder como especialista em literatura

### LLM de 1B (com RAG)
- Modelo: `tinyllama`
- Abordagem: Retrieval-Augmented Generation (RAG)
- Banco vetorial: FAISS com embeddings de trechos das obras
- Prompt: Contexto específico + pergunta

## 📁 Estrutura do Projeto

```
├── machado_rag.py          # Script principal do teste comparativo
├── utils.py               # Funções auxiliares
├── setup_models.py        # Script para configurar modelos
├── analisar_resultados.py # Script para analisar resultados
├── requirements.txt       # Dependências Python
├── README.md             # Este arquivo
├── index/                # Índice FAISS (gerado automaticamente)
├── embeddings/           # Metadados dos embeddings
├── resultados/           # Resultados das comparações (JSON)
└── analises/             # Análises exportadas (CSV)
```

## 🚀 Como Usar

### 1. Instalação das Dependências

```bash
# Instalar dependências Python
pip install -r requirements.txt

# Ou instalar manualmente
pip install faiss-cpu sentence-transformers requests numpy pandas
```

### 2. Configuração dos Modelos

```bash
# Executar script de configuração automática
python setup_models.py
```

Este script irá:
- Verificar se o Ollama está instalado
- Verificar se o servidor Ollama está rodando
- Baixar automaticamente os modelos necessários:
  - `llama2:3b` (LLM de 3B parâmetros)
  - `tinyllama` (LLM de 1B parâmetros)

### 3. Preparação dos Dados

Certifique-se de que as obras estão organizadas:
```
/home/umbelito/IPCC/obras/raw/txt/
├── romance/
│   ├── dom_casmurro.txt
│   ├── memorias_postumas.txt
│   └── ...
└── cronica/
    ├── cronica1.txt
    ├── cronica2.txt
    └── ...
```

### 4. Execução do Teste Comparativo

```bash
python machado_rag.py
```

### 5. Análise dos Resultados

```bash
python analisar_resultados.py
```

## 📊 Funcionalidades

### Script Principal (`machado_rag.py`)
- Comparação lado a lado das duas abordagens
- Medição de tempo de resposta
- Salvamento automático de resultados
- Interface interativa para perguntas

### Script de Setup (`setup_models.py`)
- Verificação automática do ambiente
- Download automático dos modelos
- Validação da configuração

### Script de Análise (`analisar_resultados.py`)
- Análise estatística dos tempos
- Categorização das perguntas
- Exportação para CSV
- Relatórios de performance

## 🔧 Configurações

No arquivo `machado_rag.py`, você pode ajustar:

```python
# Modelos
LLM_3B_MODEL = "llama2:3b"    # Modelo de 3B parâmetros
LLM_1B_MODEL = "tinyllama"    # Modelo de 1B parâmetros

# Configurações RAG
TOP_K = 5                     # Número de trechos relevantes
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo de embeddings
```

## 📈 Resultados

### Formato de Saída
Os resultados são salvos em JSON com:
- Timestamp da comparação
- Pergunta realizada
- Respostas de ambas as abordagens
- Tempos de resposta
- Metadados dos modelos

### Análises Disponíveis
- **Tempo de resposta**: Média, mínimo, máximo
- **Comparação de performance**: Qual abordagem é mais rápida
- **Categorização de perguntas**: Por tipo de conteúdo
- **Exportação para CSV**: Para análise detalhada

## 🎯 Casos de Uso

Este teste é útil para:
- Avaliar trade-offs entre tamanho do modelo e qualidade
- Comparar eficiência de RAG vs geração pura
- Analisar impacto do contexto específico na qualidade das respostas
- Benchmark de performance em tarefas específicas de literatura
- Pesquisa em sistemas de recuperação de informação

## 🔍 Exemplo de Saída

```
🔬 TESTE COMPARATIVO: LLM 3B vs LLM 1B + RAG
============================================================
📖 LLM de 3B parâmetros: llama2:3b (SEM RAG)
🔍 LLM de 1B parâmetros: tinyllama (COM RAG)
============================================================

🤔 Pergunta: Quem é o protagonista de Dom Casmurro?
================================================================================

📚 LLM de 3B parâmetros (SEM RAG):
--------------------------------------------------
⏱️  Tempo de resposta: 2.34s
💬 Resposta: [Resposta baseada no conhecimento pré-treinado]

🔍 LLM de 1B parâmetros (COM RAG):
--------------------------------------------------
⏱️  Tempo de resposta: 1.87s
📖 Trechos relevantes encontrados: 5
💬 Resposta: [Resposta baseada nos trechos recuperados]

📊 COMPARAÇÃO:
--------------------------------------------------
🕐 Tempo 3B (sem RAG): 2.34s
🕐 Tempo 1B (com RAG): 1.87s
⚡ Diferença: 0.47s
🏆 LLM de 1B com RAG foi mais rápida!
```

## 🛠️ Solução de Problemas

### Ollama não encontrado
```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### Servidor Ollama não está rodando
```bash
# Iniciar servidor
ollama serve
```

### Modelos não encontrados
```bash
# Executar setup automático
python setup_models.py
```

### Erro de dependências
```bash
# Reinstalar dependências
pip install -r requirements.txt --upgrade
```

## 📝 Notas Técnicas

- O índice FAISS é criado automaticamente na primeira execução
- Os embeddings são gerados usando o modelo `all-MiniLM-L6-v2`
- Os resultados são salvos automaticamente na pasta `resultados/`
- As análises são exportadas para CSV na pasta `analises/`