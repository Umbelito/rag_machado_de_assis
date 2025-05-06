# Descrição do Projeto

Este projeto é uma implementação de um sistema de recuperação de respostas (RAG - Retrieval-Augmented Generation) que utiliza embeddings de texto para responder perguntas baseadas em documentos de texto. Ele combina a recuperação de informações com a geração de respostas utilizando modelos de linguagem. 

## Estrutura do Projeto

O projeto contém três scripts principais:

1. **machado_rag.py**: O script principal que carrega ou cria um índice FAISS a partir dos documentos de texto e permite que o usuário faça perguntas, gerando respostas com base em trechos semelhantes encontrados nos documentos.

2. **utils.py**: Um módulo utilitário que contém funções para dividir textos, carregar documentos, criar embeddings e buscar trechos similares utilizando o modelo de embeddings.

3. **avaliador_rag.py**: Um script para avaliar o desempenho do sistema de RAG utilizando métricas como BERTScore e Recall@K. Ele lê um arquivo CSV com perguntas e respostas de referência, calcula os resultados e salva em um novo arquivo CSV.

<!-- ## Pré-requisitos -->

<!-- Antes de executar os scripts, certifique-se de ter as seguintes bibliotecas instaladas:

- `faiss`
- `numpy`
- `pandas`
- `requests`
- `sentence-transformers`
- `bert-score`
- `tqdm`

Você pode instalar as dependências necessárias com o seguinte comando:

```bash
pip install faiss-cpu numpy pandas requests sentence-transformers bert-score tqdm
``` -->

## Configuração

Antes de executar o *machado_rag.py*, você deve configurar as seguintes variáveis no código:

- `DATA_DIR`: O diretório onde os documentos de texto estão armazenados.
- `CATEGORIAS`: As categorias de documentos que você deseja carregar.
- `EMBEDDING_MODEL`: O modelo de embeddings que você deseja usar (ex: "all-MiniLM-L6-v2").
- `INDEX_PATH`: O caminho onde o índice FAISS será salvo.
- `METADATA_PATH`: O caminho onde os metadados dos documentos serão salvos.
- `OLLAMA_URL`: A URL do serviço de geração de respostas.
- `LLM_MODEL`: O modelo de linguagem que você deseja utilizar.
- `TOP_K`: O número de trechos a serem recuperados para gerar a resposta.

## Como Executar

1. Para iniciar o sistema de RAG, execute o script *machado_rag.py*:

```bash
python machado_rag.py
```

2. Você será solicitado a inserir uma pergunta. Digite sua pergunta e pressione Enter. Para sair, digite 'sair'.

## Avaliação do Modelo

Para avaliar o desempenho do modelo, use o script *avaliador_rag.py.* Certifique-se de que você tenha um arquivo CSV chamado avaliacao.csv com as colunas apropriadas:

- `pergunta`: A pergunta feita.
- `resposta_gerada`: A resposta gerada pelo modelo.
- `resposta_referencia`: A resposta correta ou de referência.
- `contexto_recuperado`: O contexto que foi recuperado para a resposta.

Execute o script da seguinte forma:

```bash
python avaliador_rag.py
```

Os resultados da avaliação serão salvos em um arquivo chamado *avaliacao_com_resultados.csv*.