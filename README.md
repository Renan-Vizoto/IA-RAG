# Sistema de Análise de Consumo Energético Residencial

## 👥 Integrantes

| Nome | RA |
|------|-----------|
| Cristian Henrique Belo | 211648 |
| Fellipe Alessandro Scruph | 222378 |
| Gabriel Souza | 222709 |
| Guilherme Matheus Fonseca Waizbart | 211899 |
| Guilherme Romualdo de Oliveira Freitas | 211709 |
| João Victor de Oliveira | 212106 |
| Renan Vizoto Ferreira | 222220 |
| Vinicius Cândido das Chagas | 211873 |
| William Wurschig | 211657 |

---

## 📋 Assunto

O dataset escolhido e o **Energy consumption of the Netherlands** (Luca Basanisi), que contem dados de consumo de eletricidade e gas na Holanda, coletados por sete distribuidoras regionais (Enexis, Liander, Stedin, Enduris, Westlandinfra, Rendo e Coteq). Os dados sao agregados por faixa de CEP (minimo 10 conexoes para anonimizacao) e cobrem multiplos anos.

## 🎯 Domínio

**Eficiência energética residencial**

## 🎯 Objetivo

Desenvolver um sistema baseado em **Retrieval Augmented Generation (RAG)** capaz de consultar e analisar dados do estudo de consumo elétrico residencial, permitindo que usuários façam perguntas sobre:

- Padrões de uso de energia
- Comportamento de consumo
- Oportunidades de eficiência energética

## 🏗️ Arquitetura do Sistema

O sistema é composto por serviços orquestrados via Docker Compose, divididos em três responsabilidades principais: **ingestão de dados**, **pipeline ETL (Medallion)** e **interface de chat RAG**.

```mermaid
graph TB
    User([Usuário])

    subgraph Docker["Docker Compose"]

        subgraph API["API FastAPI — porta 3333"]
            Files["/files\nUpload e gerenciamento\nde documentos"]
            Chat["/chat/message\nLangChain Agent\n+ ferramenta de busca"]
        end

        subgraph Medallion["Pipeline Medallion — MinIO"]
            Bronze[(Bronze\nArquivos brutos\nPDFs, CSVs...)]
            Silver[(Silver\nTexto extraído\n.txt por página)]
            Gold[(Gold\nDados prontos\npara indexação)]
        end

        subgraph ETL["Workers ETL"]
            W1["bronze → silver\nExtração de texto\nPyPDF"]
            W2["silver → gold\nChunking + Embedding\nMiniLM-L12 384 dims"]
        end

        subgraph VectorDB["Banco Vetorial — Milvus :19530"]
            Collection[(Coleção\ntext_embedding\n384 dimensões\nbusca cosine)]
        end

        Ollama["Ollama — porta 11434\nQwen3.5 Unsloth\nLLM local"]
        MLflow["MLflow\nRastreamento de\nexperimentos de chat"]
        Attu["Attu — porta 3005\nUI de gestão\ndo Milvus"]
    end

    User -->|"POST /files — upload"| Files
    Files -->|"Armazena arquivo bruto"| Bronze

    Bronze -->|"Lê e extrai texto"| W1
    W1 -->|"Salva .txt"| Silver
    Silver -->|"Lê, chunka e embeda"| W2
    W2 -->|"Insere vetores"| Collection

    User -->|"POST /chat/message"| Chat
    Chat -->|"Embed da pergunta + busca vetorial"| Collection
    Chat -->|"Envia contexto + pergunta"| Ollama
    Ollama -->|"Resposta gerada"| Chat
    Chat -->|"Loga sessão e métricas"| MLflow

    Attu -.->|"Visualiza e gerencia"| Collection
```

### Fluxo de dados

1. **Ingestão** — O usuário faz upload de documentos (PDFs) via `/files`. Os arquivos são armazenados no bucket **Bronze** do MinIO.
2. **Bronze → Silver** — Um worker extrai o texto dos documentos (PyPDF) e salva como `.txt` no bucket **Silver**, uma página por arquivo.
3. **Silver → Gold** — Outro worker divide o texto em chunks (`CharacterTextSplitter`), gera embeddings de 384 dimensões com o modelo `paraphrase-multilingual-MiniLM-L12-v2` e indexa os vetores no **Milvus**. Os dados indexados representam a camada **Gold**.
4. **Chat RAG** — O usuário faz uma pergunta via `/chat/message`. O agente LangChain embeda a pergunta, realiza busca cosine no Milvus (top 3 resultados), monta o contexto e envia ao **Ollama** (Qwen3.5) para geração da resposta. Cada sessão é registrada no **MLflow**.

---

## 📌 Links Úteis

- **Dataset**: [Energy consumption of the Netherlands](https://www.kaggle.com/datasets/lucabasa/dutch-energy)
- **Trello**: [Watttrack - Projeto de Gestão](https://trello.com/invite/b/69aa0fb5e3b7f2d77018470d/ATTIf6ec617a20231c8f424b4a28aa2b04b5F5307FFE/watttrack)
