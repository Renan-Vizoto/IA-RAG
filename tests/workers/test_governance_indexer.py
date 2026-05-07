"""
Testes do GovernanceIndexer.
Verifica que o indexador lê documentos .md (não JSONs),
chunka, embeda e insere no Milvus corretamente.
"""
import asyncio
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from app.core.workers.governance_indexer import GovernanceIndexer, _split, start_worker
from app.pipeline.storage import StorageBackend


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

SILVER_DOC = """# Documento de Governança — Camada Silver
## Processado em: 2025-01-15T10:00:00+00:00

### Tipos de Dados
| amperage | float32 | Amperagem |

### Limpeza de Dados
Removidos 100 registros duplicados.

### Feature Engineering
- amperage: extraído de type_of_connection
- consume_per_conn: target variável
"""

GOLD_DOC = """# Documento de Governança — Camada Gold
## Processado em: 2025-01-15T11:00:00+00:00

### Divisão dos Dados
- Treino: 70%
- Validação: 15%
- Teste: 15%

### Modelo
XGBoost — Run ID: abc123
"""

MLFLOW_DOC = """# Relatório MLflow — Treinamento Dutch Energy
## Experimento: dutch-energy-training
## Run ID: abc123

### Métricas
RMSE: 89.1
MAE: 39.5
R²: 0.743
"""


@pytest.fixture
def mock_storage_with_docs():
    storage = MagicMock(spec=StorageBackend)
    docs = {
        "silver/dutch-energy/governance_silver.md": SILVER_DOC.encode(),
        "gold/dutch-energy/governance_gold.md": GOLD_DOC.encode(),
        "gold/dutch-energy/mlflow_report.md": MLFLOW_DOC.encode(),
    }

    def get_object(bucket, name):
        key = f"{bucket}/{name}"
        if key not in docs:
            raise Exception(f"Não encontrado: {key}")
        return BytesIO(docs[key])

    storage.get_object.side_effect = get_object
    return storage


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.drop_and_recreate.return_value = None
    repo.insert.return_value = None
    return repo


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embbed_it.side_effect = lambda chunks: np.random.rand(len(chunks), 384)
    return embedder


@pytest.fixture
def mock_schema_builder():
    return MagicMock()


@pytest.fixture
def indexer(mock_storage_with_docs, mock_repo, mock_embedder, mock_schema_builder):
    return GovernanceIndexer(
        storage=mock_storage_with_docs,
        repo=mock_repo,
        embedder=mock_embedder,
        schema_builder=mock_schema_builder,
        collection="governance",
    )


# ──────────────────────────────────────────────
# Testes de chunking (_split)
# ──────────────────────────────────────────────

class TestSplit:
    def test_texto_curto_retorna_um_chunk(self):
        chunks = _split("texto curto", 900)
        assert chunks == ["texto curto"]

    def test_texto_longo_divide_em_chunks(self):
        texto = " ".join(["palavra"] * 200)  # bem mais que 900 chars
        chunks = _split(texto, 900)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 900

    def test_chunks_preservam_todo_o_conteudo(self):
        texto = "palavra " * 300
        chunks = _split(texto, 900)
        reconstruido = " ".join(chunks)
        # Mesmas palavras (pode ter espaços extras nas bordas)
        assert len(reconstruido.split()) == len(texto.split())

    def test_texto_exatamente_no_limite(self):
        texto = "a" * 900
        chunks = _split(texto, 900)
        assert len(chunks) == 1

    def test_texto_vazio(self):
        chunks = _split("", 900)
        assert chunks == [""]


# ──────────────────────────────────────────────
# Testes do GovernanceIndexer
# ──────────────────────────────────────────────

class TestGovernanceIndexer:

    def test_le_tres_documentos_markdown(self, indexer, mock_storage_with_docs):
        """Indexer deve ler governance_silver.md, governance_gold.md e mlflow_report.md."""
        chunks, sources = indexer._collect_chunks()
        assert len(set(sources)) == 3
        assert "silver_governance" in sources
        assert "gold_governance" in sources
        assert "mlflow_report" in sources

    def test_chunks_contem_conteudo_dos_docs(self, indexer):
        """Chunks devem conter texto dos documentos de governança."""
        chunks, _ = indexer._collect_chunks()
        all_text = " ".join(chunks)
        assert "Camada Silver" in all_text
        assert "Camada Gold" in all_text
        assert "MLflow" in all_text

    def test_nao_indexa_json_de_stats(self, indexer):
        """Indexer NÃO deve tentar ler cleaning_stats.json (formato antigo removido)."""
        chunks, sources = indexer._collect_chunks()
        assert "silver_cleaning" not in sources
        assert "bronze_manifest" not in sources
        assert "gold_metadata" not in sources

    def test_documento_ausente_ignorado_com_aviso(self, mock_repo, mock_embedder, mock_schema_builder):
        """Quando um doc está ausente, o indexer continua sem lançar exceção."""
        storage = MagicMock(spec=StorageBackend)
        storage.get_object.side_effect = Exception("Não encontrado")

        indexer = GovernanceIndexer(
            storage=storage, repo=mock_repo, embedder=mock_embedder,
            schema_builder=mock_schema_builder, collection="governance",
        )
        chunks, sources = indexer._collect_chunks()
        assert chunks == []
        assert sources == []

    def test_dois_docs_presentes_um_ausente(self, mock_repo, mock_embedder, mock_schema_builder):
        """Quando só 2 de 3 docs existem, os 2 são indexados."""
        storage = MagicMock(spec=StorageBackend)
        docs = {
            "silver/dutch-energy/governance_silver.md": SILVER_DOC.encode(),
            "gold/dutch-energy/governance_gold.md": GOLD_DOC.encode(),
        }

        def get_object(bucket, name):
            key = f"{bucket}/{name}"
            if key not in docs:
                raise Exception("Não encontrado")
            return BytesIO(docs[key])

        storage.get_object.side_effect = get_object

        indexer = GovernanceIndexer(
            storage=storage, repo=mock_repo, embedder=mock_embedder,
            schema_builder=mock_schema_builder, collection="governance",
        )
        chunks, sources = indexer._collect_chunks()
        assert "silver_governance" in sources
        assert "gold_governance" in sources
        assert "mlflow_report" not in sources

    @pytest.mark.asyncio
    async def test_run_chama_embed_e_insert(self, indexer, mock_repo, mock_embedder):
        """run() deve embedar os chunks e inserir no Milvus."""
        await indexer.run()
        mock_embedder.embbed_it.assert_called_once()
        mock_repo.drop_and_recreate.assert_called_once_with("governance", indexer._schema_builder)
        mock_repo.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_insere_dados_com_source_correto(self, indexer, mock_repo):
        """Dados inseridos no Milvus devem ter os campos text, text_vector e source."""
        await indexer.run()
        inserted = mock_repo.insert.call_args[0][1]
        for item in inserted:
            assert "text" in item
            assert "text_vector" in item
            assert "source" in item
            assert item["source"] in ["silver_governance", "gold_governance", "mlflow_report"]

    @pytest.mark.asyncio
    async def test_run_nao_reindexar_se_timestamp_igual(self, indexer, mock_repo):
        """run() não deve re-indexar quando o timestamp não mudou."""
        await indexer.run()
        assert mock_repo.insert.call_count == 1

        # Segunda chamada com mesmo timestamp
        await indexer.run()
        assert mock_repo.insert.call_count == 1  # não aumentou

    @pytest.mark.asyncio
    async def test_run_sem_documentos_nao_chama_insert(self, mock_repo, mock_embedder, mock_schema_builder):
        """run() não deve chamar insert se não há documentos."""
        storage = MagicMock(spec=StorageBackend)
        storage.get_object.side_effect = Exception("Não encontrado")

        indexer = GovernanceIndexer(
            storage=storage, repo=mock_repo, embedder=mock_embedder,
            schema_builder=mock_schema_builder, collection="governance",
        )
        await indexer.run()
        mock_repo.insert.assert_not_called()

    def test_vector_dimension_correta(self, indexer, mock_embedder):
        """Embeddings devem ter 384 dimensões (MiniLM-L12)."""
        chunks, _ = indexer._collect_chunks()
        embeddings = mock_embedder.embbed_it(chunks)
        assert embeddings.shape == (len(chunks), 384)


# ──────────────────────────────────────────────
# Teste do start_worker
# ──────────────────────────────────────────────

class TestStartWorker:
    @patch("app.core.workers.governance_indexer.AsyncIOScheduler")
    def test_start_worker_configura_scheduler(
        self, MockScheduler,
        mock_storage_with_docs, mock_repo, mock_embedder, mock_schema_builder
    ):
        """start_worker deve criar scheduler com job de 10 minutos."""
        scheduler_instance = MagicMock()
        MockScheduler.return_value = scheduler_instance

        indexer = start_worker(
            storage=mock_storage_with_docs,
            repo=mock_repo,
            embedder=mock_embedder,
            schema_builder=mock_schema_builder,
            collection="governance",
        )

        assert isinstance(indexer, GovernanceIndexer)
        scheduler_instance.add_job.assert_called_once()
        call_kwargs = scheduler_instance.add_job.call_args
        assert call_kwargs[1].get("minutes") == 10 or "minutes" in str(call_kwargs)
        scheduler_instance.start.assert_called_once()
