from unittest.mock import MagicMock, patch

from app.infrastructure.clients.ollama_model_manager import OllamaModelManager


class TestOllamaModelManager:

    def test_ensure_loaded_mesmo_modelo_nao_descarrega(self):
        manager = OllamaModelManager(base_url="http://ollama:11434")
        manager._active_model = "qwen3.5-0.8b-unsloth"

        with patch.object(manager, "unload") as mock_unload:
            manager.ensure_loaded("qwen3.5-0.8b-unsloth")
            mock_unload.assert_not_called()

        assert manager.active_model == "qwen3.5-0.8b-unsloth"

    def test_ensure_loaded_troca_modelo_descarrega_anterior(self):
        manager = OllamaModelManager(base_url="http://ollama:11434")
        manager._active_model = "qwen3.5-0.8b-unsloth"

        with patch.object(manager, "unload") as mock_unload:
            manager.ensure_loaded("gemma4-unsloth")
            mock_unload.assert_called_once_with("qwen3.5-0.8b-unsloth")

        assert manager.active_model == "gemma4-unsloth"

    @patch("app.infrastructure.clients.ollama_model_manager.urllib.request.urlopen")
    def test_unload_chama_api_generate(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value = MagicMock(read=MagicMock(return_value=b""))

        manager = OllamaModelManager(base_url="http://ollama:11434")
        manager.unload("gemma4-unsloth")

        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://ollama:11434/api/generate"
        assert b'"keep_alive": 0' in request.data
