import json
import logging
import threading
import urllib.error
import urllib.request

from app.infrastructure.configs import settings

logger = logging.getLogger(__name__)


class OllamaModelManager:
    """Garante no máximo um modelo carregado na VRAM; unload só ao trocar."""

    def __init__(self, base_url: str | None = None):
        self._base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self._active_model: str | None = None
        self._lock = threading.Lock()

    @property
    def active_model(self) -> str | None:
        return self._active_model

    def _preload(self, model: str) -> None:
        payload = json.dumps({
            "model": model,
            "prompt": "warmup",
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        }).encode()
        request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                response.read()
            logger.info("[OLLAMA] Modelo pré-carregado na VRAM: %s", model)
        except urllib.error.URLError as exc:
            logger.warning("[OLLAMA] Falha ao pré-carregar %s: %s", model, exc)
            raise

    def ensure_loaded(self, model: str) -> None:
        with self._lock:
            if self._active_model == model:
                return
            if self._active_model:
                self.unload(self._active_model)
            self._preload(model)
            self._active_model = model

    def unload(self, model: str) -> None:
        if not model:
            return

        payload = json.dumps({"model": model, "prompt": "", "keep_alive": 0}).encode()
        request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
            logger.info("[OLLAMA] Modelo descarregado da VRAM: %s", model)
        except urllib.error.URLError as exc:
            logger.warning("[OLLAMA] Falha ao descarregar %s: %s", model, exc)


model_manager = OllamaModelManager()
