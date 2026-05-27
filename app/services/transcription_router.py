from pathlib import Path

from app.config import TranscriptionConfig
from app.services.transcription import (
    TranscriptionResult,
    TranscriptionService,
    UnsupportedModelSizeError,
)


class TranscriptionRouterService:
    def __init__(
        self,
        config: TranscriptionConfig,
        backends: dict[str, TranscriptionService],
    ) -> None:
        self._model_backends = {
            model.id: model.backend for model in config.supported_model_sizes
        }
        self._backends = backends

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        backend_name = self._model_backends.get(model_size)
        if backend_name is None:
            raise UnsupportedModelSizeError(f"Unsupported model size: {model_size}")

        backend = self._backends.get(backend_name)
        if backend is None:
            raise UnsupportedModelSizeError(
                f"No transcription backend configured for model size: {model_size}"
            )

        return backend.transcribe(audio_path, language, model_size)
