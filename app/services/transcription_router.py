from pathlib import Path
from threading import Condition

from app.config import TranscriptionConfig
from app.services.transcription import (
    TranscriptionError,
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
        self._active_backend_name: str | None = None
        self._active_calls = 0
        self._switching = False
        self._lifecycle_condition = Condition()

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

        previous_backend_name: str | None = None
        with self._lifecycle_condition:
            while self._switching or (
                self._active_backend_name not in {None, backend_name}
                and self._active_calls > 0
            ):
                self._lifecycle_condition.wait()

            if self._active_backend_name != backend_name:
                self._switching = True
                previous_backend_name = self._active_backend_name

        if previous_backend_name is not None:
            try:
                self._backends[previous_backend_name].release()
            except Exception as exc:
                with self._lifecycle_condition:
                    self._switching = False
                    self._lifecycle_condition.notify_all()
                raise TranscriptionError(
                    "Failed to release transcription backend "
                    f"{previous_backend_name}: {exc}"
                ) from exc

        with self._lifecycle_condition:
            self._active_backend_name = backend_name
            self._switching = False
            self._active_calls += 1
            self._lifecycle_condition.notify_all()

        try:
            return backend.transcribe(audio_path, language, model_size)
        finally:
            with self._lifecycle_condition:
                self._active_calls -= 1
                self._lifecycle_condition.notify_all()
