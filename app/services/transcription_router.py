from collections import deque
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
        self._pending_requests: deque[tuple[object, str]] = deque()
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

        request_ticket = object()
        previous_backend_name: str | None = None
        switch_owner = False
        with self._lifecycle_condition:
            if (
                not self._switching
                and not self._pending_requests
                and self._active_backend_name == backend_name
            ):
                self._active_calls += 1
            else:
                self._pending_requests.append((request_ticket, backend_name))
                self._lifecycle_condition.notify_all()
                while True:
                    owns_next_request = (
                        self._pending_requests
                        and self._pending_requests[0][0] is request_ticket
                    )
                    if owns_next_request and not self._switching:
                        if self._active_backend_name == backend_name:
                            self._pending_requests.popleft()
                            self._active_calls += 1
                            self._lifecycle_condition.notify_all()
                            break
                        if self._active_calls == 0:
                            self._pending_requests.popleft()
                            self._switching = True
                            switch_owner = True
                            previous_backend_name = self._active_backend_name
                            break
                    self._lifecycle_condition.wait()

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

        if switch_owner:
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
